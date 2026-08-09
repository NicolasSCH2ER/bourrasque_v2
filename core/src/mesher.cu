/* mesher.cu -- champ de distance signee de Zhu & Bridson (2005), pour la
 * reconstruction de surface de fluide (M7/T1). PAS de marching cubes ici :
 * seulement le champ, verifiable independamment de la polygonisation.
 *
 * Unite de compilation AUTONOME : CUDA_SEPARABLE_COMPILATION est OFF pour
 * la cible (voir CMakeLists.txt), donc aucun symbole __device__ ne peut
 * traverser mlsmpm.cu et mesher.cu. Les quelques helpers vectoriels dont ce
 * fichier a besoin sont donc redefinis ici plutot que partages -- une
 * vingtaine de lignes dupliquees, prix assume (cf. spec T1).
 *
 * Le mailleur est INDEPENDANT de BqSim (D7 du plan M7) : il consomme un
 * nuage de positions hote, d'ou qu'il vienne (etat vivant du solveur ou
 * cache .bqd relu), jamais un BqSim directement.
 *
 * phi(x) = |x - x_moy(x)| - r_moy(x)
 *   x_moy = somme(w_i x_i) / somme(w_i)
 *   r_moy = somme(w_i r_i) / somme(w_i)
 *   w_i   = k(|x - x_i| / R),   k(s) = max(0, (1 - s^2)^3)
 *
 * FORMULATION EN GATHER (decision D2, non negociable) : chaque cellule du
 * champ parcourt les particules des buckets voisins et accumule ses sommes
 * en REGISTRES, puis ecrit un seul float. Aucun tampon d'accumulateurs a
 * l'echelle de la grille -- un scatter multiplierait l'empreinte memoire
 * par 6 (cf. plan-milestone-7.md, tableau resolution/empreinte).
 *
 * Bucketing des particules : meme principe qu'en M6 pour le champ de
 * distance des colliders (cf. mlsmpm.cu, BucketGridHost / BQ_BUCKET_DX_MULT
 * / build_bucket_grid), avec une simplification qui lui est propre. Le pas
 * de bucket vaut exactement R (influence_radius) : toute particule a moins
 * de R d'un point est donc necessairement dans l'un des 27 buckets du
 * voisinage 3x3x3 du bucket contenant ce point -- pas besoin de recherche
 * par anneaux croissants comme pour les colliders (qui doivent trouver le
 * TRIANGLE LE PLUS PROCHE, une requete non bornee par un rayon fixe). La
 * grille de buckets couvre le domaine du champ de maillage lui-meme
 * (origine (0,0,0), etendue grid_res*cell_size), et non l'AABB des
 * particules : sa resolution est donc entierement determinee par la
 * config, connue des bq_mesher_create -- c'est ce qui permet a
 * bq_mesher_vram_estimate de calculer une empreinte exacte sans particules.
 *
 * Comme pour bq_set_colliders, les positions arrivent en memoire HOTE
 * (memoire ctypes cote appelant) : le CSR de bucketing est donc construit
 * cote hote (comptage, somme prefixe, remplissage), puis televerse -- pas
 * besoin de cub::DeviceScan ici, la construction hote est deja largement
 * sous la milliseconde pour les tailles de nuages visees et evite un
 * aller-retour device pour trier ce qui est deja disponible cote hote.
 */
#define BQ_BUILD
#include "bourrasque.h"
#include "internal.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

/* ------------------------------------------------------------- petite algebre
 * (dupliquee depuis mlsmpm.cu, cf. note d'autonomie en tete de fichier) */
__host__ __device__ inline float3 vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float vlen(float3 a) {
    return sqrtf(vdot(a, a));
}
__host__ __device__ inline float3 vcross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

/* --------------------------------------------------- constantes de fichier
 * (M9/T1, noyau anisotrope -- plan-milestone-9.md D2/D6). Valeurs choisies a
 * partir des percentiles mesures dans le diagnostic de la session qui a
 * introduit ce jalon (p10 = 0 voisin, mediane 5), PAS calibrees formellement
 * -- internes, non exposees dans BqMesherConfig (D6 : aucun changement
 * d'ABI). */
/* constexpr, pas static const : un static const scalaire n'est PAS
 * automatiquement visible en code __device__/__global__ sous nvcc (constate
 * a la compilation -- k_r seul a leve une erreur de lien explicite, k_n_aniso
 * et k_n_cull "passaient" par une tolerance du compilateur pour les entiers,
 * fragile et non garantie). constexpr est la forme portable standard,
 * utilisable des deux cotes sans qualificatif __device__ separe. */
constexpr int   k_n_aniso = 6;  /* seuil de voisinage (D1/D2) : en dessous, repli isotrope */
constexpr int   k_n_cull  = 1;  /* seuil de voisinage (D2/D3) : en dessous ou egal, particule exclue du CSR filtre */
constexpr float k_r       = 4.f; /* facteur de regularisation des petites valeurs propres (D1 etape 3), valeur indicative de Yu & Turk non recalibree pour ce projet */

/* --------------------------------------------------------- grille de buckets
 * Origine fixe (0,0,0), pas h = influence_radius, resolution deduite du
 * domaine du champ (grid_res * cell_size) -- entierement determinee par la
 * config, cf. note en tete de fichier. */
struct BucketGrid {
    float3 origin;
    float  h;
    int3   res;
};

static BucketGrid compute_bucket_grid(const BqMesherConfig& cfg) {
    BucketGrid g;
    g.origin = make_float3(0.f, 0.f, 0.f);
    float h = cfg.influence_radius > 1e-8f ? cfg.influence_radius : 1e-8f;
    g.h = h;
    float dom_x = cfg.grid_res[0] * cfg.cell_size;
    float dom_y = cfg.grid_res[1] * cfg.cell_size;
    float dom_z = cfg.grid_res[2] * cfg.cell_size;
    g.res.x = std::max(1, (int)ceilf(dom_x / h));
    g.res.y = std::max(1, (int)ceilf(dom_y / h));
    g.res.z = std::max(1, (int)ceilf(dom_z / h));
    return g;
}

/* Empreinte independante des particules pour une resolution cubique donnee
 * (meme formule que bq_mesher_vram_estimate, cfg.grid_res remplace par
 * {res,res,res}) -- sert au message du garde-fou VRAM (resolution maximale
 * qui tiendrait sur la carte presente). */
static int64_t indep_bytes_for_cubic_res(int res, float cell_size,
                                          float influence_radius) {
    int64_t n_cells = (int64_t)res * res * res;
    float h = influence_radius > 1e-8f ? influence_radius : 1e-8f;
    float dom = res * cell_size;
    int bres = std::max(1, (int)ceilf(dom / h));
    int64_t n_buckets = (int64_t)bres * bres * bres;
    int64_t n_edges = 3 * n_cells;
    int64_t c = std::max(0, res - 1);
    int64_t n_cubes = c * c * c;
    /* Champ + tampon de lissage + buckets + marching cubes (D4). Le marching
     * cubes demande deux tableaux d'entiers sur les ARETES (3 par cellule :
     * drapeau et scan) et deux sur les cubes, soit 32 octets par cellule --
     * quatre fois le champ+tampon (2 * 4 octets). C'est donc lui qui borne
     * la resolution atteignable, ce que la premiere redaction du plan
     * sous-estimait. */
    return n_cells * 4 /* champ */ + n_cells * 4 /* tampon de lissage */ +
           n_buckets * 4 + n_buckets * 4 /* d_bucket_off2, M9/T1 */ +
           (n_edges + 1) * 8 + (n_cubes + 1) * 8;
}

static int max_cubic_res_for_budget(float cell_size, float influence_radius,
                                    size_t budget) {
    int lo = 1, hi = 4096, best = 0;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int64_t b = indep_bytes_for_cubic_res(mid, cell_size, influence_radius);
        if ((uint64_t)b <= (uint64_t)budget) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    return best;
}

/* Hash entier bon marche (variante Wang hash) pour un jitter positionnel
 * fonction de l'indice de particule -- meme indice, meme decalage, d'un appel
 * a l'autre de bq_mesher_run, tant que l'appelant fournit les positions dans
 * le meme ordre (deja le cas : cf. bq_read_positions). Trois appels avec des
 * sels differents donnent trois composantes independantes. */
__device__ inline float mc_hash01(unsigned int x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    return (float)(x & 0x00FFFFFFu) * (1.f / 16777216.f); /* [0,1) */
}

/* ---------------------------------------------- eigendecomposition 3x3 symetrique
 * (M9/T1, plan-milestone-9.md D5). Methode trigonometrique en forme fermee
 * pour matrice symetrique reelle (Smith 1961 / page Wikipedia "Eigenvalue
 * algorithm", section matrices 3x3), utilisee telle quelle. Valeurs propres
 * triees decroissant (l1>=l2>=l3), vecteurs propres extraits par produit
 * vectoriel de deux lignes de (C - lambda*I) -- les trois paires sont
 * testees, la plus grande norme est retenue pour la stabilite numerique. */
struct Eig3 {
    float  l1, l2, l3;
    float3 v1, v2, v3;
    bool   degenerate; /* vrai si un vecteur propre n'a pas pu etre extrait
                         * (valeurs propres repetees, repere indetermine) --
                         * appelle un repli isotrope cote appelant. */
};

__device__ inline Eig3 mc_eigen_sym3(float Cxx, float Cyy, float Czz,
                                     float Cxy, float Cxz, float Cyz) {
    Eig3 e;
    e.degenerate = false;
    float q  = (Cxx + Cyy + Czz) / 3.f;
    float p1 = Cxy * Cxy + Cxz * Cxz + Cyz * Cyz;
    if (p1 < 1e-14f) {
        /* C deja quasi diagonale : eigvals = (Cxx,Cyy,Czz) tries decroissant,
         * eigvecs = base canonique dans l'ordre correspondant au tri. */
        float vals[3]  = {Cxx, Cyy, Czz};
        float3 basis[3] = {make_float3(1.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f),
                           make_float3(0.f, 0.f, 1.f)};
        int order[3] = {0, 1, 2};
        for (int a = 0; a < 3; ++a)
            for (int b = a + 1; b < 3; ++b)
                if (vals[order[b]] > vals[order[a]]) { int t = order[a]; order[a] = order[b]; order[b] = t; }
        e.l1 = vals[order[0]]; e.l2 = vals[order[1]]; e.l3 = vals[order[2]];
        e.v1 = basis[order[0]]; e.v2 = basis[order[1]]; e.v3 = basis[order[2]];
        return e;
    }
    float p2 = (Cxx - q) * (Cxx - q) + (Cyy - q) * (Cyy - q) + (Czz - q) * (Czz - q) + 2.f * p1;
    float p  = sqrtf(p2 / 6.f);
    float Bxx = (Cxx - q) / p, Byy = (Cyy - q) / p, Bzz = (Czz - q) / p;
    float Bxy = Cxy / p, Bxz = Cxz / p, Byz = Cyz / p;
    float detB = Bxx * (Byy * Bzz - Byz * Byz) - Bxy * (Bxy * Bzz - Byz * Bxz) +
                Bxz * (Bxy * Byz - Byy * Bxz);
    float r = detB * 0.5f;
    r = fminf(1.f, fmaxf(-1.f, r));
    float phi = acosf(r) / 3.f;
    float l1 = q + 2.f * p * cosf(phi);
    float l3 = q + 2.f * p * cosf(phi + 2.f * 3.14159265358979323846f / 3.f);
    float l2 = 3.f * q - l1 - l3;
    e.l1 = l1; e.l2 = l2; e.l3 = l3;

    float lambdas[3] = {l1, l2, l3};
    float3 vecs[3];
    for (int k = 0; k < 3; ++k) {
        float lk = lambdas[k];
        float3 row0 = make_float3(Cxx - lk, Cxy, Cxz);
        float3 row1 = make_float3(Cxy, Cyy - lk, Cyz);
        float3 row2 = make_float3(Cxz, Cyz, Czz - lk);
        float3 c01 = vcross(row0, row1);
        float3 c02 = vcross(row0, row2);
        float3 c12 = vcross(row1, row2);
        float n01 = vdot(c01, c01), n02 = vdot(c02, c02), n12 = vdot(c12, c12);
        float3 best; float bestn;
        if (n01 >= n02 && n01 >= n12) { best = c01; bestn = n01; }
        else if (n02 >= n12)          { best = c02; bestn = n02; }
        else                           { best = c12; bestn = n12; }
        if (bestn < 1e-14f) {
            e.degenerate = true;
            best = make_float3(0.f, 0.f, 0.f);
        } else {
            float invn = 1.f / sqrtf(bestn);
            best.x *= invn; best.y *= invn; best.z *= invn;
        }
        vecs[k] = best;
    }
    e.v1 = vecs[0]; e.v2 = vecs[1]; e.v3 = vecs[2];
    return e;
}

/* ------------------------------------------------------------ k_mc_compute_aniso
 * Pre-passe (M9/T1) construisant la matrice anisotrope G_i de chaque
 * particule fluide, cf. Yu & Turk, "Reconstructing Surfaces of
 * Particle-Based Fluids Using Anisotropic Kernels", 2013, et
 * plan-milestone-9.md D1/D2/D5. Un thread par particule i, CSR COMPLET
 * (toutes les particules, y compris celles qui seront ensuite exclues du
 * champ par le filtrage D3 -- une particule exclue reste un voisin valide
 * pour compter/ponderer les autres).
 *
 * Covariance ponderee accumulee en DEUX passages sur le voisinage bucketise
 * (moyenne d'abord, puis somme centree apres calcul de la moyenne, D1) :
 * la formule mathematiquement equivalente en un seul passage
 * (E[xx^T] - E[x]E[x]^T) subit une annulation catastrophique en float32 sur
 * les nappes fines (mesure : ratio d'anisotropie a 4,5e26 au lieu de ~121000
 * sur un cas synthetique reproduisant une nappe fluide d'une seule particule
 * d'epaisseur), d'ou le retour a la formule centree malgre le cout d'une
 * seconde boucle sur les 27 buckets.
 *
 * La normalisation par moyenne geometrique (etape 4 de D1) est un choix
 * PROPRE A CE PROJET, pas celui du papier : elle garantit qu'un voisinage
 * isotrope redonne EXACTEMENT G_i=(1/R)*Identite -- cf. plan-milestone-9.md
 * D1 point 4 pour la justification complete, non reformulee ici. */
__global__ void k_mc_compute_aniso(const float3* __restrict__ pos,
                                   const int* __restrict__ bucket_off,
                                   const int* __restrict__ bucket_idx,
                                   int3 bucket_res, float bucket_h,
                                   float3 bucket_origin,
                                   float influence_radius, int n,
                                   float* __restrict__ aniso,
                                   int* __restrict__ neighbor_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 xi = pos[i];
    float inv_R = 1.f / influence_radius;
    float R2 = influence_radius * influence_radius;

    int3 bc = make_int3((int)floorf((xi.x - bucket_origin.x) / bucket_h),
                        (int)floorf((xi.y - bucket_origin.y) / bucket_h),
                        (int)floorf((xi.z - bucket_origin.z) / bucket_h));

    float sum_w = 0.f;
    float3 sum_wx = make_float3(0.f, 0.f, 0.f);
    int ncount = 0;

    for (int di = -1; di <= 1; ++di) {
        int bi = bc.x + di;
        if (bi < 0 || bi >= bucket_res.x) continue;
        for (int dj = -1; dj <= 1; ++dj) {
            int bj = bc.y + dj;
            if (bj < 0 || bj >= bucket_res.y) continue;
            for (int dk = -1; dk <= 1; ++dk) {
                int bk = bc.z + dk;
                if (bk < 0 || bk >= bucket_res.z) continue;
                int bidx = (bi * bucket_res.y + bj) * bucket_res.z + bk;
                int off0 = bucket_off[bidx], off1 = bucket_off[bidx + 1];
                for (int e = off0; e < off1; ++e) {
                    int j = bucket_idx[e];
                    float3 xj = pos[j];
                    float3 d = vsub(xi, xj);
                    float d2 = vdot(d, d);
                    if (d2 >= R2) continue;
                    if (j != i) ++ncount;
                    float s = sqrtf(d2) * inv_R;
                    float t = 1.f - s * s;
                    float w = t * t * t; /* w_ii = k(0) = 1, j=i inclus (D1) */
                    sum_w += w;
                    sum_wx.x += w * xj.x; sum_wx.y += w * xj.y; sum_wx.z += w * xj.z;
                }
            }
        }
    }

    neighbor_count[i] = ncount;

    /* Repli isotrope (D1) : moins de k_n_aniso voisins reels, ou (garde-fou
     * pratique) sum_w nul -- ne devrait pas arriver puisque w_ii=1 est
     * toujours inclus, gardee pour ne jamais diviser par zero. */
    if (ncount < k_n_aniso || sum_w <= 0.f) {
        aniso[6 * i + 0] = inv_R; aniso[6 * i + 1] = inv_R; aniso[6 * i + 2] = inv_R;
        aniso[6 * i + 3] = 0.f;   aniso[6 * i + 4] = 0.f;   aniso[6 * i + 5] = 0.f;
        return;
    }

    float inv_sw = 1.f / sum_w;
    float3 xbar = make_float3(sum_wx.x * inv_sw, sum_wx.y * inv_sw, sum_wx.z * inv_sw);

    /* Deuxieme boucle sur le meme voisinage bucketise : la covariance est
     * accumulee DIRECTEMENT centree (xj - xbar avant le carre), et non via
     * E[xx^T] - E[x]E[x]^T -- ce dernier calcul en un seul passage subit une
     * annulation catastrophique en float32 sur les nappes fines (mesure :
     * ratio d'anisotropie a 4,5e26 au lieu de ~121000 attendu, sur un cas
     * synthetique reproduisant une nappe fluide d'une seule particule
     * d'epaisseur). */
    float sxx = 0.f, syy = 0.f, szz = 0.f, sxy = 0.f, sxz = 0.f, syz = 0.f;
    for (int di = -1; di <= 1; ++di) {
        int bi = bc.x + di;
        if (bi < 0 || bi >= bucket_res.x) continue;
        for (int dj = -1; dj <= 1; ++dj) {
            int bj = bc.y + dj;
            if (bj < 0 || bj >= bucket_res.y) continue;
            for (int dk = -1; dk <= 1; ++dk) {
                int bk = bc.z + dk;
                if (bk < 0 || bk >= bucket_res.z) continue;
                int bidx = (bi * bucket_res.y + bj) * bucket_res.z + bk;
                int off0 = bucket_off[bidx], off1 = bucket_off[bidx + 1];
                for (int e = off0; e < off1; ++e) {
                    int j = bucket_idx[e];
                    float3 xj = pos[j];
                    float3 d = vsub(xi, xj);
                    float d2 = vdot(d, d);
                    if (d2 >= R2) continue;
                    float s = sqrtf(d2) * inv_R;
                    float t = 1.f - s * s;
                    float w = t * t * t;
                    float3 dc = vsub(xj, xbar); /* xj - xbar, PAS xj - xi */
                    sxx += w * dc.x * dc.x; syy += w * dc.y * dc.y; szz += w * dc.z * dc.z;
                    sxy += w * dc.x * dc.y; sxz += w * dc.x * dc.z; syz += w * dc.y * dc.z;
                }
            }
        }
    }

    float Cxx = sxx * inv_sw;
    float Cyy = syy * inv_sw;
    float Czz = szz * inv_sw;
    float Cxy = sxy * inv_sw;
    float Cxz = sxz * inv_sw;
    float Cyz = syz * inv_sw;

    Eig3 eig = mc_eigen_sym3(Cxx, Cyy, Czz, Cxy, Cxz, Cyz);
    if (eig.degenerate) {
        aniso[6 * i + 0] = inv_R; aniso[6 * i + 1] = inv_R; aniso[6 * i + 2] = inv_R;
        aniso[6 * i + 3] = 0.f;   aniso[6 * i + 4] = 0.f;   aniso[6 * i + 5] = 0.f;
        return;
    }

    /* Regularisation (D1 etape 3) : seules les deux plus petites valeurs
     * propres sont planchees a l1/k_r, l1 lui-meme n'est jamais touche. */
    float l1 = eig.l1;
    float l2r = fmaxf(eig.l2, l1 * (1.f / k_r));
    float l3r = fmaxf(eig.l3, l1 * (1.f / k_r));
    float prod = l1 * l2r * l3r;
    /* Garde-fou numerique (pas dans le papier ni le plan) : une covariance
     * quasi degeneree malgre p1 non nul pourrait donner un produit quasi nul
     * -- repli isotrope plutot qu'une racine cubique/carree de quasi-zero. */
    if (!(prod > 1e-20f)) {
        aniso[6 * i + 0] = inv_R; aniso[6 * i + 1] = inv_R; aniso[6 * i + 2] = inv_R;
        aniso[6 * i + 3] = 0.f;   aniso[6 * i + 4] = 0.f;   aniso[6 * i + 5] = 0.f;
        return;
    }

    /* Normalisation par moyenne geometrique (D1 etape 4, cf. commentaire en
     * tete de kernel) puis construction de G_i = (1/R)*Rot*diag(1/sqrt)*Rot^T
     * (D1 etape 5). */
    float g = cbrtf(prod);
    float l1n = l1 / g, l2n = l2r / g, l3n = l3r / g;
    float s1 = inv_R / sqrtf(l1n);
    float s2 = inv_R / sqrtf(l2n);
    float s3 = inv_R / sqrtf(l3n);

    float3 v1 = eig.v1, v2 = eig.v2, v3 = eig.v3;
    float Gxx = s1 * v1.x * v1.x + s2 * v2.x * v2.x + s3 * v3.x * v3.x;
    float Gyy = s1 * v1.y * v1.y + s2 * v2.y * v2.y + s3 * v3.y * v3.y;
    float Gzz = s1 * v1.z * v1.z + s2 * v2.z * v2.z + s3 * v3.z * v3.z;
    float Gxy = s1 * v1.x * v1.y + s2 * v2.x * v2.y + s3 * v3.x * v3.y;
    float Gxz = s1 * v1.x * v1.z + s2 * v2.x * v2.z + s3 * v3.x * v3.z;
    float Gyz = s1 * v1.y * v1.z + s2 * v2.y * v2.z + s3 * v3.y * v3.z;
    aniso[6 * i + 0] = Gxx; aniso[6 * i + 1] = Gyy; aniso[6 * i + 2] = Gzz;
    aniso[6 * i + 3] = Gxy; aniso[6 * i + 4] = Gxz; aniso[6 * i + 5] = Gyz;
}

/* ------------------------------------------------------------------ kernel
 * Gather : une cellule = un thread, accumulation en registres, une seule
 * ecriture. cf. note D2 en tete de fichier. */
__global__ void k_zhu_bridson_field(float* __restrict__ field,
                                    const float3* __restrict__ pos,
                                    const float* __restrict__ aniso,
                                    const int* __restrict__ bucket_off,
                                    const int* __restrict__ bucket_idx,
                                    int3 bucket_res, float bucket_h,
                                    float3 bucket_origin,
                                    int3 grid_res, float cell_size,
                                    float influence_radius,
                                    float particle_radius, int n_cells) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cells) return;

    int i = id / (grid_res.y * grid_res.z);
    int j = (id / grid_res.z) % grid_res.y;
    int k = id % grid_res.z;
    float3 p = make_float3((i + 0.5f) * cell_size, (j + 0.5f) * cell_size,
                           (k + 0.5f) * cell_size);

    int3 bc = make_int3((int)floorf((p.x - bucket_origin.x) / bucket_h),
                        (int)floorf((p.y - bucket_origin.y) / bucket_h),
                        (int)floorf((p.z - bucket_origin.z) / bucket_h));

    float sum_w = 0.f;
    float3 sum_wx = make_float3(0.f, 0.f, 0.f);
    float sum_wr = 0.f;
    /* R2_stretch borne le rejet rapide par distance euclidienne (optimisation
     * facultative, cf. spec T2) : k_r=4 est le facteur MAX de regularisation
     * des valeurs propres (k_mc_compute_aniso), donc l'etirement maximal du
     * noyau anisotrope vaut sqrt(k_r)=2 -- une particule ne peut jamais
     * contribuer au-dela de 2R en distance euclidienne, meme dans la
     * direction la plus etiree. */
    float R2_stretch = influence_radius * influence_radius * k_r;

    /* Recherche par buckets ELARGIE a [-2,2] (125 buckets, rayon de bucket =
     * R) au lieu de [-1,1] (27 buckets) -- CORRECTIF DE CORRECTION, pas une
     * option : avec un noyau anisotrope etire d'un facteur max sqrt(k_r)=2
     * (cf. R2_stretch ci-dessus), le support d'une particule peut depasser R
     * en distance euclidienne dans la direction d'etirement, donc deborder
     * du voisinage 3x3x3 qui suffisait au noyau isotrope. k_mc_compute_aniso
     * reste, lui, en [-1,1] : il n'a pas besoin de cet elargissement,
     * puisqu'il ne fait que compter/ponderer des voisins a distance < R,
     * jamais evaluer un noyau deja etire. */
    for (int di = -2; di <= 2; ++di) {
        int bi = bc.x + di;
        if (bi < 0 || bi >= bucket_res.x) continue;
        for (int dj = -2; dj <= 2; ++dj) {
            int bj = bc.y + dj;
            if (bj < 0 || bj >= bucket_res.y) continue;
            for (int dk = -2; dk <= 2; ++dk) {
                int bk = bc.z + dk;
                if (bk < 0 || bk >= bucket_res.z) continue;
                int bidx = (bi * bucket_res.y + bj) * bucket_res.z + bk;
                int off0 = bucket_off[bidx], off1 = bucket_off[bidx + 1];
                for (int e = off0; e < off1; ++e) {
                    int pidx = bucket_idx[e];
                    float3 xp = pos[pidx];
                    /* Jitter positionnel deterministe, amplitude = 0.02 *
                     * particle_radius. Mesure (nappe au repos, meme scene que
                     * balayage_rayon.py) : au facteur d'influence par defaut
                     * (3.0), le maillage reste une seule composante connexe,
                     * amplitude 0.05 comme 0.02. En-dessous de 3.0, TOUTE
                     * amplitude non nulle degrade fortement la connexite
                     * (facteur 1.0 : 2 composantes sans jitter -> 55 a
                     * amplitude 0.02, 69 a 0.05, 133 a 0.25) -- la mesure
                     * "1.5 et au-dela = 1 composante" documentee dans
                     * extension/props.py (mesh_influence_factor) etait un
                     * artefact du reseau d'emission PARFAITEMENT regulier de
                     * ce cas de test, pas une vraie marge : le jitter la
                     * revele plutot qu'il ne la cree. Voir
                     * valide_lissage_jitter.py (scratchpad de la session qui
                     * a introduit ce jitter) pour le detail du balayage. Le
                     * plancher expose a l'artiste dans props.py a ete
                     * remonte en consequence -- ne pas re-descendre l'un sans
                     * revalider l'autre. */
                    float amp = 0.02f * particle_radius;
                    xp.x += amp * (2.f * mc_hash01((unsigned)pidx * 0x9E3779B1u + 1u) - 1.f);
                    xp.y += amp * (2.f * mc_hash01((unsigned)pidx * 0x9E3779B1u + 2u) - 1.f);
                    xp.z += amp * (2.f * mc_hash01((unsigned)pidx * 0x9E3779B1u + 3u) - 1.f);
                    float3 d = vsub(p, xp);
                    float d2 = vdot(d, d);
                    if (d2 >= R2_stretch) continue; /* rejet rapide facultatif, cf. R2_stretch */
                    float Gxx = aniso[6 * pidx + 0], Gyy = aniso[6 * pidx + 1], Gzz = aniso[6 * pidx + 2];
                    float Gxy = aniso[6 * pidx + 3], Gxz = aniso[6 * pidx + 4], Gyz = aniso[6 * pidx + 5];
                    float3 gd = make_float3(Gxx * d.x + Gxy * d.y + Gxz * d.z,
                                            Gxy * d.x + Gyy * d.y + Gyz * d.z,
                                            Gxz * d.x + Gyz * d.y + Gzz * d.z);
                    float s2 = vdot(gd, gd);
                    if (s2 >= 1.f) continue;
                    float t = 1.f - s2; /* > 0 puisque s2 < 1 */
                    float w = t * t * t;
                    sum_w += w;
                    sum_wx.x += w * xp.x; sum_wx.y += w * xp.y; sum_wx.z += w * xp.z;
                    sum_wr += w * particle_radius;
                }
            }
        }
    }

    if (sum_w <= 0.f) {
        /* aucune particule dans le rayon : grande valeur positive, jamais
         * zero ni NaN (cf. spec T1). */
        field[id] = influence_radius;
        return;
    }
    float inv_sum = 1.f / sum_w;
    float3 xmean = make_float3(sum_wx.x * inv_sum, sum_wx.y * inv_sum,
                               sum_wx.z * inv_sum);
    float rmean = sum_wr * inv_sum;
    field[id] = vlen(vsub(p, xmean)) - rmean;
}

/* ============================================================ marching cubes
 * (M7/T2). Maillage indexe, sommets dedupliques par arete (decision D4 du
 * plan). Le champ (ci-dessus) reste inchange -- cette section consomme
 * uniquement m->d_field.
 *
 * CONVENTION DE NUMEROTATION DES COINS DU CUBE (i,j,k)..(i+1,j+1,k+1) :
 * c'est EXACTEMENT celle de l'implementation de reference domaine public
 * (Bourke/Bloyd, a2fVertexOffset/a2iEdgeConnection,
 * http://paulbourke.net/geometry/polygonise/), ce qui permet de recopier
 * edgeTable/triTable tels quels sans reindexation :
 *
 *   coin 0 = (i,   j,   k)      coin 4 = (i,   j,   k+1)
 *   coin 1 = (i+1, j,   k)      coin 5 = (i+1, j,   k+1)
 *   coin 2 = (i+1, j+1, k)      coin 6 = (i+1, j+1, k+1)
 *   coin 3 = (i,   j+1, k)      coin 7 = (i,   j+1, k+1)
 *
 *        4----------5
 *       /|         /|
 *      7----------6 |
 *      | |        | |
 *      | 0--------|-1
 *      |/         |/
 *      3----------2
 *
 * bit c du cube-index vaut 1 si field[coin c] < 0 (interieur, cf. convention
 * stricte phi<0 = interieur / phi>=0 = exterieur, appliquee identiquement au
 * marquage des aretes ci-dessous : aucune divergence entre les deux tests,
 * qui doivent s'accorder pour que arete marquee <=> cas produit un triangle
 * sur cette arete).
 *
 * Les 12 aretes MC (a2iEdgeConnection) relient les coins :
 *   e0:0-1  e1:1-2  e2:2-3  e3:3-0  e4:4-5  e5:5-6  e6:6-7  e7:7-4
 *   e8:0-4  e9:1-5  e10:2-6 e11:3-7
 *
 * DEDUPLICATION PAR ARETE (D4) : chaque arete de GRILLE (pas de cube) est
 * possedee par la cellule dont elle part vers +x/+y/+z, avec l'identifiant
 * global 3*((i*res.y+j)*res.z+k) + axe (axe 0=x,1=y,2=z). g_mc_edge_owner
 * traduit chaque arete MC (0..11) en (di,dj,dk,axe) : l'arete du cube
 * (i,j,k) revient donc a la cellule (i+di,j+dj,k+dk), axe. Cette table est
 * derivee directement de a2iEdgeConnection + la convention de coins
 * ci-dessus (coin bas = coin possedant l'arete), PAS une donnee separee a
 * maintenir en synchronisation manuelle -- verifiee par un maillage fermee
 * sur sphere de reference (V - E + F = 2) avant integration.
 *
 * BORD DU CHAMP : les cubes s'arretent a res-2 (dernier indice de coin haut
 * = res-1), donc un fluide qui atteint le bord du champ de maillage produit
 * un maillage OUVERT a cet endroit -- accepte pour ce jalon (cf. spec T2).
 */
struct McEdgeOwner { int di, dj, dk, axis; };
__device__ __constant__ McEdgeOwner g_mc_edge_owner[12] = {
    {0, 0, 0, 0}, /* e0:  0-1 */
    {1, 0, 0, 1}, /* e1:  1-2 */
    {0, 1, 0, 0}, /* e2:  2-3 */
    {0, 0, 0, 1}, /* e3:  3-0 */
    {0, 0, 1, 0}, /* e4:  4-5 */
    {1, 0, 1, 1}, /* e5:  5-6 */
    {0, 1, 1, 0}, /* e6:  6-7 */
    {0, 0, 1, 1}, /* e7:  7-4 */
    {0, 0, 0, 2}, /* e8:  0-4 */
    {1, 0, 0, 2}, /* e9:  1-5 */
    {1, 1, 0, 2}, /* e10: 2-6 */
    {0, 1, 0, 2}, /* e11: 3-7 */
};

/* edgeTable standard de marching cubes (domaine public, Lorensen & Cline via
 * Bourke/Bloyd, http://paulbourke.net/geometry/polygonise/) -- recopiee telle
 * quelle. Indexee par le cube-index 8 bits (bit c = 1 si le coin c est
 * interieur, cf. convention de coins ci-dessus). Bit e du resultat = 1 si
 * l'arete MC e est traversee par l'isosurface pour ce cas. Non utilisee dans
 * le pipeline (le marquage des aretes, pass 1, est fait independamment par
 * cellule/axe), gardee pour la lisibilite et la parente avec triTable. */
__device__ __constant__ int g_mc_edge_table[256] = {
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
};

/* triTable standard de marching cubes, meme source. triTable[c] liste les
 * aretes MC (0..11) formant les triangles du cas c, par groupes de 3,
 * terminee par -1. Au plus 5 triangles (15 entrees utiles) par cas. */
__device__ __constant__ int g_mc_tri_table[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
};

/* Cube-index a partir des 8 coins (convention en tete de section) et du
 * champ deja calcule. Partagee entre la passe de comptage et la passe
 * d'emission -- recalculee dans chaque thread plutot que memorisee, moins
 * cher qu'un tampon supplementaire a l'echelle des cubes. */
__device__ inline int mc_cube_index(const float* __restrict__ field, int i, int j,
                                    int k, int3 res) {
    int idx000 = (i * res.y + j) * res.z + k;
    int idx100 = ((i + 1) * res.y + j) * res.z + k;
    int idx110 = ((i + 1) * res.y + (j + 1)) * res.z + k;
    int idx010 = (i * res.y + (j + 1)) * res.z + k;
    int idx001 = (i * res.y + j) * res.z + (k + 1);
    int idx101 = ((i + 1) * res.y + j) * res.z + (k + 1);
    int idx111 = ((i + 1) * res.y + (j + 1)) * res.z + (k + 1);
    int idx011 = (i * res.y + (j + 1)) * res.z + (k + 1);
    int c = 0;
    if (field[idx000] < 0.f) c |= 1;
    if (field[idx100] < 0.f) c |= 2;
    if (field[idx110] < 0.f) c |= 4;
    if (field[idx010] < 0.f) c |= 8;
    if (field[idx001] < 0.f) c |= 16;
    if (field[idx101] < 0.f) c |= 32;
    if (field[idx111] < 0.f) c |= 64;
    if (field[idx011] < 0.f) c |= 128;
    return c;
}

/* Passe 1 -- marquage des aretes. Un thread par (cellule, axe), id in
 * [0, 3*n_cells). L'arete est traversee si les signes des deux extremites
 * different (convention stricte phi<0 interieur / phi>=0 exterieur, cf.
 * en-tete de section). Pas de voisin (bord de grille sur cet axe) => 0,
 * jamais de lecture hors bornes. */
__global__ void k_mc_mark_edges(int* __restrict__ edge_flag,
                                const float* __restrict__ field, int3 res,
                                int64_t n_edges) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_edges) return;
    int64_t cell = id / 3;
    int axis = (int)(id - cell * 3);
    int i = (int)(cell / (res.y * res.z));
    int j = (int)((cell / res.z) % res.y);
    int k = (int)(cell % res.z);

    int ni = i + (axis == 0 ? 1 : 0);
    int nj = j + (axis == 1 ? 1 : 0);
    int nk = k + (axis == 2 ? 1 : 0);
    if (ni >= res.x || nj >= res.y || nk >= res.z) {
        edge_flag[id] = 0;
        return;
    }
    float va = field[cell];
    float vb = field[(ni * res.y + nj) * res.z + nk];
    edge_flag[id] = ((va < 0.f) != (vb < 0.f)) ? 1 : 0;
}

/* Passe 3 -- emission des sommets, un thread par arete marquee (les autres
 * ne font rien : le scan garantit que seules les aretes a edge_flag=1 ont
 * un index de sortie valide). Interpolation lineaire du zero entre les
 * centres de cellule des deux extremites. */
__global__ void k_mc_emit_vertices(float* __restrict__ verts,
                                   const int* __restrict__ edge_flag,
                                   const int* __restrict__ edge_scan,
                                   const float* __restrict__ field, int3 res,
                                   float cell_size, int64_t n_edges) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_edges) return;
    if (!edge_flag[id]) return;
    int64_t cell = id / 3;
    int axis = (int)(id - cell * 3);
    int i = (int)(cell / (res.y * res.z));
    int j = (int)((cell / res.z) % res.y);
    int k = (int)(cell % res.z);

    int ni = i + (axis == 0 ? 1 : 0);
    int nj = j + (axis == 1 ? 1 : 0);
    int nk = k + (axis == 2 ? 1 : 0);
    float va = field[cell];
    float vb = field[(ni * res.y + nj) * res.z + nk];
    float t = va / (va - vb); /* va,vb de signes opposes (garanti par le
                                  marquage) : denominateur non nul */
    float3 pa = make_float3((i + 0.5f) * cell_size, (j + 0.5f) * cell_size,
                            (k + 0.5f) * cell_size);
    float3 pb = make_float3((ni + 0.5f) * cell_size, (nj + 0.5f) * cell_size,
                            (nk + 0.5f) * cell_size);
    float3 p = make_float3(pa.x + t * (pb.x - pa.x), pa.y + t * (pb.y - pa.y),
                           pa.z + t * (pb.z - pa.z));
    int vidx = edge_scan[id];
    verts[3 * vidx + 0] = p.x;
    verts[3 * vidx + 1] = p.y;
    verts[3 * vidx + 2] = p.z;
}

/* Passe 4 -- comptage des triangles par cube, un thread par cube. */
__global__ void k_mc_count_tris(int* __restrict__ cube_tricount,
                                const float* __restrict__ field, int3 res,
                                int3 cube_res, int64_t n_cubes) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cubes) return;
    int i = (int)(id / (cube_res.y * cube_res.z));
    int j = (int)((id / cube_res.z) % cube_res.y);
    int k = (int)(id % cube_res.z);
    int c = mc_cube_index(field, i, j, k, res);
    int n = 0;
    for (int t = 0; t < 15 && g_mc_tri_table[c][t] != -1; t += 3) ++n;
    cube_tricount[id] = n;
}

/* Passe 6 -- emission des indices, un thread par cube. Pour chaque triangle
 * du cas, chacune des 3 aretes MC locales est traduite en identifiant
 * d'arete globale via g_mc_edge_owner puis relue dans edge_scan (D4). */
__global__ void k_mc_emit_indices(int* __restrict__ tris,
                                  const int* __restrict__ cube_tricount,
                                  const int* __restrict__ cube_scan,
                                  const int* __restrict__ edge_scan,
                                  const float* __restrict__ field, int3 res,
                                  int3 cube_res, int64_t n_cubes) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cubes) return;
    int n = cube_tricount[id];
    if (n == 0) return;
    int i = (int)(id / (cube_res.y * cube_res.z));
    int j = (int)((id / cube_res.z) % cube_res.y);
    int k = (int)(id % cube_res.z);
    int c = mc_cube_index(field, i, j, k, res);
    int out_tri0 = cube_scan[id];
    for (int t = 0; t < n; ++t) {
        for (int v = 0; v < 3; ++v) {
            int e = g_mc_tri_table[c][3 * t + v];
            McEdgeOwner o = g_mc_edge_owner[e];
            int64_t owner_cell = (int64_t)((i + o.di) * res.y + (j + o.dj)) * res.z +
                                 (k + o.dk);
            int64_t gedge = 3 * owner_cell + o.axis;
            /* Flip global d'orientation (mesure, pas suppose -- cf.
             * docs/plan-milestone-14.md, section "Constat > 1. Normales") :
             * g_mc_tri_table est recopiee telle quelle depuis la reference
             * domaine public Bourke/Bloyd, dont la convention de champ
             * scalaire (densite, haut = solide) est opposee a la notre
             * (distance signee, negatif = fluide/interieur). Le test de bit
             * de mc_cube_index reste identique, mais l'orientation de
             * winding qu'il induit via triTable est inversee par rapport a
             * notre polarite -- mesure sur une sphere pleine : volume signe
             * = -1.042x le volume theorique, 100% des triangles avec
             * normale vers l'interieur. Correctif verifie algebriquement :
             * echanger les sommets 1 et 2 de chaque triangle a l'emission
             * inverse exactement le signe (-1.042 -> +1.042), sans toucher
             * g_mc_tri_table ni g_mc_edge_owner. */
            int v_out = (v == 0) ? 0 : (3 - v);
            tris[3 * (out_tri0 + t) + v_out] = edge_scan[gedge];
        }
    }
}

/* ==================================================== rognage collider (M7/T4)
 * Le champ collider est fourni deja echantillonne par l'appelant, sur SA
 * PROPRE grille (en general celle du solveur, resolution differente de celle
 * du mailleur) -- ce fichier ne reconstruit rien (D7 : le mailleur est
 * independant de BqSim et de mlsmpm.cu, cf. note d'autonomie en tete de
 * fichier). Convention : idx = (i*res[1]+j)*res[2]+k, valeur a (i,j,k) =
 * centre de cellule ((i+0.5)*cell_size, ...) -- EXACTEMENT la meme convention
 * que le champ Zhu-Bridson et la grille de marching cubes ci-dessus
 * (k_zhu_bridson_field, p = (i+0.5)*cell_size), afin que l'interpolation
 * trilineaire ci-dessous et le champ rogne restent coherents entre eux dans
 * CE fichier. Signe identique partout dans le projet : negatif = interieur
 * du solide.
 *
 * Hors du domaine du champ collider fourni (aucun des 8 coins necessaires a
 * l'interpolation n'est disponible) : grande valeur POSITIVE, jamais une
 * extrapolation -- une cellule du champ de maillage peut legitimement tomber
 * hors de la grille du solveur (cf. spec T4). */
__device__ inline float mc_sample_collider_sdf(const float* __restrict__ csdf,
                                               int3 cres, float ccell, float3 p) {
    /* Le champ collider est echantillonne AUX NOEUDS (i*ccell), pas au centre
     * de cellule : c'est la convention du solveur, etablie et justifiee dans
     * k_sdf_unsigned (mlsmpm.cu, « Echantillonnage AUX NOEUDS (i*dx), et non au
     * centre de cellule »), parce que c'est k_grid_update qui consomme ce champ
     * et qu'il est indexe comme la grille MPM dont les noeuds sont en i*dx.
     *
     * Le champ du MAILLEUR, lui, est au centre de cellule. Les deux conventions
     * coexistent donc, et c'est ici -- au seul point de contact entre elles --
     * qu'il faut convertir. Retrancher 0.5 comme pour un champ centre
     * decalerait le rognage d'un demi-pas par axe, soit 0.87 pas en diagonale :
     * le fluide serait rogne a cote de la surface du collider. */
    float gx = p.x / ccell;
    float gy = p.y / ccell;
    float gz = p.z / ccell;
    int i0 = (int)floorf(gx), j0 = (int)floorf(gy), k0 = (int)floorf(gz);
    if (i0 < 0 || i0 + 1 >= cres.x || j0 < 0 || j0 + 1 >= cres.y ||
        k0 < 0 || k0 + 1 >= cres.z) {
        return 1e6f;
    }
    float tx = gx - i0, ty = gy - j0, tz = gz - k0;
    int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
    float c000 = csdf[(i0 * cres.y + j0) * cres.z + k0];
    float c100 = csdf[(i1 * cres.y + j0) * cres.z + k0];
    float c010 = csdf[(i0 * cres.y + j1) * cres.z + k0];
    float c110 = csdf[(i1 * cres.y + j1) * cres.z + k0];
    float c001 = csdf[(i0 * cres.y + j0) * cres.z + k1];
    float c101 = csdf[(i1 * cres.y + j0) * cres.z + k1];
    float c011 = csdf[(i0 * cres.y + j1) * cres.z + k1];
    float c111 = csdf[(i1 * cres.y + j1) * cres.z + k1];
    float c00 = c000 * (1.f - tx) + c100 * tx;
    float c10 = c010 * (1.f - tx) + c110 * tx;
    float c01 = c001 * (1.f - tx) + c101 * tx;
    float c11 = c011 * (1.f - tx) + c111 * tx;
    float c0 = c00 * (1.f - ty) + c10 * ty;
    float c1 = c01 * (1.f - ty) + c11 * ty;
    return c0 * (1.f - tz) + c1 * tz;
}

/* Un thread par cellule du champ de maillage. phi_final = max(phi_fluide,
 * -phi_solide + collider_offset) : a l'interieur du solide phi_solide < 0
 * donc -phi_solide > 0, le max rend la cellule exterieure au fluide ; hors du
 * solide -phi_solide < 0 (ou 1e6 si hors domaine collider) et le max laisse
 * le champ de fluide inchange. Lance seulement si un collider est defini
 * (cf. bq_mesher_run) : cout nul dans le cas courant sans collider. */
__global__ void k_mc_crop_collider(float* __restrict__ field,
                                   const float* __restrict__ collider_sdf,
                                   int3 res, float cell_size,
                                   int3 collider_res, float collider_cell_size,
                                   float collider_offset, int64_t n_cells) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cells) return;
    int i = (int)(id / (res.y * res.z));
    int j = (int)((id / res.z) % res.y);
    int k = (int)(id % res.z);
    float3 p = make_float3((i + 0.5f) * cell_size, (j + 0.5f) * cell_size,
                           (k + 0.5f) * cell_size);
    float phi_solid = mc_sample_collider_sdf(collider_sdf, collider_res,
                                             collider_cell_size, p);
    field[id] = fmaxf(field[id], -phi_solid + collider_offset);
}

/* Constantes du filtre de Taubin (M11/T1, D1), article fondateur Taubin 1995
 * -- valeurs classiques non recalibrees pour ce projet, meme discipline que
 * k_r/k_n_aniso ci-dessus (|mu| > lambda, condition necessaire a la
 * non-amplification du filtre). constexpr pour la meme raison que k_r :
 * visibilite garantie en code __global__ sous nvcc. */
constexpr float BQ_TAUBIN_LAMBDA = 0.5f;
constexpr float BQ_TAUBIN_MU     = -0.53f;

/* Lissage de Taubin (lambda|mu) sur grille reguliere (M11/T1, D1) : passe de
 * diffusion explicite a 6 voisins, comme avant, mais avec un terme de rappel
 * vers la valeur centrale pondere par `step` : dst = c + step*(avg - c). Le
 * Laplace pur precedent (dst = avg, sans rappel) erodait l'interface a
 * chaque iteration -- un flot par courbure moyenne applique au champ de
 * distance signee, mathematiquement identique au retrecissement bien connu
 * du lissage de Laplace sur maillage. Avec step = 1, cette formule se reduit
 * EXACTEMENT a l'ancien comportement (c + 1*(avg-c) = avg) : verification de
 * non-regression algebrique immediate. Le filtre de Taubin alterne une passe
 * a step = BQ_TAUBIN_LAMBDA (>0, lissage) et une passe a step = BQ_TAUBIN_MU
 * (<0, "anti-lissage" de meme grandeur mais |mu| > lambda) pour annuler le
 * retrecissement en volume tout en conservant l'effet anti-bruit (cf.
 * bq_mesher_run pour l'enchainement des deux passes par unite de
 * smoothing_iters). Voisin hors grille : reflechi (on reutilise la valeur du
 * centre), pour ne pas tirer le champ vers une valeur arbitraire au bord du
 * domaine de maillage. cfg.smoothing_iters pilote le nombre de PAIRES de
 * passes ; 0 = desactive (cout nul, aucun kernel lance, cf. bq_mesher_run).
 * Lance AVANT le rognage collider : k_mc_crop_collider reimpose ensuite la
 * surface exacte du solide via son max(), donc un eventuel flou introduit
 * pres d'un collider est corrige apres coup -- lisser apres le rognage
 * laisserait au contraire le collider s'eroder. */
__global__ void k_mc_smooth_field(float* __restrict__ dst,
                                  const float* __restrict__ src,
                                  int3 res, int64_t n_cells, float step) {
    int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_cells) return;
    int i = (int)(id / (res.y * res.z));
    int j = (int)((id / res.z) % res.y);
    int k = (int)(id % res.z);
    float c = src[id];
    float sum = 0.f;
    sum += (i > 0)          ? src[id - res.y * res.z] : c;
    sum += (i < res.x - 1)  ? src[id + res.y * res.z] : c;
    sum += (j > 0)          ? src[id - res.z]          : c;
    sum += (j < res.y - 1)  ? src[id + res.z]          : c;
    sum += (k > 0)          ? src[id - 1]               : c;
    sum += (k < res.z - 1)  ? src[id + 1]               : c;
    float avg = sum * (1.f / 6.f);
    dst[id] = c + step * (avg - c);
}

/* ------------------------------------------------------------------ BqMesher */
struct BqMesher {
    BqMesherConfig cfg;
    int64_t n_cells = 0;
    BucketGrid bucket;
    int64_t n_buckets = 0;

    float*  d_field = nullptr;      /* n_cells floats, fixe a la creation */
    float*  d_field_tmp = nullptr;  /* n_cells floats, tampon de lissage (cf. k_mc_smooth_field) */
    int*    d_bucket_off = nullptr; /* n_buckets+1, fixe a la creation */
    /* dependants du nombre de particules : realloues (taille exacte) quand
     * la capacite courante est depassee, jamais retreci -- meme politique
     * que bq_set_colliders dans mlsmpm.cu. */
    int*    d_bucket_idx = nullptr;
    float3* d_pos = nullptr;
    int particle_cap = 0;

    /* --- noyau anisotrope + filtrage des particules isolees (M9/T1) ---
     * d_aniso/d_neighbor_count : particule-dependants, meme politique de
     * reallocation grow-only que d_pos/d_bucket_idx ci-dessus (taille
     * exacte, jamais retrecie). d_bucket_idx2 aussi (CSR filtre, au plus
     * particle_cap entrees). d_bucket_off2 en revanche a la MEME taille que
     * d_bucket_off (n_buckets+1, determinee par la grille de buckets, pas
     * par le nombre de particules) : alloue une seule fois a la creation,
     * comme d_bucket_off, pas de compteur de capacite separe. */
    float*  d_aniso = nullptr;          /* n*6 floats, G_i (Gxx,Gyy,Gzz,Gxy,Gxz,Gyz) */
    int*    d_neighbor_count = nullptr; /* n ints, voisins reels hors soi-meme dans R */
    int*    d_bucket_off2 = nullptr;    /* n_buckets+1, CSR FILTRE (D3) */
    int*    d_bucket_idx2 = nullptr;    /* particle_cap au plus, CSR FILTRE (D3) */

    /* --- marching cubes (D4) --- */
    int64_t n_edges = 0;            /* 3 * n_cells */
    int64_t n_cubes = 0;
    int3    cube_res = make_int3(0, 0, 0);
    /* Tableaux dimensionnes a n+1 : la derniere case, jamais ecrite par les
     * kernels, est mise a zero une fois pour toutes a la creation. La somme
     * prefixe exclusive porte donc sur n+1 elements et sa derniere valeur EST
     * le total, qu'on relit d'un seul memcpy de 4 octets -- sans quoi il
     * faudrait relire a la fois le dernier scan et le dernier drapeau. */
    int*    d_edge_flag = nullptr;  /* n_edges + 1 */
    int*    d_edge_scan = nullptr;  /* n_edges + 1 */
    int*    d_cube_tri = nullptr;   /* n_cubes + 1 */
    int*    d_cube_scan = nullptr;  /* n_cubes + 1 */
    void*   d_cub_tmp = nullptr;
    size_t  cub_tmp_bytes = 0;
    /* sorties, dimensionnees a la demande (jamais retrecies) */
    float*  d_verts = nullptr;
    int*    d_tris = nullptr;
    int     vert_cap = 0;
    int     tri_cap = 0;
    int     n_verts = 0;
    int     n_tris = 0;

    /* --- rognage collider (M7/T4) --- persistant entre les appels a
     * bq_mesher_run (les colliders statiques ne sont fournis qu'une fois),
     * realloue seulement si sa resolution change. */
    float*  d_collider_sdf = nullptr;
    int3    collider_res = make_int3(0, 0, 0);
    float   collider_cell_size = 0.f;
    bool    has_collider = false;

    /* --- suppression des petites composantes connexes (M11/T1, D2/D3) ---
     * tampons hote scratch pour l'union-find post-marching-cubes, memes
     * conventions grow-only que le reste de la struct : jamais retrecis
     * entre deux appels (resize suffit, pas de shrink_to_fit), pour eviter
     * une reallocation hote a chaque frame de bake tant que la taille du
     * maillage ne croit pas. */
    std::vector<float> h_verts_scratch;   /* copie D2H de d_verts, puis sortie filtree */
    std::vector<int>   h_tris_scratch;    /* copie D2H de d_tris, puis sortie filtree */
    std::vector<int>   h_uf_parent;       /* union-find, n_verts entrees */
    std::vector<int>   h_uf_size;         /* union-find, poids par racine (union by size) */
};

/* find avec compression de chemin (path halving) : a chaque pas, chaque noeud
 * saute directement au grand-parent, ce qui aplatit l'arbre au fil des
 * appels sans recursion (profondeur de pile non bornee a eviter sur des
 * maillages a quelques centaines de milliers de sommets). */
static int uf_find(std::vector<int>& parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

/* union par taille (pas par rang) : le poids (h_uf_size) sert directement de
 * proxy de taille pour l'union-find ET, plus tard, pour le comptage de
 * triangles par composante -- reutiliser la meme notion de "poids" pour les
 * deux evite deux passes conceptuellement distinctes. */
static void uf_union(std::vector<int>& parent, std::vector<int>& size, int a, int b) {
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    if (size[a] < size[b]) std::swap(a, b);
    parent[b] = a;
    size[a] += size[b];
}

/* Suppression des petites composantes connexes du maillage (M11/T1, D2/D3) :
 * le marching cubes produit parfois des dizaines a plus d'un millier d'ilots
 * microscopiques (quelques particules quasi isolees qui survivent au filtre
 * k_n_cull), un defaut topologique de la reconstruction locale qu'aucun
 * filtre passe-bas sur le champ (Laplace ou Taubin) ne peut resorber -- il
 * faut le traiter au niveau du graphe du maillage, apres polygonisation.
 *
 * Union-find HOTE (pas un kernel GPU) : a la taille de maillage visee ici
 * (jusqu'a quelques centaines de milliers de sommets/triangles), un
 * union-find sequentiel a compression de chemin + union par taille est de
 * l'ordre de la milliseconde, largement sous le cout des passes de
 * lissage/marching cubes qui l'entourent -- pas de raison de risquer un
 * labeling parallele (convergence, races, iterations non bornees) pour une
 * operation qui tourne une fois par frame de bake, jamais dans la boucle
 * interne du solveur.
 *
 * GARDE-FOU (non contournable par un seuil eleve) : la composante de plus
 * grand poids (nombre de triangles) n'est JAMAIS supprimee, quel que soit
 * min_component_tris -- un OU inconditionnel avec le critere de seuil,
 * pas une exception au-dessus du seuil. Sans cela un seuil mal regle par
 * l'utilisateur pourrait vider accidentellement tout le corps principal du
 * fluide (scene ou le fluide entier tiendrait sous le seuil).
 *
 * cfg.min_component_tris <= 0 desactive entierement le passage : aucune
 * copie D2H/H2D, aucun cout, meme convention que smoothing_iters == 0
 * ailleurs dans ce fichier. */
static int bq_mesher_cleanup_small_components(BqMesher* m) {
    if (m->cfg.min_component_tris <= 0 || m->n_tris == 0) return 0;

    int nv = m->n_verts, nt = m->n_tris;

    m->h_verts_scratch.resize((size_t)nv * 3);
    m->h_tris_scratch.resize((size_t)nt * 3);
    BQ_CUDA_CHECK(cudaMemcpy(m->h_verts_scratch.data(), m->d_verts,
                             (size_t)nv * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    BQ_CUDA_CHECK(cudaMemcpy(m->h_tris_scratch.data(), m->d_tris,
                             (size_t)nt * 3 * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int>& parent = m->h_uf_parent;
    std::vector<int>& size = m->h_uf_size;
    parent.resize(nv);
    size.resize(nv);
    for (int i = 0; i < nv; ++i) { parent[i] = i; size[i] = 1; }

    const int* tris = m->h_tris_scratch.data();
    for (int t = 0; t < nt; ++t) {
        int a = tris[t * 3 + 0], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
        uf_union(parent, size, a, b);
        uf_union(parent, size, a, c);
    }

    /* poids par racine, en nombre de TRIANGLES (proxy plus fidele du poids
     * visuel/cout de rendu qu'un compte de sommets) -- tableau dense de
     * taille nv, plus simple qu'une table de hachage a ce volume. */
    std::vector<int> tri_count(nv, 0);
    int main_root = 0, main_count = -1;
    for (int t = 0; t < nt; ++t) {
        int root = uf_find(parent, tris[t * 3 + 0]);
        int c = ++tri_count[root];
        if (c > main_count) { main_count = c; main_root = root; }
    }

    /* Ensemble des racines conservees : la composante principale (garde-fou
     * D2, OU inconditionnel, jamais contournable) PLUS toute composante dont
     * le poids atteint le seuil configure. */
    std::vector<char> keep_root(nv, 0);
    for (int r = 0; r < nv; ++r) {
        if (tri_count[r] == 0) continue;
        if (r == main_root || tri_count[r] >= m->cfg.min_component_tris) keep_root[r] = 1;
    }

    /* Compaction : remappe les sommets survivants vers un espace d'indices
     * contigu en preservant leur ORDRE D'ORIGINE (pas l'ordre d'apparition
     * dans la liste de triangles) -- passe dediee sur les sommets d'abord,
     * triangles ensuite. Important pour un futur canal vitesse par sommet
     * (bq_mesher_read, parametre vel deja reserve) : celui-ci sera indexe
     * comme d_verts, donc doit rester dans le meme ordre que la sortie brute
     * du marching cubes, pas un ordre derive du parcours des triangles. */
    std::vector<int> remap(nv, -1);
    std::vector<float> out_verts;
    std::vector<int> out_tris;
    out_verts.reserve((size_t)nv * 3);
    out_tris.reserve((size_t)nt * 3);
    const float* verts = m->h_verts_scratch.data();
    int nv_out = 0, nt_out = 0;
    for (int v = 0; v < nv; ++v) {
        if (!keep_root[uf_find(parent, v)]) continue;
        remap[v] = nv_out++;
        out_verts.push_back(verts[v * 3 + 0]);
        out_verts.push_back(verts[v * 3 + 1]);
        out_verts.push_back(verts[v * 3 + 2]);
    }
    for (int t = 0; t < nt; ++t) {
        int a = tris[t * 3 + 0], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
        if (remap[a] < 0) continue; /* composante filtree */
        out_tris.push_back(remap[a]);
        out_tris.push_back(remap[b]);
        out_tris.push_back(remap[c]);
        ++nt_out;
    }

    if (nt_out == nt) return 0; /* rien a supprimer, aucune copie inutile */

    /* Recopie (pas std::move/swap) : preserve la capacite deja allouee des
     * tampons membres (politique grow-only) au lieu de la remplacer par
     * celle, plus petite, des tampons de sortie locaux. resize() vers une
     * taille plus petite ne libere jamais la capacite sous-jacente. */
    m->h_verts_scratch.resize(out_verts.size());
    std::copy(out_verts.begin(), out_verts.end(), m->h_verts_scratch.begin());
    m->h_tris_scratch.resize(out_tris.size());
    std::copy(out_tris.begin(), out_tris.end(), m->h_tris_scratch.begin());

    BQ_CUDA_CHECK(cudaMemcpy(m->d_verts, m->h_verts_scratch.data(),
                             (size_t)nv_out * 3 * sizeof(float), cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(m->d_tris, m->h_tris_scratch.data(),
                             (size_t)nt_out * 3 * sizeof(int), cudaMemcpyHostToDevice));
    m->n_verts = nv_out;
    m->n_tris = nt_out;
    return 0;
}

/* -------------------------------------------------------------------- API */
extern "C" {

BQ_API int bq_mesher_config_size(void) {
    return (int)sizeof(BqMesherConfig);
}

BQ_API void bq_mesher_default_config(BqMesherConfig* cfg) {
    if (!cfg) return;
    cfg->grid_res[0] = cfg->grid_res[1] = cfg->grid_res[2] = 128;
    cfg->cell_size        = 1.f / 128.f;
    cfg->influence_radius = 3.f / 128.f;
    cfg->particle_radius  = 1.f / 128.f;
    cfg->collider_offset  = 0.f;
    cfg->smoothing_iters  = 0;
    cfg->min_component_tris = 50;
    cfg->channels         = 0;
}

BQ_API int bq_mesher_vram_estimate(const BqMesherConfig* cfg, int n_particles,
                                   int64_t* bytes) {
    if (!cfg || !bytes) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_vram_estimate: pointeur nul");
        return -1;
    }
    int64_t n_cells = (int64_t)cfg->grid_res[0] * cfg->grid_res[1] * cfg->grid_res[2];
    if (n_cells <= 0) {
        snprintf(g_error, sizeof(g_error),
                 "bq_mesher_vram_estimate: grid_res invalide (%d,%d,%d)",
                 cfg->grid_res[0], cfg->grid_res[1], cfg->grid_res[2]);
        return -1;
    }
    BucketGrid bg = compute_bucket_grid(*cfg);
    int64_t n_buckets = (int64_t)bg.res.x * bg.res.y * bg.res.z;
    int64_t np = n_particles > 0 ? (int64_t)n_particles : 0;
    /* Doit compter TOUT ce que le mailleur alloue, pas seulement le champ :
     * c'est cette fonction que l'interface utilise pour avertir l'artiste
     * AVANT qu'il lance un bake. Le marching cubes (deux tableaux d'entiers sur
     * les aretes, 3 par cellule, et deux sur les cubes) pese 32 octets par
     * cellule, soit quatre fois le champ + le tampon de lissage -- l'oublier
     * sous-estimerait l'empreinte et l'avertissement ne servirait a rien. */
    int64_t n_edges = 3 * n_cells;
    int64_t cx = cfg->grid_res[0] > 1 ? cfg->grid_res[0] - 1 : 0;
    int64_t cy = cfg->grid_res[1] > 1 ? cfg->grid_res[1] - 1 : 0;
    int64_t cz = cfg->grid_res[2] > 1 ? cfg->grid_res[2] - 1 : 0;
    int64_t n_cubes = cx * cy * cz;
    /* M9/T1 (noyau anisotrope + filtrage) : la partie dependante des
     * particules gagne d_aniso (24 octets/particule, 6 floats) et
     * d_neighbor_count (4 octets/particule) ; le second CSR filtre pese
     * comme le premier (n_buckets*4 pour d_bucket_off2, np*4 au plus pour
     * d_bucket_idx2) -- meme rigueur que la correction deja faite une fois
     * cette session sur ce calcul (tampon de lissage), a ne pas
     * sous-estimer une deuxieme fois. */
    *bytes = n_cells * 4 + n_cells * 4 + n_buckets * 4 + np * 4 + np * 12 +
             (n_edges + 1) * 8 + (n_cubes + 1) * 8 +
             n_buckets * 4 /* d_bucket_off2 */ +
             np * 24 /* d_aniso */ + np * 4 /* d_neighbor_count */ +
             np * 4 /* d_bucket_idx2, au plus np */;
    return 0;
}

BQ_API BqMesher* bq_mesher_create(const BqMesherConfig* cfg) {
    BqMesherConfig c;
    if (cfg) c = *cfg; else bq_mesher_default_config(&c);

    int64_t n_cells = (int64_t)c.grid_res[0] * c.grid_res[1] * c.grid_res[2];
    if (n_cells <= 0) {
        snprintf(g_error, sizeof(g_error),
                 "bq_mesher_create: grid_res invalide (%d,%d,%d)",
                 c.grid_res[0], c.grid_res[1], c.grid_res[2]);
        return nullptr;
    }

    BucketGrid bg = compute_bucket_grid(c);
    int64_t n_buckets = (int64_t)bg.res.x * bg.res.y * bg.res.z;

    /* garde-fou VRAM : part independante des particules uniquement (le
     * nombre de particules n'est pas encore connu, cf. spec T1). Jamais un
     * cudaMalloc qui echoue en silence. d_bucket_off2 (M9/T1) est alloue ici
     * meme, a taille fixe n_buckets+1 comme d_bucket_off (cf. BqMesher) --
     * donc compte dans la part independante des particules. */
    int64_t bytes_indep = n_cells * 4 + n_buckets * 4 + n_cells * 4 + n_buckets * 4;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_create: cudaMemGetInfo echoue");
        return nullptr;
    }
    if ((uint64_t)bytes_indep > (uint64_t)free_b) {
        int max_res = max_cubic_res_for_budget(c.cell_size, c.influence_radius, free_b);
        snprintf(g_error, sizeof(g_error),
                 "bq_mesher_create: VRAM insuffisante (demande %.1f Mo, libre "
                 "%.1f Mo) -- resolution cubique maximale estimee sur cette "
                 "carte : %d^3",
                 bytes_indep / 1e6, (double)free_b / 1e6, max_res);
        return nullptr;
    }

    BqMesher* m = new BqMesher();
    m->cfg = c;
    m->n_cells = n_cells;
    m->bucket = bg;
    m->n_buckets = n_buckets;

    m->n_edges = 3 * n_cells;
    m->cube_res = make_int3(std::max(0, c.grid_res[0] - 1),
                            std::max(0, c.grid_res[1] - 1),
                            std::max(0, c.grid_res[2] - 1));
    m->n_cubes = (int64_t)m->cube_res.x * m->cube_res.y * m->cube_res.z;

    if (cudaMalloc(&m->d_field, (size_t)n_cells * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&m->d_field_tmp, (size_t)n_cells * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&m->d_bucket_off, (size_t)(n_buckets + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&m->d_bucket_off2, (size_t)(n_buckets + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&m->d_edge_flag, (size_t)(m->n_edges + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&m->d_edge_scan, (size_t)(m->n_edges + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&m->d_cube_tri, (size_t)(m->n_cubes + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&m->d_cube_scan, (size_t)(m->n_cubes + 1) * sizeof(int)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_create: cudaMalloc echoue");
        bq_mesher_destroy(m);
        return nullptr;
    }

    /* sentinelles a zero : cf. commentaire sur d_edge_flag dans BqMesher */
    if (cudaMemset(m->d_edge_flag + m->n_edges, 0, sizeof(int)) != cudaSuccess ||
        cudaMemset(m->d_cube_tri + m->n_cubes, 0, sizeof(int)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_create: cudaMemset echoue");
        bq_mesher_destroy(m);
        return nullptr;
    }

    /* Espace de travail de CUB, dimensionne une fois pour le plus grand des
     * deux scans. Le motif en deux appels (le premier avec un tampon nul ne
     * fait que renseigner la taille requise) est celui de CUB. */
    size_t tmp_e = 0, tmp_c = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_e, (int*)nullptr, (int*)nullptr,
                                  (int)(m->n_edges + 1));
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_c, (int*)nullptr, (int*)nullptr,
                                  (int)(m->n_cubes + 1));
    m->cub_tmp_bytes = std::max(tmp_e, tmp_c);
    if (cudaMalloc(&m->d_cub_tmp, m->cub_tmp_bytes) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error),
                 "bq_mesher_create: cudaMalloc echoue (espace de travail CUB, "
                 "%.1f Mo)", m->cub_tmp_bytes / 1e6);
        bq_mesher_destroy(m);
        return nullptr;
    }
    return m;
}

BQ_API void bq_mesher_destroy(BqMesher* m) {
    if (!m) return;
    cudaFree(m->d_field);
    cudaFree(m->d_field_tmp);
    cudaFree(m->d_bucket_off);
    cudaFree(m->d_bucket_idx);
    cudaFree(m->d_pos);
    cudaFree(m->d_aniso);
    cudaFree(m->d_neighbor_count);
    cudaFree(m->d_bucket_off2);
    cudaFree(m->d_bucket_idx2);
    cudaFree(m->d_edge_flag);
    cudaFree(m->d_edge_scan);
    cudaFree(m->d_cube_tri);
    cudaFree(m->d_cube_scan);
    cudaFree(m->d_cub_tmp);
    cudaFree(m->d_verts);
    cudaFree(m->d_tris);
    cudaFree(m->d_collider_sdf);
    delete m;
}

BQ_API int bq_mesher_set_collider_sdf(BqMesher* m, const float* sdf,
                                      const int res[3], float cell_size) {
    if (!m) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_set_collider_sdf: mesher nul");
        return -1;
    }
    if (!sdf) {
        /* efface le collider courant (cf. spec T4) */
        cudaFree(m->d_collider_sdf);
        m->d_collider_sdf = nullptr;
        m->collider_res = make_int3(0, 0, 0);
        m->collider_cell_size = 0.f;
        m->has_collider = false;
        return 0;
    }
    if (!res || res[0] <= 0 || res[1] <= 0 || res[2] <= 0 || !(cell_size > 0.f)) {
        snprintf(g_error, sizeof(g_error),
                 "bq_mesher_set_collider_sdf: parametres invalides (res=(%d,%d,%d), "
                 "cell_size=%f)", res ? res[0] : -1, res ? res[1] : -1,
                 res ? res[2] : -1, cell_size);
        return -1;
    }

    int3 new_res = make_int3(res[0], res[1], res[2]);
    if (new_res.x != m->collider_res.x || new_res.y != m->collider_res.y ||
        new_res.z != m->collider_res.z) {
        cudaFree(m->d_collider_sdf);
        m->d_collider_sdf = nullptr;
        m->collider_res = make_int3(0, 0, 0);
        int64_t n = (int64_t)new_res.x * new_res.y * new_res.z;
        if (cudaMalloc(&m->d_collider_sdf, (size_t)n * sizeof(float)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_mesher_set_collider_sdf: cudaMalloc echoue (%.1f Mo)",
                     n * sizeof(float) / 1e6);
            return -1;
        }
    }

    int64_t n = (int64_t)new_res.x * new_res.y * new_res.z;
    BQ_CUDA_CHECK(cudaMemcpy(m->d_collider_sdf, sdf, (size_t)n * sizeof(float),
                             cudaMemcpyHostToDevice));
    m->collider_res = new_res;
    m->collider_cell_size = cell_size;
    m->has_collider = true;
    return 0;
}

BQ_API int bq_mesher_run(BqMesher* m, const float* pos, int n) {
    if (!m) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_run: mesher nul");
        return -1;
    }
    if (n < 0) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_run: n negatif (%d)", n);
        return -1;
    }
    if (n > 0 && !pos) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_run: pos nul (n=%d)", n);
        return -1;
    }

    /* garde-fou VRAM complet (champ + buckets, deja alloues, + part
     * dependante des particules) -- meme soin qu'a la creation. M9/T1 :
     * d_bucket_off2 (deja alloue, taille fixe n_buckets+1) + d_aniso,
     * d_neighbor_count, d_bucket_idx2 (dependants des particules). */
    int64_t bytes_total = m->n_cells * 4 + m->n_buckets * 4 +
                          (int64_t)n * 4 + (int64_t)n * 12 +
                          m->n_buckets * 4 /* d_bucket_off2 */ +
                          (int64_t)n * 24 /* d_aniso */ +
                          (int64_t)n * 4  /* d_neighbor_count */ +
                          (int64_t)n * 4  /* d_bucket_idx2, au plus n */;
    size_t free_b = 0, total_b = 0;
    BQ_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if ((uint64_t)bytes_total > (uint64_t)free_b) {
        int max_res = max_cubic_res_for_budget(m->cfg.cell_size,
                                               m->cfg.influence_radius, free_b);
        snprintf(g_error, sizeof(g_error),
                 "bq_mesher_run: VRAM insuffisante (demande %.1f Mo, libre "
                 "%.1f Mo) -- resolution cubique maximale estimee sur cette "
                 "carte : %d^3",
                 bytes_total / 1e6, (double)free_b / 1e6, max_res);
        return -1;
    }

    /* reallocation seulement quand la capacite courante est depassee, taille
     * exacte -- meme politique que bq_set_colliders (mlsmpm.cu). */
    if (n > m->particle_cap) {
        cudaFree(m->d_pos); cudaFree(m->d_bucket_idx);
        cudaFree(m->d_aniso); cudaFree(m->d_neighbor_count); cudaFree(m->d_bucket_idx2);
        m->d_pos = nullptr; m->d_bucket_idx = nullptr;
        m->d_aniso = nullptr; m->d_neighbor_count = nullptr; m->d_bucket_idx2 = nullptr;
        m->particle_cap = 0;
        if (cudaMalloc(&m->d_pos, (size_t)n * sizeof(float3)) != cudaSuccess ||
            cudaMalloc(&m->d_bucket_idx, (size_t)n * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&m->d_aniso, (size_t)n * 6 * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&m->d_neighbor_count, (size_t)n * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&m->d_bucket_idx2, (size_t)n * sizeof(int)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_mesher_run: cudaMalloc echoue (n=%d)", n);
            return -1;
        }
        m->particle_cap = n;
    }

    /* bucketing CSR cote hote (pos est en memoire hote, cf. note en tete de
     * fichier), puis televersement. */
    int nb = (int)m->n_buckets;
    std::vector<int> off((size_t)nb + 1, 0);
    std::vector<int> bidx_of_p(n);
    for (int p = 0; p < n; ++p) {
        float x = pos[3 * p + 0], y = pos[3 * p + 1], z = pos[3 * p + 2];
        int bi = (int)floorf((x - m->bucket.origin.x) / m->bucket.h);
        int bj = (int)floorf((y - m->bucket.origin.y) / m->bucket.h);
        int bk = (int)floorf((z - m->bucket.origin.z) / m->bucket.h);
        bi = std::min(std::max(bi, 0), m->bucket.res.x - 1);
        bj = std::min(std::max(bj, 0), m->bucket.res.y - 1);
        bk = std::min(std::max(bk, 0), m->bucket.res.z - 1);
        int b = (bi * m->bucket.res.y + bj) * m->bucket.res.z + bk;
        bidx_of_p[p] = b;
        off[b + 1]++;
    }
    for (int b = 0; b < nb; ++b) off[b + 1] += off[b];

    std::vector<int> idx(n);
    std::vector<int> cursor(off.begin(), off.end());
    for (int p = 0; p < n; ++p) {
        int b = bidx_of_p[p];
        idx[cursor[b]++] = p;
    }

    BQ_CUDA_CHECK(cudaMemcpy(m->d_bucket_off, off.data(),
                             (size_t)(nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (n > 0) {
        BQ_CUDA_CHECK(cudaMemcpy(m->d_pos, pos, (size_t)n * sizeof(float3),
                                 cudaMemcpyHostToDevice));
        BQ_CUDA_CHECK(cudaMemcpy(m->d_bucket_idx, idx.data(), (size_t)n * sizeof(int),
                                 cudaMemcpyHostToDevice));
    }

    int3 res3 = make_int3(m->cfg.grid_res[0], m->cfg.grid_res[1],
                          m->cfg.grid_res[2]);
    dim3 bs(256), gs((unsigned)((m->n_cells + 255) / 256));

    /* ------------------------------------ pre-passe anisotrope (M9/T1, D1/D5)
     * Un thread par particule fluide, CSR COMPLET (d_bucket_off/d_bucket_idx,
     * TOUTES les particules) -- cf. en-tete de k_mc_compute_aniso. */
    if (n > 0) {
        dim3 gp((unsigned)((n + 255) / 256));
        k_mc_compute_aniso<<<gp, bs>>>(
            m->d_pos, m->d_bucket_off, m->d_bucket_idx,
            m->bucket.res, m->bucket.h, m->bucket.origin,
            m->cfg.influence_radius, n, m->d_aniso, m->d_neighbor_count);
        BQ_CUDA_CHECK(cudaGetLastError());
    }

    /* ------------------------- second CSR, filtre des particules isolees (D3)
     * Meme algorithme EXACT que la construction du premier CSR ci-dessus
     * (comptage/somme prefixe/remplissage), restreint aux particules dont
     * neighbor_count > k_n_cull. bidx_of_p (deja calcule pour le premier CSR)
     * est reutilise : meme grille de buckets, pas besoin de le recalculer. */
    std::vector<int> ncount(n);
    if (n > 0) {
        BQ_CUDA_CHECK(cudaMemcpy(ncount.data(), m->d_neighbor_count,
                                 (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
    }
    std::vector<int> off2((size_t)nb + 1, 0);
    for (int p = 0; p < n; ++p) {
        if (ncount[p] <= k_n_cull) continue;
        off2[bidx_of_p[p] + 1]++;
    }
    for (int b = 0; b < nb; ++b) off2[b + 1] += off2[b];
    int n_survive = off2[nb];

    std::vector<int> idx2((size_t)n_survive);
    std::vector<int> cursor2(off2.begin(), off2.end());
    for (int p = 0; p < n; ++p) {
        if (ncount[p] <= k_n_cull) continue;
        int b = bidx_of_p[p];
        idx2[cursor2[b]++] = p;
    }

    BQ_CUDA_CHECK(cudaMemcpy(m->d_bucket_off2, off2.data(),
                             (size_t)(nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (n_survive > 0) {
        BQ_CUDA_CHECK(cudaMemcpy(m->d_bucket_idx2, idx2.data(),
                                 (size_t)n_survive * sizeof(int), cudaMemcpyHostToDevice));
    }

    /* Champ Zhu-Bridson anisotrope (D1) : consomme d_aniso et le CSR FILTRE
     * (d_bucket_off2/d_bucket_idx2, D3) -- les particules quasi isolees ne
     * contribuent plus jamais au champ. */
    k_zhu_bridson_field<<<gs, bs>>>(
        m->d_field, m->d_pos, m->d_aniso, m->d_bucket_off2, m->d_bucket_idx2,
        m->bucket.res, m->bucket.h, m->bucket.origin,
        res3,
        m->cfg.cell_size, m->cfg.influence_radius, m->cfg.particle_radius,
        (int)m->n_cells);
    BQ_CUDA_CHECK(cudaGetLastError());

    /* Lissage du champ, AVANT le rognage collider -- cf. justification en
     * tete de k_mc_smooth_field. Taubin (D1) : chaque unite de
     * smoothing_iters declenche DEUX lancements de kernel (passe lambda puis
     * passe mu), sur le meme ping-pong d_field/d_field_tmp -- aucune
     * nouvelle allocation. Le nombre total de lancements est donc pair,
     * `cur` retombe systematiquement sur m->d_field a la fin d'une iteration
     * complete quand smoothing_iters > 0 ; le test `cur != m->d_field` est
     * conserve tel quel comme garde-fou (il resterait correct meme si ce
     * nombre devenait impair). */
    if (m->cfg.smoothing_iters > 0) {
        float* cur = m->d_field;
        float* nxt = m->d_field_tmp;
        for (int it = 0; it < m->cfg.smoothing_iters; ++it) {
            k_mc_smooth_field<<<gs, bs>>>(nxt, cur, res3, m->n_cells, BQ_TAUBIN_LAMBDA);
            BQ_CUDA_CHECK(cudaGetLastError());
            float* t = cur; cur = nxt; nxt = t;

            k_mc_smooth_field<<<gs, bs>>>(nxt, cur, res3, m->n_cells, BQ_TAUBIN_MU);
            BQ_CUDA_CHECK(cudaGetLastError());
            t = cur; cur = nxt; nxt = t;
        }
        if (cur != m->d_field) {
            BQ_CUDA_CHECK(cudaMemcpy(m->d_field, cur,
                                     (size_t)m->n_cells * sizeof(float),
                                     cudaMemcpyDeviceToDevice));
        }
    }

    /* Rognage collider (M7/T4, D5) -- avant le marching cubes. Lance
     * seulement si un collider est defini : cout nul dans le cas courant
     * (pas de collider). */
    if (m->has_collider) {
        k_mc_crop_collider<<<gs, bs>>>(
            m->d_field, m->d_collider_sdf,
            res3,
            m->cfg.cell_size, m->collider_res, m->collider_cell_size,
            m->cfg.collider_offset, m->n_cells);
        BQ_CUDA_CHECK(cudaGetLastError());
    }

    /* ------------------------------------------ marching cubes (D4), 6 passes
     * Sommets dedupliques par arete : chaque arete de grille traversee par
     * l'isosurface produit UN sommet, et une arete appartient a la cellule dont
     * elle part. Sans cela chaque triangle porterait ses 3 sommets propres, le
     * maillage serait deux a trois fois plus lourd et Blender ne pourrait pas
     * lisser les normales. */
    dim3 ge((unsigned)((m->n_edges + 255) / 256));
    dim3 gc((unsigned)((m->n_cubes + 255) / 256));

    k_mc_mark_edges<<<ge, bs>>>(m->d_edge_flag, m->d_field, res3, m->n_edges);
    BQ_CUDA_CHECK(cudaGetLastError());
    BQ_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        m->d_cub_tmp, m->cub_tmp_bytes, m->d_edge_flag, m->d_edge_scan,
        (int)(m->n_edges + 1)));

    k_mc_count_tris<<<gc, bs>>>(m->d_cube_tri, m->d_field, res3, m->cube_res,
                                m->n_cubes);
    BQ_CUDA_CHECK(cudaGetLastError());
    BQ_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        m->d_cub_tmp, m->cub_tmp_bytes, m->d_cube_tri, m->d_cube_scan,
        (int)(m->n_cubes + 1)));

    int nv = 0, nt = 0;
    BQ_CUDA_CHECK(cudaMemcpy(&nv, m->d_edge_scan + m->n_edges, sizeof(int),
                             cudaMemcpyDeviceToHost));
    BQ_CUDA_CHECK(cudaMemcpy(&nt, m->d_cube_scan + m->n_cubes, sizeof(int),
                             cudaMemcpyDeviceToHost));
    m->n_verts = nv;
    m->n_tris = nt;

    /* Champ entierement positif (aucun fluide) : cas NORMAL, pas une erreur.
     * On sort sans lancer de kernel sur des tableaux vides ni allouer 0 octet. */
    if (nv == 0 || nt == 0) {
        m->n_verts = 0;
        m->n_tris = 0;
        BQ_CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    }

    if (nv > m->vert_cap) {
        cudaFree(m->d_verts); m->d_verts = nullptr; m->vert_cap = 0;
        if (cudaMalloc(&m->d_verts, (size_t)nv * 3 * sizeof(float)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_mesher_run: cudaMalloc echoue (%d sommets, %.1f Mo)",
                     nv, nv * 3.0 * sizeof(float) / 1e6);
            return -1;
        }
        m->vert_cap = nv;
    }
    if (nt > m->tri_cap) {
        cudaFree(m->d_tris); m->d_tris = nullptr; m->tri_cap = 0;
        if (cudaMalloc(&m->d_tris, (size_t)nt * 3 * sizeof(int)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_mesher_run: cudaMalloc echoue (%d triangles, %.1f Mo)",
                     nt, nt * 3.0 * sizeof(int) / 1e6);
            return -1;
        }
        m->tri_cap = nt;
    }

    k_mc_emit_vertices<<<ge, bs>>>(m->d_verts, m->d_edge_flag, m->d_edge_scan,
                                   m->d_field, res3, m->cfg.cell_size,
                                   m->n_edges);
    BQ_CUDA_CHECK(cudaGetLastError());
    k_mc_emit_indices<<<gc, bs>>>(m->d_tris, m->d_cube_tri, m->d_cube_scan,
                                  m->d_edge_scan, m->d_field, res3, m->cube_res,
                                  m->n_cubes);
    BQ_CUDA_CHECK(cudaGetLastError());
    BQ_CUDA_CHECK(cudaDeviceSynchronize());

    /* Suppression des petites composantes connexes (M11/T1, D2/D3) -- apres
     * que nv/nt bruts et m->n_verts/m->n_tris aient ete fixes ci-dessus. */
    if (bq_mesher_cleanup_small_components(m) != 0) return -1;

    return 0;
}

BQ_API int bq_mesher_counts(const BqMesher* m, int* n_verts, int* n_tris) {
    if (!m) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_counts: mesher nul");
        return -1;
    }
    if (n_verts) *n_verts = m->n_verts;
    if (n_tris) *n_tris = m->n_tris;
    return 0;
}

BQ_API int bq_mesher_read(BqMesher* m, float* verts, int* tris, float* vel) {
    if (!m) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_read: mesher nul");
        return -1;
    }
    /* vel est ignore tant que le canal de vitesse par sommet n'est pas
     * implemente (D9) ; le parametre existe pour ne pas casser l'ABI ensuite. */
    (void)vel;
    if (m->n_verts > 0 && verts) {
        BQ_CUDA_CHECK(cudaMemcpy(verts, m->d_verts,
                                 (size_t)m->n_verts * 3 * sizeof(float),
                                 cudaMemcpyDeviceToHost));
    }
    if (m->n_tris > 0 && tris) {
        BQ_CUDA_CHECK(cudaMemcpy(tris, m->d_tris,
                                 (size_t)m->n_tris * 3 * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    }
    return 0;
}

BQ_API int bq_mesher_read_field(BqMesher* m, float* dst) {
    if (!m || !dst) {
        snprintf(g_error, sizeof(g_error), "bq_mesher_read_field: pointeur nul");
        return -1;
    }
    BQ_CUDA_CHECK(cudaMemcpy(dst, m->d_field, (size_t)m->n_cells * sizeof(float),
                             cudaMemcpyDeviceToHost));
    return 0;
}

} /* extern "C" */

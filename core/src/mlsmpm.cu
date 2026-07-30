/* mlsmpm.cu -- solveur MLS-MPM 3D (Hu et al. 2018), B-splines quadratiques.
 *
 * REFERENCE : scripts/ref_mlsmpm.py est la specification executable de ce
 * fichier. Chaque kernel est la transcription d'un bloc de Sim.substep().
 * En cas de doute sur une formule, la reference NumPy fait foi.
 *
 * Pipeline par substep :
 *   1. k_clear_grid  : remise a zero (masse, quantite de mouvement)
 *   2. k_p2g         : maj de F, contrainte de Cauchy, scatter atomique
 *   3. k_grid_update : v = mv/m, gravite, conditions aux limites separantes
 *   4. k_g2p         : gather v et C (APIC), advection, maj de J (eau)
 */
#define BQ_BUILD
#include "bourrasque.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

/* ------------------------------------------------------------------ erreurs */
static char g_error[512] = "";

#define BQ_CUDA_CHECK(call)                                                  \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            snprintf(g_error, sizeof(g_error), "%s:%d CUDA: %s", __FILE__,   \
                     __LINE__, cudaGetErrorString(err_));                    \
            return -1;                                                       \
        }                                                                    \
    } while (0)

/* ------------------------------------------------------------- petite algebre */
struct mat3 {
    float m[9]; /* row-major */
    __host__ __device__ static mat3 zero() { mat3 r{}; return r; }
    __host__ __device__ static mat3 identity() {
        mat3 r{}; r.m[0] = r.m[4] = r.m[8] = 1.f; return r;
    }
};

__device__ inline mat3 operator+(const mat3& a, const mat3& b) {
    mat3 r; for (int i = 0; i < 9; ++i) r.m[i] = a.m[i] + b.m[i]; return r;
}
__device__ inline mat3 operator*(float s, const mat3& a) {
    mat3 r; for (int i = 0; i < 9; ++i) r.m[i] = s * a.m[i]; return r;
}
__device__ inline mat3 matmul(const mat3& a, const mat3& b) {
    mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float s = 0.f;
            for (int k = 0; k < 3; ++k) s += a.m[3 * i + k] * b.m[3 * k + j];
            r.m[3 * i + j] = s;
        }
    return r;
}
__device__ inline mat3 transpose(const mat3& a) {
    mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[3 * i + j] = a.m[3 * j + i];
    return r;
}
__device__ inline float det(const mat3& a) {
    return a.m[0] * (a.m[4] * a.m[8] - a.m[5] * a.m[7])
         - a.m[1] * (a.m[3] * a.m[8] - a.m[5] * a.m[6])
         + a.m[2] * (a.m[3] * a.m[7] - a.m[4] * a.m[6]);
}
__device__ inline mat3 inverse(const mat3& a) {
    float d = det(a);
    float id = (fabsf(d) > 1e-20f) ? 1.f / d : 0.f;
    mat3 r;
    r.m[0] =  (a.m[4] * a.m[8] - a.m[5] * a.m[7]) * id;
    r.m[1] = -(a.m[1] * a.m[8] - a.m[2] * a.m[7]) * id;
    r.m[2] =  (a.m[1] * a.m[5] - a.m[2] * a.m[4]) * id;
    r.m[3] = -(a.m[3] * a.m[8] - a.m[5] * a.m[6]) * id;
    r.m[4] =  (a.m[0] * a.m[8] - a.m[2] * a.m[6]) * id;
    r.m[5] = -(a.m[0] * a.m[5] - a.m[2] * a.m[3]) * id;
    r.m[6] =  (a.m[3] * a.m[7] - a.m[4] * a.m[6]) * id;
    r.m[7] = -(a.m[0] * a.m[7] - a.m[1] * a.m[6]) * id;
    r.m[8] =  (a.m[0] * a.m[4] - a.m[1] * a.m[3]) * id;
    return r;
}
__device__ inline float3 matvec(const mat3& a, float3 v) {
    return make_float3(a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z,
                       a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z,
                       a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z);
}
__device__ inline mat3 outer(float3 a, float3 b) {
    mat3 r;
    r.m[0] = a.x * b.x; r.m[1] = a.x * b.y; r.m[2] = a.x * b.z;
    r.m[3] = a.y * b.x; r.m[4] = a.y * b.y; r.m[5] = a.y * b.z;
    r.m[6] = a.z * b.x; r.m[7] = a.z * b.y; r.m[8] = a.z * b.z;
    return r;
}

/* ------------------------------------------------------- colliders : geo */
/* Plus proche point sur le triangle (a,b,c), Ericson "Real-Time Collision
 * Detection" 5.1.5. Renvoie aussi les poids barycentriques (u,v,w) du point
 * trouve, pour interpoler la vitesse aux sommets. */
__device__ inline float3 closest_pt_triangle(float3 p, float3 a, float3 b,
                                              float3 c, float& u, float& v,
                                              float& w) {
    float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
    float3 ap = make_float3(p.x - a.x, p.y - a.y, p.z - a.z);
    float d1 = ab.x * ap.x + ab.y * ap.y + ab.z * ap.z;
    float d2 = ac.x * ap.x + ac.y * ap.y + ac.z * ap.z;
    if (d1 <= 0.f && d2 <= 0.f) { u = 1.f; v = 0.f; w = 0.f; return a; }

    float3 bp = make_float3(p.x - b.x, p.y - b.y, p.z - b.z);
    float d3 = ab.x * bp.x + ab.y * bp.y + ab.z * bp.z;
    float d4 = ac.x * bp.x + ac.y * bp.y + ac.z * bp.z;
    if (d3 >= 0.f && d4 <= d3) { u = 0.f; v = 1.f; w = 0.f; return b; }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        float t = d1 / (d1 - d3);
        u = 1.f - t; v = t; w = 0.f;
        return make_float3(a.x + t * ab.x, a.y + t * ab.y, a.z + t * ab.z);
    }

    float3 cp = make_float3(p.x - c.x, p.y - c.y, p.z - c.z);
    float d5 = ab.x * cp.x + ab.y * cp.y + ab.z * cp.z;
    float d6 = ac.x * cp.x + ac.y * cp.y + ac.z * cp.z;
    if (d6 >= 0.f && d5 <= d6) { u = 0.f; v = 0.f; w = 1.f; return c; }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        float t = d2 / (d2 - d6);
        u = 1.f - t; v = 0.f; w = t;
        return make_float3(a.x + t * ac.x, a.y + t * ac.y, a.z + t * ac.z);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        u = 0.f; v = 1.f - t; w = t;
        return make_float3(b.x + t * (c.x - b.x), b.y + t * (c.y - b.y),
                           b.z + t * (c.z - b.z));
    }

    float denom = 1.f / (va + vb + vc);
    v = vb * denom; w = vc * denom; u = 1.f - v - w;
    return make_float3(a.x + ab.x * v + ac.x * w, a.y + ab.y * v + ac.y * w,
                       a.z + ab.z * v + ac.z * w);
}

/* NOTE : l'ancien calcul de signe par winding number generalise (Jacobson et
 * al. 2013, somme d'angles solides via Van Oosterom & Strackee) a ete retire.
 * C'etait une somme GLOBALE sur tous les triangles, impossible a tronquer
 * spatialement -- c'est elle qui rendait k_collider_sdf lineaire en n_tri et
 * dominait son cout (chaque paire (cellule, triangle) evaluait un atan2 en
 * plus de closest_pt_triangle). Le signe est desormais obtenu par un test
 * local pres de la surface (graines) suivi d'une propagation par connexite
 * (voir k_sdf_propagate_sign plus bas), un cout qui ne depend plus du nombre
 * de triangles.
 *
 * IMPORTANT : l'amorce du signe ne repose plus sur l'AABB des colliders (elle
 * ne sert plus qu'a borner spatialement la recherche du triangle le plus
 * proche, pour la performance). Amorcer depuis "hors de l'AABB = exterieur"
 * echouait des que l'AABB dilatee couvrait tout le domaine (aucune cellule
 * d'amorce, propagation qui ne demarre jamais, domaine entier signe solide).
 * L'amorce vient desormais du test local de signe dans la bande etroite
 * pres de la surface (fiable la ou le triangle le plus proche est
 * representatif), et la propagation relaie ce signe aux cellules plus
 * loin (fiable la ou le test local ne l'est plus). Cf. BQ_SDF_STATE_* et
 * les commentaires de k_sdf_unsigned / k_sdf_propagate_sign / k_sdf_finalize_sign. */

/* Etat de signe d'une cellule pendant la resolution (tableau d_ext, un octet
 * par cellule) :
 *   UNKNOWN  : signe pas encore determine, sera resolu par propagation
 *              depuis un voisin deja resolu (ou restera UNKNOWN si aucune
 *              graine ne l'atteint -- cf. regle de securite dans
 *              k_sdf_finalize_sign).
 *   EXTERIOR : hors du collider, determine soit par le test local pres de
 *              la surface (graine), soit par propagation depuis un voisin
 *              EXTERIOR.
 *   INTERIOR : dans le collider, meme origine (graine locale ou
 *              propagation) que EXTERIOR mais signe oppose. */
#define BQ_SDF_STATE_UNKNOWN  0
#define BQ_SDF_STATE_EXTERIOR 1
#define BQ_SDF_STATE_INTERIOR 2

/* Seuil de bande, en unites de dx : une cellule dont la distance non signee
 * est sous ce seuil est consideree assez pres de la surface pour que le
 * triangle le plus proche soit geometriquement representatif -- le test
 * local de signe y est fiable et sert de graine. La moitie de la diagonale
 * d'une cellule cubique vaut (sqrt(3)/2)*dx ~= 0.866*dx : c'est la distance
 * maximale entre le centre de la cellule et un point de surface qui la
 * traverse effectivement (pire cas : la surface coupe un coin). 1*dx majore
 * cette borne avec marge, donc couvre systematiquement toute cellule
 * traversee par la surface. Un seuil plus petit (par ex. 0.3*dx) laisserait
 * des cellules traversees sans graine fiable ; un seuil plus grand (par ex.
 * 3*dx) risquerait de faire porter le test local a des cellules ou le
 * triangle le plus proche n'est plus representatif (cavites etroites). */
#define BQ_SDF_WALL_EPS_MULT 1.0f

/* --------------------------------------------- colliders : grille de buckets
 * Distance non signee : recherche par grille uniforme de buckets plutot que
 * parcours exhaustif de tous les triangles. Construite cote HOTE (cout
 * negligeable : quelques dizaines de milliers de triangles, largement sous
 * la milliseconde, contre ~100 ms cote GPU pour l'ancien parcours exhaustif)
 * puis televersee -- ce qui evite un tri parallele et un scan sur device
 * pour une structure qui n'a pas besoin d'etre construite a chaque frame par
 * le GPU. Format CSR classique : bucket_off[b .. b+1[ indexe bucket_tri pour
 * les triangles du bucket b.
 *
 * Pas de bucket h = BQ_BUCKET_DX_MULT * dx = 2*dx. La bande active des
 * cellules considerees (bq_set_colliders, pad AABB) fait 3*dx d'epaisseur :
 * avec h = 2*dx, 1 a 2 anneaux de buckets suffisent a la couvrir, donc peu
 * d'anneaux a visiter par cellule dans le cas courant. Un h plus fin (1*dx)
 * multiplierait les buckets vides a traverser en anneaux pour un meme rayon
 * de recherche ; un h plus grossier (4*dx) fait croitre le nombre de
 * triangles par bucket avec la densite du maillage collider (a 50000
 * triangles sur la sphere du banc d'essai, un bucket de 4*dx contient deja
 * environ 150 triangles, on retombe pres du cout du parcours exhaustif dans
 * chaque bucket visite). 2*dx est le compromis retenu ; il reste correct
 * (pas seulement rapide) pour tout h > 0, c'est le choix qui ne change que
 * le cout, jamais le resultat. */
#define BQ_BUCKET_DX_MULT 2.0f

/* Plafonds de la grille de buckets : sans eux, un collider tres etendu (plan
 * de sol de plusieurs dizaines de metres, cas d'usage courant) produit une
 * resolution deduite de l'AABB sans aucune borne -- des millions de buckets
 * alloues et copies vers le device a CHAQUE bq_set_colliders, et en 3D un
 * collider volumineux peut meme faire deborder l'entier utilise pour
 * dimensionner le tableau (nb negatif, cudaMalloc corrompu). BQ_BUCKET_MAX_
 * AXIS_RES borne la resolution par axe (utile pour un collider tres allonge
 * sur un seul axe -- un sol fin et tres etendu -- ou l'AABB elle-meme ne
 * dirait rien du volume total) ; BQ_BUCKET_MAX_TOTAL_BUCKETS (leur cube)
 * borne le produit. 128 par axe donne au plus 128^3 = 2 097 152 buckets,
 * soit ~8 Mo pour bucket_off (int32) -- une allocation qui reste largement
 * sous la barre du "delirant" meme au pire cas, alors que 512 (le prochain
 * palier naturel) ferait deja 512^3 ~= 134M buckets, ~537 Mo, par frame.
 * Depasser ce plafond ne degrade que la SELECTIVITE de la recherche (des
 * buckets plus gros contiennent plus de triangles a filtrer un par un dans
 * closest_pt_triangle), jamais la justesse : la recherche par anneaux reste
 * correcte pour tout h > 0 (cf. note sur BQ_BUCKET_DX_MULT plus haut). */
#define BQ_BUCKET_MAX_AXIS_RES 128
#define BQ_BUCKET_MAX_TOTAL_BUCKETS \
    ((int64_t)BQ_BUCKET_MAX_AXIS_RES * BQ_BUCKET_MAX_AXIS_RES * BQ_BUCKET_MAX_AXIS_RES)

/* Demi-epaisseur de la couche de contact, en multiples de dx.
 *
 * 1.5 dx est le rayon du stencil B-spline quadratique : la couche couvre donc
 * exactement l'ensemble des noeuds qu'une particule au contact peut influencer.
 * Un seul plan de noeuds contraints ne suffit PAS a arreter le fluide -- le
 * transfert grille-particule etant une moyenne ponderee sur 3 noeuds par axe,
 * la particule ne fait que ralentir puis s'infiltre. C'est la meme echelle que
 * c_p.bound = 3 cellules, utilisee par les parois du domaine.
 *
 * Mesure a l'appui, sur un contenant ferme dont on fait varier l'epaisseur de
 * paroi (fuite du fluide hors du contenant apres 5 s) :
 *   h = 0.5 dx -> 24.7 % de fuite   |   h = 1.5 dx -> 0.1 %
 * Et aucun epaississement apparent mesurable sur un collider epais : le fluide
 * s'arrete a la surface exacte dans les deux cas, parce que les noeuds
 * exterieurs y sont en contact unilateral (cf. normal_corroborated) et ne
 * bloquent donc que l'entree. */
#define BQ_CONTACT_BAND_MULT 1.5f

/* Distance minimale, en multiples de dx, a laquelle la contrainte de position de
 * k_g2p maintient une particule de la surface d'un collider. La condition aux
 * limites de k_grid_update agit sur les vitesses de grille : elle est molle par
 * nature (le transfert grille-particule moyenne les noeuds contraints avec les
 * noeuds libres). Cette contrainte-ci agit sur les positions et est dure ; c'est
 * elle qui garantit qu'aucune particule ne franchit une paroi, quelle que soit
 * son epaisseur devant dx. */
#define BQ_CONTACT_PUSH_MULT 0.5f

/* Decalage d'echantillonnage du champ de collider, en multiples de dx, destine a
 * lever les coincidences exactes noeud/surface (cf. k_sdf_unsigned). Assez grand
 * pour que le carre de la distance reste tres au-dessus du seuil de degenerescence
 * en float32, assez petit pour etre physiquement insignifiant (15 um a dx = 15 mm). */
#define BQ_SDF_NODE_EPS 1e-3f

struct BucketGridHost {
    float3 origin;
    float  h;
    int3   res;
};

/* Calcule l'AABB de tous les triangles, dimensionne la grille de buckets
 * (h de depart BQ_BUCKET_DX_MULT*dx, agrandi si necessaire pour respecter
 * les plafonds ci-dessus -- degradation propre : le pas grossit, la
 * resolution baisse, la recherche reste correcte), puis range chaque
 * triangle dans tous les buckets que son AABB recouvre (CSR : deux passes,
 * comptage puis remplissage). tri : 9*n_tri floats (3 sommets xyz par
 * triangle). Renvoie false (et remplit g_error) si la configuration reste
 * hors bornes meme apres agrandissement de h -- ne doit normalement jamais
 * se produire, garde-fou plutot que debordement silencieux. */
static bool build_bucket_grid(const float* tri, int n_tri, float h_base,
                              BucketGridHost& g, std::vector<int>& offsets,
                              std::vector<int>& tri_idx) {
    float3 lo = make_float3(3.4e38f, 3.4e38f, 3.4e38f);
    float3 hi = make_float3(-3.4e38f, -3.4e38f, -3.4e38f);
    for (int i = 0; i < 3 * n_tri; ++i) {
        float x = tri[3 * i], y = tri[3 * i + 1], z = tri[3 * i + 2];
        lo.x = fminf(lo.x, x); lo.y = fminf(lo.y, y); lo.z = fminf(lo.z, z);
        hi.x = fmaxf(hi.x, x); hi.y = fmaxf(hi.y, y); hi.z = fmaxf(hi.z, z);
    }
    g.origin = lo;

    float ext_x = fmaxf(hi.x - lo.x, 1e-6f);
    float ext_y = fmaxf(hi.y - lo.y, 1e-6f);
    float ext_z = fmaxf(hi.z - lo.z, 1e-6f);

    /* h de depart eventuellement agrandi : par axe (utile si un seul axe est
     * demesure, ex. un sol fin et tres etendu) et par volume total (cas d'un
     * collider volumineux dans les trois dimensions). Arithmetique double
     * pour le calcul de volume/cube : les extents peuvent etre tres grands,
     * un calcul en simple precision risquerait de saturer avant meme d'avoir
     * une reponse utile. */
    float h = h_base;
    h = fmaxf(h, ext_x / (float)BQ_BUCKET_MAX_AXIS_RES);
    h = fmaxf(h, ext_y / (float)BQ_BUCKET_MAX_AXIS_RES);
    h = fmaxf(h, ext_z / (float)BQ_BUCKET_MAX_AXIS_RES);
    double vol = (double)ext_x * (double)ext_y * (double)ext_z;
    double h_vol = cbrt(vol / (double)BQ_BUCKET_MAX_TOTAL_BUCKETS);
    if (h_vol > (double)h) h = (float)h_vol;

    auto compute_res = [&]() {
        g.res.x = (int)floorf(ext_x / h) + 1; if (g.res.x < 1) g.res.x = 1;
        g.res.y = (int)floorf(ext_y / h) + 1; if (g.res.y < 1) g.res.y = 1;
        g.res.z = (int)floorf(ext_z / h) + 1; if (g.res.z < 1) g.res.z = 1;
    };
    g.h = h;
    compute_res();
    int64_t nb64 = (int64_t)g.res.x * g.res.y * g.res.z;

    /* garde residuelle : l'arrondi floor()+1 par axe peut, dans de rares cas,
     * faire deborder legerement les plafonds vises ci-dessus -- un dernier
     * agrandissement direct de h les absorbe. */
    if (nb64 > BQ_BUCKET_MAX_TOTAL_BUCKETS) {
        double scale = cbrt((double)nb64 / (double)BQ_BUCKET_MAX_TOTAL_BUCKETS) * 1.01;
        h = (float)((double)h * scale);
        g.h = h;
        compute_res();
        nb64 = (int64_t)g.res.x * g.res.y * g.res.z;
    }
    if (nb64 <= 0 || nb64 > BQ_BUCKET_MAX_TOTAL_BUCKETS) {
        snprintf(g_error, sizeof(g_error),
                 "bq_set_colliders: grille de buckets hors bornes meme apres "
                 "degradation (nb=%lld, max=%lld)",
                 (long long)nb64, (long long)BQ_BUCKET_MAX_TOTAL_BUCKETS);
        return false;
    }
    int nb = (int)nb64; /* borne par BQ_BUCKET_MAX_TOTAL_BUCKETS, tient dans un int */
    offsets.assign((size_t)nb + 1, 0);

    auto bucket_of = [&](float x, float y, float z, int& bi, int& bj, int& bk) {
        bi = (int)floorf((x - g.origin.x) / h);
        if (bi < 0) bi = 0; else if (bi >= g.res.x) bi = g.res.x - 1;
        bj = (int)floorf((y - g.origin.y) / h);
        if (bj < 0) bj = 0; else if (bj >= g.res.y) bj = g.res.y - 1;
        bk = (int)floorf((z - g.origin.z) / h);
        if (bk < 0) bk = 0; else if (bk >= g.res.z) bk = g.res.z - 1;
    };

    std::vector<int> imin(n_tri), imax(n_tri), jmin(n_tri), jmax(n_tri),
        kmin(n_tri), kmax(n_tri);
    for (int t = 0; t < n_tri; ++t) {
        float3 a = make_float3(tri[9*t+0], tri[9*t+1], tri[9*t+2]);
        float3 b = make_float3(tri[9*t+3], tri[9*t+4], tri[9*t+5]);
        float3 c = make_float3(tri[9*t+6], tri[9*t+7], tri[9*t+8]);
        float txl = fminf(a.x, fminf(b.x, c.x)), txh = fmaxf(a.x, fmaxf(b.x, c.x));
        float tyl = fminf(a.y, fminf(b.y, c.y)), tyh = fmaxf(a.y, fmaxf(b.y, c.y));
        float tzl = fminf(a.z, fminf(b.z, c.z)), tzh = fmaxf(a.z, fmaxf(b.z, c.z));
        int bi0, bj0, bk0, bi1, bj1, bk1;
        bucket_of(txl, tyl, tzl, bi0, bj0, bk0);
        bucket_of(txh, tyh, tzh, bi1, bj1, bk1);
        imin[t] = bi0; imax[t] = bi1; jmin[t] = bj0; jmax[t] = bj1;
        kmin[t] = bk0; kmax[t] = bk1;
        for (int ii = bi0; ii <= bi1; ++ii)
            for (int jj = bj0; jj <= bj1; ++jj)
                for (int kk = bk0; kk <= bk1; ++kk)
                    offsets[(ii * g.res.y + jj) * g.res.z + kk + 1]++;
    }
    for (int b = 0; b < nb; ++b) offsets[b + 1] += offsets[b];

    tri_idx.resize(offsets[nb]);
    std::vector<int> cursor(offsets.begin(), offsets.end());
    for (int t = 0; t < n_tri; ++t) {
        for (int ii = imin[t]; ii <= imax[t]; ++ii)
            for (int jj = jmin[t]; jj <= jmax[t]; ++jj)
                for (int kk = kmin[t]; kk <= kmax[t]; ++kk) {
                    int b = (ii * g.res.y + jj) * g.res.z + kk;
                    tri_idx[cursor[b]++] = t;
                }
    }
    return true;
}

/* Decomposition polaire par iteration de Higham : R <- (R + R^-T)/2.
 * Suffisant pour le corotationnel fixe (on ne veut que R, pas la SVD
 * complete). La SVD 3x3 (McAdams) arrivera en M3 pour la plasticite. */
__device__ inline mat3 polar_rotation(const mat3& F) {
    if (det(F) < 1e-9f) return mat3::identity(); /* garde-fou inversion */
    mat3 R = F;
    for (int i = 0; i < 10; ++i)
        R = 0.5f * (R + transpose(inverse(R)));
    return R;
}

/* --------------------------------------------------------------- parametres */
#define BQ_MAX_MATERIALS 8

struct MaterialGpu {
    int   model;
    float p_mass;         /* rho * p_vol */
    float mu, lam;        /* Lame (ELASTIC) */
    float bulk, gamma;    /* Tait (WATER)   */
};

struct SimParamsGpu {
    int3  res;
    float dx, inv_dx, dt;
    float gravity_y;
    int   bound;
    float p_vol;
    MaterialGpu mats[BQ_MAX_MATERIALS];
};
__constant__ SimParamsGpu c_p;

/* ------------------------------------------------------------------ kernels */
__global__ void k_clear_grid(float4* grid, int ncell) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ncell) grid[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}

__device__ inline void bspline_weights(float3 fx, float w[3][3]) {
    /* B-spline quadratique, fx dans [0.5, 1.5] par axe */
    w[0][0] = 0.5f * (1.5f - fx.x) * (1.5f - fx.x);
    w[0][1] = 0.5f * (1.5f - fx.y) * (1.5f - fx.y);
    w[0][2] = 0.5f * (1.5f - fx.z) * (1.5f - fx.z);
    w[1][0] = 0.75f - (fx.x - 1.f) * (fx.x - 1.f);
    w[1][1] = 0.75f - (fx.y - 1.f) * (fx.y - 1.f);
    w[1][2] = 0.75f - (fx.z - 1.f) * (fx.z - 1.f);
    w[2][0] = 0.5f * (fx.x - 0.5f) * (fx.x - 0.5f);
    w[2][1] = 0.5f * (fx.y - 0.5f) * (fx.y - 0.5f);
    w[2][2] = 0.5f * (fx.z - 0.5f) * (fx.z - 0.5f);
}

__global__ void k_p2g(const float3* __restrict__ x,
                      const float3* __restrict__ v,
                      const float* __restrict__ Cbuf,
                      float* __restrict__ Fbuf,
                      const float* __restrict__ Jw,
                      const uint8_t* __restrict__ mat,
                      float4* grid, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    const MaterialGpu m = c_p.mats[mat[p]];
    float3 xp = x[p];
    int3 base = make_int3((int)floorf(xp.x * c_p.inv_dx - 0.5f),
                          (int)floorf(xp.y * c_p.inv_dx - 0.5f),
                          (int)floorf(xp.z * c_p.inv_dx - 0.5f));
    float3 fx = make_float3(xp.x * c_p.inv_dx - base.x,
                            xp.y * c_p.inv_dx - base.y,
                            xp.z * c_p.inv_dx - base.z);
    float w[3][3];
    bspline_weights(fx, w);

    mat3 C;
    memcpy(C.m, Cbuf + 9 * p, 9 * sizeof(float));

    /* --- contrainte de Cauchy selon le modele constitutif */
    mat3 stress = mat3::zero();
    if (m.model == BQ_MODEL_ELASTIC) {
        mat3 F;
        memcpy(F.m, Fbuf + 9 * p, 9 * sizeof(float));
        F = matmul(mat3::identity() + c_p.dt * C, F);   /* F <- (I + dt C) F */
        memcpy(Fbuf + 9 * p, F.m, 9 * sizeof(float));
        mat3 R = polar_rotation(F);
        float J = det(F);
        /* corotationnel fixe : 2mu (F - R) F^T + lam J (J - 1) I */
        stress = 2.f * m.mu * matmul(F + (-1.f * R), transpose(F));
        float diag = m.lam * J * (J - 1.f);
        stress.m[0] += diag; stress.m[4] += diag; stress.m[8] += diag;
    } else { /* BQ_MODEL_WATER : EOS de Tait, sigma = -p I */
        float J = Jw[p];
        float pr = (m.bulk / m.gamma) * (powf(J, -m.gamma) - 1.f);
        stress.m[0] = stress.m[4] = stress.m[8] = -pr;
    }

    /* affine = (-dt vol 4/dx^2) sigma + m C  -- cf. reference NumPy */
    float coeff = -c_p.dt * c_p.p_vol * 4.f * c_p.inv_dx * c_p.inv_dx;
    mat3 affine = coeff * stress + m.p_mass * C;

    float3 vp = v[p];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                int gx = base.x + i, gy = base.y + j, gz = base.z + k;
                if (gx < 0 || gx >= c_p.res.x || gy < 0 || gy >= c_p.res.y ||
                    gz < 0 || gz >= c_p.res.z)
                    continue; /* noeud hors grille : ignore (UB evite) */
                float3 dpos = make_float3((i - fx.x) * c_p.dx,
                                          (j - fx.y) * c_p.dx,
                                          (k - fx.z) * c_p.dx);
                float weight = w[i][0] * w[j][1] * w[k][2];
                float3 mom = matvec(affine, dpos);
                mom.x = weight * (m.p_mass * vp.x + mom.x);
                mom.y = weight * (m.p_mass * vp.y + mom.y);
                mom.z = weight * (m.p_mass * vp.z + mom.z);
                int idx = (gx * c_p.res.y + gy) * c_p.res.z + gz;
                atomicAdd(&grid[idx].x, mom.x);
                atomicAdd(&grid[idx].y, mom.y);
                atomicAdd(&grid[idx].z, mom.z);
                atomicAdd(&grid[idx].w, weight * m.p_mass);
            }
}

/* Initialise (ou reinitialise) le champ de distance a "pas de collider" :
 * grande valeur positive partout, vitesse/friction nulles, couche de contact
 * (normale nulle, distance non signee 1e6) vide elle aussi. */
__global__ void k_fill_sdf(float* sdf, float4* cvel, float4* cnrm, int ncell) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    sdf[id] = 1e6f;
    cvel[id] = make_float4(0.f, 0.f, 0.f, 0.f);
    cnrm[id] = make_float4(0.f, 0.f, 0.f, 1e6f);
}

/* Index de bucket (non borne : peut deborder de [0,res) si p est hors de la
 * grille de buckets, ce qui arrive puisque la bande active des cellules
 * (pad = 3*dx) peut deborder legerement de l'AABB stricte des triangles sur
 * laquelle la grille de buckets est dimensionnee). Les acces qui en
 * decoulent sont bornes explicitement au moment de l'indexation. */
__device__ inline int3 bucket_index(float3 p, float3 origin, float h) {
    return make_int3((int)floorf((p.x - origin.x) / h),
                     (int)floorf((p.y - origin.y) / h),
                     (int)floorf((p.z - origin.z) / h));
}

/* Distance non signee au collider le plus proche + vitesse/friction
 * interpolees, une fois par frame (pas par substep). Bande etroite de
 * PERFORMANCE (pas de signe) : toute cellule hors de l'AABB dilatee des
 * colliders sort immediatement avec une grande distance positive, sans
 * chercher de triangle -- sans cela le cout serait ncell * n_tri sur tout le
 * domaine. Cette AABB ne joue plus aucun role dans la determination du
 * signe (cf. note plus haut) : une cellule hors AABB est laissee a l'etat
 * UNKNOWN, resolue plus tard par propagation depuis les graines de la bande,
 * jamais amorcee directement comme exterieure ici.
 *
 * Recherche par anneaux de buckets croissants (Chebyshev) autour du bucket
 * de la cellule : a chaque anneau r >= 1, la distance minimale atteignable
 * par un point de ce bucket est (r-1)*h (borne atteinte quand p est au bord
 * du bucket central le plus proche de l'anneau). Des qu'elle depasse la
 * meilleure distance deja trouvee, aucun anneau plus loin ne peut ameliorer
 * le resultat : on s'arrete. C'est cette troncature spatiale -- impossible
 * avec le winding number global -- qui remplace le parcours exhaustif. */
__global__ void k_sdf_unsigned(float* __restrict__ sdf, float4* __restrict__ cvel,
                               float4* __restrict__ cnrm,
                               uint8_t* __restrict__ state,
                               const float3* __restrict__ tri,
                               const float3* __restrict__ trivel,
                               const float* __restrict__ trifric,
                               const int* __restrict__ bucket_off,
                               const int* __restrict__ bucket_tri,
                               float3 bucket_origin, float bucket_h, int3 bucket_res,
                               float3 aabb_lo, float3 aabb_hi, int ncell) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;

    int3 res = c_p.res;
    int i = id / (res.y * res.z);
    int j = (id / res.z) % res.y;
    int k = id % res.z;
    /* Echantillonnage AUX NOEUDS (i*dx), et non au centre de cellule : c'est
     * k_grid_update qui consomme ce champ, indexe exactement comme la grille
     * MPM, dont les noeuds sont en i*dx -- convention etablie par k_p2g
     * (dpos = (i - fx)*dx), par les parois du domaine (i < bound <=> paroi en
     * bound*dx) et par le clamp de k_g2p. Un echantillonnage au centre de
     * cellule decalerait le champ de contact d'un demi-pas par axe (0.87 dx en
     * diagonale) par rapport aux noeuds qui l'utilisent, et rendrait la bande
     * de contact anisotrope : plus large d'un cote de la paroi que de l'autre.
     *
     * Decalage infinitesimal BQ_SDF_NODE_EPS : une geometrie posee sur des
     * multiples exacts de dx (cas courant -- un artiste aligne ses objets sur
     * la grille, et nos propres scenes de test le font) place des noeuds PILE
     * sur une face. La direction depuis le point le plus proche est alors
     * indeterminee (dlen2 nul), aucune graine de signe n'est posee, et un
     * solide entier peut se retrouver non signe. Le decalage brise cette
     * coincidence : le noeud tombe a 0.1 % de dx de la face, ce qui suffit a
     * orienter le test de signe sans deplacer le champ de facon perceptible. */
    const float e = BQ_SDF_NODE_EPS * c_p.dx;
    float3 p = make_float3(i * c_p.dx + e, j * c_p.dx + e, k * c_p.dx + e);
    bool active = !(p.x < aabb_lo.x || p.x > aabb_hi.x || p.y < aabb_lo.y ||
                    p.y > aabb_hi.y || p.z < aabb_lo.z || p.z > aabb_hi.z);
    if (!active) {
        sdf[id] = 1e6f;
        cvel[id] = make_float4(0.f, 0.f, 0.f, 0.f);
        cnrm[id] = make_float4(0.f, 0.f, 0.f, 1e6f);
        state[id] = BQ_SDF_STATE_UNKNOWN; /* resolu par propagation, cf. note plus haut */
        return;
    }

    int3 bc = bucket_index(p, bucket_origin, bucket_h);
    float best_d2 = 3.4e38f;
    int best_t = -1;
    float best_u = 0.f, best_v = 0.f, best_w = 0.f;
    float3 best_cp = make_float3(0.f, 0.f, 0.f);

    int max_ring = bucket_res.x + bucket_res.y + bucket_res.z; /* borne large, sortie anticipee */
    for (int ring = 0; ring <= max_ring; ++ring) {
        if (ring > 0) {
            float lb = (ring - 1) * bucket_h;
            if (lb * lb > best_d2) break; /* aucun bucket plus loin ne peut ameliorer best_d2 */
        }
        int lo_i = bc.x - ring, hi_i = bc.x + ring;
        int lo_j = bc.y - ring, hi_j = bc.y + ring;
        int lo_k = bc.z - ring, hi_k = bc.z + ring;
        for (int ii = max(lo_i, 0); ii <= min(hi_i, bucket_res.x - 1); ++ii) {
            bool ii_edge = (ii == lo_i || ii == hi_i);
            for (int jj = max(lo_j, 0); jj <= min(hi_j, bucket_res.y - 1); ++jj) {
                bool jj_edge = (jj == lo_j || jj == hi_j);
                for (int kk = max(lo_k, 0); kk <= min(hi_k, bucket_res.z - 1); ++kk) {
                    bool kk_edge = (kk == lo_k || kk == hi_k);
                    /* pour ring > 0, ne visiter que la coquille de l'anneau
                     * (l'interieur du cube a deja ete visite aux anneaux
                     * precedents) */
                    if (ring > 0 && !(ii_edge || jj_edge || kk_edge)) continue;
                    int bidx = (ii * bucket_res.y + jj) * bucket_res.z + kk;
                    int off0 = bucket_off[bidx], off1 = bucket_off[bidx + 1];
                    for (int e = off0; e < off1; ++e) {
                        int t = bucket_tri[e];
                        float3 a = tri[3 * t + 0], b = tri[3 * t + 1], c3 = tri[3 * t + 2];
                        float u, v, w;
                        float3 cp = closest_pt_triangle(p, a, b, c3, u, v, w);
                        float3 d = make_float3(p.x - cp.x, p.y - cp.y, p.z - cp.z);
                        float d2 = d.x * d.x + d.y * d.y + d.z * d.z;
                        if (d2 < best_d2) {
                            best_d2 = d2; best_t = t;
                            best_u = u; best_v = v; best_w = w;
                            best_cp = cp;
                        }
                    }
                }
            }
        }
    }

    float3 best_vel = make_float3(0.f, 0.f, 0.f);
    float best_fric = 0.f;
    float3 nu = make_float3(0.f, 0.f, 0.f); /* normale de la couche de contact */
    int8_t sign_local = 0; /* 0 = pas de test local fiable, repli sur la propagation */
    if (best_t >= 0) {
        float3 va = trivel[3 * best_t + 0], vb = trivel[3 * best_t + 1],
               vc = trivel[3 * best_t + 2];
        best_vel = make_float3(best_u * va.x + best_v * vb.x + best_w * vc.x,
                               best_u * va.y + best_v * vb.y + best_w * vc.y,
                               best_u * va.z + best_v * vb.z + best_w * vc.z);
        best_fric = trifric[best_t];

        /* Test local de signe : (p - cp) . normale_du_triangle_le_plus_proche.
         * Positif = dehors, negatif = dedans. Fiable pres de la surface (bande
         * de BQ_SDF_WALL_EPS_MULT*dx) car le triangle le plus proche y est
         * representatif de la geometrie locale -- c'est de la que partent les
         * graines de la propagation (cf. k_sdf_propagate_sign). Loin de la
         * surface ce triangle n'est plus forcement representatif : on ne s'y
         * fie pas, la cellule reste UNKNOWN et attend la propagation. */
        float3 a = tri[3 * best_t + 0], b = tri[3 * best_t + 1], c3 = tri[3 * best_t + 2];
        float3 e1 = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
        float3 e2 = make_float3(c3.x - a.x, c3.y - a.y, c3.z - a.z);
        float3 nrm = make_float3(e1.y * e2.z - e1.z * e2.y,
                                 e1.z * e2.x - e1.x * e2.z,
                                 e1.x * e2.y - e1.y * e2.x);
        float nlen2 = nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z;
        float3 diff = make_float3(p.x - best_cp.x, p.y - best_cp.y, p.z - best_cp.z);
        float dlen2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        /* cas degeneres : triangle degenere (normale nulle) ou p == cp
         * (produit scalaire non significatif) -- on laisse sign_local a 0,
         * la cellule n'est pas amorcee du tout (ni graine exterieure, ni
         * graine interieure) : regle de securite anti-Defaut-2, un triangle
         * degenere isole (Decimate/Remesh/Merge by Distance) ne doit jamais
         * fabriquer une coquille de cellules solides parasites. */
        if (nlen2 > 1e-20f && dlen2 > 1e-20f) {
            float dot = diff.x * nrm.x + diff.y * nrm.y + diff.z * nrm.z;
            sign_local = (dot >= 0.f) ? 1 : -1;
        }

        /* Normale de la couche de contact : direction depuis le point le plus
         * proche sur le triangle vers la cellule -- exacte que ce point soit
         * sur une face, une arete ou un sommet (contrairement a la normale de
         * face, fausse pres des aretes/sommets). diff et dlen2 sont deja
         * calcules ci-dessus pour le test de signe local, reutilises ici. */
        if (dlen2 > 1e-20f) {
            float dinv = 1.f / sqrtf(dlen2);
            nu = make_float3(diff.x * dinv, diff.y * dinv, diff.z * dinv);
        } else if (nlen2 > 1e-20f) {
            /* cellule pile sur la surface : direction indeterminee, repli sur
             * la normale du triangle. */
            float ninv = 1.f / sqrtf(nlen2);
            nu = make_float3(nrm.x * ninv, nrm.y * ninv, nrm.z * ninv);
        }
    }

    sdf[id] = sqrtf(best_d2); /* non signee pour l'instant, cf. k_sdf_finalize_sign */
    cvel[id] = make_float4(best_vel.x, best_vel.y, best_vel.z, best_fric);
    cnrm[id] = (best_t >= 0) ? make_float4(nu.x, nu.y, nu.z, sqrtf(best_d2))
                              : make_float4(0.f, 0.f, 0.f, 1e6f);

    /* Amorce (graine) uniquement dans la bande, et seulement si le test local
     * a pu trancher. Hors bande, ou test local degenere : UNKNOWN, resolu
     * plus tard par propagation depuis une graine voisine (jamais force a
     * "interieur" par defaut -- cf. regle de securite en tete de fichier). */
    bool near_band = sdf[id] < BQ_SDF_WALL_EPS_MULT * c_p.dx;
    if (near_band && sign_local != 0) {
        state[id] = (sign_local > 0) ? BQ_SDF_STATE_EXTERIOR : BQ_SDF_STATE_INTERIOR;
    } else {
        state[id] = BQ_SDF_STATE_UNKNOWN;
    }
}

/* Propagation par 6-voisinage, une passe Jacobi : state ne passe jamais de
 * EXTERIOR/INTERIOR a UNKNOWN (monotone -- une fois resolue, une cellule ne
 * change plus), donc la lecture de voisins pas encore mis a jour dans la
 * meme passe ne fait au pire que retarder la convergence d'une iteration,
 * jamais la corrompre -- pas besoin de double buffer. Ecrit 1 dans *changed
 * (via atomicOr) si au moins une cellule a bascule, pour que l'hote sache
 * s'il doit relancer une passe.
 *
 * Les seules cellules qui bloquent la propagation sont celles deja resolues
 * (EXTERIOR ou INTERIOR) : ce sont les graines de la bande posees par
 * k_sdf_unsigned, qui forment naturellement une coquille autour de la
 * surface et separent interieur/exterieur -- plus besoin de tester la
 * distance non signee ici (cf. defaut 1 : l'ancienne version bloquait sur
 * un seuil de distance, ce qui couplait a tort le signe a l'AABB des
 * colliders). Une cellule UNKNOWN adjacente a la fois a un voisin EXTERIOR
 * et a un voisin INTERIOR (gap issu d'un triangle degenere isole, cf.
 * defaut 2) est resolue EXTERIOR en priorite -- regle de securite : ne
 * jamais fabriquer de solide non justifie.
 *
 * Limite persistante (grille reguliere) : une paroi plus fine qu'une
 * cellule ne bloque toujours pas la propagation -- l'interieur "fuit" au
 * travers et le signe de sdf devient localement aveugle a l'obstacle.
 * Cependant l'etancheite du contact ne repose plus sur le signe seul : la
 * couche de contact de k_grid_update (BQ_CONTACT_BAND_MULT, distance non
 * signee cnrm.w) prend le relais dans ce cas et bloque le fluide meme la ou
 * aucune cellule n'est marquee solide. */
__global__ void k_sdf_propagate_sign(uint8_t* __restrict__ state,
                                     int ncell, int* changed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    if (state[id] != BQ_SDF_STATE_UNKNOWN) return; /* deja resolue */

    int3 res = c_p.res;
    int i = id / (res.y * res.z);
    int j = (id / res.z) % res.y;
    int k = id % res.z;

    bool has_ext = false, has_int = false;
#define BQ_SDF_CHECK_NEIGHBOR(nid)                        \
    do {                                                  \
        uint8_t ns = state[nid];                          \
        if (ns == BQ_SDF_STATE_EXTERIOR) has_ext = true;   \
        else if (ns == BQ_SDF_STATE_INTERIOR) has_int = true; \
    } while (0)
    if (i > 0)          BQ_SDF_CHECK_NEIGHBOR(id - res.y * res.z);
    if (i < res.x - 1)  BQ_SDF_CHECK_NEIGHBOR(id + res.y * res.z);
    if (j > 0)          BQ_SDF_CHECK_NEIGHBOR(id - res.z);
    if (j < res.y - 1)  BQ_SDF_CHECK_NEIGHBOR(id + res.z);
    if (k > 0)          BQ_SDF_CHECK_NEIGHBOR(id - 1);
    if (k < res.z - 1)  BQ_SDF_CHECK_NEIGHBOR(id + 1);
#undef BQ_SDF_CHECK_NEIGHBOR

    if (has_ext) {
        state[id] = BQ_SDF_STATE_EXTERIOR;
        atomicOr(changed, 1);
    } else if (has_int) {
        state[id] = BQ_SDF_STATE_INTERIOR;
        atomicOr(changed, 1);
    }
}

/* Une fois la propagation convergee (ou tronquee, cf. borne d'iteration cote
 * hote) : sdf contient encore une distance non signee partout, il faut lui
 * appliquer un signe a partir de l'etat resolu par k_sdf_unsigned (graines)
 * puis k_sdf_propagate_sign (relais par connexite).
 *
 * Regle de securite, non negociable : une cellule encore UNKNOWN a ce stade
 * (aucune graine ne l'a atteinte -- collider absent, tous les triangles
 * voisins degeneres, region isolee de toute surface, ou troncature de la
 * boucle de propagation cote hote) est declaree EXTERIEURE, jamais solide.
 * Un obstacle qui manque est un desagrement ; un domaine entierement fige en
 * solide est une simulation perdue. D'ou le test : seul INTERIOR nege le
 * signe, tout le reste (EXTERIOR et UNKNOWN) reste positif. */
__global__ void k_sdf_finalize_sign(float* __restrict__ sdf,
                                    float4* __restrict__ cnrm,
                                    const uint8_t* __restrict__ state,
                                    int ncell) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    if (state[id] == BQ_SDF_STATE_INTERIOR) {
        sdf[id] = -sdf[id];
        /* cnrm.xyz doit rester la normale SORTANTE du solide quel que soit le
         * cote ou se trouve la cellule : ici la cellule est du cote interieur,
         * la normale calculee en k_sdf_unsigned pointait donc vers l'interieur
         * -- on l'inverse. w (distance non signee) est inchange. */
        float4 cn = cnrm[id];
        cnrm[id] = make_float4(-cn.x, -cn.y, -cn.z, cn.w);
    }
}

/* La normale du collider en (i,j,k) est-elle CORROBOREE par du solide derriere
 * elle ? On regarde la cellule voisine dans la direction -n, c'est-a-dire du
 * cote ou la normale pretend que se trouve le solide.
 *
 * Corroboree : la normale est fiable, l'obstacle est resolu par la grille de ce
 * cote-la. Non corroboree : la normale peut pointer du mauvais cote, et deux
 * geometries y menent :
 *   - paroi plus fine que dx : aucun centre de cellule dedans, le champ signe
 *     est integralement aveugle ;
 *   - paroi d'environ 1 dx : la cellule est solide mais equidistante des deux
 *     faces, le point le plus proche est donc arbitraire. Tester la seule
 *     presence de solide au voisinage n'attrape pas ce cas (la cellule est
 *     elle-meme solide) -- d'ou la corroboration DIRECTIONNELLE.
 *
 * Bornage par indices et non par offset lineaire : un pas de +/-1 sur k
 * franchirait sinon la frontiere d'axe et lirait la ligne voisine. Voisin hors
 * grille : non corrobore, les appelants retiennent alors le comportement sur. */
__device__ inline bool normal_corroborated(const float* __restrict__ sdf,
                                           int3 res, int i, int j, int k,
                                           float3 n) {
    int di = 0, dj = 0, dk = 0;
    if (fabsf(n.x) >= fabsf(n.y) && fabsf(n.x) >= fabsf(n.z))
        di = (n.x > 0.f) ? -1 : 1;
    else if (fabsf(n.y) >= fabsf(n.z))
        dj = (n.y > 0.f) ? -1 : 1;
    else
        dk = (n.z > 0.f) ? -1 : 1;
    int bi = i + di, bj = j + dj, bk = k + dk;
    return bi >= 0 && bi < res.x && bj >= 0 && bj < res.y &&
           bk >= 0 && bk < res.z &&
           sdf[(bi * res.y + bj) * res.z + bk] < 0.f;
}

__global__ void k_grid_update(float4* grid, const float* __restrict__ sdf,
                              const float4* __restrict__ cvel,
                              const float4* __restrict__ cnrm, int ncell) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    float4 g = grid[id];
    if (g.w <= 0.f) return;

    float3 v = make_float3(g.x / g.w, g.y / g.w, g.z / g.w);
    v.y += c_p.dt * c_p.gravity_y;

    int3 res = c_p.res; int b = c_p.bound;
    int i = id / (res.y * res.z);
    int j = (id / res.z) % res.y;
    int k = id % res.z;

    /* condition aux limites du collider : friction de Coulomb (Stomakhin et
     * al. 2013) sur la composante normale, appliquee soit quand le champ de
     * distance signale l'interieur du solide, soit quand la cellule est dans
     * la couche de contact (distance non signee cnrm.w < h) -- cette derniere
     * rattrape les parois plus fines que dx, invisibles au signe seul. */
    float phi = sdf[id];
    float4 cn = cnrm[id];
    float h = BQ_CONTACT_BAND_MULT * c_p.dx;
    bool solid_here = (phi < 0.f);
    if (solid_here || cn.w < h) {
        float3 n = make_float3(cn.x, cn.y, cn.z);
        if (n.x != 0.f || n.y != 0.f || n.z != 0.f) {
            /* Normale corroboree : contact UNILATERAL, le fluide qui s'eloigne
             * de l'obstacle n'y colle pas. Non corroboree : contact
             * BIDIRECTIONNEL, un noeud unique devant bloquer le fluide des deux
             * cotes d'une paroi que la grille ne resout pas (cf.
             * normal_corroborated). */
            bool bidir = !normal_corroborated(sdf, res, i, j, k, n);
            float4 cv = cvel[id];
            float3 vc = make_float3(cv.x, cv.y, cv.z);
            float fr = cv.w;
            float3 vrel = make_float3(v.x - vc.x, v.y - vc.y, v.z - vc.z);
            float vn = vrel.x * n.x + vrel.y * n.y + vrel.z * n.z;
            if (vn < 0.f || bidir) {
                float3 vt = make_float3(vrel.x - vn * n.x, vrel.y - vn * n.y,
                                        vrel.z - vn * n.z);
                float vt_norm = sqrtf(vt.x * vt.x + vt.y * vt.y + vt.z * vt.z);
                /* friction de Coulomb (Stomakhin et al. 2013), ecrite avec |vn| :
                 * strictement equivalente a la forme -fr*vn dans le cas vn < 0, et
                 * valable aussi pour le sens sortant du mode bidirectionnel. */
                float vn_abs = fabsf(vn);
                if (vt_norm <= fr * vn_abs) {
                    vt = make_float3(0.f, 0.f, 0.f);
                } else if (vt_norm > 1e-8f) {
                    float scale = 1.f - fr * vn_abs / vt_norm;
                    vt.x *= scale; vt.y *= scale; vt.z *= scale;
                }
                v = make_float3(vc.x + vt.x, vc.y + vt.y, vc.z + vt.z);
            }
        }
    }

    /* conditions separantes : composante normale annulee vers la paroi */
    if (i < b && v.x < 0.f) v.x = 0.f;
    if (i >= res.x - b && v.x > 0.f) v.x = 0.f;
    if (j < b && v.y < 0.f) v.y = 0.f;
    if (j >= res.y - b && v.y > 0.f) v.y = 0.f;
    if (k < b && v.z < 0.f) v.z = 0.f;
    if (k >= res.z - b && v.z > 0.f) v.z = 0.f;

    grid[id] = make_float4(v.x, v.y, v.z, g.w);
}

__global__ void k_g2p(float3* __restrict__ x,
                      float3* __restrict__ v,
                      float* __restrict__ Cbuf,
                      float* __restrict__ Jw,
                      const uint8_t* __restrict__ mat,
                      const float4* __restrict__ grid,
                      const float* __restrict__ sdf,
                      const float4* __restrict__ cnrm, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;

    float3 xp = x[p];
    int3 base = make_int3((int)floorf(xp.x * c_p.inv_dx - 0.5f),
                          (int)floorf(xp.y * c_p.inv_dx - 0.5f),
                          (int)floorf(xp.z * c_p.inv_dx - 0.5f));
    float3 fx = make_float3(xp.x * c_p.inv_dx - base.x,
                            xp.y * c_p.inv_dx - base.y,
                            xp.z * c_p.inv_dx - base.z);
    float w[3][3];
    bspline_weights(fx, w);

    float3 nv = make_float3(0.f, 0.f, 0.f);
    mat3 B = mat3::zero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                int gx = base.x + i, gy = base.y + j, gz = base.z + k;
                if (gx < 0 || gx >= c_p.res.x || gy < 0 || gy >= c_p.res.y ||
                    gz < 0 || gz >= c_p.res.z)
                    continue; /* noeud hors grille : contribution nulle */
                float3 dpos = make_float3((i - fx.x) * c_p.dx,
                                          (j - fx.y) * c_p.dx,
                                          (k - fx.z) * c_p.dx);
                float weight = w[i][0] * w[j][1] * w[k][2];
                int idx = (gx * c_p.res.y + gy) * c_p.res.z + gz;
                float4 g = grid[idx];
                float3 gv = make_float3(g.x, g.y, g.z);
                nv.x += weight * gv.x;
                nv.y += weight * gv.y;
                nv.z += weight * gv.z;
                B = B + weight * outer(gv, dpos);
            }

    mat3 C = 4.f * c_p.inv_dx * c_p.inv_dx * B;   /* D^-1 quadratique */
    memcpy(Cbuf + 9 * p, C.m, 9 * sizeof(float));
    v[p] = nv;

    float lo = c_p.bound * c_p.dx;
    float hi_x = c_p.res.x * c_p.dx - lo;
    float hi_y = c_p.res.y * c_p.dx - lo;
    float hi_z = c_p.res.z * c_p.dx - lo;
    float3 xnew = make_float3(fminf(fmaxf(xp.x + c_p.dt * nv.x, lo), hi_x),
                              fminf(fmaxf(xp.y + c_p.dt * nv.y, lo), hi_y),
                              fminf(fmaxf(xp.z + c_p.dt * nv.z, lo), hi_z));

    /* Correction de position (projection sur la surface du collider) : le
     * contact MPM passe par la grille et reste "mou" -- une particule dont
     * le stencil chevauche solide et fluide peut deriver dans l'obstacle
     * malgre la condition aux limites de k_grid_update. Si la particule se
     * retrouve dans le solide (phi < 0), on la repousse exactement a la
     * surface le long du gradient du champ de distance : xnew -= phi * n.
     *
     * Lecture de phi et de la normale a la cellule la plus proche, sans
     * interpolation trilineaire : coherent avec k_grid_update, qui traite
     * deja le champ indexe par NOEUD pour sa propre condition aux limites --
     * d'ou la lecture au noeud le plus proche ici, floorf(x*inv_dx + 0.5),
     * et non a la cellule contenant la particule : meme convention de part et
     * d'autre du contact, sans quoi le mecanisme mou (vitesses de grille) et
     * le mecanisme dur (positions) travailleraient sur deux champs decales
     * d'un demi-pas. Moins couteux aussi : ce test tourne par particule et par
     * substep (bien plus souvent que k_grid_update, qui tourne par
     * cellule). La normale geometrique exacte est precalculee dans cnrm
     * (direction depuis le point le plus proche du collider vers le
     * noeud, cf. k_sdf_unsigned) : une simple lecture, plus de gradient
     * par differences finies a recalculer ici. En l'absence de collider,
     * sdf vaut 1e6 partout (k_fill_sdf) : le test phi < 0 echoue
     * immediatement, chemin quasi gratuit. */
    int3 res = c_p.res;
    int ci = min(max((int)floorf(xnew.x * c_p.inv_dx + 0.5f), 0), res.x - 1);
    int cj = min(max((int)floorf(xnew.y * c_p.inv_dx + 0.5f), 0), res.y - 1);
    int ck = min(max((int)floorf(xnew.z * c_p.inv_dx + 0.5f), 0), res.z - 1);
    int idx_new = (ci * res.y + cj) * res.z + ck;
    float phi = sdf[idx_new];

    /* Contrainte de position, du COTE d'ou vient la particule.
     *
     * Le signe du champ de distance ne peut pas servir ici : une paroi plus fine
     * que dx n'a pas d'interieur discretise, et une paroi d'environ 1 dx a des
     * cellules equidistantes de ses deux faces dont la normale est arbitraire.
     * On se passe donc du signe : la normale lue a la position PRECEDENTE de la
     * particule (cn_old) designe le cote de la paroi ou elle se trouvait, et on
     * la maintient de ce cote-la.
     *
     * Detection de franchissement sans signe : les normales des cellules situees
     * de part et d'autre d'une paroi pointent en sens opposes, donc un produit
     * scalaire negatif entre la normale du cote d'origine et celle de la cellule
     * d'arrivee signale que la particule a change de cote. Elle est alors
     * ramenee de l'autre cote, a la distance de securite.
     *
     * La CFL bornant le deplacement d'un substep a environ dx, une particule qui
     * arrive dans une paroi venait forcement d'au plus dx de celle-ci, donc d'une
     * cellule ou cnrm est renseigne : le cote d'origine est toujours disponible.
     * Garde-fou geometrique : sur une surface courbe ou pres d'une arete, deux
     * cellules d'un MEME cote peuvent avoir des normales divergentes ; on
     * n'interprete donc un produit scalaire negatif comme un franchissement que
     * dans la bande de contact, et le deplacement est borne a dx pour ne jamais
     * teleporter une particule. */
    float push = BQ_CONTACT_PUSH_MULT * c_p.dx;
    int oi = min(max((int)floorf(xp.x * c_p.inv_dx + 0.5f), 0), res.x - 1);
    int oj = min(max((int)floorf(xp.y * c_p.inv_dx + 0.5f), 0), res.y - 1);
    int ok_ = min(max((int)floorf(xp.z * c_p.inv_dx + 0.5f), 0), res.z - 1);
    float4 cn_old = cnrm[(oi * res.y + oj) * res.z + ok_];
    float4 cn_new = cnrm[idx_new];
    float3 n_side = make_float3(cn_old.x, cn_old.y, cn_old.z);
    float3 n_new = make_float3(cn_new.x, cn_new.y, cn_new.z);
    bool ok_side = (n_side.x != 0.f || n_side.y != 0.f || n_side.z != 0.f);
    bool ok_new = (n_new.x != 0.f || n_new.y != 0.f || n_new.z != 0.f);
    bool traite = false;
    if (ok_side && ok_new && cn_new.w < BQ_CONTACT_BAND_MULT * c_p.dx) {
        float dot_side = n_new.x * n_side.x + n_new.y * n_side.y +
                         n_new.z * n_side.z;
        bool franchi = (dot_side < 0.f);
        if (franchi || cn_new.w < push) {
            float depl = franchi ? (push + cn_new.w) : (push - cn_new.w);
            depl = fminf(depl, c_p.dx);
            xnew.x += depl * n_side.x;
            xnew.y += depl * n_side.y;
            xnew.z += depl * n_side.z;
            traite = true;
        }
    }

    if (!traite && phi < 0.f) {
        float4 cn = cnrm[idx_new];
        float3 n = make_float3(cn.x, cn.y, cn.z);
        /* Ne projeter que sur une normale DEGENEREE-libre et CORROBOREE. Une
         * normale non corroboree peut pointer du mauvais cote de la paroi (cas
         * d'une paroi d'environ 1 dx, cellule equidistante de ses deux faces) :
         * la projection deviendrait alors un mecanisme de fuite, poussant
         * activement la particule au travers de l'obstacle -- mesure : 4.9 % de
         * fuite sur un contenant a paroi de 1 dx, contre 0.1 % sans cette
         * projection. Dans le doute on ne deplace rien : la condition aux
         * limites bidirectionnelle de k_grid_update tient deja ce cas. */
        if ((n.x != 0.f || n.y != 0.f || n.z != 0.f) &&
            normal_corroborated(sdf, res, ci, cj, ck, n)) {
            /* Deplacement plafonne a dx, comme la contrainte de bande ci-dessus.
             * |phi| est petit tant que le signe du champ est juste, mais il
             * existe un mode d'echec ou il ne l'est pas : un collider dont les
             * normales de faces sont inversees (Solidify, extrusion, faces
             * retournees -- erreur de modelisation banale) trompe TOUTES les
             * graines de signe de k_sdf_unsigned. L'exterieur devient alors
             * INTERIOR, phi vaut plusieurs dizaines de centimetres, et une
             * projection non bornee teleporterait tout le fluide d'un coup. La
             * regle de securite de k_sdf_finalize_sign ne protege que les
             * cellules UNKNOWN, pas les graines fausses : ce plafond est la
             * seule garde sur ce chemin. */
            float depl_p = fminf(-phi, c_p.dx);
            xnew.x += depl_p * n.x;
            xnew.y += depl_p * n.y;
            xnew.z += depl_p * n.z;
        }
    }

    /* re-clamp final : une particule repoussee par la projection ne doit
     * jamais ressortir de la grille. Note : si le collider deborde tres
     * profondement sous le domaine, ce re-clamp peut ramener la particule
     * dans le solide -- c'est accepte, inevitable (on ne peut pas a la fois
     * rester dans le domaine et sortir d'un solide qui occupe le bord du
     * domaine), et de toute facon marginal par rapport a la fuite corrigee
     * ici. La vitesse n'est volontairement pas touchee : c'est le role de
     * k_grid_update, la retoucher ici injecterait de l'energie et ferait
     * vibrer le contact. */
    x[p] = make_float3(fminf(fmaxf(xnew.x, lo), hi_x),
                       fminf(fmaxf(xnew.y, lo), hi_y),
                       fminf(fmaxf(xnew.z, lo), hi_z));

    if (c_p.mats[mat[p]].model == BQ_MODEL_WATER) {
        float tr = C.m[0] + C.m[4] + C.m[8];
        Jw[p] = fminf(fmaxf(Jw[p] * (1.f + c_p.dt * tr), 0.5f), 1.5f);
    }
}

/* ------------------------------------------------------------------- BqSim */
struct BqSim {
    BqConfig cfg;
    SimParamsGpu prm;
    int n = 0;
    int n_mats = 0;
    BqMaterial mats_host[BQ_MAX_MATERIALS];
    float dt = 0.f; /* recalcule quand la liste de materiaux change */

    float3* d_x = nullptr;
    float3* d_v = nullptr;
    float*  d_C = nullptr;
    float*  d_F = nullptr;
    float*  d_J = nullptr;
    uint8_t* d_mat = nullptr;
    float4* d_grid = nullptr;

    /* colliders : champ de distance signee + vitesse/friction, taille ncell */
    float*  d_sdf = nullptr;
    float4* d_cvel = nullptr;
    /* couche de contact : xyz = normale unitaire SORTANTE du solide, w =
     * distance NON SIGNEE au collider (cf. k_sdf_unsigned, k_sdf_finalize_sign,
     * k_grid_update). Rattrape les parois plus fines que dx, invisibles au
     * champ signe sdf seul -- taille ncell. */
    float4* d_cnrm = nullptr;
    /* geometrie triangulaire brute, reallouee seulement quand n_tri augmente */
    float3* d_tri = nullptr;
    float3* d_trivel = nullptr;
    float*  d_trifric = nullptr;
    int tri_cap = 0;
    int n_tri = 0;

    /* grille de buckets (CSR), reconstruite sur l'hote a chaque appel de
     * bq_set_colliders puis televersee ; les buffers device ne sont
     * realloues que si la capacite courante est depassee. */
    int* d_bucket_off = nullptr; /* taille nb+1 */
    int* d_bucket_tri = nullptr; /* taille bucket_off[nb], indices de triangles */
    int bucket_off_cap = 0;
    int bucket_tri_cap = 0;

    /* etat de signe par cellule (ncell) : BQ_SDF_STATE_UNKNOWN / _EXTERIOR /
     * _INTERIOR (cf. mlsmpm.cu, section SDF), et flag device de convergence
     * pour k_sdf_propagate_sign */
    uint8_t* d_ext = nullptr;
    int* d_changed = nullptr;
};

static float material_sound_speed(const BqMaterial& m) {
    float stiff = (m.model == BQ_MODEL_ELASTIC) ? m.E : m.bulk;
    return sqrtf(stiff / m.rho);
}

static int upload_params(BqSim* s) {
    float c_max = 1e-3f;
    for (int i = 0; i < s->n_mats; ++i)
        c_max = fmaxf(c_max, material_sound_speed(s->mats_host[i]));
    float dx = s->cfg.cell_size;
    s->dt = s->cfg.cfl * dx / c_max;

    SimParamsGpu& p = s->prm;
    p.res = make_int3(s->cfg.grid_res[0], s->cfg.grid_res[1], s->cfg.grid_res[2]);
    p.dx = dx;
    p.inv_dx = 1.f / dx;
    p.dt = s->dt;
    p.gravity_y = s->cfg.gravity_y;
    p.bound = 3;
    float spacing = dx / s->cfg.ppc_axis;
    p.p_vol = spacing * spacing * spacing;
    for (int i = 0; i < s->n_mats; ++i) {
        const BqMaterial& m = s->mats_host[i];
        MaterialGpu& g = p.mats[i];
        g.model = m.model;
        g.p_mass = m.rho * p.p_vol;
        g.mu = m.E / (2.f * (1.f + m.nu));
        g.lam = m.E * m.nu / ((1.f + m.nu) * (1.f - 2.f * m.nu));
        g.bulk = m.bulk;
        g.gamma = m.gamma;
    }
    BQ_CUDA_CHECK(cudaMemcpyToSymbol(c_p, &p, sizeof(p)));
    return 0;
}

/* -------------------------------------------------------------------- API */
extern "C" {

/* A appeler en tout premier, avant bq_create : ne touchent a aucun etat ni
   au GPU, donc ne peuvent jamais echouer. */
BQ_API int bq_abi_version(void) {
    return BQ_ABI_VERSION;
}

BQ_API int bq_config_size(void) {
    return (int)sizeof(BqConfig);
}

BQ_API void bq_default_config(BqConfig* cfg) {
    cfg->grid_res[0] = cfg->grid_res[1] = cfg->grid_res[2] = 64;
    cfg->cell_size = 1.f / 64.f;
    cfg->gravity_y = -9.8f;
    cfg->cfl = 0.3f;
    cfg->ppc_axis = 2;
    cfg->max_particles = 2000000;
}

BQ_API BqSim* bq_create(const BqConfig* cfg) {
    BqSim* s = new BqSim();
    s->cfg = cfg ? *cfg : (bq_default_config(&s->cfg), s->cfg);
    int cap = s->cfg.max_particles;
    int ncell = s->cfg.grid_res[0] * s->cfg.grid_res[1] * s->cfg.grid_res[2];
    if (cudaMalloc(&s->d_x, cap * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&s->d_v, cap * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&s->d_C, cap * 9 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_F, cap * 9 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_J, cap * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_mat, cap * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(&s->d_grid, ncell * sizeof(float4)) != cudaSuccess ||
        cudaMalloc(&s->d_sdf, ncell * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_cvel, ncell * sizeof(float4)) != cudaSuccess ||
        cudaMalloc(&s->d_cnrm, ncell * sizeof(float4)) != cudaSuccess ||
        cudaMalloc(&s->d_ext, ncell * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(&s->d_changed, sizeof(int)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "cudaMalloc: memoire insuffisante");
        bq_destroy(s);
        return nullptr;
    }
    /* pas de collider au depart : sdf grand partout, vitesse/friction nulles */
    dim3 bp(256), gc((ncell + 255) / 256);
    k_fill_sdf<<<gc, bp>>>(s->d_sdf, s->d_cvel, s->d_cnrm, ncell);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "k_fill_sdf: echec init");
        bq_destroy(s);
        return nullptr;
    }
    return s;
}

BQ_API void bq_destroy(BqSim* s) {
    if (!s) return;
    cudaFree(s->d_x);   cudaFree(s->d_v);   cudaFree(s->d_C);
    cudaFree(s->d_F);   cudaFree(s->d_J);   cudaFree(s->d_mat);
    cudaFree(s->d_grid);
    cudaFree(s->d_sdf); cudaFree(s->d_cvel); cudaFree(s->d_cnrm);
    cudaFree(s->d_tri); cudaFree(s->d_trivel); cudaFree(s->d_trifric);
    cudaFree(s->d_bucket_off); cudaFree(s->d_bucket_tri);
    cudaFree(s->d_ext); cudaFree(s->d_changed);
    delete s;
}

BQ_API int bq_add_material(BqSim* s, const BqMaterial* mat) {
    if (s->n_mats >= BQ_MAX_MATERIALS) {
        snprintf(g_error, sizeof(g_error), "max %d materiaux", BQ_MAX_MATERIALS);
        return -1;
    }
    s->mats_host[s->n_mats] = *mat;
    int id = s->n_mats++;
    if (upload_params(s) < 0) return -1;
    return id;
}

/* Chemin d'initialisation/upload partage par bq_emit_box, bq_emit_points et
 * bq_emit_points_vel : verifie mat_id, capacite et que chaque point est dans
 * le domaine valide, puis initialise F=I, C=0, J=1, v=pv[i], mat=mat_id et
 * televerse vers le device. pv doit pointer sur count vitesses (une par
 * particule ; l'appelant duplique une vitesse uniforme si besoin).
 * Retourne le nombre de particules emises, ou -1 (g_error rempli). */
static int emit_particles(BqSim* s, int mat_id, const float3* px,
                           const float3* pv, int count) {
    if (mat_id < 0 || mat_id >= s->n_mats) {
        snprintf(g_error, sizeof(g_error), "mat_id %d invalide", mat_id);
        return -1;
    }
    if (s->n + count > s->cfg.max_particles) {
        snprintf(g_error, sizeof(g_error), "capacite depassee (%d + %d > %d)",
                 s->n, count, s->cfg.max_particles);
        return -1;
    }
    {
        float valid_lo = s->prm.bound * s->prm.dx;
        float valid_hi_x = s->cfg.grid_res[0] * s->prm.dx - s->prm.bound * s->prm.dx;
        float valid_hi_y = s->cfg.grid_res[1] * s->prm.dx - s->prm.bound * s->prm.dx;
        float valid_hi_z = s->cfg.grid_res[2] * s->prm.dx - s->prm.bound * s->prm.dx;
        for (int i = 0; i < count; ++i) {
            const float3& p = px[i];
            if (p.x < valid_lo || p.x > valid_hi_x ||
                p.y < valid_lo || p.y > valid_hi_y ||
                p.z < valid_lo || p.z > valid_hi_z) {
                snprintf(g_error, sizeof(g_error),
                         "point %d hors domaine (x=%g y=%g z=%g), "
                         "bornes valides [%g, %g]x[%g, %g]x[%g, %g]",
                         i, p.x, p.y, p.z, valid_lo, valid_hi_x,
                         valid_lo, valid_hi_y, valid_lo, valid_hi_z);
                return -1;
            }
        }
    }

    std::vector<float> id9(count * 9, 0.f), ones(count, 1.f);
    for (int i = 0; i < count; ++i) { id9[9 * i] = id9[9 * i + 4] = id9[9 * i + 8] = 1.f; }
    std::vector<float> zero9(count * 9, 0.f);
    std::vector<uint8_t> mid(count, (uint8_t)mat_id);

    int off = s->n;
    BQ_CUDA_CHECK(cudaMemcpy(s->d_x + off, px, count * sizeof(float3),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_v + off, pv, count * sizeof(float3),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_F + 9 * off, id9.data(),
                             count * 9 * sizeof(float), cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_C + 9 * off, zero9.data(),
                             count * 9 * sizeof(float), cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_J + off, ones.data(), count * sizeof(float),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_mat + off, mid.data(), count * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));
    s->n += count;
    return count;
}

BQ_API int bq_emit_box(BqSim* s, int mat_id, const float lo[3],
                       const float hi[3], const float vel[3]) {
    if (mat_id < 0 || mat_id >= s->n_mats) {
        snprintf(g_error, sizeof(g_error), "mat_id %d invalide", mat_id);
        return -1;
    }
    {
        static const char* axis_name[3] = { "x", "y", "z" };
        float valid_lo = s->prm.bound * s->prm.dx;
        for (int a = 0; a < 3; ++a) {
            float valid_hi = s->cfg.grid_res[a] * s->prm.dx - s->prm.bound * s->prm.dx;
            if (lo[a] >= hi[a]) {
                snprintf(g_error, sizeof(g_error),
                         "bq_emit_box: boite degeneree sur l'axe %s (lo=%g >= hi=%g)",
                         axis_name[a], lo[a], hi[a]);
                return -1;
            }
            if (lo[a] < valid_lo || hi[a] > valid_hi) {
                snprintf(g_error, sizeof(g_error),
                         "bq_emit_box: boite hors domaine sur l'axe %s "
                         "(lo=%g hi=%g, bornes valides [%g, %g])",
                         axis_name[a], lo[a], hi[a], valid_lo, valid_hi);
                return -1;
            }
        }
    }
    float spacing = s->prm.dx / s->cfg.ppc_axis;
    std::vector<float3> px;
    for (float x = lo[0] + spacing / 2; x < hi[0]; x += spacing)
        for (float y = lo[1] + spacing / 2; y < hi[1]; y += spacing)
            for (float z = lo[2] + spacing / 2; z < hi[2]; z += spacing)
                px.push_back(make_float3(x, y, z));

    std::vector<float3> pv(px.size(), make_float3(vel[0], vel[1], vel[2]));
    return emit_particles(s, mat_id, px.data(), pv.data(), (int)px.size());
}

BQ_API int bq_emit_points(BqSim* s, int mat_id, const float* pos, int count,
                          const float vel[3]) {
    if (mat_id < 0 || mat_id >= s->n_mats) {
        snprintf(g_error, sizeof(g_error), "mat_id %d invalide", mat_id);
        return -1;
    }
    if (count <= 0 || pos == NULL) {
        snprintf(g_error, sizeof(g_error),
                 "bq_emit_points: count invalide ou pos nul (count=%d)", count);
        return -1;
    }
    std::vector<float3> px(count);
    for (int i = 0; i < count; ++i)
        px[i] = make_float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);

    std::vector<float3> pv(count, make_float3(vel[0], vel[1], vel[2]));
    return emit_particles(s, mat_id, px.data(), pv.data(), count);
}

/* Comme bq_emit_points, mais avec une vitesse propre a chaque particule
 * (vel entrelace vx,vy,vz, meme indexation que pos). Meme validation que
 * bq_emit_points, plus le rejet de vel nul. */
BQ_API int bq_emit_points_vel(BqSim* s, int mat_id, const float* pos,
                              const float* vel, int count) {
    if (mat_id < 0 || mat_id >= s->n_mats) {
        snprintf(g_error, sizeof(g_error), "mat_id %d invalide", mat_id);
        return -1;
    }
    if (count <= 0 || pos == NULL) {
        snprintf(g_error, sizeof(g_error),
                 "bq_emit_points_vel: count invalide ou pos nul (count=%d)", count);
        return -1;
    }
    if (vel == NULL) {
        snprintf(g_error, sizeof(g_error),
                 "bq_emit_points_vel: vel nul (count=%d)", count);
        return -1;
    }
    std::vector<float3> px(count), pv(count);
    for (int i = 0; i < count; ++i) {
        px[i] = make_float3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
        pv[i] = make_float3(vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]);
    }

    return emit_particles(s, mat_id, px.data(), pv.data(), count);
}

/* Remplace l'ensemble des colliders et recalcule le champ de distance signee.
 * A appeler une fois par frame, avant bq_step (voir bourrasque.h).
 *
 * Requiert qu'au moins un materiau ait deja ete enregistre (bq_add_material)
 * : prm (donc res et dx) n'est renseigne que par upload_params, declenche
 * par le premier ajout de materiau. Sans cette garde, un appel precoce lit
 * res=(0,0,0), calcule ncell=0 et lance une grille de 0 bloc -- ce qui
 * remonte un "CUDA: invalid argument" totalement opaque sur le premier
 * kernel venu plus loin dans la fonction. */
BQ_API int bq_set_colliders(BqSim* s, const float* tri, const float* tri_vel,
                            const float* tri_friction, int n_tri) {
    if (s->n_mats == 0) {
        snprintf(g_error, sizeof(g_error),
                 "bq_set_colliders: aucun materiau enregistre -- appeler "
                 "bq_add_material au moins une fois avant de definir des colliders");
        return -1;
    }
    int ncell = s->prm.res.x * s->prm.res.y * s->prm.res.z;
    dim3 bp(256), gc((ncell + 255) / 256);

    if (n_tri < 0) {
        snprintf(g_error, sizeof(g_error), "bq_set_colliders: n_tri negatif (%d)", n_tri);
        return -1;
    }
    if (n_tri == 0) {
        s->n_tri = 0;
        k_fill_sdf<<<gc, bp>>>(s->d_sdf, s->d_cvel, s->d_cnrm, ncell);
        BQ_CUDA_CHECK(cudaGetLastError());
        BQ_CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    }
    if (tri == NULL || tri_vel == NULL || tri_friction == NULL) {
        snprintf(g_error, sizeof(g_error),
                 "bq_set_colliders: pointeur nul (n_tri=%d)", n_tri);
        return -1;
    }

    /* reallocation seulement quand la capacite courante est depassee */
    if (n_tri > s->tri_cap) {
        cudaFree(s->d_tri); cudaFree(s->d_trivel); cudaFree(s->d_trifric);
        s->d_tri = nullptr; s->d_trivel = nullptr; s->d_trifric = nullptr;
        s->tri_cap = 0;
        if (cudaMalloc(&s->d_tri, (size_t)n_tri * 3 * sizeof(float3)) != cudaSuccess ||
            cudaMalloc(&s->d_trivel, (size_t)n_tri * 3 * sizeof(float3)) != cudaSuccess ||
            cudaMalloc(&s->d_trifric, (size_t)n_tri * sizeof(float)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_set_colliders: cudaMalloc echoue (n_tri=%d)", n_tri);
            return -1;
        }
        s->tri_cap = n_tri;
    }
    s->n_tri = n_tri;

    BQ_CUDA_CHECK(cudaMemcpy(s->d_tri, tri, (size_t)n_tri * 9 * sizeof(float),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_trivel, tri_vel, (size_t)n_tri * 9 * sizeof(float),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_trifric, tri_friction, (size_t)n_tri * sizeof(float),
                             cudaMemcpyHostToDevice));

    /* AABB des colliders, cote hote, dilatee de quelques dx pour la bande
     * etroite du kernel de distance. */
    float3 lo = make_float3(3.4e38f, 3.4e38f, 3.4e38f);
    float3 hi = make_float3(-3.4e38f, -3.4e38f, -3.4e38f);
    for (int i = 0; i < 3 * n_tri; ++i) {
        float x = tri[3 * i], y = tri[3 * i + 1], z = tri[3 * i + 2];
        lo.x = fminf(lo.x, x); lo.y = fminf(lo.y, y); lo.z = fminf(lo.z, z);
        hi.x = fmaxf(hi.x, x); hi.y = fmaxf(hi.y, y); hi.z = fmaxf(hi.z, z);
    }
    float pad = 3.f * s->prm.dx;
    lo = make_float3(lo.x - pad, lo.y - pad, lo.z - pad);
    hi = make_float3(hi.x + pad, hi.y + pad, hi.z + pad);

    /* grille de buckets : construite sur l'hote (voir build_bucket_grid),
     * puis televersee. Realloc device seulement si la capacite courante
     * (nb de buckets, nb de references de triangles) est depassee. */
    BucketGridHost bg;
    std::vector<int> bucket_off, bucket_tri;
    if (!build_bucket_grid(tri, n_tri, BQ_BUCKET_DX_MULT * s->prm.dx, bg,
                           bucket_off, bucket_tri)) {
        return -1; /* g_error deja rempli par build_bucket_grid */
    }
    int nb = bg.res.x * bg.res.y * bg.res.z; /* borne par BQ_BUCKET_MAX_TOTAL_BUCKETS */

    if (nb + 1 > s->bucket_off_cap) {
        cudaFree(s->d_bucket_off);
        s->d_bucket_off = nullptr; s->bucket_off_cap = 0;
        if (cudaMalloc(&s->d_bucket_off, (size_t)(nb + 1) * sizeof(int)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_set_colliders: cudaMalloc bucket_off echoue (nb=%d)", nb);
            return -1;
        }
        s->bucket_off_cap = nb + 1;
    }
    int n_refs = (int)bucket_tri.size();
    if (n_refs > s->bucket_tri_cap) {
        cudaFree(s->d_bucket_tri);
        s->d_bucket_tri = nullptr; s->bucket_tri_cap = 0;
        if (n_refs > 0 &&
            cudaMalloc(&s->d_bucket_tri, (size_t)n_refs * sizeof(int)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_set_colliders: cudaMalloc bucket_tri echoue (n_refs=%d)", n_refs);
            return -1;
        }
        s->bucket_tri_cap = n_refs;
    }
    BQ_CUDA_CHECK(cudaMemcpy(s->d_bucket_off, bucket_off.data(),
                             (size_t)(nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (n_refs > 0)
        BQ_CUDA_CHECK(cudaMemcpy(s->d_bucket_tri, bucket_tri.data(),
                                 (size_t)n_refs * sizeof(int), cudaMemcpyHostToDevice));

    /* 1. distance non signee + vitesse/friction, par recherche en anneaux
     *    de buckets. Amorce aussi l'etat de signe (d_ext) : graines EXTERIOR
     *    / INTERIOR dans la bande proche de la surface, UNKNOWN partout
     *    ailleurs (cf. k_sdf_unsigned). */
    k_sdf_unsigned<<<gc, bp>>>(s->d_sdf, s->d_cvel, s->d_cnrm, s->d_ext,
                               s->d_tri, s->d_trivel,
                               s->d_trifric, s->d_bucket_off, s->d_bucket_tri,
                               bg.origin, bg.h, bg.res, lo, hi, ncell);
    BQ_CUDA_CHECK(cudaGetLastError());

    /* 2. signe par propagation depuis les graines, jusqu'a convergence
     * (drapeau "changed" relu par l'hote toutes les BQ_SDF_CHECK_EVERY
     * passes, pas a chaque passe -- un cudaMemcpy synchrone par passe serait
     * jusqu'a ~200 allers-retours bloquants par frame sur une grille 64^3).
     *
     * Borne d'iteration max_iter : PAS une garantie theorique. Le nombre de
     * passes necessaire pour qu'une graine atteigne une cellule donnee est
     * la distance geodesique en 6-voisinage jusqu'a la graine la plus
     * proche ; pour une region convexe elle est bornee par la somme des
     * dimensions de la grille (diametre de Manhattan), mais pour un
     * exterieur tortueux (cavites, obstacles imbriques) elle peut la
     * depasser. On borne quand meme la boucle avec une marge heuristique
     * (protection contre une boucle infinie), et une eventuelle troncature
     * est sans danger grace a la regle de securite de k_sdf_finalize_sign :
     * toute cellule encore UNKNOWN a la sortie est traitee comme exterieure,
     * jamais comme solide. On la rend neanmoins detectable (message sur
     * stderr) plutot que silencieuse. */
    int max_iter = 2 * (s->prm.res.x + s->prm.res.y + s->prm.res.z);
    const int BQ_SDF_CHECK_EVERY = 8;
    BQ_CUDA_CHECK(cudaMemset(s->d_changed, 0, sizeof(int)));
    bool converged = false;
    for (int it = 0; it < max_iter; ++it) {
        k_sdf_propagate_sign<<<gc, bp>>>(s->d_ext, ncell, s->d_changed);
        BQ_CUDA_CHECK(cudaGetLastError());
        bool check_now = ((it + 1) % BQ_SDF_CHECK_EVERY == 0) || (it + 1 == max_iter);
        if (check_now) {
            int h_changed = 0;
            BQ_CUDA_CHECK(cudaMemcpy(&h_changed, s->d_changed, sizeof(int),
                                     cudaMemcpyDeviceToHost));
            if (!h_changed) { converged = true; break; }
            BQ_CUDA_CHECK(cudaMemset(s->d_changed, 0, sizeof(int)));
        }
    }
    if (!converged) {
        fprintf(stderr,
                "bourrasque: bq_set_colliders: propagation du signe du SDF "
                "tronquee apres %d passes (geometrie tortueuse ?) -- les "
                "cellules encore indeterminees sont traitees comme "
                "exterieures (regle de securite), jamais comme solides.\n",
                max_iter);
    }

    /* 3. applique le signe : negatif la ou l'etat resolu est INTERIOR,
     * positif partout ailleurs (EXTERIOR ou UNKNOWN, cf. regle de securite
     * dans k_sdf_finalize_sign). */
    k_sdf_finalize_sign<<<gc, bp>>>(s->d_sdf, s->d_cnrm, s->d_ext, ncell);
    BQ_CUDA_CHECK(cudaGetLastError());
    BQ_CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

BQ_API int bq_step(BqSim* s, float frame_dt) {
    if (s->n == 0 || s->n_mats == 0) return 0;
    int substeps = (int)ceilf(frame_dt / s->dt);
    int ncell = s->prm.res.x * s->prm.res.y * s->prm.res.z;
    dim3 bp(256), gp((s->n + 255) / 256), gc((ncell + 255) / 256);

    for (int i = 0; i < substeps; ++i) {
        k_clear_grid<<<gc, bp>>>(s->d_grid, ncell);
        k_p2g<<<gp, bp>>>(s->d_x, s->d_v, s->d_C, s->d_F, s->d_J, s->d_mat,
                          s->d_grid, s->n);
        k_grid_update<<<gc, bp>>>(s->d_grid, s->d_sdf, s->d_cvel, s->d_cnrm, ncell);
        k_g2p<<<gp, bp>>>(s->d_x, s->d_v, s->d_C, s->d_J, s->d_mat,
                          s->d_grid, s->d_sdf, s->d_cnrm, s->n);
    }
    BQ_CUDA_CHECK(cudaGetLastError());
    BQ_CUDA_CHECK(cudaDeviceSynchronize());
    return substeps;
}

BQ_API int bq_particle_count(const BqSim* s) { return s->n; }

BQ_API int bq_read_positions(BqSim* s, float* dst) {
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_x, s->n * sizeof(float3),
                             cudaMemcpyDeviceToHost));
    return s->n;
}

BQ_API int bq_read_materials(BqSim* s, uint8_t* dst) {
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_mat, s->n * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost));
    return s->n;
}

/* Copie le champ de distance signee courant vers dst (ncell floats, meme
 * indexation ((i*res.y+j)*res.z+k) que le reste du solveur). Diagnostic et
 * validation (voir bq_read_positions pour la meme convention). */
BQ_API int bq_read_sdf(BqSim* s, float* dst) {
    int ncell = s->prm.res.x * s->prm.res.y * s->prm.res.z;
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_sdf, ncell * sizeof(float),
                             cudaMemcpyDeviceToHost));
    return ncell;
}

BQ_API const char* bq_last_error(void) { return g_error; }

} /* extern "C" */

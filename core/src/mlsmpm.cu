/* mlsmpm.cu -- solveur MLS-MPM 3D (Hu et al. 2018), B-splines quadratiques.
 *
 * REFERENCE : scripts/ref_mlsmpm.py est la specification executable de ce
 * fichier. Chaque kernel est la transcription d'un bloc de Sim.substep().
 * En cas de doute sur une formule, la reference NumPy fait foi.
 *
 * Pipeline par substep (fluide seul -- cf. bq_step pour l'ordre complet
 * quand des corps rigides sont declares, M17) :
 *   1. k_clear_grid        : remise a zero (masse, quantite de mouvement)
 *   2. k_p2g                : maj de F, contrainte de Cauchy, scatter atomique
 *   3. k_grid_apply_gravity : v = mv/m, gravite
 *   4. k_grid_update         : conditions aux limites separantes (contact)
 *   5. k_g2p                 : gather v et C (APIC), advection, maj de J (eau)
 */
#define BQ_BUILD
#include "bourrasque.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* ------------------------------------------------------------------ erreurs */
#include "internal.h"
char g_error[512] = "";

/* ------------------------------------------------------------- petite algebre */
/* mat3 et ses operateurs de base vivent maintenant dans svd3.cuh (M18/S1) --
 * definition UNIQUE, partagee avec le harnais de validation de la SVD
 * (tools/repro/svd3_harness.cu) qui compile sans lier tout le solveur. */
#include "svd3.cuh"

/* Rotation depuis un quaternion (w, x, y, z) -- convention BqRigidBody::q,
 * partagee avec l'extension Python (une divergence de convention serait un
 * bug tres penible a diagnostiquer, cf. plan-milestone-17.md D5). */
__device__ inline mat3 quat_to_mat3(const float q[4]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    mat3 r;
    r.m[0] = 1.f - 2.f * (y * y + z * z); r.m[1] = 2.f * (x * y - w * z);       r.m[2] = 2.f * (x * z + w * y);
    r.m[3] = 2.f * (x * y + w * z);       r.m[4] = 1.f - 2.f * (x * x + z * z); r.m[5] = 2.f * (y * z - w * x);
    r.m[6] = 2.f * (x * z - w * y);       r.m[7] = 2.f * (y * z + w * x);       r.m[8] = 1.f - 2.f * (x * x + y * y);
    return r;
}

/* Petits helpers vectoriels, utilises par la CCD Moller-Trumbore (cf.
 * ccd_segment_tri plus bas) -- absents jusqu'ici du fichier, le reste du
 * code construisant ses float3 intermediaires a la main via make_float3. */
__device__ inline float3 vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 vadd(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 vcross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
__device__ inline float vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* Matrice de produit vectoriel [a]x, telle que [a]x * v == a x v pour tout v
 * (utilisee par le couplage implicite fluide-solide, cf. k_body_solve). */
__device__ inline mat3 skew(float3 a) {
    mat3 r{};
    r.m[0] =  0.f;  r.m[1] = -a.z;  r.m[2] =  a.y;
    r.m[3] =  a.z;  r.m[4] =  0.f;  r.m[5] = -a.x;
    r.m[6] = -a.y;  r.m[7] =  a.x;  r.m[8] =  0.f;
    return r;
}

/* Resolution d'un systeme lineaire 6x6 par elimination de Gauss avec pivot
 * partiel -- utilise par k_body_solve pour le couplage implicite corps/fluide
 * (M17/A5). n_bodies est petit (<= BQ_MAX_BODIES = 64) : un thread par corps,
 * elimination sequentielle, largement suffisant. Le systeme construit par
 * k_body_solve est symetrique defini positif (cf. commentaire de ce noyau),
 * mais le pivot partiel est conserve comme garde-fou peu couteux plutot que
 * de s'appuyer sur Cholesky, qui suppose la SPD-tude exacte et diverge
 * silencieusement au moindre residu numerique. A[6][6] est detruite par
 * l'appel (elimination en place). */
__device__ inline void solve6x6(float A[6][6], float b[6], float x[6]) {
    for (int col = 0; col < 6; ++col) {
        int piv = col;
        float best = fabsf(A[col][col]);
        for (int r = col + 1; r < 6; ++r) {
            float v = fabsf(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) {
            for (int c = 0; c < 6; ++c) { float t = A[col][c]; A[col][c] = A[piv][c]; A[piv][c] = t; }
            float t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        float diag = A[col][col];
        /* garde-fou : un systeme degenere (corps sans inertie propre sur un
         * axe, par exemple) ne doit jamais produire une division par ~0 --
         * seul un corps mal configure cote Python en paierait le prix (une
         * reponse figee sur cet axe), jamais un NaN qui contaminerait toute
         * la boucle de sous-pas. */
        if (fabsf(diag) < 1e-12f) diag = (diag >= 0.f) ? 1e-12f : -1e-12f;
        for (int r = col + 1; r < 6; ++r) {
            float f = A[r][col] / diag;
            if (f == 0.f) continue;
            for (int c = col; c < 6; ++c) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    for (int r = 5; r >= 0; --r) {
        float s = b[r];
        for (int c = r + 1; c < 6; ++c) s -= A[r][c] * x[c];
        float diag = A[r][r];
        if (fabsf(diag) < 1e-12f) diag = (diag >= 0.f) ? 1e-12f : -1e-12f;
        x[r] = s / diag;
    }
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

/* Intersection segment-triangle, Moller & Trumbore 1997 -- algorithme
 * standard d'intersection rayon-triangle, ici applique a un SEGMENT borne
 * (p0 -> p1) plutot qu'a un rayon infini : le test t in [0,1] remplace le
 * test t >= 0 habituel. Utilise par la CCD de k_g2p (cf. plus bas) comme
 * filet de securite SUPPLEMENTAIRE avant le mecanisme D7-D9 existant, pas un
 * remplacement -- ce dernier reste un test ponctuel aux extremites d'un
 * sous-pas, la CCD teste tout le segment de trajectoire. */
__device__ inline bool ccd_segment_tri(float3 p0, float3 p1, float3 v0, float3 v1, float3 v2,
                                       float* out_t) {
    float3 dir = vsub(p1, p0);
    float3 edge1 = vsub(v1, v0);
    float3 edge2 = vsub(v2, v0);
    float3 h = vcross(dir, edge2);
    float a = vdot(edge1, h);
    if (fabsf(a) < 1e-10f) return false; /* segment parallele au triangle */
    float f = 1.f / a;
    float3 s = vsub(p0, v0);
    float u = f * vdot(s, h);
    if (u < 0.f || u > 1.f) return false;
    float3 q = vcross(s, edge1);
    float v = f * vdot(dir, q);
    if (v < 0.f || u + v > 1.f) return false;
    float t = f * vdot(edge2, q);
    if (t < 0.f || t > 1.f) return false; /* hors du segment [p0,p1] */
    *out_t = t;
    return true;
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
#define BQ_CONTACT_BAND_MULT 0.5f /* experimental (session 2026-08-03) : 1.5f
    donnait ~0.95dx de vide de repos, 1.0f ET 0.75f donnent EXACTEMENT 0.5dx
    (palier, pas de reduction supplementaire entre les deux) -- sans fuite
    sur le test paroi fine dans les deux cas. On descend encore pour voir si
    0.5dx est un plancher de quantification de grille ou si ca continue a
    baisser. Voir diag_gap_collider.py / diag_leak_thinwall.py. */

/* Distance minimale, en multiples de dx, a laquelle la contrainte de position de
 * k_g2p maintient une particule de la surface d'un collider. La condition aux
 * limites de k_grid_update agit sur les vitesses de grille : elle est molle par
 * nature (le transfert grille-particule moyenne les noeuds contraints avec les
 * noeuds libres). Cette contrainte-ci agit sur les positions et est dure ; c'est
 * elle qui garantit qu'aucune particule ne franchit une paroi, quelle que soit
 * son epaisseur devant dx. */
#define BQ_CONTACT_PUSH_MULT 0.5f /* teste a 0.25f (session 2026-08-03) : aucun
    effet mesure sur le vide de repos (reste a 0.5dx pile, identique) ni sur
    le rebond -- ce mecanisme ne s'active pas dans le regime de repos sous
    gravite residuelle (voir diag_gap_collider.py / diag_bounce.py). Remis a
    sa valeur d'origine, aucun gain a le changer. */

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

/* Plafond de corps rigides declares (cf. bq_set_collider_bodies) : les
 * tampons device (etat, wrench) sont alloues une fois a cette capacite fixe
 * a bq_create, jamais realloues -- meme politique que mats_host ci-dessus,
 * un nombre de corps a deux chiffres n'exige aucune reallocation dynamique. */
#define BQ_MAX_BODIES 64

/* Plafond du tampon de contacts corps<->corps (M17, phase B, B3a). Large
 * devant l'usage attendu (une poignee de corps, un contact reel par paire de
 * cellules en recouvrement) mais fixe et petit en VRAM (BQ_MAX_CONTACTS *
 * sizeof(BqContact) ~ 300 Ko) -- au-dela, la generation REFUSE les contacts
 * en trop et le signale (bq_contacts_last_overflow), jamais une troncature
 * silencieuse (meme discipline que bq_whitewater_last_refused). */
#define BQ_MAX_CONTACTS 8192

/* Pas de temps de repli quand AUCUN materiau n'est enregistre (M17/B3b,
 * point 0 de la spec) : upload_params derive normalement dt de la vitesse du
 * son du materiau le plus raide (CFL acoustique), donc une scene de corps
 * rigides purs -- sans fluide, sans materiau -- n'a rien dont deriver un dt.
 * Sans repli, le plancher c_max=1e-3f de upload_params donnerait dt =
 * cfl*dx/1e-3 (plusieurs SECONDES a la config par defaut), bien trop grand
 * pour integrer un contact stable. Valeur choisie : 1/120s, un ordre de
 * grandeur standard pour un solveur de corps rigides a la Box2D/Bullet
 * (8 iterations de Gauss-Seidel par sous-pas convergent bien a cette
 * cadence). Sans effet des qu'un materiau est enregistre. */
#define BQ_NO_FLUID_DT (1.0f / 120.f)

/* --------------------------------------- solveur de contact corps<->corps (M17, B3b, D11)
 *
 * Capacite du solveur d'impulsions sequentielles : DISTINCTE du plafond de
 * GENERATION BQ_MAX_CONTACTS (8192) ci-dessus. k_contact_solve est un noyau
 * MONO-BLOC (cf. son commentaire) qui garde tous les contacts qu'il traite
 * en memoire PARTAGEE pour la duree du Gauss-Seidel sequentiel -- une
 * contrainte de capacite fixe et bien plus etroite que la simple VRAM du
 * tampon de generation. 256 contacts actifs SIMULTANEMENT est deja large
 * devant l'usage cible du jalon ("peu de corps et peu de contacts", D11) :
 * quelques caisses posees ou empilees produisent des dizaines de contacts,
 * pas des centaines. Au-dela, les contacts en exces (au-dela de
 * BQ_CONTACT_SOLVE_CAP parmi ceux generes) sont simplement IGNORES par le
 * solveur ce sous-pas -- stabilite verifiee malgre l'omission (verification
 * 7 de la porte de phase B), a distinguer de bq_contacts_last_overflow qui
 * lui rapporte la saturation de la GENERATION. */
#define BQ_CONTACT_SOLVE_CAP 256
#define BQ_CONTACT_ITERATIONS 8   /* D11 : "iterations exposees, defaut 8" --
                                     constante documentee ; aucune fonction
                                     API n'est demandee par ce lot pour la
                                     rendre reglable a l'execution. */
#define BQ_CONTACT_SLOP_FRAC 0.1f /* D11 : penetration toleree, "de l'ordre de
                                     10% d'un voxel" -- reference = c_p.dx, le
                                     voxel du DOMAINE (coherent avec D9, qui
                                     vise la meme taille pour les SDF locaux). */
#define BQ_CONTACT_BETA 0.2f      /* Facteur de correction du canal SEPARE
                                     (split impulse) -- PAS un biais de
                                     Baumgarte injecte dans la vitesse REELLE
                                     (cf. commentaire de k_contact_solve pour
                                     la distinction, qui est le coeur de D11). */
#define BQ_RESTITUTION_VEL_EPS 1.0f /* m/s : sous ce seuil de vitesse de
                                     fermeture, un contact est traite comme
                                     parfaitement inelastique meme si
                                     restitution > 0 -- sans ce garde-fou, le
                                     micro-choc que la gravite inflige a un
                                     corps au repos CHAQUE sous-pas serait
                                     amplifie par la restitution et le
                                     corps ne s'immobiliserait jamais
                                     (frémissement permanent, cf.
                                     verification 1). Valeur usuelle
                                     (Box2D/Bullet). */

/* --------------------------------------------------- mise en sommeil (M17, B3b, D11) */
#define BQ_SLEEP_LIN_THRESH 0.01f  /* m/s */
#define BQ_SLEEP_ANG_THRESH 0.01f  /* rad/s */
#define BQ_SLEEP_SUBSTEPS 30       /* sous-pas consecutifs sous le seuil avant endormissement */
#define BQ_WAKE_LIN_THRESH 0.05f   /* m/s : au-dela, un corps endormi est reveille */
#define BQ_WAKE_ANG_THRESH 0.05f   /* rad/s */

struct MaterialGpu {
    int   model;
    float p_mass;         /* rho * p_vol */
    float mu, lam;        /* Lame (ELASTIC et SAND) */
    float bulk, gamma;    /* Tait (WATER)   */
    /* SAND uniquement (Drucker-Prager) : coefficient de frottement du cone,
     * precalcule cote hote (upload_params) a partir de friction_angle -- pas
     * de trigonometrie par particule et par sous-pas. Cf. plan-milestone-18.md
     * D2 : alpha = sqrt(2/3) * 2 sin(phi) / (3 - sin(phi)). */
    float alpha;
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
    } else if (m.model == BQ_MODEL_SAND) {
        /* Drucker-Prager elastoplastique (Klar et al. 2016), cf.
         * plan-milestone-18.md D2. Travaille dans l'espace des valeurs
         * singulieres de F (deformation elastique) : Hencky (log des valeurs
         * singulieres) donne un espace ou le critere de Drucker-Prager (un
         * cone dans l'espace des contraintes principales) devient un
         * "return mapping" simple -- projection radiale de la partie
         * deviatorique de eps_log. */
        mat3 F;
        memcpy(F.m, Fbuf + 9 * p, 9 * sizeof(float));
        F = matmul(mat3::identity() + c_p.dt * C, F);   /* F <- (I + dt C) F */

        mat3 U, V; float3 S;
        svd3(F, U, S, V);
        /* Borne AVANT le log : svd3 ne clampe jamais (documente dans
         * svd3.cuh), c'est a l'appelant de le faire. |S| : la SVD peut
         * ressortir S.z negatif (reflexion absorbee), le log ne s'applique
         * qu'a une magnitude. */
        const float eps_svd = 1e-6f;
        S.x = fmaxf(fabsf(S.x), eps_svd);
        S.y = fmaxf(fabsf(S.y), eps_svd);
        S.z = fmaxf(fabsf(S.z), eps_svd);

        float3 eps_log = make_float3(logf(S.x), logf(S.y), logf(S.z));
        float tr = eps_log.x + eps_log.y + eps_log.z;
        float inv3 = tr * (1.f / 3.f);
        float3 dev = make_float3(eps_log.x - inv3, eps_log.y - inv3, eps_log.z - inv3);
        float devnorm = sqrtf(dev.x * dev.x + dev.y * dev.y + dev.z * dev.z);

        if (tr > 0.f) {
            /* SOMMET DU CONE : dilatation pure, le sable ne TIRE pas. C'est
             * le trait qui distingue visuellement le sable d'un solide --
             * une poignee lachee se disperse au lieu de rester en bloc,
             * parce qu'ici toute trace elastique est effacee (S -> 1,
             * eps_log -> 0) et la contrainte qui en decoule plus bas est
             * nulle : la particule ne "tient" plus rien. */
            S = make_float3(1.f, 1.f, 1.f);
            eps_log = make_float3(0.f, 0.f, 0.f);
        } else if (devnorm > 1e-12f) {
            /* cf. D2 : dgamma = ||dev|| + alpha * (3 lam + 2 mu)/(2 mu) * tr */
            float dgamma = devnorm + m.alpha * (3.f * m.lam + 2.f * m.mu) /
                                          (2.f * m.mu) * tr;
            if (dgamma > 0.f) {
                /* hors du cone : projection radiale sur sa surface */
                float scale = dgamma / devnorm;
                eps_log.x -= scale * dev.x;
                eps_log.y -= scale * dev.y;
                eps_log.z -= scale * dev.z;
                S = make_float3(expf(eps_log.x), expf(eps_log.y), expf(eps_log.z));
            }
            /* dgamma <= 0 : dans le cone, purement elastique, S/eps_log
             * inchanges (deja la valeur predite ci-dessus) */
        }
        /* devnorm == 0 (deformation isotrope pure, tr <= 0) : inchange */

        mat3 Sdiag = mat3::zero();
        Sdiag.m[0] = S.x; Sdiag.m[4] = S.y; Sdiag.m[8] = S.z;
        F = matmul(U, matmul(Sdiag, transpose(V)));
        memcpy(Fbuf + 9 * p, F.m, 9 * sizeof(float));

        /* tau = U (2 mu eps_log + lam tr I) U^T -- contrainte de Kirchhoff,
         * exprimee dans le repere de U (principal du F elastique) puis
         * ramenee au repere monde. */
        float tr_final = eps_log.x + eps_log.y + eps_log.z;
        mat3 sigma_diag = mat3::zero();
        sigma_diag.m[0] = 2.f * m.mu * eps_log.x + m.lam * tr_final;
        sigma_diag.m[4] = 2.f * m.mu * eps_log.y + m.lam * tr_final;
        sigma_diag.m[8] = 2.f * m.mu * eps_log.z + m.lam * tr_final;
        stress = matmul(U, matmul(sigma_diag, transpose(U)));
    } else { /* BQ_MODEL_WATER : EOS de Tait, sigma = -p I.
              * Un device kernel ne peut pas lever d'erreur : ce else reste
              * implicitement "sinon WATER", mais il est desormais garde par
              * bq_add_material (cote hote) qui refuse tout modele inconnu
              * avant qu'il n'atteigne jamais ce code -- cf. D2/plan M18. */
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
 * (normale nulle, distance non signee 1e6) vide elle aussi, et aucune
 * cellule attribuee a un corps (cbody = -1, cf. D2 du plan M17). */
__global__ void k_fill_sdf(float* sdf, float4* cvel, float4* cnrm,
                           int* __restrict__ cbody, int ncell) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    sdf[id] = 1e6f;
    cvel[id] = make_float4(0.f, 0.f, 0.f, 0.f);
    cnrm[id] = make_float4(0.f, 0.f, 0.f, 1e6f);
    cbody[id] = -1;
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
                               const int* __restrict__ tri_body,
                               int default_body,
                               int* __restrict__ cbody,
                               const int* __restrict__ bucket_off,
                               const int* __restrict__ bucket_tri,
                               float3 bucket_origin, float bucket_h, int3 bucket_res,
                               float3 aabb_lo, float3 aabb_hi, int ncell,
                               int3 res, float dx) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;

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
    const float e = BQ_SDF_NODE_EPS * dx;
    float3 p = make_float3(i * dx + e, j * dx + e, k * dx + e);
    bool active = !(p.x < aabb_lo.x || p.x > aabb_hi.x || p.y < aabb_lo.y ||
                    p.y > aabb_hi.y || p.z < aabb_lo.z || p.z > aabb_hi.z);
    if (!active) {
        sdf[id] = 1e6f;
        cvel[id] = make_float4(0.f, 0.f, 0.f, 0.f);
        cnrm[id] = make_float4(0.f, 0.f, 0.f, 1e6f);
        state[id] = BQ_SDF_STATE_UNKNOWN; /* resolu par propagation, cf. note plus haut */
        cbody[id] = -1; /* hors AABB : jamais attribuee a un corps (D2, plan M17) */
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

    /* Identite de corps (D2, plan M17) : derivee gratuitement du triangle
     * gagnant deja trouve ci-dessus, aucune recherche supplementaire. Le
     * couplage fluide -> solide de k_grid_update s'en sert pour attribuer
     * l'impulsion recoltee au bon corps. tri_body == NULL (aucun tableau
     * fourni par l'appelant) retombe sur default_body, calcule cote hote :
     * 0 si des corps ont ete declares (compatibilite -- une Sim qui ne
     * distingue pas ses colliders les traite comme un unique corps 0), -1
     * sinon (aucun corps declare : comportement cinematique actuel inchange,
     * cf. non-regression). */
    cbody[id] = (best_t >= 0) ? ((tri_body != nullptr) ? tri_body[best_t] : default_body)
                              : -1;

    /* Amorce (graine) uniquement dans la bande, et seulement si le test local
     * a pu trancher. Hors bande, ou test local degenere : UNKNOWN, resolu
     * plus tard par propagation depuis une graine voisine (jamais force a
     * "interieur" par defaut -- cf. regle de securite en tete de fichier). */
    bool near_band = sdf[id] < BQ_SDF_WALL_EPS_MULT * dx;
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
                                     int ncell, int* changed, int3 res) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    if (state[id] != BQ_SDF_STATE_UNKNOWN) return; /* deja resolue */

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

/* ------------------------------------------------- SDF locaux par corps (M17, phase B, D9)
 *
 * Un corps rigide porte son PROPRE champ de distance signee, en repere de
 * CORPS (origine au centre de masse), construit UNE SEULE FOIS au demarrage
 * du bake (bq_build_body_sdf) puis jamais reconstruit -- contrairement au
 * champ collider fusionne de bq_set_colliders (espace monde, reconstruit a
 * chaque frame), qui vaut zero a la surface de son propre corps et ne peut
 * donc pas servir au contact corps <-> corps.
 *
 * Convention de repere local : les noyaux k_sdf_unsigned / k_sdf_propagate_sign
 * (reutilises tels quels, cf. D0/plan-milestone-17.md) echantillonnent leurs
 * noeuds a p = (i*dx, j*dx, k*dx) SANS parametre d'origine -- ils ne
 * connaissent que la resolution et le pas passes en argument. Pour batir un
 * SDF dont le coin min n'est pas l'origine du repere de corps (le cas
 * general : un centre de masse n'est presque jamais au coin de son AABB), les
 * triangles sont donc TRANSLATES avant construction (cf. bq_build_body_sdf),
 * de sorte que le coin min de l'AABB dilatee tombe exactement sur (0,0,0). Ce
 * decalage est memorise dans BqBodySdf::origin et reapplique en sens inverse
 * a chaque requete (bq_body_sdf_sample ci-dessous). L'oublier ferait
 * interroger le champ a la mauvaise coordonnee -- silencieusement, puisque le
 * champ existe bel et bien, juste decale : c'est le mode d'echec le plus
 * probable de ce mecanisme, verifie explicitement par invariance sous
 * transformation rigide (cf. tools/repro/verify_body_sdf.py). */
struct BqBodySdf {
    float3 origin; /* coin min de la grille locale, repere de CORPS (le decalage
                       de translation applique aux triangles avant construction) */
    float  cell;   /* taille de voxel EFFECTIVE (peut depasser target_cell si le
                       plafond max_res a ete declenche, cf. bq_build_body_sdf) */
    int3   res;    /* resolution (<= max_res par axe) */
    float* phi;    /* res.x*res.y*res.z valeurs, pointeur DEVICE. nullptr = pas
                       encore construit (corps sans SDF local, cf. body_sdf) */
};

/* Echantillonnage trilineaire brut, sans repli hors grille (appele par
 * body_sdf ci-dessous, qui gere lui le hors-grille et le gradient). Suppose
 * p_local DEJA verifie dans [origin, origin+(res-1)*cell] par l'appelant --
 * le clamp d'indice ci-dessous n'est qu'une garde contre l'arrondi flottant
 * au bord exact de cet intervalle, pas une politique de hors-grille. */
__device__ inline float bq_body_sdf_sample(const BqBodySdf& s, float3 p_local) {
    float qx = (p_local.x - s.origin.x) / s.cell;
    float qy = (p_local.y - s.origin.y) / s.cell;
    float qz = (p_local.z - s.origin.z) / s.cell;
    int i0 = (int)floorf(qx); if (i0 < 0) i0 = 0; if (i0 > s.res.x - 2) i0 = s.res.x - 2;
    int j0 = (int)floorf(qy); if (j0 < 0) j0 = 0; if (j0 > s.res.y - 2) j0 = s.res.y - 2;
    int k0 = (int)floorf(qz); if (k0 < 0) k0 = 0; if (k0 > s.res.z - 2) k0 = s.res.z - 2;
    float fx = qx - (float)i0, fy = qy - (float)j0, fz = qz - (float)k0;
    fx = fminf(fmaxf(fx, 0.f), 1.f);
    fy = fminf(fmaxf(fy, 0.f), 1.f);
    fz = fminf(fmaxf(fz, 0.f), 1.f);

    int strideI = s.res.y * s.res.z, strideJ = s.res.z;
    int idx = i0 * strideI + j0 * strideJ + k0;
    float c000 = s.phi[idx],               c100 = s.phi[idx + strideI];
    float c010 = s.phi[idx + strideJ],     c110 = s.phi[idx + strideI + strideJ];
    float c001 = s.phi[idx + 1],           c101 = s.phi[idx + strideI + 1];
    float c011 = s.phi[idx + strideJ + 1], c111 = s.phi[idx + strideI + strideJ + 1];

    float c00 = c000 * (1.f - fx) + c100 * fx;
    float c10 = c010 * (1.f - fx) + c110 * fx;
    float c01 = c001 * (1.f - fx) + c101 * fx;
    float c11 = c011 * (1.f - fx) + c111 * fx;
    float c0 = c00 * (1.f - fy) + c10 * fy;
    float c1 = c01 * (1.f - fy) + c11 * fy;
    return c0 * (1.f - fz) + c1 * fz;
}

/* Requete du SDF local d'un corps en un point DEJA EN REPERE DE CORPS (p_local
 * = R^T(p_monde - x_corps), a la charge de l'appelant -- ce champ, rigide,
 * n'a besoin de rien connaitre de la pose monde). Interpolation trilineaire ;
 * hors de la grille locale (ou corps sans SDF construit) : grande valeur
 * positive, JAMAIS zero, jamais une extrapolation -- meme discipline que
 * bq_read_sdf/bq_mesher_set_collider_sdf. Gradient par differences centrees,
 * calcule A LA VOLEE (aucun stockage supplementaire) si grad_or_null non nul
 * ; pas = un voxel, sondes repliees (clamp) a l'interieur de la grille pres
 * du bord plutot que de laisser la sentinelle hors-grille contaminer une
 * difference finie. */
__device__ inline float body_sdf(const BqBodySdf& s, float3 p_local, float3* grad_or_null) {
    if (s.phi == nullptr) {
        if (grad_or_null) *grad_or_null = make_float3(0.f, 0.f, 0.f);
        return 1e6f;
    }
    float3 lo = s.origin;
    float3 hi = make_float3(s.origin.x + (float)(s.res.x - 1) * s.cell,
                            s.origin.y + (float)(s.res.y - 1) * s.cell,
                            s.origin.z + (float)(s.res.z - 1) * s.cell);
    bool inside = p_local.x >= lo.x && p_local.x <= hi.x &&
                  p_local.y >= lo.y && p_local.y <= hi.y &&
                  p_local.z >= lo.z && p_local.z <= hi.z;
    if (!inside) {
        if (grad_or_null) *grad_or_null = make_float3(0.f, 0.f, 0.f);
        return 1e6f;
    }
    float phi = bq_body_sdf_sample(s, p_local);
    if (grad_or_null) {
        float h = s.cell;
        /* Sondes repliees (clamp) sur chaque axe : pres du bord de la grille,
         * un pas complet h deborderait, ce qui reviendrait a lire la
         * sentinelle hors-grille et a fabriquer un gradient enorme et faux.
         * Le pas effectif (denominateur) se reduit alors en consequence --
         * difference decentree plutot que centree tout pres du bord, jamais
         * de contamination par la sentinelle. */
        float3 cl_xp = make_float3(fminf(p_local.x + h, hi.x), p_local.y, p_local.z);
        float3 cl_xm = make_float3(fmaxf(p_local.x - h, lo.x), p_local.y, p_local.z);
        float3 cl_yp = make_float3(p_local.x, fminf(p_local.y + h, hi.y), p_local.z);
        float3 cl_ym = make_float3(p_local.x, fmaxf(p_local.y - h, lo.y), p_local.z);
        float3 cl_zp = make_float3(p_local.x, p_local.y, fminf(p_local.z + h, hi.z));
        float3 cl_zm = make_float3(p_local.x, p_local.y, fmaxf(p_local.z - h, lo.z));
        float dx2 = cl_xp.x - cl_xm.x, dy2 = cl_yp.y - cl_ym.y, dz2 = cl_zp.z - cl_zm.z;
        float gx = (dx2 > 1e-12f) ? (bq_body_sdf_sample(s, cl_xp) - bq_body_sdf_sample(s, cl_xm)) / dx2 : 0.f;
        float gy = (dy2 > 1e-12f) ? (bq_body_sdf_sample(s, cl_yp) - bq_body_sdf_sample(s, cl_ym)) / dy2 : 0.f;
        float gz = (dz2 > 1e-12f) ? (bq_body_sdf_sample(s, cl_zp) - bq_body_sdf_sample(s, cl_zm)) / dz2 : 0.f;
        *grad_or_null = make_float3(gx, gy, gz);
    }
    return phi;
}

/* Applique la masse (mv -> v) et la gravite a chaque noeud de grille -- ex-
 * premiere moitie de k_grid_update, sortie en noyau independant (M17/A5)
 * parce que l'ordre du sous-pas exige desormais que la grille porte deja
 * "la vitesse apres gravite" AVANT la recolte du couplage implicite
 * (k_grid_gather, cf. plus bas) et AVANT la resolution du systeme 6x6
 * (k_body_solve) -- la condition de contact (ce que fait encore
 * k_grid_update) ne peut, elle, s'appliquer qu'APRES : c'est l'etat resolu
 * du corps qui doit fournir la vitesse de mur.
 *
 * Comportement NUMERIQUEMENT IDENTIQUE a l'ancien bloc inline : memes
 * operations flottantes, dans le meme ordre, sur la meme cellule -- scinder
 * ce calcul en son propre noyau ne change aucun bit de resultat, seulement
 * le sous-pas ou il s'execute par rapport au couplage corps rigide. Tourne
 * sur TOUTE la grille, avec ou sans corps declare (n_bodies == 0 compris) :
 * c'est le meme calcul qu'avant pour ce cas, cf. non-regression D14. */
__global__ void k_grid_apply_gravity(float4* grid, int ncell) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    float4 g = grid[id];
    if (g.w <= 0.f) return;
    float3 v = make_float3(g.x / g.w, g.y / g.w, g.z / g.w);
    v.y += c_p.dt * c_p.gravity_y;
    grid[id] = make_float4(v.x, v.y, v.z, g.w);
}

__global__ void k_grid_update(float4* grid, const float* __restrict__ sdf,
                              const float4* __restrict__ cvel,
                              const float4* __restrict__ cnrm,
                              const int* __restrict__ cbody,
                              const BqRigidBody* __restrict__ bodies,
                              int ncell) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    float4 g = grid[id];
    if (g.w <= 0.f) return;

    /* La grille porte deja "vitesse apres gravite" (k_grid_apply_gravity,
     * execute plus tot dans le sous-pas -- cf. bq_step). Ce noyau n'a plus
     * qu'a resoudre la condition de contact ; la recolte d'impulsion (D1
     * jusqu'a A1) a ete DEPLACEE vers k_grid_gather/k_body_solve (M17/A5,
     * couplage implicite) -- ce noyau ne touche donc plus a d_body_wrench. */
    float3 v = make_float3(g.x, g.y, g.z);

    int3 res = c_p.res; int bnd = c_p.bound;
    int i = id / (res.y * res.z);
    int j = (id / res.z) % res.y;
    int k = id % res.z;

    /* Identite de corps de la cellule (D2) : -1 si aucun corps (hors AABB
     * collider ou aucun corps declare), sinon indice dans `bodies`. */
    int body_id = cbody[id];
    bool body_dyn = (body_id >= 0) && bodies[body_id].dynamic;

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
            float fr = cv.w; /* friction : toujours issue de cvel, quel que soit le corps */
            float3 vc;
            if (body_dyn) {
                /* Vitesse de mur VIVE (D4, plan M17) : recalculee depuis
                 * l'etat courant du corps plutot que lue dans cvel (figee a
                 * la derniere frame). C'est ce qui stabilise le couplage --
                 * un corps qui accelere est moins pousse des le sous-pas
                 * suivant. Echantillonnage AUX NOEUDS (i*dx), meme
                 * convention que k_sdf_unsigned ; x_corps est l'origine du
                 * repere de corps (centre de masse), donc le bras de levier
                 * est bien x_noeud - x_corps. cnrm.w (friction) reste issu
                 * de cvel dans tous les cas : seule la partie rigide de la
                 * vitesse de mur est vive. */
                float3 xnode = make_float3(i * c_p.dx, j * c_p.dx, k * c_p.dx);
                const BqRigidBody& bd = bodies[body_id];
                float3 bx = make_float3(bd.x[0], bd.x[1], bd.x[2]);
                float3 bv = make_float3(bd.v[0], bd.v[1], bd.v[2]);
                float3 bw = make_float3(bd.w[0], bd.w[1], bd.w[2]);
                float3 r = vsub(xnode, bx);
                float3 wxr = vcross(bw, r);
                vc = make_float3(bv.x + wxr.x, bv.y + wxr.y, bv.z + wxr.z);
            } else {
                /* Corps absent ou cinematique : chemin actuel inchange, lu
                 * dans cvel (cf. non-regression, plan M17 verification 1). */
                vc = make_float3(cv.x, cv.y, cv.z);
            }
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

    /* Conditions separantes : composante normale annulee vers la paroi.
     *
     * Le test porte sur j <= b, et non j < b : le clamp de position de k_g2p
     * retient les particules a y = b*dx, c'est-a-dire SUR le noeud d'indice b.
     * Avec j < b ce noeud restait libre, donc les particules maintenues par le
     * clamp y recevaient la gravite a chaque substep, repressaient, et se
     * faisaient re-clamper. Le clamp deplace la particule sans toucher son J :
     * ce volume-la etait detruit sans que la loi de comportement le voie, et le
     * fond d'une colonne au repos se tassait indefiniment (mesure : 38
     * particules par cellule au lieu de 8 apres 4 s, Jw restant a 1.00 donc
     * pression nulle, voire negative). Contraindre le noeud b aligne le plan
     * ou la vitesse est annulee sur le plan ou les positions sont retenues.
     *
     * Ce n'est PAS le fait que la particule repousse qui compte : annuler sa
     * vitesse normale quand le clamp mord a ete essaye et ne change rien (meme
     * densite de fond, meme Jw, a la troisieme decimale). Ce qui compte est que
     * le clamp ABSORBE. Tant que c'est lui qui arrete le fluide, la deceleration
     * se produit dans un plan infiniment mince, sous la resolution de la grille :
     * tr(C) ne la voit pas, donc J ne descend pas, donc aucune pression ne nait.
     * Il faut que ce soit la condition de vitesse qui arrete le fluide, sur une
     * bande que la grille resout.
     *
     * Mesure, colonne d'eau au repos couvrant le fond du domaine (bulk = 4e4,
     * H = 0.25 m), avant / apres :
     *   densite du fond a 4 s   38 part./cellule (J_geo 0.21) -> 8 (J_geo 1.0)
     *   Jw au fond              1.004 (donc en TRACTION)      -> 0.950
     *   pression au fond        -0.14 kPa                     -> 2.2 kPa
     *                           (il en faut 2.45 pour porter la colonne)
     *   centre de masse a 20 s  -86 mm, non convergent        -> -21 mm, stable
     *
     * Contrepartie assumee : la bande contrainte gagne un noeud sur chaque face,
     * donc un fluide rapide s'arrete un peu plus tot devant la paroi. C'est le
     * comportement correct -- c'est la condition de vitesse qui doit arreter le
     * fluide, pas le clamp de position, qui n'est qu'un filet de securite. */
    if (i <= bnd && v.x < 0.f) v.x = 0.f;
    if (i >= res.x - bnd - 1 && v.x > 0.f) v.x = 0.f;
    if (j <= bnd && v.y < 0.f) v.y = 0.f;
    if (j >= res.y - bnd - 1 && v.y > 0.f) v.y = 0.f;
    if (k <= bnd && v.z < 0.f) v.z = 0.f;
    if (k >= res.z - bnd - 1 && v.z > 0.f) v.z = 0.f;

    grid[id] = make_float4(v.x, v.y, v.z, g.w);
}

/* Prediction des corps rigides (M17/A5), un thread par corps -- ex-
 * k_integrate_bodies, ampute de tout ce qui touchait l'impulsion fluide
 * (celle-ci est desormais integree implicitement par k_body_solve, apres
 * k_grid_gather). Ne fait rien si le corps est cinematique/statique : sa
 * position/orientation est fournie par l'appelant (Blender), le solveur n'y
 * touche jamais.
 *
 * Ce noyau applique les SEULES forces qui ne viennent pas du fluide :
 * gravite et terme gyroscopique. C'est le "v_b, w_b" de la derivation du
 * couplage implicite (cf. k_body_solve) -- l'etat du corps juste avant
 * qu'il ne "percute" la masse de fluide en contact.
 *
 * Rotation : l'inertie inverse est portee en repere de CORPS
 * (BqRigidBody::inv_inertia), tournee en repere MONDE via R * I_inv * R^T
 * avant usage (I_w_inv). Le terme gyroscopique w x (I_w * w) est ajoute
 * explicitement (Euler semi-implicite classique pour un corps rigide libre,
 * necessaire des qu'un corps n'a pas une inertie isotrope -- une toupie qui
 * ne le recevrait pas ne precederait jamais).
 *
 * Verrous d'axe appliques ICI (etat predit) ET A NOUVEAU apres k_body_solve
 * (D13 du plan M17) : la resolution du systeme 6x6 peut reintroduire une
 * composante verrouillee (le fluide pousse sur un axe bloque -> la ligne/
 * colonne correspondante du systeme n'est pas annulee, seul le verrou final
 * l'est). Les appliquer aussi ici evite que le "v_b, w_b" servant de membre
 * de droite au solve porte deja une composante qui devrait etre nulle.
 *
 * Mise en sommeil (M17/B3b, D11) : un corps endormi (asleep[b] != 0, decide
 * par k_body_sleep_update a la FIN du sous-pas precedent) ne recoit NI
 * gravite NI terme gyroscopique -- ses vitesses sont forcees a zero et le
 * noyau sort tot. Il peut neanmoins etre reveille PLUS LOIN dans ce meme
 * sous-pas par une impulsion de contact (k_contact_solve) ou par le couplage
 * fluide implicite (k_body_solve) : les deux ecrivent directement
 * bodies[b].v/w si le corps repond a une force, et k_body_sleep_update juge
 * sur l'etat resultant, pas sur celui-ci. Sans ce gel, un tas de corps au
 * repos "respirerait" indefiniment sous l'effet de la gravite reappliquee
 * chaque sous-pas puis annulee par le contact -- le defaut le plus visible
 * au rendu (cf. plan-milestone-17.md D11). */
__global__ void k_body_predict(BqRigidBody* __restrict__ bodies,
                               const uint8_t* __restrict__ asleep, int n_bodies) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_bodies) return;
    if (!bodies[b].dynamic) return;
    if (asleep[b]) {
        bodies[b].v[0] = bodies[b].v[1] = bodies[b].v[2] = 0.f;
        bodies[b].w[0] = bodies[b].w[1] = bodies[b].w[2] = 0.f;
        return;
    }

    float3 v = make_float3(bodies[b].v[0], bodies[b].v[1], bodies[b].v[2]);
    if (bodies[b].use_gravity) v.y += c_p.dt * c_p.gravity_y;

    mat3 Ib_inv;
    for (int e = 0; e < 9; ++e) Ib_inv.m[e] = bodies[b].inv_inertia[e];
    mat3 R = quat_to_mat3(bodies[b].q);
    mat3 Iw_inv = matmul(matmul(R, Ib_inv), transpose(R)); /* inertie inverse, repere MONDE */
    mat3 Iw = inverse(Iw_inv); /* seulement pour le terme gyroscopique */

    float3 w = make_float3(bodies[b].w[0], bodies[b].w[1], bodies[b].w[2]);
    float3 gyro = vcross(w, matvec(Iw, w));

    float3 dw = matvec(Iw_inv, make_float3(-c_p.dt * gyro.x,
                                           -c_p.dt * gyro.y,
                                           -c_p.dt * gyro.z));
    w.x += dw.x; w.y += dw.y; w.z += dw.z;

    /* Verrous d'axe (repere MONDE) : composante mise a zero apres
     * integration -- une contrainte dure, pas une force de rappel. */
    if (bodies[b].lock_lin[0]) v.x = 0.f;
    if (bodies[b].lock_lin[1]) v.y = 0.f;
    if (bodies[b].lock_lin[2]) v.z = 0.f;
    if (bodies[b].lock_ang[0]) w.x = 0.f;
    if (bodies[b].lock_ang[1]) w.y = 0.f;
    if (bodies[b].lock_ang[2]) w.z = 0.f;

    bodies[b].v[0] = v.x; bodies[b].v[1] = v.y; bodies[b].v[2] = v.z;
    bodies[b].w[0] = w.x; bodies[b].w[1] = w.y; bodies[b].w[2] = w.z;
}

/* Recolte de grille pour le couplage implicite (M17/A5), un thread par
 * cellule. Remplace la recolte d'impulsion explicite de A1 (masse *
 * (v_pre - v) apres coup) par les cinq sommes necessaires a poser le choc
 * parfaitement inelastique corps/fluide-en-contact (cf. derivation complete
 * dans le commentaire de k_body_solve) :
 *
 *   S_m  = somme des masses de noeud                    (scalaire)
 *   S_p  = somme m_n * v_n                               (quantite de
 *          mouvement du fluide en contact, AVANT le choc)
 *   S_L  = somme m_n * (r x v_n)                          (moment cinetique
 *          du fluide en contact autour du centre de masse du corps, AVANT)
 *   S_mr = somme m_n * r                                  (premier moment,
 *          couple les deux blocs du systeme 6x6 -- nul si le fluide en
 *          contact est distribue symetriquement autour du corps)
 *   S_rr = somme m_n * (dot(r,r)*Id - r r^T)               ("tenseur
 *          d'inertie" du fluide en contact autour du meme point, meme
 *          formule que l'inertie d'un nuage de points ponctuels)
 *
 * avec r = x_noeud - x_corps, v_n LUE APRES GRAVITE (k_grid_apply_gravity
 * s'est deja execute ce sous-pas, cf. bq_step) -- un corps qui porte une
 * colonne d'eau doit en sentir le poids, exactement comme en A1.
 *
 * Ensemble de cellules : la bande de contact ETROITE (solid_here == phi<0,
 * OU cnrm.w < h -- EXACTEMENT le meme garde-fou que le bloc de contact de
 * k_grid_update), phi <= -h (interieur profond) exclu en plus. PAS "toute
 * cellule non profondement interieure dans l'AABB du corps" : cbody est
 * attribue par k_sdf_unsigned sur toute l'AABB des triangles DILATEE de
 * 3*dx (cf. bq_set_colliders) pour les besoins de la recherche du SDF, une
 * region bien plus large que la bande de contact reelle (h =
 * BQ_CONTACT_BAND_MULT*dx = 0.5*dx, largement plus etroite que le pad de
 * 3*dx). Sans cette restriction, m_contact mesure ~10 pour un cube de
 * 0.12 m (la region dilatee entiere) au lieu d'une fraction de cette
 * valeur, et le corps s'enfonce bien au-dela de l'equilibre d'Archimede :
 * le signal de pression a la surface du corps est noye dans la moyenne de
 * mouvement du fluide ambiant sur toute la region dilatee.
 *
 * SUPPLEMENT INDISPENSABLE, decouvert par la mesure (le test de conservation
 * D13/A1, 10^-6 en A1, degradait a plus de 50% sans ce filtre) : au sein
 * meme de la bande de contact, il faut EXCLURE les noeuds qui ne seront
 * PAS effectivement bloques par k_grid_update -- c'est-a-dire reproduire
 * ici le test UNILATERAL "vn < 0 ou bidir" de k_grid_update, avec la
 * vitesse de corps PREDITE (avant solve, seule disponible a ce stade) en
 * lieu et place de la vitesse resolue. Pourquoi cette exclusion est
 * necessaire et pas seulement souhaitable : le systeme 6x6 de k_body_solve
 * est un TAUTOLOGIE de conservation -- "corps + integralite de la bande
 * recoltee, fusionnes" conserve exactement sa propre quantite de mouvement.
 * Mais si un noeud de la bande n'est ensuite PAS touche par k_grid_update
 * (parce qu'il s'eloignait du corps, vn >= 0, mode unilateral), sa vitesse
 * REELLE reste celle d'AVANT le choc -- alors que le corps, lui, a deja
 * absorbe la part de quantite de mouvement que ce noeud etait cense
 * apporter au choc fusionne. Le systeme perd alors sa propriete
 * conservative : le corps gagne une quantite de mouvement qu'aucun noeud
 * de fluide n'a reellement cedee. Mesure sur le jet de la verification de
 * conservation (fluide tres agite pres du corps, beaucoup de noeuds
 * "rebondissent" hors de la bande a chaque sous-pas) : l'ecart entre noeuds
 * bloques et noeuds simplement proches est loin d'etre negligeable --
 * d'ou la derive a 52%. Avec ce filtre, la recolte redevient une
 * SURESTIMATION LEGERE (et non plus une fuite) de la masse effectivement
 * entrainee : la vitesse de corps utilisee ici est PREDITE, pas RESOLUE, et
 * la friction de k_grid_update peut encore laisser glisser un noeud declare
 * "bloque" ici -- mais l'ecart entre les deux etats du corps sur un seul
 * sous-pas est petit, contrairement a l'ecart entre "s'eloigne" et
 * "s'approche" du corps, qui est la vraie source de la fuite corrigee
 * ci-dessus. */
__global__ void k_grid_gather(const float4* __restrict__ grid,
                              const float* __restrict__ sdf,
                              const float4* __restrict__ cnrm,
                              const int* __restrict__ cbody,
                              const BqRigidBody* __restrict__ bodies,
                              float* __restrict__ gather, int ncell) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    float4 g = grid[id];
    if (g.w <= 0.f) return;

    int body_id = cbody[id];
    if (body_id < 0 || !bodies[body_id].dynamic) return;

    float h = BQ_CONTACT_BAND_MULT * c_p.dx;
    float phi = sdf[id];
    if (phi <= -h) return; /* interieur profond : jamais un contact reel */
    float4 cn = cnrm[id];
    bool solid_here = (phi < 0.f);
    if (!(solid_here || cn.w < h)) return; /* hors bande de contact -- meme garde-fou que k_grid_update */

    int3 res = c_p.res;
    int i = id / (res.y * res.z);
    int j = (id / res.z) % res.y;
    int k = id % res.z;
    float3 xnode = make_float3(i * c_p.dx, j * c_p.dx, k * c_p.dx);

    const BqRigidBody& bd = bodies[body_id];
    float3 bx = make_float3(bd.x[0], bd.x[1], bd.x[2]);
    float3 r = vsub(xnode, bx);

    float3 v_n = make_float3(g.x, g.y, g.z); /* grille deja en vitesse (post-gravite) */
    float  m_n = g.w;

    /* Filtre unilateral (cf. commentaire du noyau) : normale nulle -> pas de
     * contact geometrique fiable, rien a recolter. Sinon, meme test que
     * k_grid_update (vn < 0 OU normale non corroboree = bidirectionnel),
     * mais avec la vitesse de corps PREDITE (bd.v/bd.w, pas encore resolue
     * par k_body_solve -- ce noyau s'execute avant). */
    float3 n = make_float3(cn.x, cn.y, cn.z);
    if (n.x == 0.f && n.y == 0.f && n.z == 0.f) return;
    bool bidir = !normal_corroborated(sdf, res, i, j, k, n);
    float3 bv = make_float3(bd.v[0], bd.v[1], bd.v[2]);
    float3 bw = make_float3(bd.w[0], bd.w[1], bd.w[2]);
    float3 wxr = vcross(bw, r);
    float3 vc = make_float3(bv.x + wxr.x, bv.y + wxr.y, bv.z + wxr.z);
    float vn = vdot(vsub(v_n, vc), n);
    if (!(vn < 0.f || bidir)) return; /* le noeud s'eloigne : jamais bloque, jamais recolte */

    float* acc = gather + 16 * body_id;
    atomicAdd(&acc[0], m_n);

    float3 mv = make_float3(m_n * v_n.x, m_n * v_n.y, m_n * v_n.z);
    atomicAdd(&acc[1], mv.x); atomicAdd(&acc[2], mv.y); atomicAdd(&acc[3], mv.z);

    float3 L = vcross(r, v_n);
    atomicAdd(&acc[4], m_n * L.x); atomicAdd(&acc[5], m_n * L.y); atomicAdd(&acc[6], m_n * L.z);

    atomicAdd(&acc[7], m_n * r.x); atomicAdd(&acc[8], m_n * r.y); atomicAdd(&acc[9], m_n * r.z);

    /* S_rr, symetrique : xx, yy, zz, xy, xz, yz (6 composantes independantes
     * de dot(r,r)*Id - r r^T, mise a l'echelle par m_n). */
    float rr = vdot(r, r);
    atomicAdd(&acc[10], m_n * (rr - r.x * r.x));
    atomicAdd(&acc[11], m_n * (rr - r.y * r.y));
    atomicAdd(&acc[12], m_n * (rr - r.z * r.z));
    atomicAdd(&acc[13], m_n * (-r.x * r.y));
    atomicAdd(&acc[14], m_n * (-r.x * r.z));
    atomicAdd(&acc[15], m_n * (-r.y * r.z));
}

/* Resolution du couplage implicite fluide-solide (M17/A5), un thread par
 * corps -- coeur de la tache A5. Remplace la masse ajoutee (D5/A1), une
 * approximation structurellement fausse (elle divisait l'impulsion recue
 * sans jamais la restituer, cf. plan-milestone-17.md D5) par un CHOC
 * PARFAITEMENT INELASTIQUE entre le corps et la masse de fluide qui le
 * touche : au lieu d'appliquer une impulsion puis d'esperer que ca
 * converge, on resout directement la vitesse commune (v_new, w_new) que le
 * corps ET le fluide en contact adopteraient s'ils ne faisaient plus qu'un
 * solide rigide le temps de ce sous-pas.
 *
 * DERIVATION (a refaire a la main avant de faire confiance aux signes
 * ci-dessous -- c'est l'exigence de la spec de cette tache, et la raison
 * d'etre de ce commentaire).
 *
 * Choc inelastique : le corps (masse m_b, inertie I_b, vitesses v_b/w_b
 * APRES gravite et terme gyroscopique -- cf. k_body_predict) et le fluide en
 * contact (masse totale S_m repartie sur des noeuds de position r_i =
 * x_i - x_b et de vitesse v_i, cf. k_grid_gather) fusionnent en un seul
 * solide rigide instantane de vitesses (v_new, w_new) autour du centre de
 * masse du CORPS (pas du systeme fusionne -- x_b ne bouge pas pendant un
 * sous-pas, on peut donc exprimer toutes les quantites autour de ce point
 * fixe sans avoir a re-deriver un centre de masse combine).
 *
 * Quantite de mouvement lineaire du solide fusionne, vitesse d'un point
 * rigide etant v_new + w_new x r_i au noeud i :
 *   P = m_b*v_new + somme m_i*(v_new + w_new x r_i)
 *     = (m_b + S_m)*v_new + w_new x (somme m_i*r_i)
 *     = (m_b + S_m)*v_new + w_new x S_mr
 *     = (m_b + S_m)*v_new - S_mr x w_new                    (a x b = -b x a)
 *     = (m_b + S_m)*v_new - [S_mr]x * w_new
 * Egalee a la quantite de mouvement AVANT le choc : m_b*v_b + S_p.
 *   => ligne 1 du systeme : (m_b+S_m)*v_new - [S_mr]x*w_new = m_b*v_b + S_p
 *
 * Moment cinetique autour de x_b, meme construction : la contribution du
 * corps lui-meme est I_b*w_new (son centre de masse EST x_b, pas de terme
 * m_b*r x v puisque r=0 pour le corps) ; celle du noeud i est
 * r_i x [m_i*(v_new + w_new x r_i)] = m_i*(r_i x v_new) + m_i*(r_i x (w_new x r_i)).
 * Le premier terme sur tous les noeuds : somme m_i*(r_i x v_new) = S_mr x v_new
 * = -v_new x S_mr = [S_mr]x * v_new... attention au sens : r x v_new =
 * -(v_new x r) = -[v_new]x*r, mais on veut factoriser par v_new PAS par r ;
 * l'identite utile est a x b = -[b]x*a, donc r_i x v_new = -[v_new]x*r_i =
 * [r_i]x*v_new (car [a]x*b = a x b = -(b x a) = -[b]x*a) : somme m_i*[r_i]x*v_new
 * = [S_mr]x * v_new. Le second terme est le tenseur d'inertie standard d'un
 * nuage de points autour de l'origine : somme m_i*(r_i x (w x r_i)) =
 * (somme m_i*(dot(r_i,r_i)*Id - r_i r_i^T)) * w_new = S_rr * w_new
 * (identite vectorielle a x (b x a) = dot(a,a)*b - dot(a,b)*a, appliquee a
 * chaque noeud puis sommee -- c'est exactement la definition du tenseur
 * d'inertie d'une masse ponctuelle autour d'un axe passant par l'origine).
 * Moment cinetique total fusionne :
 *   L = I_b*w_new + [S_mr]x*v_new + S_rr*w_new
 *     = [S_mr]x*v_new + (I_b + S_rr)*w_new
 * Egale au moment cinetique AVANT le choc : I_b*w_b + S_L (S_L = somme
 * m_i*(r_i x v_i), calcule directement dans k_grid_gather).
 *   => ligne 2 du systeme : [S_mr]x*v_new + (I_b+S_rr)*w_new = I_b*w_b + S_L
 *
 * D'ou le systeme 6x6 assemble ci-dessous -- il correspond EXACTEMENT a
 * celui de la spec de la tache, blocs hors-diagonale antisymetriques et
 * transposes l'un de l'autre ([S_mr]x^T = -[S_mr]x), donc symetrique defini
 * positif tant que m_b > 0 (toujours vrai pour un corps dynamique) et I_b
 * non singuliere (garanti par construction cote Python, cf. rigidbody.py).
 *
 * POURQUOI c'est inconditionnellement stable : v_new est une MOYENNE
 * PONDEREE des vitesses avant choc (ponderation par les masses), jamais une
 * extrapolation au-dela. Un corps tres leger (m_b << S_m) voit v_new tendre
 * vers la vitesse du fluide, jamais au-dela -- contrairement a la masse
 * ajoutee (A1), qui appliquait l'integralite de l'impulsion recoltee a une
 * masse effective plus grande SANS jamais restituer le reste au fluide (un
 * puits de quantite de mouvement, cf. plan D5). Ici, rien n'est perdu ni
 * cree : ce qui est retire au fluide par la condition de contact de
 * k_grid_update (qui vient APRES, avec la vitesse resolue comme vitesse de
 * mur) est exactement ce que ce choc lui avait deja impute -- conservatif
 * par construction, pas par chance.
 *
 * Wrench de diagnostic (bq_read_collider_wrench) : impulsion EFFECTIVE
 * m_b*(v_new-v_b) et moment effectif I_b*(w_new-w_b) -- ce que le corps a
 * REELLEMENT recu ce sous-pas, pas la somme brute recoltee (qui n'est plus
 * calculee telle quelle : S_p/S_L sont une quantite de mouvement, pas une
 * impulsion). S_m (masse de fluide couplee) reporte tel quel en 7e valeur,
 * comme en A1. */
__global__ void k_body_solve(BqRigidBody* __restrict__ bodies,
                             const float* __restrict__ gather,
                             float* __restrict__ wrench, int n_bodies) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_bodies) return;
    if (!bodies[b].dynamic) return;

    const float* acc = gather + 16 * b;
    float  S_m  = acc[0];
    float3 S_p  = make_float3(acc[1], acc[2], acc[3]);
    float3 S_L  = make_float3(acc[4], acc[5], acc[6]);
    float3 S_mr = make_float3(acc[7], acc[8], acc[9]);
    mat3 S_rr;
    S_rr.m[0] = acc[10]; S_rr.m[4] = acc[11]; S_rr.m[8] = acc[12];
    S_rr.m[1] = S_rr.m[3] = acc[13];
    S_rr.m[2] = S_rr.m[6] = acc[14];
    S_rr.m[5] = S_rr.m[7] = acc[15];

    float mass = bodies[b].mass;
    float3 v_b = make_float3(bodies[b].v[0], bodies[b].v[1], bodies[b].v[2]);
    float3 w_b = make_float3(bodies[b].w[0], bodies[b].w[1], bodies[b].w[2]);

    mat3 Ib_inv;
    for (int e = 0; e < 9; ++e) Ib_inv.m[e] = bodies[b].inv_inertia[e];
    mat3 R = quat_to_mat3(bodies[b].q);
    mat3 Iw_inv = matmul(matmul(R, Ib_inv), transpose(R));
    mat3 Iw = inverse(Iw_inv); /* I_b en repere MONDE, cf. derivation ci-dessus */

    mat3 Smr_x = skew(S_mr);

    /* assemblage du systeme 6x6, cf. derivation dans le commentaire du noyau */
    float A[6][6];
    float rhs[6], sol[6];
    float mtot = mass + S_m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            A[r][c]         = (r == c) ? mtot : 0.f;   /* (m_b+S_m)*Id */
            A[r][3 + c]     = -Smr_x.m[3 * r + c];      /* -[S_mr]x */
            A[3 + r][c]     =  Smr_x.m[3 * r + c];      /* +[S_mr]x */
            A[3 + r][3 + c] =  Iw.m[3 * r + c] + S_rr.m[3 * r + c]; /* I_b+S_rr */
        }

    float3 rhs_lin = make_float3(mass * v_b.x + S_p.x, mass * v_b.y + S_p.y, mass * v_b.z + S_p.z);
    float3 Ib_wb = matvec(Iw, w_b);
    float3 rhs_ang = make_float3(Ib_wb.x + S_L.x, Ib_wb.y + S_L.y, Ib_wb.z + S_L.z);
    rhs[0] = rhs_lin.x; rhs[1] = rhs_lin.y; rhs[2] = rhs_lin.z;
    rhs[3] = rhs_ang.x; rhs[4] = rhs_ang.y; rhs[5] = rhs_ang.z;

    solve6x6(A, rhs, sol);
    float3 v_new = make_float3(sol[0], sol[1], sol[2]);
    float3 w_new = make_float3(sol[3], sol[4], sol[5]);

    /* Verrous d'axe (repere MONDE) A NOUVEAU apres le solve (D13, plan M17) :
     * le systeme ci-dessus ne "connait" pas les verrous, il peut donc
     * reintroduire une composante que k_body_predict avait annulee (le
     * fluide pousse sur l'axe bloque -> couplage vers les autres composantes
     * via les blocs hors-diagonale). Contrainte dure, pas une force de
     * rappel : mise a zero apres coup, comme en A1. */
    if (bodies[b].lock_lin[0]) v_new.x = 0.f;
    if (bodies[b].lock_lin[1]) v_new.y = 0.f;
    if (bodies[b].lock_lin[2]) v_new.z = 0.f;
    if (bodies[b].lock_ang[0]) w_new.x = 0.f;
    if (bodies[b].lock_ang[1]) w_new.y = 0.f;
    if (bodies[b].lock_ang[2]) w_new.z = 0.f;

    /* wrench de diagnostic : impulsion/moment EFFECTIFS (cf. commentaire du
     * noyau) -- calcules AVANT d'ecraser bodies[b].v/w avec l'etat resolu. */
    float* wr = wrench + 7 * b;
    wr[0] = mass * (v_new.x - v_b.x);
    wr[1] = mass * (v_new.y - v_b.y);
    wr[2] = mass * (v_new.z - v_b.z);
    float3 dw_eff = make_float3(w_new.x - w_b.x, w_new.y - w_b.y, w_new.z - w_b.z);
    float3 tau_eff = matvec(Iw, dw_eff);
    wr[3] = tau_eff.x; wr[4] = tau_eff.y; wr[5] = tau_eff.z;
    wr[6] = S_m;

    bodies[b].v[0] = v_new.x; bodies[b].v[1] = v_new.y; bodies[b].v[2] = v_new.z;
    bodies[b].w[0] = w_new.x; bodies[b].w[1] = w_new.y; bodies[b].w[2] = w_new.z;
}

/* ------------------------------------------- contact corps<->corps (M17, phase B, B3a)
 *
 * DETECTION SEULEMENT (lot B3a, plan-milestone-17.md D10) : cette section
 * genere des contacts et les expose en diagnostic, mais ne modifie AUCUN
 * etat de corps -- la resolution (impulsions sequentielles, D11) est le lot
 * suivant. Tourne a la cadence du sous-pas (D13), entre k_body_solve et
 * k_advance_bodies, SANS synchronisation hote (memes contraintes de
 * performance que le reste de la boucle de bq_step).
 *
 * Un contact par (corps A source de l'echantillon, corps B proprietaire du
 * SDF interroge) : cf. bq_read_contacts dans bourrasque.h pour le format
 * expose a l'appelant. */
struct BqContact {
    int    bodyA;   /* corps source de l'echantillon de surface */
    int    bodyB;   /* corps dont le SDF local a ete interroge */
    float3 point;   /* position monde de l'echantillon */
    float3 normal;  /* normale de contact monde, unitaire, dirigee de B vers A */
    float  depth;   /* profondeur de penetration (-phi_B), > 0 */
    int    sampleIdx; /* indice de l'echantillon de surface DANS bodyA (B3b) --
                         cle d'appariement du warm starting de k_contact_solve,
                         stable puisque les points d'echantillonnage sont fixes
                         en repere de corps (cf. commentaire de k_gen_contacts). */
};

/* Cache de warm starting (M17/B3b, D11) : lambda accumules du sous-pas
 * PRECEDENT, apparies par (corps A, corps B, indice d'echantillon de surface
 * sur A) -- stable d'un sous-pas a l'autre puisque les points
 * d'echantillonnage sont fixes en repere de corps. Cle complete (pas
 * seulement bodyA+sampleIdx) : un meme echantillon de A peut en principe
 * toucher un corps B different d'un sous-pas a l'autre (rare mais possible
 * si plusieurs corps se recouvrent), la cle complete evite un faux
 * appariement qui reinjecterait un lambda venu d'un AUTRE contact. */
struct BqContactCacheEntry {
    int   bodyA, bodyB, sampleIdx;
    float lambda_n, lambda_t1, lambda_t2;
};

/* AABB monde de chaque corps (broadphase, D10), obtenue de l'AABB LOCALE du
 * SDF du corps (BqBodySdf::origin/res/cell -- deja dilatee d'au moins 4
 * voxels a la construction, cf. bq_build_body_sdf) transformee par la pose
 * courante (x, q). Transformation conservative standard (centre + demi-
 * etendue tournee par |R|, composante par composante) : peut etre plus
 * large que l'AABB exacte apres rotation, jamais plus etroite -- suffisant
 * pour une broadphase, qui n'a besoin d'aucun test serre. Un corps sans SDF
 * local construit (phi == nullptr) recoit une AABB vide (lo > hi), qui ne
 * recouvre jamais rien : aucun cas particulier requis en aval. */
__global__ void k_body_world_aabb(const BqRigidBody* __restrict__ bodies,
                                  const BqBodySdf* __restrict__ sdf_table,
                                  float3* __restrict__ lo_out,
                                  float3* __restrict__ hi_out,
                                  int n_bodies) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_bodies) return;
    const BqBodySdf& s = sdf_table[b];
    if (s.phi == nullptr) {
        lo_out[b] = make_float3(3.4e38f, 3.4e38f, 3.4e38f);
        hi_out[b] = make_float3(-3.4e38f, -3.4e38f, -3.4e38f);
        return;
    }
    float3 lo_local = s.origin;
    float3 hi_local = make_float3(s.origin.x + (float)(s.res.x - 1) * s.cell,
                                  s.origin.y + (float)(s.res.y - 1) * s.cell,
                                  s.origin.z + (float)(s.res.z - 1) * s.cell);
    float3 c_local = make_float3(0.5f * (lo_local.x + hi_local.x),
                                 0.5f * (lo_local.y + hi_local.y),
                                 0.5f * (lo_local.z + hi_local.z));
    float3 h_local = make_float3(0.5f * (hi_local.x - lo_local.x),
                                 0.5f * (hi_local.y - lo_local.y),
                                 0.5f * (hi_local.z - lo_local.z));

    mat3 R = quat_to_mat3(bodies[b].q);
    mat3 Rabs;
    for (int e = 0; e < 9; ++e) Rabs.m[e] = fabsf(R.m[e]);
    float3 bx = make_float3(bodies[b].x[0], bodies[b].x[1], bodies[b].x[2]);
    float3 c_world = vadd(bx, matvec(R, c_local));
    float3 h_world = matvec(Rabs, h_local);

    lo_out[b] = vsub(c_world, h_world);
    hi_out[b] = vadd(c_world, h_world);
}

/* Test de recouvrement par paires, O(n^2) (D10) : n_bodies^2 threads (n de
 * l'ordre de la dizaine, plafonne a BQ_MAX_BODIES = 64 -- aucune structure
 * d'acceleration ne se justifie). Ecarte la diagonale (corps contre lui-
 * meme) et les paires statique/statique (aucun mouvement relatif possible,
 * cf. D10). pair_valid est indexe [BQ_MAX_BODIES*BQ_MAX_BODIES], symetrique
 * par construction (le test d'AABB et le garde-fou statique/statique le
 * sont tous les deux) -- la generation de contacts en profite pour iterer
 * les DEUX sens (A contre B, B contre A) via une seule table. */
__global__ void k_broadphase_pairs(const BqRigidBody* __restrict__ bodies,
                                   const float3* __restrict__ lo,
                                   const float3* __restrict__ hi,
                                   uint8_t* __restrict__ pair_valid,
                                   int n_bodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_bodies * n_bodies;
    if (idx >= total) return;
    int i = idx / n_bodies, j = idx % n_bodies;
    if (i == j) { pair_valid[i * BQ_MAX_BODIES + j] = 0; return; }
    if (!bodies[i].dynamic && !bodies[j].dynamic) {
        pair_valid[i * BQ_MAX_BODIES + j] = 0; return; /* D10 : paire ecartee */
    }
    float3 loi = lo[i], hii = hi[i], loj = lo[j], hij = hi[j];
    bool overlap = loi.x <= hij.x && hii.x >= loj.x &&
                  loi.y <= hij.y && hii.y >= loj.y &&
                  loi.z <= hij.z && hii.z >= loj.z;
    pair_valid[i * BQ_MAX_BODIES + j] = overlap ? 1 : 0;
}

/* Generation de contacts (D10) : pour chaque paire retenue par la
 * broadphase, teste chaque echantillon de surface de A contre le SDF local
 * de B. Grille de lancement fixe (n_bodies * n_bodies * max_samples,
 * max_samples = plus grand nombre d'echantillons sur un seul corps parmi
 * les n_bodies actifs, calcule cote hote sans synchronisation device -- les
 * comptes d'echantillons sont deja residents cote hote depuis
 * bq_set_body_samples) : chaque thread se voit attribuer (a, b, indice
 * d'echantillon dans a) par decodage de son indice global, et sort tot si
 * hors bornes ou paire non retenue. Le parcours de TOUS les couples ordonnes
 * (a, b), a != b, teste les DEUX SENS sans code duplique : quand a=A, b=B on
 * teste les echantillons de A contre le SDF de B ; quand a=B, b=A (autre
 * thread) l'inverse -- exactement l'exigence D10 (contact sommet/face
 * asymetrique).
 *
 * phi_B(p) < 0 => contact : point = p (monde), normale = normalize(R_B *
 * grad_B), profondeur = -phi_B. Stocke aussi (bodyA, indice d'echantillon)
 * -- pas utilise par CE lot, mais c'est la cle d'appariement du warm
 * starting du solveur d'impulsions du lot suivant (D11), stable puisque les
 * points d'echantillonnage sont fixes en repere de corps.
 *
 * Tampon plafonne (BQ_MAX_CONTACTS) : un slot est reserve par atomicAdd
 * AVANT d'ecrire, jamais l'inverse -- une ecriture a un index >= cap
 * ecraserait de la memoire hors tampon. Au-dela du plafond, le contact est
 * simplement perdu et *overflow est leve (cf. bq_contacts_last_overflow) :
 * saturation rapportee, jamais silencieuse (D10, meme discipline que
 * bq_whitewater_last_refused). */
__global__ void k_gen_contacts(const BqRigidBody* __restrict__ bodies,
                               const BqBodySdf* __restrict__ sdf_table,
                               const float3* const* __restrict__ samples_table,
                               const int* __restrict__ samples_n,
                               const uint8_t* __restrict__ pair_valid,
                               int n_bodies, int max_samples,
                               BqContact* __restrict__ contacts,
                               int* __restrict__ count, int cap,
                               int* __restrict__ overflow) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long per_body = (long long)n_bodies * (long long)max_samples;
    long long total = (long long)n_bodies * per_body;
    if (idx >= total) return;

    int a = (int)(idx / per_body);
    long long rem = idx % per_body;
    int b = (int)(rem / max_samples);
    int sidx = (int)(rem % max_samples);
    if (a == b) return;
    if (sidx >= samples_n[a]) return;
    if (!pair_valid[a * BQ_MAX_BODIES + b]) return;

    const BqBodySdf& sb = sdf_table[b];
    if (sb.phi == nullptr) return;
    const float3* samples_a = samples_table[a];
    if (samples_a == nullptr) return;

    /* echantillon (repere de corps A) -> monde -> repere de corps B (cf.
     * section D10 de la spec, formules exactes du milestone). */
    mat3 Ra = quat_to_mat3(bodies[a].q);
    float3 xa = make_float3(bodies[a].x[0], bodies[a].x[1], bodies[a].x[2]);
    float3 p_world = vadd(xa, matvec(Ra, samples_a[sidx]));

    mat3 Rb = quat_to_mat3(bodies[b].q);
    float3 xb = make_float3(bodies[b].x[0], bodies[b].x[1], bodies[b].x[2]);
    float3 p_localB = matvec(transpose(Rb), vsub(p_world, xb));

    float3 grad;
    float phi = body_sdf(sb, p_localB, &grad);
    if (phi >= 0.f) return;

    float3 n_world = matvec(Rb, grad);
    float  len2 = vdot(n_world, n_world);
    if (len2 > 1e-24f) {
        float inv_len = 1.f / sqrtf(len2);
        n_world.x *= inv_len; n_world.y *= inv_len; n_world.z *= inv_len;
    } else {
        n_world = make_float3(0.f, 0.f, 0.f); /* gradient degenere -- normale non fiable */
    }

    int slot = atomicAdd(count, 1);
    if (slot >= cap) {
        atomicExch(overflow, 1);
        return;
    }
    contacts[slot].bodyA = a;
    contacts[slot].bodyB = b;
    contacts[slot].point = p_world;
    contacts[slot].normal = n_world;
    contacts[slot].depth = -phi;
    contacts[slot].sampleIdx = sidx;
}

/* Proprietes inverses UNIFORMES (D12, plan M17) : un corps statique ou
 * cinematique (dynamic == 0) a une masse et une inertie infinies -- inv_mass
 * = 0, inv_inertia = matrice nulle, SANS lire bodies[b].mass/inv_inertia
 * (qui peuvent n'avoir jamais ete renseignes cote Python pour un collider
 * statique : rien ne les exige, cf. D7/D12). C'est ce traitement uniforme,
 * sans aucun cas particulier dans k_contact_solve, qui fait qu'une caisse se
 * pose sur un sol statique exactement comme sur un autre corps dynamique. */
__device__ inline void body_inv_props(const BqRigidBody& bd, float& inv_mass, mat3& inv_iw) {
    if (!bd.dynamic || bd.mass <= 0.f) {
        inv_mass = 0.f;
        inv_iw = mat3::zero();
        return;
    }
    inv_mass = 1.f / bd.mass;
    mat3 Ib_inv;
    for (int e = 0; e < 9; ++e) Ib_inv.m[e] = bd.inv_inertia[e];
    mat3 R = quat_to_mat3(bd.q);
    inv_iw = matmul(matmul(R, Ib_inv), transpose(R));
}

/* Masse effective (reciproque) d'une contrainte scalaire de contact le long
 * de la direction `dir` (normale ou tangente) -- formule standard 3D d'un
 * solveur d'impulsions sequentielles (Catto/Box2D/Bullet) :
 *   k = invMassA + invMassB
 *     + dot(dir, cross(invIwA * cross(rA, dir), rA))
 *     + dot(dir, cross(invIwB * cross(rB, dir), rB))
 * Le retour est k lui-meme (pas son inverse) -- l'appelant garde le controle
 * du garde-fou de division par ~0 (contrainte degeneree, ex. deux masses
 * nulles). */
__device__ inline float contact_k(float invMassA, const mat3& invIwA, float3 rA,
                                  float invMassB, const mat3& invIwB, float3 rB,
                                  float3 dir) {
    float3 termA = vcross(matvec(invIwA, vcross(rA, dir)), rA);
    float3 termB = vcross(matvec(invIwB, vcross(rB, dir)), rB);
    return invMassA + invMassB + vdot(dir, termA) + vdot(dir, termB);
}

/* Solveur de contact corps<->corps (M17, phase B, B3b, D11) : impulsions
 * SEQUENTIELLES, Gauss-Seidel projete, motif standard (Catto). Un seul
 * BLOC (D11 : "peu de corps et peu de contacts... garde le caractere
 * sequentiel du Gauss-Seidel" -- une version Jacobi parallele CHANGERAIT le
 * resultat, ce n'est pas juste une question de vitesse), contacts et corps
 * en memoire PARTAGEE pour la duree des BQ_CONTACT_ITERATIONS iterations
 * (cf. BQ_CONTACT_SOLVE_CAP pour la justification de la capacite).
 *
 * Trois phases synchronisees par __syncthreads (la boucle GS elle-meme,
 * phase 2, reste sequentielle -- un seul thread) :
 *   1. PARALLELE : chaque thread charge SON contact (bras de levier rA/rB,
 *      base tangente, masses effectives, cible de restitution, cible de
 *      poussee de separation) et cherche son lambda de warm start dans le
 *      cache du sous-pas PRECEDENT (recherche lineaire -- n petit, cf. D11).
 *   2. SEQUENTIEL (thread 0) : applique les impulsions de warm start, puis
 *      BQ_CONTACT_ITERATIONS passes de Gauss-Seidel sur tous les contacts.
 *   3. PARALLELE : ecrit les vitesses resolues dans `bodies`, le canal de
 *      poussee separee dans `pushout` (consomme par k_advance_bodies), et le
 *      nouveau cache de warm start pour le sous-pas suivant.
 *
 * SPLIT IMPULSE -- pourquoi un canal separe existe (D11) : sans lui, la
 * correction de penetration (necessaire, sinon les corps s'enfoncent
 * indefiniment sous l'effet cumule de la gravite et d'un contact legerement
 * elastique numeriquement) devrait passer par la MEME vitesse que la
 * physique reelle -- un biais de Baumgarte classique. Ce biais AJOUTE de
 * l'energie a chaque sous-pas ou il y a penetration residuelle (meme sous le
 * slop) et la retire au sous-pas suivant quand la penetration diminue : le
 * corps "respire" visiblement, l'empilement vibre au lieu de tenir (D11).
 * Le split impulse resout un SECOND systeme, formellement identique (meme
 * k_n, meme geometrie de contrainte) mais sur des vitesses "fantomes"
 * (svp/swp) qui ne sont JAMAIS copiees dans bodies[b].v/w -- seulement
 * utilisees par k_advance_bodies pour deplacer la POSITION ce sous-pas.
 * Elles disparaissent ensuite sans laisser de trace dans la quantite de
 * mouvement reelle du corps : la poussee de recouvrement ne "fuit" jamais
 * dans les vitesses qui persistent au sous-pas suivant.
 *
 * WARM STARTING -- pourquoi (D11) : sans lui, chaque sous-pas repart de
 * lambda=0 et doit reconverger de zero vers l'impulsion d'equilibre (le
 * poids du corps, contact apres contact) en BQ_CONTACT_ITERATIONS passes --
 * a un contact statique (caisse posee), c'est exactement l'impulsion qui
 * annule la gravite, RECALCULEE identiquement a chaque sous-pas. Repartir de
 * la solution du sous-pas precedent (les contacts et la geometrie ayant a
 * peine bouge) fait converger en 1-2 iterations au lieu de 8, et surtout
 * stabilise l'empilement : sans lui, la caisse du bas d'une pile de trois
 * n'a jamais le temps de "sentir" tout le poids au-dessus avant que le
 * sous-pas suivant ne remette les compteurs a zero. */
__global__ void k_contact_solve(BqRigidBody* __restrict__ bodies,
                                const BqContact* __restrict__ contacts,
                                const int* __restrict__ contact_count,
                                BqContactCacheEntry* __restrict__ prev_cache,
                                int* __restrict__ prev_cache_count,
                                float* __restrict__ pushout, /* BQ_MAX_BODIES*6 : vp[3], wp[3] */
                                int n_bodies) {
    __shared__ int    scA[BQ_CONTACT_SOLVE_CAP];
    __shared__ int    scB[BQ_CONTACT_SOLVE_CAP];
    __shared__ int    scSample[BQ_CONTACT_SOLVE_CAP];
    __shared__ float3 scRA[BQ_CONTACT_SOLVE_CAP];
    __shared__ float3 scRB[BQ_CONTACT_SOLVE_CAP];
    __shared__ float3 scN[BQ_CONTACT_SOLVE_CAP];
    __shared__ float3 scT1[BQ_CONTACT_SOLVE_CAP];
    __shared__ float3 scT2[BQ_CONTACT_SOLVE_CAP];
    __shared__ float  scInvMn[BQ_CONTACT_SOLVE_CAP];
    __shared__ float  scInvMt1[BQ_CONTACT_SOLVE_CAP];
    __shared__ float  scInvMt2[BQ_CONTACT_SOLVE_CAP];
    __shared__ float  scLambdaN[BQ_CONTACT_SOLVE_CAP];
    __shared__ float  scLambdaT1[BQ_CONTACT_SOLVE_CAP];
    __shared__ float  scLambdaT2[BQ_CONTACT_SOLVE_CAP];
    __shared__ float  scLambdaBias[BQ_CONTACT_SOLVE_CAP];
    __shared__ float  scBiasVel[BQ_CONTACT_SOLVE_CAP];
    __shared__ float  scPushout[BQ_CONTACT_SOLVE_CAP];

    __shared__ float3 sv[BQ_MAX_BODIES], sw[BQ_MAX_BODIES];
    __shared__ float3 svp[BQ_MAX_BODIES], swp[BQ_MAX_BODIES];
    __shared__ float3 sx[BQ_MAX_BODIES];
    __shared__ float  sInvM[BQ_MAX_BODIES];
    __shared__ mat3   sInvI[BQ_MAX_BODIES];

    int t = threadIdx.x;
    int n_contacts = *contact_count;
    if (n_contacts > BQ_CONTACT_SOLVE_CAP) n_contacts = BQ_CONTACT_SOLVE_CAP;
    int prev_n = *prev_cache_count;

    /* Phase 1a (parallele) : etat des corps en memoire partagee. */
    for (int b = t; b < n_bodies; b += blockDim.x) {
        sx[b]  = make_float3(bodies[b].x[0], bodies[b].x[1], bodies[b].x[2]);
        sv[b]  = make_float3(bodies[b].v[0], bodies[b].v[1], bodies[b].v[2]);
        sw[b]  = make_float3(bodies[b].w[0], bodies[b].w[1], bodies[b].w[2]);
        svp[b] = make_float3(0.f, 0.f, 0.f);
        swp[b] = make_float3(0.f, 0.f, 0.f);
        body_inv_props(bodies[b], sInvM[b], sInvI[b]);
    }
    __syncthreads();

    /* Phase 1b (parallele) : geometrie et masses effectives par contact,
     * recherche du warm start dans le cache du sous-pas precedent. */
    if (t < n_contacts) {
        BqContact c = contacts[t];
        int A = c.bodyA, B = c.bodyB;
        float3 rA = vsub(c.point, sx[A]);
        float3 rB = vsub(c.point, sx[B]);
        float3 n = c.normal;

        /* Base tangente orthonormee (n, t1, t2) -- axe de reference choisi
         * pour eviter un produit vectoriel degenere quand n est quasi
         * colineaire a un axe monde. */
        float3 up = (fabsf(n.x) > 0.9f) ? make_float3(0.f, 1.f, 0.f) : make_float3(1.f, 0.f, 0.f);
        float3 t1 = vcross(n, up);
        float l1 = sqrtf(vdot(t1, t1));
        t1 = (l1 > 1e-12f) ? make_float3(t1.x / l1, t1.y / l1, t1.z / l1) : make_float3(0.f, 0.f, 0.f);
        float3 t2 = vcross(n, t1);

        float kN  = contact_k(sInvM[A], sInvI[A], rA, sInvM[B], sInvI[B], rB, n);
        float kT1 = contact_k(sInvM[A], sInvI[A], rA, sInvM[B], sInvI[B], rB, t1);
        float kT2 = contact_k(sInvM[A], sInvI[A], rA, sInvM[B], sInvI[B], rB, t2);

        scA[t] = A; scB[t] = B; scSample[t] = c.sampleIdx;
        scRA[t] = rA; scRB[t] = rB; scN[t] = n; scT1[t] = t1; scT2[t] = t2;
        scInvMn[t]  = (kN  > 1e-9f) ? 1.f / kN  : 0.f;
        scInvMt1[t] = (kT1 > 1e-9f) ? 1.f / kT1 : 0.f;
        scInvMt2[t] = (kT2 > 1e-9f) ? 1.f / kT2 : 0.f;

        /* Vitesse de fermeture initiale (avant tout warm start) : cible de
         * restitution, cf. BQ_RESTITUTION_VEL_EPS. Convention normale =
         * "dirigee de B vers A" (cf. BqContact) : vn0 < 0 => A se rapproche
         * de B, la restitution doit repousser A. */
        float3 vpA0 = vadd(sv[A], vcross(sw[A], rA));
        float3 vpB0 = vadd(sv[B], vcross(sw[B], rB));
        float vn0 = vdot(vsub(vpA0, vpB0), n);
        float rest = fmaxf(bodies[A].restitution, bodies[B].restitution);
        scBiasVel[t] = (vn0 < -BQ_RESTITUTION_VEL_EPS) ? (-rest * vn0) : 0.f;

        /* Cible du canal SEPARE (split impulse) : vitesse de separation
         * necessaire pour resorber la penetration au-dela du slop en UN
         * sous-pas, cf. BQ_CONTACT_BETA. N'affecte jamais scBiasVel/sv/sw. */
        float slop = BQ_CONTACT_SLOP_FRAC * c_p.dx;
        float pen = fmaxf(c.depth - slop, 0.f);
        scPushout[t] = (c_p.dt > 1e-9f) ? (BQ_CONTACT_BETA / c_p.dt) * pen : 0.f;

        /* Warm start (D11) : recherche lineaire dans le cache du sous-pas
         * precedent -- n petit ("peu de contacts"), cout negligeable. */
        float wn = 0.f, wt1 = 0.f, wt2 = 0.f;
        for (int p = 0; p < prev_n; ++p) {
            if (prev_cache[p].bodyA == A && prev_cache[p].bodyB == B &&
                prev_cache[p].sampleIdx == c.sampleIdx) {
                wn = prev_cache[p].lambda_n; wt1 = prev_cache[p].lambda_t1; wt2 = prev_cache[p].lambda_t2;
                break;
            }
        }
        scLambdaN[t] = wn; scLambdaT1[t] = wt1; scLambdaT2[t] = wt2;
        scLambdaBias[t] = 0.f; /* jamais de warm start pour le canal separe : purement local a ce sous-pas */
    }
    __syncthreads();

    /* Phase 2 (SEQUENTIEL, thread 0 uniquement) : Gauss-Seidel projete. */
    if (t == 0) {
        /* Applique les impulsions de warm start AVANT la premiere iteration
         * -- c'est ce qui fait converger en 1-2 passes au lieu de 8 pour un
         * contact deja a l'equilibre (cf. commentaire du noyau). */
        for (int i = 0; i < n_contacts; ++i) {
            int A = scA[i], B = scB[i];
            float3 P = make_float3(scLambdaN[i] * scN[i].x + scLambdaT1[i] * scT1[i].x + scLambdaT2[i] * scT2[i].x,
                                   scLambdaN[i] * scN[i].y + scLambdaT1[i] * scT1[i].y + scLambdaT2[i] * scT2[i].y,
                                   scLambdaN[i] * scN[i].z + scLambdaT1[i] * scT1[i].z + scLambdaT2[i] * scT2[i].z);
            sv[A] = vadd(sv[A], make_float3(sInvM[A] * P.x, sInvM[A] * P.y, sInvM[A] * P.z));
            sw[A] = vadd(sw[A], matvec(sInvI[A], vcross(scRA[i], P)));
            sv[B] = vsub(sv[B], make_float3(sInvM[B] * P.x, sInvM[B] * P.y, sInvM[B] * P.z));
            sw[B] = vsub(sw[B], matvec(sInvI[B], vcross(scRB[i], P)));
        }

        for (int iter = 0; iter < BQ_CONTACT_ITERATIONS; ++iter) {
            for (int i = 0; i < n_contacts; ++i) {
                int A = scA[i], B = scB[i];
                float3 rA = scRA[i], rB = scRB[i], n = scN[i], t1 = scT1[i], t2 = scT2[i];

                /* ---- canal normal : non-penetration, lambda_n >= 0 (D11) ---- */
                float3 vpA = vadd(sv[A], vcross(sw[A], rA));
                float3 vpB = vadd(sv[B], vcross(sw[B], rB));
                float vn = vdot(vsub(vpA, vpB), n);
                float dLambda = -(vn - scBiasVel[i]) * scInvMn[i];
                float newLambda = fmaxf(scLambdaN[i] + dLambda, 0.f);
                dLambda = newLambda - scLambdaN[i];
                scLambdaN[i] = newLambda;
                float3 Pn = make_float3(dLambda * n.x, dLambda * n.y, dLambda * n.z);
                sv[A] = vadd(sv[A], make_float3(sInvM[A] * Pn.x, sInvM[A] * Pn.y, sInvM[A] * Pn.z));
                sw[A] = vadd(sw[A], matvec(sInvI[A], vcross(rA, Pn)));
                sv[B] = vsub(sv[B], make_float3(sInvM[B] * Pn.x, sInvM[B] * Pn.y, sInvM[B] * Pn.z));
                sw[B] = vsub(sw[B], matvec(sInvI[B], vcross(rB, Pn)));

                /* ---- friction de Coulomb, |lambda_t| <= mu*lambda_n (D11) ----
                 * Combinaison de paire : moyenne geometrique (D11 du plan),
                 * PAS arithmetique. Clamp "boite" (par axe tangent
                 * independamment), approximation courante du cone de
                 * friction exact -- suffisante ici, jamais source
                 * d'instabilite (contrairement a un clamp absent). */
                float mu = sqrtf(fmaxf(bodies[A].friction, 0.f) * fmaxf(bodies[B].friction, 0.f));
                float maxT = mu * scLambdaN[i];

                vpA = vadd(sv[A], vcross(sw[A], rA));
                vpB = vadd(sv[B], vcross(sw[B], rB));
                float vt1 = vdot(vsub(vpA, vpB), t1);
                float dT1 = -vt1 * scInvMt1[i];
                float newT1 = fminf(fmaxf(scLambdaT1[i] + dT1, -maxT), maxT);
                dT1 = newT1 - scLambdaT1[i]; scLambdaT1[i] = newT1;
                float3 Pt1 = make_float3(dT1 * t1.x, dT1 * t1.y, dT1 * t1.z);
                sv[A] = vadd(sv[A], make_float3(sInvM[A] * Pt1.x, sInvM[A] * Pt1.y, sInvM[A] * Pt1.z));
                sw[A] = vadd(sw[A], matvec(sInvI[A], vcross(rA, Pt1)));
                sv[B] = vsub(sv[B], make_float3(sInvM[B] * Pt1.x, sInvM[B] * Pt1.y, sInvM[B] * Pt1.z));
                sw[B] = vsub(sw[B], matvec(sInvI[B], vcross(rB, Pt1)));

                vpA = vadd(sv[A], vcross(sw[A], rA));
                vpB = vadd(sv[B], vcross(sw[B], rB));
                float vt2 = vdot(vsub(vpA, vpB), t2);
                float dT2 = -vt2 * scInvMt2[i];
                float newT2 = fminf(fmaxf(scLambdaT2[i] + dT2, -maxT), maxT);
                dT2 = newT2 - scLambdaT2[i]; scLambdaT2[i] = newT2;
                float3 Pt2 = make_float3(dT2 * t2.x, dT2 * t2.y, dT2 * t2.z);
                sv[A] = vadd(sv[A], make_float3(sInvM[A] * Pt2.x, sInvM[A] * Pt2.y, sInvM[A] * Pt2.z));
                sw[A] = vadd(sw[A], matvec(sInvI[A], vcross(rA, Pt2)));
                sv[B] = vsub(sv[B], make_float3(sInvM[B] * Pt2.x, sInvM[B] * Pt2.y, sInvM[B] * Pt2.z));
                sw[B] = vsub(sw[B], matvec(sInvI[B], vcross(rB, Pt2)));

                /* ---- canal SEPARE (split impulse) : pousse svp/swp hors de
                 * la penetration SANS jamais toucher sv/sw (D11 -- cf.
                 * commentaire du noyau pour la raison d'etre de ce canal). */
                float3 vpAp = vadd(svp[A], vcross(swp[A], rA));
                float3 vpBp = vadd(svp[B], vcross(swp[B], rB));
                float vnP = vdot(vsub(vpAp, vpBp), n);
                float dBias = -(vnP - scPushout[i]) * scInvMn[i];
                float newBias = fmaxf(scLambdaBias[i] + dBias, 0.f);
                dBias = newBias - scLambdaBias[i];
                scLambdaBias[i] = newBias;
                float3 Pb = make_float3(dBias * n.x, dBias * n.y, dBias * n.z);
                svp[A] = vadd(svp[A], make_float3(sInvM[A] * Pb.x, sInvM[A] * Pb.y, sInvM[A] * Pb.z));
                swp[A] = vadd(swp[A], matvec(sInvI[A], vcross(rA, Pb)));
                svp[B] = vsub(svp[B], make_float3(sInvM[B] * Pb.x, sInvM[B] * Pb.y, sInvM[B] * Pb.z));
                swp[B] = vsub(swp[B], matvec(sInvI[B], vcross(rB, Pb)));
            }
        }
    }
    __syncthreads();

    /* Phase 3 (parallele) : ecriture des resultats. */
    for (int b = t; b < n_bodies; b += blockDim.x) {
        if (bodies[b].dynamic) {
            bodies[b].v[0] = sv[b].x; bodies[b].v[1] = sv[b].y; bodies[b].v[2] = sv[b].z;
            bodies[b].w[0] = sw[b].x; bodies[b].w[1] = sw[b].y; bodies[b].w[2] = sw[b].z;
        }
        pushout[6 * b + 0] = svp[b].x; pushout[6 * b + 1] = svp[b].y; pushout[6 * b + 2] = svp[b].z;
        pushout[6 * b + 3] = swp[b].x; pushout[6 * b + 4] = swp[b].y; pushout[6 * b + 5] = swp[b].z;
    }
    if (t < n_contacts) {
        prev_cache[t].bodyA = scA[t]; prev_cache[t].bodyB = scB[t]; prev_cache[t].sampleIdx = scSample[t];
        prev_cache[t].lambda_n = scLambdaN[t]; prev_cache[t].lambda_t1 = scLambdaT1[t]; prev_cache[t].lambda_t2 = scLambdaT2[t];
    }
    if (t == 0) *prev_cache_count = n_contacts;
}

/* Mise en sommeil (M17/B3b, D11), un thread par corps -- APRES k_contact_solve
 * (qui a deja ecrit la vitesse RESOLUE, contact compris) et AVANT
 * k_advance_bodies (pour qu'un corps qui RESTE endormi voie sa vitesse mise
 * a zero avant que cette vitesse ne serve a integrer sa position -- c'est ce
 * qui elimine le fremissement residuel une fois endormi, verification 1 de
 * la porte de phase B). Un corps endormi est reveille par le seul critere
 * "vitesse resolue au-dessus du seuil de reveil", qu'elle vienne d'une
 * impulsion de contact (k_contact_solve) ou du couplage fluide implicite
 * (k_body_solve) -- les deux ecrivent bodies[b].v/w AVANT ce noyau, un seul
 * critere unifie suffit, pas deux chemins de reveil distincts a maintenir. */
__global__ void k_body_sleep_update(BqRigidBody* __restrict__ bodies,
                                    float* __restrict__ sleep_timer,
                                    uint8_t* __restrict__ asleep, int n_bodies) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_bodies) return;
    if (!bodies[b].dynamic) return;

    float3 v = make_float3(bodies[b].v[0], bodies[b].v[1], bodies[b].v[2]);
    float3 w = make_float3(bodies[b].w[0], bodies[b].w[1], bodies[b].w[2]);
    float speed = sqrtf(vdot(v, v));
    float angspeed = sqrtf(vdot(w, w));

    if (asleep[b]) {
        if (speed > BQ_WAKE_LIN_THRESH || angspeed > BQ_WAKE_ANG_THRESH) {
            asleep[b] = 0;
            sleep_timer[b] = 0.f;
        } else {
            /* Reste endormi : ecrase le residu numerique (friction/contact
             * peuvent laisser une vitesse non nulle mais sous le seuil de
             * reveil) plutot que de le laisser s'accumuler en frequence --
             * c'est precisement ce qui evite le fremissement. */
            bodies[b].v[0] = bodies[b].v[1] = bodies[b].v[2] = 0.f;
            bodies[b].w[0] = bodies[b].w[1] = bodies[b].w[2] = 0.f;
        }
        return;
    }

    if (speed < BQ_SLEEP_LIN_THRESH && angspeed < BQ_SLEEP_ANG_THRESH) {
        sleep_timer[b] += 1.f;
        if (sleep_timer[b] >= (float)BQ_SLEEP_SUBSTEPS) {
            asleep[b] = 1;
            bodies[b].v[0] = bodies[b].v[1] = bodies[b].v[2] = 0.f;
            bodies[b].w[0] = bodies[b].w[1] = bodies[b].w[2] = 0.f;
        }
    } else {
        sleep_timer[b] = 0.f;
    }
}

/* Avancee des corps rigides (D5, plan M17), un thread par corps -- separe
 * de k_body_solve (et non fusionne) parce que k_contact_solve (B3b) s'insere
 * maintenant entre les deux (cf. D13 du plan). Ne fait rien si le corps est
 * cinematique/statique.
 *
 * SPLIT IMPULSE (D11, cf. commentaire complet dans k_contact_solve) : la
 * position/orientation avance avec (v + vp, w + wp) -- vp/wp (canal
 * `pushout`) est la vitesse "fantome" qui resorbe la penetration residuelle,
 * jamais copiee dans bodies[b].v/w. Consommee une seule fois ici, elle ne
 * persiste pas au-dela de ce sous-pas. */
__global__ void k_advance_bodies(BqRigidBody* __restrict__ bodies,
                                 const float* __restrict__ pushout, int n_bodies) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_bodies) return;
    if (!bodies[b].dynamic) return;

    float dt = c_p.dt;
    float3 vp = make_float3(pushout[6 * b + 0], pushout[6 * b + 1], pushout[6 * b + 2]);
    float3 wp = make_float3(pushout[6 * b + 3], pushout[6 * b + 4], pushout[6 * b + 5]);
    bodies[b].x[0] += dt * (bodies[b].v[0] + vp.x);
    bodies[b].x[1] += dt * (bodies[b].v[1] + vp.y);
    bodies[b].x[2] += dt * (bodies[b].v[2] + vp.z);

    /* q += dt * 0.5 * quat(0, w+wp) (x) q -- produit de Hamilton, w purement
     * imaginaire A GAUCHE. Convention (w, x, y, z), IMPERATIVE : le module
     * Python cote extension utilise deja exactement celle-ci. */
    float qw = bodies[b].q[0], qx = bodies[b].q[1], qy = bodies[b].q[2], qz = bodies[b].q[3];
    float wx = bodies[b].w[0] + wp.x, wy = bodies[b].w[1] + wp.y, wz = bodies[b].w[2] + wp.z;
    float dqw = -wx * qx - wy * qy - wz * qz;
    float dqx =  wx * qw + wy * qz - wz * qy;
    float dqy = -wx * qz + wy * qw + wz * qx;
    float dqz =  wx * qy - wy * qx + wz * qw;
    qw += dt * 0.5f * dqw; qx += dt * 0.5f * dqx;
    qy += dt * 0.5f * dqy; qz += dt * 0.5f * dqz;
    float qn = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qn > 1e-12f) {
        float qinv = 1.f / qn;
        qw *= qinv; qx *= qinv; qy *= qinv; qz *= qinv;
    }
    bodies[b].q[0] = qw; bodies[b].q[1] = qx; bodies[b].q[2] = qy; bodies[b].q[3] = qz;
}

__global__ void k_g2p(float3* __restrict__ x,
                      float3* __restrict__ v,
                      float* __restrict__ Cbuf,
                      float* __restrict__ Jw,
                      const uint8_t* __restrict__ mat,
                      const float4* __restrict__ grid,
                      const float* __restrict__ sdf,
                      const float4* __restrict__ cnrm,
                      const float3* __restrict__ ccd_tri,
                      const int* __restrict__ ccd_bucket_off,
                      const int* __restrict__ ccd_bucket_tri,
                      float3 ccd_bucket_origin, float ccd_bucket_h,
                      int3 ccd_bucket_res, int n_tri, int n) {
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

    /* CCD (detection de collision continue), Moller & Trumbore 1997 --
     * filet de securite SUPPLEMENTAIRE avant le mecanisme D7-D9 ci-dessous
     * (contrainte de position / projection sur le champ de distance), qui ne
     * teste que les deux EXTREMITES du sous-pas (xp et xnew) : un
     * deplacement qui traverse une paroi fine sans que le champ signe change
     * de signe aux points echantillonnes, ou dont les normales ne se
     * contredisent pas franchement, passe encore au travers. La CCD teste le
     * SEGMENT xp -> xnew lui-meme contre les triangles du collider, et
     * corrige xnew AVANT que D7-D9 ne le consomme (ci/cj/ck/idx_new/phi sont
     * calcules plus bas, sur la position CCD-corrigee).
     *
     * Recherche des triangles candidats par la meme grille de buckets que le
     * SDF (build_bucket_grid, cf. plus haut) : union du voisinage 27-buckets
     * autour de xp ET autour de xnew (deux recherches, pas de deduplication
     * -- tester deux fois le meme triangle ne change pas le resultat, cout
     * redondant accepte). Suffisant par le meme argument CFL que D9 : le
     * deplacement d'un sous-pas est borne a environ dx, donc le segment reste
     * court par rapport a un bucket (BQ_BUCKET_DX_MULT*dx = 2*dx).
     *
     * Position UNIQUEMENT : comme le mecanisme D7-D9 (cf. note plus bas,
     * juste avant le re-clamp final -- "La vitesse n'est volontairement pas
     * touchee : c'est le role de k_grid_update, la retoucher ici injecterait
     * de l'energie et ferait vibrer le contact"), la CCD ne touche jamais nv
     * ni v[p]. La particule repositionnee est naturellement freinee au pas
     * SUIVANT par k_grid_update.
     *
     * Recul le long du SEGMENT (xp -> xnew), pas le long de la normale du
     * triangle touche : evite toute question d'orientation/sens de la
     * normale -- xp est par construction une position valide (elle vient du
     * substep precedent), donc reculer VERS xp le long du segment reste
     * toujours du bon cote, quelle que soit l'orientation du triangle.
     *
     * Cout en l'absence de collider (n_tri == 0) : chemin quasi gratuit,
     * meme discipline que le reste du fichier pour ce cas (cf. k_sdf_unsigned,
     * D7-D9 ci-dessous). */
    if (n_tri > 0) {
        float3 ccd_dir = make_float3(xnew.x - xp.x, xnew.y - xp.y, xnew.z - xp.z);
        float ccd_best_t = 2.f; /* sentinelle > 1 : aucune intersection valide */
        bool ccd_hit = false;
        for (int ccd_pass = 0; ccd_pass < 2; ++ccd_pass) {
            float3 ccd_q = (ccd_pass == 0) ? xp : xnew;
            int3 ccd_bc = bucket_index(ccd_q, ccd_bucket_origin, ccd_bucket_h);
            int ccd_lo_i = max(ccd_bc.x - 1, 0), ccd_hi_i = min(ccd_bc.x + 1, ccd_bucket_res.x - 1);
            int ccd_lo_j = max(ccd_bc.y - 1, 0), ccd_hi_j = min(ccd_bc.y + 1, ccd_bucket_res.y - 1);
            int ccd_lo_k = max(ccd_bc.z - 1, 0), ccd_hi_k = min(ccd_bc.z + 1, ccd_bucket_res.z - 1);
            for (int ii = ccd_lo_i; ii <= ccd_hi_i; ++ii)
                for (int jj = ccd_lo_j; jj <= ccd_hi_j; ++jj)
                    for (int kk = ccd_lo_k; kk <= ccd_hi_k; ++kk) {
                        int ccd_bidx = (ii * ccd_bucket_res.y + jj) * ccd_bucket_res.z + kk;
                        int ccd_off0 = ccd_bucket_off[ccd_bidx];
                        int ccd_off1 = ccd_bucket_off[ccd_bidx + 1];
                        for (int e = ccd_off0; e < ccd_off1; ++e) {
                            int t = ccd_bucket_tri[e];
                            float3 v0 = ccd_tri[3 * t + 0], v1 = ccd_tri[3 * t + 1],
                                   v2 = ccd_tri[3 * t + 2];
                            float tt;
                            if (ccd_segment_tri(xp, xnew, v0, v1, v2, &tt) && tt < ccd_best_t) {
                                ccd_best_t = tt;
                                ccd_hit = true;
                            }
                        }
                    }
        }
        if (ccd_hit) {
            float t_safe = ccd_best_t * 0.99f;
            xnew.x = xp.x + t_safe * ccd_dir.x;
            xnew.y = xp.y + t_safe * ccd_dir.y;
            xnew.z = xp.z + t_safe * ccd_dir.z;
        }
    }

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

/* --------------------------------------------------------- reseeding (M10)
 * Requilibre la population de particules par cellule une fois par frame,
 * apres le dernier sous-pas -- corrige la fragmentation en zones de forte
 * deformation (mesuree en M9, cf. plan-milestone-10.md D1-D5). Motif valide
 * en production par deux moteurs : Houdini FLIP (section "Reseeding",
 * `Particles Per Voxel` + seuils naissance/mort) et Mantaflow, le moteur
 * fluide natif de Blender, notre cible directe (seuils min/max par cellule).
 *
 * Cible = ppc_axis^3 (deja expose dans BqConfig, pas un nouveau parametre).
 * Seuils de naissance/mort internes, non exposes -- meme philosophie que les
 * autres constantes internes du fichier (l'utilisateur ne doit rien regler).
 *
 * Meme motif de compaction par flux deja utilise trois fois dans ce projet
 * (marching cubes M7, generation whitewater M8, compaction whitewater M8) :
 * comptage par thread, cub::DeviceScan::ExclusiveSum, emission aux offsets
 * calcules, tampon sentinelle mis a zero avant chaque scan.
 *
 * Frequence : une fois par appel a bq_step (une fois par frame, apres le
 * dernier sous-pas), jamais a chaque sous-pas -- cf. reseed() plus bas et
 * son appel en fin de bq_step. */
#define BQ_RESEED_BIRTH_MULT 0.5f
#define BQ_RESEED_DEATH_MULT 2.0f

/* Hash entier bon marche (variante Wang hash), duplique ici sous un nom
 * distinct : meme motif que mc_hash01 de mesher.cu, mais chaque unite de
 * compilation reste autonome (CUDA_SEPARABLE_COMPILATION OFF, aucun symbole
 * partage entre .cu). Utilise pour le jitter de position des naissances et
 * la selection aleatoire des morts. */
__device__ inline float reseed_hash01(unsigned int x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    return (float)(x & 0x00FFFFFFu) * (1.f / 16777216.f); /* [0,1) */
}

/* Indexe une position sur la cellule (noeud) la plus proche, EXACTEMENT la
 * meme convention que celle deja utilisee pour indexer sdf/cnrm dans k_g2p
 * (floorf(x*inv_dx + 0.5), clampee) -- la coherence entre le comptage de
 * reseeding et le contact du collider est requise par la spec (D2). */
__device__ inline int reseed_cell_index(float3 xp) {
    int3 res = c_p.res;
    int ci = min(max((int)floorf(xp.x * c_p.inv_dx + 0.5f), 0), res.x - 1);
    int cj = min(max((int)floorf(xp.y * c_p.inv_dx + 0.5f), 0), res.y - 1);
    int ck = min(max((int)floorf(xp.z * c_p.inv_dx + 0.5f), 0), res.z - 1);
    return (ci * res.y + cj) * res.z + ck;
}

/* Vitesse G2P (meme stencil B-spline quadratique 3x3x3, memes poids que
 * k_g2p) evaluee a un point de requete ARBITRAIRE plutot qu'a une particule
 * existante -- utilise pour la vitesse initiale d'une particule nee par
 * reseeding. Pas de nouvelle formule : copie du calcul de `nv` de k_g2p,
 * sans le calcul de C (non requis pour une naissance, cf. D3). */
__device__ inline float3 reseed_g2p_velocity(float3 xp, const float4* __restrict__ grid) {
    int3 base = make_int3((int)floorf(xp.x * c_p.inv_dx - 0.5f),
                          (int)floorf(xp.y * c_p.inv_dx - 0.5f),
                          (int)floorf(xp.z * c_p.inv_dx - 0.5f));
    float3 fx = make_float3(xp.x * c_p.inv_dx - base.x,
                            xp.y * c_p.inv_dx - base.y,
                            xp.z * c_p.inv_dx - base.z);
    float w[3][3];
    bspline_weights(fx, w);

    float3 nv = make_float3(0.f, 0.f, 0.f);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                int gx = base.x + i, gy = base.y + j, gz = base.z + k;
                if (gx < 0 || gx >= c_p.res.x || gy < 0 || gy >= c_p.res.y ||
                    gz < 0 || gz >= c_p.res.z)
                    continue;
                float weight = w[i][0] * w[j][1] * w[k][2];
                int idx = (gx * c_p.res.y + gy) * c_p.res.z + gz;
                float4 g = grid[idx];
                nv.x += weight * g.x;
                nv.y += weight * g.y;
                nv.z += weight * g.z;
            }
    return nv;
}

/* D1/D2 -- comptage par cellule : un thread par particule active, atomicAdd
 * sur le compte de la cellule (meme convention d'indexation que
 * reseed_cell_index), somme des J existants (pour la moyenne de naissance,
 * D3) et indice de la premiere particule trouvee dans la cellule (pour
 * heriter son materiau, D3) -- via atomicCAS sur la sentinelle -1, la
 * premiere ecriture gagnante (ordre d'execution des threads, pas ordre
 * d'indice). count/jsum doivent etre remis a zero et first a -1 avant
 * l'appel (cf. reseed()). */
__global__ void k_reseed_count(const float3* __restrict__ x,
                               const float* __restrict__ Jw,
                               int* __restrict__ count, float* __restrict__ jsum,
                               int* __restrict__ first, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    int idx = reseed_cell_index(x[p]);
    atomicAdd(&count[idx], 1);
    atomicAdd(&jsum[idx], Jw[p]);
    atomicCAS(&first[idx], -1, p);
}

/* D1/D3/D4 -- decision par cellule : combien de naissances (jusqu'a la
 * cible, seulement si la cellule contient deja au moins une particule), et
 * quelle probabilite de survie appliquer si la cellule est au-dessus du
 * seuil de mort (1 = aucune mort). birth[ncell] (sentinelle du scan) doit
 * etre mise a zero avant l'appel (cf. reseed()). */
__global__ void k_reseed_plan(const int* __restrict__ count, int* __restrict__ birth,
                              float* __restrict__ keepprob, int target, int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    int cnt = count[c];
    float ft = (float)target;

    /* Correctif regression mesuree (session reseeding) : autoriser la
     * naissance des qu'une cellule contenait >=1 particule et etait sous le
     * seuil densifiait artificiellement toute la couche de surface -- une
     * cellule d'interface fluide/air est LEGITIMEMENT sous la cible (une
     * partie de son volume est de l'air, pas une fragmentation numerique a
     * corriger). Mesure : 430592 -> 854843 particules en 48 frames (+98%) sur
     * une nappe au repos, maillage reconstruit passant de 1 a 30 composantes
     * connexes, ecart-type de hauteur de 2.6mm a 19.4mm. La naissance ne doit
     * s'appliquer qu'a l'interieur reel du fluide, jamais a sa frontiere :
     * une cellule n'est eligible que si ses 6 voisins directs (face, pas
     * diagonale) contiennent chacun au moins une particule -- un voisin hors
     * grille compte comme vide (le bord du domaine est aussi une surface). */
    int3 res = c_p.res;
    int i = c / (res.y * res.z);
    int j = (c / res.z) % res.y;
    int k = c % res.z;
    bool entouree = true;
    {
        int ni, nj, nk;
        ni = i - 1; entouree = entouree && (ni >= 0) && (count[(ni * res.y + j) * res.z + k] >= 1);
        ni = i + 1; entouree = entouree && (ni < res.x) && (count[(ni * res.y + j) * res.z + k] >= 1);
        nj = j - 1; entouree = entouree && (nj >= 0) && (count[(i * res.y + nj) * res.z + k] >= 1);
        nj = j + 1; entouree = entouree && (nj < res.y) && (count[(i * res.y + nj) * res.z + k] >= 1);
        nk = k - 1; entouree = entouree && (nk >= 0) && (count[(i * res.y + j) * res.z + nk] >= 1);
        nk = k + 1; entouree = entouree && (nk < res.z) && (count[(i * res.y + j) * res.z + nk] >= 1);
    }
    /* Correctif M12 (investigation de la derive volume/masse mesuree sur une
     * nappe au repos, 100s simulees : +9% puis -4% de population, jamais
     * stabilise) : la correction visait le CENTRE (target), pas le SEUIL de
     * declenchement -- une cellule a peine sous 0.5*target sautait d'un coup
     * a 100% de la cible (gros correctif pour un petit ecart), alors qu'une
     * cellule ne perdait des particules qu'au-dela de 2*target, et seulement
     * en esperance vers la cible (petit correctif pour un gros ecart). Cette
     * asymetrie amplifie le bruit de tassement en oscillation de population
     * (controle bang-bang), confirme empiriquement : le meme run SANS
     * reseeding (commutateur de diagnostic BQ_DISABLE_RESEED) ne derive que
     * de +0.8% et converge, la derive n'est donc pas une derive de J
     * independante. Correctif : chaque correction ne vise plus que SON PROPRE
     * seuil de declenchement (ceil(0.5*target) pour la naissance,
     * 2.0*target pour la mort), jamais le centre -- correction proportionnelle
     * a l'ecart plutot qu'un reset complet, meme esprit que les seuils
     * min/max de Mantaflow (pas une remise a la cible nominale). */
    int birth_threshold = (int)ceilf(BQ_RESEED_BIRTH_MULT * ft);
    bool can_birth = entouree && (cnt >= 1) && (cnt < birth_threshold);
    birth[c] = can_birth ? (birth_threshold - cnt) : 0;
    bool dying = (float)cnt > BQ_RESEED_DEATH_MULT * ft;
    keepprob[c] = (dying && cnt > 0) ? (BQ_RESEED_DEATH_MULT * ft / (float)cnt) : 1.f;
}

/* D4 -- mort : suppression ALEATOIRE, pas de fusion (cf. plan, ecarte
 * explicitement). Approche probabiliste plutot qu'une selection exacte des
 * `target` survivants (qui demanderait un classement par cellule, motif
 * bien plus lourd pour un correctif de population) : chaque particule d'une
 * cellule au-dessus du seuil de mort survit avec probabilite
 * target/count(cellule), via un hash deterministe par indice de particule
 * -- en esperance la cellule revient a la cible, sans garantie exacte au
 * tirage pres. Decision prise faute d'indication plus precise dans la spec
 * sur l'exactitude requise ; a signaler si une cible EXACTE s'avere
 * necessaire (cf. V3/V4 du plan). */
__global__ void k_reseed_mark_alive(const float3* __restrict__ x,
                                    const float* __restrict__ keepprob,
                                    int* __restrict__ alive, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    int idx = reseed_cell_index(x[p]);
    float kp = keepprob[idx];
    int a;
    if (kp >= 1.f) {
        a = 1;
    } else {
        float r = reseed_hash01((unsigned int)p * 2654435761u ^ 0x51ed270bu);
        a = (r < kp) ? 1 : 0;
    }
    alive[p] = a;
}

/* D5 -- compaction des survivantes (celles PAS marquees pour la mort) vers
 * le second jeu de tampons (ping-pong), a l'offset donne par le scan
 * exclusif de alive_flag. */
__global__ void k_reseed_compact_survivors(
    const float3* __restrict__ old_x, const float3* __restrict__ old_v,
    const float* __restrict__ old_C, const float* __restrict__ old_F,
    const float* __restrict__ old_J, const uint8_t* __restrict__ old_mat,
    const int* __restrict__ alive, const int* __restrict__ alive_scan,
    float3* __restrict__ new_x, float3* __restrict__ new_v,
    float* __restrict__ new_C, float* __restrict__ new_F,
    float* __restrict__ new_J, uint8_t* __restrict__ new_mat, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    if (!alive[p]) return;
    int idx = alive_scan[p];
    new_x[idx] = old_x[p];
    new_v[idx] = old_v[p];
    for (int c9 = 0; c9 < 9; ++c9) {
        new_C[9 * idx + c9] = old_C[9 * p + c9];
        new_F[9 * idx + c9] = old_F[9 * p + c9];
    }
    new_J[idx] = old_J[p];
    new_mat[idx] = old_mat[p];
}

/* D3 -- naissance : un thread par cellule, emet ses `birth[cell]` nouvelles
 * particules a la suite des survivantes (n_survivors + birth_scan[cell] +
 * rang local), plafonnees a `room` (capacite restante, cf. reseed()) --
 * naissances excedentaires omises silencieusement, meme politique que le
 * whitewater (D6 de M8). Position : jitter uniforme dans le volume de la
 * cellule (noeud i*dx +/- dx/2 par axe, meme convention que
 * reseed_cell_index). Vitesse : G2P au point jitte (reseed_g2p_velocity).
 * J : moyenne des J existants de la cellule, uniquement si le materiau herite
 * (celui de la premiere particule trouvee, cell_first) est WATER -- sinon 1.
 * C=0 : meme initialisation par defaut qu'une particule fraiche de
 * bq_emit_box (emit_particles). Masse : pas de champ par particule dans ce
 * solveur, deja derivee du materiau (MaterialGpu.p_mass) a chaque substep --
 * rien a initialiser ici.
 *
 * F (D5, M18) : herite du F de la particule-graine de la cellule (fi =
 * cell_first[cell], deja utilisee ci-dessous pour le materiau), PAS
 * l'identite -- sauf pour l'eau, qui n'utilise jamais F (seul Jw compte,
 * cf. k_p2g/k_g2p) et pour laquelle l'identite reste le choix le plus simple.
 * Sans ce soin, une naissance dans un tas de sable compacte relacherait
 * localement l'etat plastique/elastique accumule dans F -- meme motif que
 * javg pour J juste au-dessus, mais SANS moyenne : moyenner des tenseurs de
 * deformation n'a pas de sens tensoriel evident, on herite donc d'une seule
 * particule plutot que d'en fabriquer une hybride. */
__global__ void k_reseed_emit_births(
    const int* __restrict__ cell_count, const int* __restrict__ cell_first,
    const float* __restrict__ cell_jsum, const int* __restrict__ birth,
    const int* __restrict__ birth_scan, const uint8_t* __restrict__ old_mat,
    const float* __restrict__ old_F,
    const float4* __restrict__ grid, float3* __restrict__ new_x,
    float3* __restrict__ new_v, float* __restrict__ new_C,
    float* __restrict__ new_F, float* __restrict__ new_J,
    uint8_t* __restrict__ new_mat, int n_survivors, int room, int ncell) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;
    int nb = birth[cell];
    if (nb <= 0) return;

    int3 res = c_p.res;
    int i = cell / (res.y * res.z);
    int j = (cell / res.z) % res.y;
    int k = cell % res.z;
    float3 corner = make_float3((i - 0.5f) * c_p.dx, (j - 0.5f) * c_p.dx,
                                (k - 0.5f) * c_p.dx);

    int fi = cell_first[cell];
    if (fi < 0) return; /* garde-fou : ne devrait pas arriver (nb>0 => count>=1) */
    uint8_t mid = old_mat[fi];
    bool water = (c_p.mats[mid].model == BQ_MODEL_WATER);
    int cnt = cell_count[cell];
    float javg = (cnt > 0) ? (cell_jsum[cell] / (float)cnt) : 1.f;

    float lo = c_p.bound * c_p.dx;
    float hi_x = res.x * c_p.dx - lo, hi_y = res.y * c_p.dx - lo,
          hi_z = res.z * c_p.dx - lo;

    int base = birth_scan[cell];
    for (int r = 0; r < nb; ++r) {
        int g = base + r;
        if (g >= room) return; /* capacite atteinte : reste de cette cellule omis */
        unsigned int seed = (unsigned int)cell * 9781u + (unsigned int)r * 6271u + 12345u;
        float jx = reseed_hash01(seed);
        float jy = reseed_hash01(seed ^ 0x9e3779b9u);
        float jz = reseed_hash01(seed ^ 0x85ebca6bu);
        float3 xp = make_float3(fminf(fmaxf(corner.x + jx * c_p.dx, lo), hi_x),
                                fminf(fmaxf(corner.y + jy * c_p.dx, lo), hi_y),
                                fminf(fmaxf(corner.z + jz * c_p.dx, lo), hi_z));
        float3 v = reseed_g2p_velocity(xp, grid);

        int idx = n_survivors + g;
        new_x[idx] = xp;
        new_v[idx] = v;
        for (int c9 = 0; c9 < 9; ++c9) {
            new_C[9 * idx + c9] = 0.f;
            /* eau : identite (F non utilise) ; sinon herite de la graine
             * (fi), cf. commentaire de tete -- pas de moyenne. */
            new_F[9 * idx + c9] = water
                ? ((c9 == 0 || c9 == 4 || c9 == 8) ? 1.f : 0.f)
                : old_F[9 * fi + c9];
        }
        new_J[idx] = water ? javg : 1.f;
        new_mat[idx] = mid;
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
    /* indice de corps rigide par triangle (D2, plan M17), meme politique de
     * reallocation que d_tri ci-dessus (tri_cap partage) -- upload seulement
     * si l'appelant fournit un tableau non NULL a bq_set_colliders, sinon le
     * contenu est ignore (default_body passe directement au kernel, cf.
     * k_sdf_unsigned). */
    int*    d_tri_body = nullptr;
    int tri_cap = 0;
    int n_tri = 0;

    /* corps rigides (M17, phase A) : identite de corps par cellule, derivee
     * du triangle gagnant (cf. k_sdf_unsigned) -- taille ncell, -1 = aucun
     * corps. */
    int* d_cbody = nullptr;
    /* etat + parametres des corps rigides, capacite fixe BQ_MAX_BODIES (cf.
     * sa def) : x/q/v/w sont mutes en place par k_body_predict, k_body_solve
     * et k_advance_bodies a chaque sous-pas, dynamic/mass/inv_inertia/... sont
     * fournis une fois par bq_set_collider_bodies et jamais modifies par le
     * solveur. */
    BqRigidBody* d_bodies = nullptr;
    /* impulsion EFFECTIVE par corps, diagnostic (M17/A5) : 7 floats/corps
     * (impulsion lineaire [3], couple [3], masse de fluide couplee [1]).
     * Ecrit en une seule fois par k_body_solve (pas d'atomicAdd -- un seul
     * thread par corps), remis a zero a CHAQUE sous-pas (cf. bq_step) pour
     * qu'un corps cinematique/statique (jamais touche par k_body_solve, qui
     * sort tot pour dynamic=0) lise 0 plutot qu'une valeur perimee du
     * sous-pas precedent. */
    float* d_body_wrench = nullptr;
    /* accumulateur des cinq sommes de la recolte de grille (M17/A5, cf.
     * k_grid_gather) : 16 floats/corps -- S_m [1], S_p [3], S_L [3],
     * S_mr [3], S_rr [6, symetrique xx,yy,zz,xy,xz,yz]. Remis a zero a
     * CHAQUE sous-pas, jamais alloue au-dela de BQ_MAX_BODIES. */
    float* d_body_gather = nullptr;
    int n_bodies = 0;

    /* SDF locaux par corps (M17, phase B, D9) : construits une seule fois
     * par bq_build_body_sdf, jamais reconstruits ensuite -- contrairement au
     * champ collider fusionne ci-dessus. Miroir hote (origine, pas,
     * resolution, pointeur DEVICE vers le champ, un par corps) + copie
     * device de la MEME structure (d_body_sdf_table, BQ_MAX_BODIES entrees)
     * pour que body_sdf() (cf. plus haut) puisse l'indexer par corps depuis
     * un kernel. phi == nullptr (mise a zero par bq_create) = corps sans
     * SDF local construit -- body_sdf le traite comme "pas de collider",
     * jamais comme une erreur. */
    BqBodySdf body_sdf_host[BQ_MAX_BODIES] = {};
    BqBodySdf* d_body_sdf_table = nullptr;

    /* Points d'echantillonnage de surface par corps (D10), repere de CORPS
     * -- stockes ici, consommes par la detection de contact (M17, phase B,
     * B3a, k_gen_contacts). Reconstruits au complet a chaque
     * bq_set_body_samples (pas de politique de capacite partagee : n petit,
     * appele une fois par bake, jamais par frame). */
    float3* d_body_samples[BQ_MAX_BODIES] = {};
    int     body_samples_n[BQ_MAX_BODIES] = {};
    /* Miroirs DEVICE des deux tableaux ci-dessus (un kernel ne peut pas
     * indexer un tableau hote de pointeurs device) : d_body_samples_table[i]
     * == d_body_samples[i], d_body_samples_n_table[i] == body_samples_n[i],
     * tenus a jour a chaque bq_set_body_samples par une copie d'un seul
     * element (pas de retelevesement complet). Mis a zero a la creation
     * (tous nullptr / 0, "pas d'echantillons"). */
    float3** d_body_samples_table = nullptr;
    int*     d_body_samples_n_table = nullptr;

    /* Detection de contact corps<->corps (M17, phase B, B3a) : AABB monde
     * par corps (broadphase), masque de paires retenues, et tampon de
     * contacts du DERNIER sous-pas execute -- meme politique que
     * d_body_wrench/d_body_gather (remis a zero a chaque sous-pas, pas une
     * somme sur la frame). d_contact_count/d_contact_overflow sont lus par
     * bq_read_contacts / bq_contacts_last_overflow (copie hote explicite a
     * cet appel, hors de la boucle de sous-pas). */
    float3*     d_body_lo = nullptr;      /* BQ_MAX_BODIES */
    float3*     d_body_hi = nullptr;      /* BQ_MAX_BODIES */
    uint8_t*    d_pair_valid = nullptr;   /* BQ_MAX_BODIES*BQ_MAX_BODIES */
    BqContact*  d_contacts = nullptr;     /* BQ_MAX_CONTACTS */
    int*        d_contact_count = nullptr;
    int*        d_contact_overflow = nullptr;

    /* Solveur de contact corps<->corps (M17, phase B, B3b, D11) : cache de
     * warm starting PERSISTANT entre sous-pas (contrairement aux tampons
     * ci-dessus, remis a zero a chaque sous-pas) -- capacite
     * BQ_CONTACT_SOLVE_CAP, ecrit et relu par k_contact_solve uniquement,
     * jamais expose a l'appelant. d_body_pushout (BQ_MAX_BODIES*6 floats :
     * vp[3], wp[3] par corps) porte le canal de vitesse SEPAREE du split
     * impulse, ecrit par k_contact_solve et consomme par k_advance_bodies
     * dans le meme sous-pas -- pas besoin de remise a zero explicite, le
     * noyau reecrit systematiquement les 6 floats de chaque corps actif a
     * chaque appel. Sommeil (D11) : sleep_timer/asleep sont eux aussi
     * PERSISTANTS (contrairement au reste de cette section), mis a jour par
     * k_body_sleep_update a la fin de chaque sous-pas et lus par
     * k_body_predict au sous-pas SUIVANT. */
    BqContactCacheEntry* d_prev_contact_cache = nullptr; /* BQ_CONTACT_SOLVE_CAP */
    int*        d_prev_contact_count = nullptr;
    float*      d_body_pushout = nullptr;    /* BQ_MAX_BODIES*6 */
    float*      d_body_sleep_timer = nullptr; /* BQ_MAX_BODIES */
    uint8_t*    d_body_asleep = nullptr;      /* BQ_MAX_BODIES */

    /* grille de buckets (CSR), reconstruite sur l'hote a chaque appel de
     * bq_set_colliders puis televersee ; les buffers device ne sont
     * realloues que si la capacite courante est depassee. */
    int* d_bucket_off = nullptr; /* taille nb+1 */
    int* d_bucket_tri = nullptr; /* taille bucket_off[nb], indices de triangles */
    int bucket_off_cap = 0;
    int bucket_tri_cap = 0;
    /* geometrie de la grille de buckets (origine, pas, resolution) : la
     * structure CSR (d_bucket_off/d_bucket_tri) ne suffit pas a elle seule a
     * indexer un point, il faut aussi ces trois champs -- construits sur
     * l'hote dans bq_set_colliders (build_bucket_grid) mais jusqu'ici jamais
     * conserves au-dela de cet appel (variable locale `bg`). Persistes ici
     * pour que k_g2p puisse les reutiliser a chaque substep pour la CCD. */
    float3 bucket_origin = make_float3(0.f, 0.f, 0.f);
    float  bucket_h = 0.f;
    int3   bucket_res = make_int3(0, 0, 0);

    /* etat de signe par cellule (ncell) : BQ_SDF_STATE_UNKNOWN / _EXTERIOR /
     * _INTERIOR (cf. mlsmpm.cu, section SDF), et flag device de convergence
     * pour k_sdf_propagate_sign */
    uint8_t* d_ext = nullptr;
    int* d_changed = nullptr;

    /* reseeding (M10, T1-T3) : cf. section "reseeding" plus haut et reseed()
     * plus bas. Tampons ping-pong par particule (capacite fixe =
     * max_particles, alloues une fois a bq_create, jamais realloues -- meme
     * politique que le whitewater) et tampons de comptage par cellule
     * (capacite fixe = ncell, alloues une fois). */
    float3* d_x2 = nullptr;
    float3* d_v2 = nullptr;
    float*  d_C2 = nullptr;
    float*  d_F2 = nullptr;
    float*  d_J2 = nullptr;
    uint8_t* d_mat2 = nullptr;

    int*   d_reseed_count = nullptr;      /* ncell : particules actives par cellule */
    float* d_reseed_jsum = nullptr;       /* ncell : somme des J existants (moyenne de naissance) */
    int*   d_reseed_first = nullptr;      /* ncell : indice de la 1ere particule trouvee (materiau herite), sentinelle -1 */
    int*   d_reseed_birth = nullptr;      /* ncell+1 : naissances par cellule, case ncell = sentinelle du scan */
    int*   d_reseed_birth_scan = nullptr; /* ncell+1 : scan exclusif de d_reseed_birth */
    float* d_reseed_keepprob = nullptr;   /* ncell : probabilite de survie a la mort (1 = pas de mort) */
    int*   d_reseed_alive = nullptr;      /* max_particles+1 : marquage de survie, case n = sentinelle */
    int*   d_reseed_alive_scan = nullptr; /* max_particles+1 : scan exclusif de d_reseed_alive */
    void*  d_reseed_cub_tmp = nullptr;    /* espace de travail CUB, dimensionne une fois a bq_create */
    size_t reseed_cub_tmp_bytes = 0;

    /* Tri spatial par cellule (optimisation perf, cf. sort_particles() plus
     * bas) : reutilise EXACTEMENT les tampons ping-pong *2 de reseed()
     * ci-dessus -- libres de nouveau une fois reseed() revenu (son propre
     * swap en a deja vide le contenu utile). Comptage/offset/curseur dedies
     * (capacite fixe = ncell, alloues une fois a bq_create, meme politique
     * que d_reseed_count et consorts). */
    int*   d_sort_count = nullptr;   /* ncell : particules actives par cellule */
    int*   d_sort_offset = nullptr;  /* ncell : scan exclusif de d_sort_count (debut de plage par cellule) */
    int*   d_sort_cursor = nullptr;  /* ncell : curseur d'ecriture par cellule pendant le scatter */
    void*  d_sort_cub_tmp = nullptr; /* espace de travail CUB, dimensionne une fois a bq_create */
    size_t sort_cub_tmp_bytes = 0;

    /* plancher de dt sur la vitesse reelle des particules (garde-fou CCD) :
     * dt n'est aujourd'hui derive que de material_sound_speed, jamais de la
     * vitesse effective des particules. En regime extreme (materiau tres mou
     * + vitesse elevee), le deplacement par sous-pas peut alors depasser
     * largement le rayon de recherche de la CCD contre les colliders (cf.
     * commentaire CCD dans k_g2p) et laisser des particules traverser un mur
     * fin sans jamais etre rattrapees. Reduction cub::DeviceReduce::Max sur
     * |v[p]| de toutes les particules actives, calculee une fois par frame a
     * la fin de bq_step (apres le dernier sous-pas, avant reseed) et
     * consommee par upload_params au step suivant. Au tout premier appel,
     * prev_frame_max_speed vaut 0 : le plancher ne change donc rien tant
     * qu'aucune frame n'a encore ete simulee. */
    float* d_speed = nullptr;          /* cap : |v[p]| par particule, tampon scratch */
    float* d_max_speed = nullptr;      /* 1 : resultat de cub::DeviceReduce::Max */
    void*  d_speed_cub_tmp = nullptr;  /* espace de travail CUB, dimensionne une fois a bq_create */
    size_t speed_cub_tmp_bytes = 0;
    float  prev_frame_max_speed = 0.f;
};

/* Switch EXHAUSTIF sur le modele -- cf. plan-milestone-18.md, section "Le
 * vrai risque : le else silencieux". C'est le site le plus dangereux : un
 * sable de bulk nul tombant dans un else "sinon WATER" donnerait une vitesse
 * du son nulle, donc un dt non borne (cfl*dx/c_max -> +inf), donc une
 * simulation qui explose loin de sa cause. Le modele est deja valide par
 * bq_add_material avant d'atteindre cette fonction ; le defaut ci-dessous
 * n'est donc qu'une seconde ligne de defense, jamais le chemin normal. */
static float material_sound_speed(const BqMaterial& m) {
    float stiff;
    if (m.model == BQ_MODEL_ELASTIC) {
        stiff = m.E;
    } else if (m.model == BQ_MODEL_WATER) {
        stiff = m.bulk;
    } else if (m.model == BQ_MODEL_SAND) {
        /* c = sqrt((lam + 2 mu) / rho) -- vitesse d'onde P elastique,
         * derivee des memes Lame que la contrainte de Kirchhoff de k_p2g. */
        float mu = m.E / (2.f * (1.f + m.nu));
        float lam = m.E * m.nu / ((1.f + m.nu) * (1.f - 2.f * m.nu));
        return sqrtf((lam + 2.f * mu) / m.rho);
    } else {
        /* modele inconnu : ne devrait jamais arriver (bq_add_material
         * refuse ce cas), garde-fou silencieux impossible a eviter ici --
         * une raideur nulle est le comportement le PLUS sur en dernier
         * recours (dt petit, pas dt infini). */
        stiff = 0.f;
    }
    return sqrtf(stiff / m.rho);
}

static int upload_params(BqSim* s) {
    float dx = s->cfg.cell_size;
    if (s->n_mats == 0) {
        /* Aucun materiau enregistre : rien dont deriver une vitesse du son,
         * donc rien dont deriver un dt par CFL acoustique (M17/B3b, point 0
         * de la spec). Une scene de corps rigides purs (sans fluide) a
         * quand meme besoin d'un dt stable pour integrer le contact --
         * repli EXPLICITE et documente (cf. BQ_NO_FLUID_DT), jamais le
         * plancher c_max=1e-3f ci-dessous qui donnerait un dt de plusieurs
         * secondes (cfl*dx/1e-3), bien trop grand pour un Gauss-Seidel de
         * contact stable. */
        s->dt = BQ_NO_FLUID_DT;
    } else {
        float c_max = 1e-3f;
        for (int i = 0; i < s->n_mats; ++i)
            c_max = fmaxf(c_max, material_sound_speed(s->mats_host[i]));
        /* plancher de vitesse reelle (garde-fou CCD, cf. commentaire sur
         * prev_frame_max_speed dans BqSim) : vaut 0 tant qu'aucune frame n'a
         * ete simulee, donc sans effet au demarrage. */
        c_max = fmaxf(c_max, s->prev_frame_max_speed);
        s->dt = s->cfg.cfl * dx / c_max;
    }

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
        /* alpha du cone de Drucker-Prager (SAND uniquement) : precalcule ici
         * une fois par upload, jamais par particule/sous-pas (cf. k_p2g).
         * Sans objet pour ELASTIC/WATER, laisse a 0. */
        g.alpha = 0.f;
        if (m.model == BQ_MODEL_SAND) {
            const float deg2rad = 3.14159265358979323846f / 180.f;
            float phi = m.friction_angle * deg2rad;
            g.alpha = sqrtf(2.f / 3.f) * 2.f * sinf(phi) / (3.f - sinf(phi));
        }
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
        cudaMalloc(&s->d_cbody, ncell * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_bodies, BQ_MAX_BODIES * sizeof(BqRigidBody)) != cudaSuccess ||
        cudaMalloc(&s->d_body_wrench, BQ_MAX_BODIES * 7 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_body_gather, BQ_MAX_BODIES * 16 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_ext, ncell * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(&s->d_changed, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_x2, cap * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&s->d_v2, cap * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&s->d_C2, cap * 9 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_F2, cap * 9 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_J2, cap * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_mat2, cap * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(&s->d_reseed_count, ncell * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_reseed_jsum, ncell * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_reseed_first, ncell * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_reseed_birth, (ncell + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_reseed_birth_scan, (ncell + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_reseed_keepprob, ncell * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_reseed_alive, (cap + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_reseed_alive_scan, (cap + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_speed, cap * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_max_speed, sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_sort_count, ncell * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_sort_offset, ncell * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_sort_cursor, ncell * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_body_samples_table, BQ_MAX_BODIES * sizeof(float3*)) != cudaSuccess ||
        cudaMalloc(&s->d_body_samples_n_table, BQ_MAX_BODIES * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_body_lo, BQ_MAX_BODIES * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&s->d_body_hi, BQ_MAX_BODIES * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&s->d_pair_valid, (size_t)BQ_MAX_BODIES * BQ_MAX_BODIES * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(&s->d_contacts, BQ_MAX_CONTACTS * sizeof(BqContact)) != cudaSuccess ||
        cudaMalloc(&s->d_contact_count, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_contact_overflow, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_prev_contact_cache, BQ_CONTACT_SOLVE_CAP * sizeof(BqContactCacheEntry)) != cudaSuccess ||
        cudaMalloc(&s->d_prev_contact_count, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&s->d_body_pushout, BQ_MAX_BODIES * 6 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_body_sleep_timer, BQ_MAX_BODIES * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_body_asleep, BQ_MAX_BODIES * sizeof(uint8_t)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "cudaMalloc: memoire insuffisante");
        bq_destroy(s);
        return nullptr;
    }
    /* espace de travail CUB pour les deux scans de reseed() (naissances par
     * cellule, ncell+1 ; survie par particule, cap+1) -- dimensionne une
     * fois pour le plus grand des deux, jamais realloue ensuite (capacite
     * fixe, cf. commentaires de BqSim). */
    {
        size_t tmp_cell = 0, tmp_part = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_cell, (int*)nullptr,
                                      (int*)nullptr, ncell + 1);
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_part, (int*)nullptr,
                                      (int*)nullptr, cap + 1);
        s->reseed_cub_tmp_bytes = (tmp_cell > tmp_part) ? tmp_cell : tmp_part;
        if (s->reseed_cub_tmp_bytes == 0) s->reseed_cub_tmp_bytes = 1;
        if (cudaMalloc(&s->d_reseed_cub_tmp, s->reseed_cub_tmp_bytes) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "cudaMalloc: espace de travail CUB (reseeding) insuffisant");
            bq_destroy(s);
            return nullptr;
        }
    }
    /* espace de travail CUB pour la reduction max de vitesse (garde-fou dt,
     * cf. commentaire sur prev_frame_max_speed dans BqSim) -- dimensionne une
     * fois pour cap elements, jamais realloue ensuite. */
    {
        size_t tmp_speed = 0;
        cub::DeviceReduce::Max(nullptr, tmp_speed, (float*)nullptr,
                               (float*)nullptr, cap);
        s->speed_cub_tmp_bytes = tmp_speed;
        if (s->speed_cub_tmp_bytes == 0) s->speed_cub_tmp_bytes = 1;
        if (cudaMalloc(&s->d_speed_cub_tmp, s->speed_cub_tmp_bytes) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "cudaMalloc: espace de travail CUB (vitesse max) insuffisant");
            bq_destroy(s);
            return nullptr;
        }
    }
    /* espace de travail CUB pour le scan du tri spatial (sort_particles,
     * ncell elements) -- dimensionne une fois, jamais realloue ensuite. Non
     * partage avec d_reseed_cub_tmp : celui-ci est deja dimensionne pour
     * ncell+1/cap+1, ce qui suffirait probablement, mais un tampon dedie
     * evite toute hypothese fragile sur la relation taille/octets de CUB
     * entre deux appels de tailles differentes. */
    {
        size_t tmp_sort = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_sort, (int*)nullptr,
                                      (int*)nullptr, ncell);
        s->sort_cub_tmp_bytes = tmp_sort;
        if (s->sort_cub_tmp_bytes == 0) s->sort_cub_tmp_bytes = 1;
        if (cudaMalloc(&s->d_sort_cub_tmp, s->sort_cub_tmp_bytes) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "cudaMalloc: espace de travail CUB (tri spatial) insuffisant");
            bq_destroy(s);
            return nullptr;
        }
    }
    /* table device des SDF locaux par corps (M17, phase B, D9) : capacite
     * fixe BQ_MAX_BODIES, comme d_bodies -- mise a zero (tous phi = nullptr,
     * "pas de SDF construit") tant que bq_build_body_sdf n'a pas ete appele
     * pour un corps donne. */
    if (cudaMalloc(&s->d_body_sdf_table, BQ_MAX_BODIES * sizeof(BqBodySdf)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "cudaMalloc: table SDF locale (corps) insuffisante");
        bq_destroy(s);
        return nullptr;
    }
    if (cudaMemset(s->d_body_sdf_table, 0, BQ_MAX_BODIES * sizeof(BqBodySdf)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "cudaMemset: table SDF locale (corps) echoue");
        bq_destroy(s);
        return nullptr;
    }
    /* Miroirs device des echantillons de surface (D10, B3a) : nullptr/0 tant
     * que bq_set_body_samples n'a pas ete appele pour un corps -- meme
     * discipline que d_body_sdf_table ci-dessus. Compteurs de contact remis
     * a zero : rien de lu avant le premier bq_step avec des corps declares. */
    if (cudaMemset(s->d_body_samples_table, 0, BQ_MAX_BODIES * sizeof(float3*)) != cudaSuccess ||
        cudaMemset(s->d_body_samples_n_table, 0, BQ_MAX_BODIES * sizeof(int)) != cudaSuccess ||
        cudaMemset(s->d_contact_count, 0, sizeof(int)) != cudaSuccess ||
        cudaMemset(s->d_contact_overflow, 0, sizeof(int)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "cudaMemset: tables de contact (corps) echoue");
        bq_destroy(s);
        return nullptr;
    }
    /* Solveur de contact (B3b) : cache de warm start vide, aucun corps
     * endormi au depart -- meme discipline que ci-dessus. */
    if (cudaMemset(s->d_prev_contact_count, 0, sizeof(int)) != cudaSuccess ||
        cudaMemset(s->d_body_pushout, 0, BQ_MAX_BODIES * 6 * sizeof(float)) != cudaSuccess ||
        cudaMemset(s->d_body_sleep_timer, 0, BQ_MAX_BODIES * sizeof(float)) != cudaSuccess ||
        cudaMemset(s->d_body_asleep, 0, BQ_MAX_BODIES * sizeof(uint8_t)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "cudaMemset: solveur de contact (B3b) echoue");
        bq_destroy(s);
        return nullptr;
    }

    /* pas de collider au depart : sdf grand partout, vitesse/friction nulles */
    dim3 bp(256), gc((ncell + 255) / 256);
    k_fill_sdf<<<gc, bp>>>(s->d_sdf, s->d_cvel, s->d_cnrm, s->d_cbody, ncell);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "k_fill_sdf: echec init");
        bq_destroy(s);
        return nullptr;
    }
    /* Initialise s->dt AVANT le premier bq_step (M17/B3b, correctif) :
     * jusqu'ici s->dt restait a sa valeur par defaut 0.f tant qu'aucun
     * bq_add_material n'avait ete appele (seul upload_params le calculait,
     * et seul bq_add_material/bq_step l'invoquaient). Une scene de corps
     * rigides purs (aucun materiau JAMAIS enregistre, cf. point 0 de la
     * spec) atteignait alors bq_step avec dt=0, substeps = frame_dt/0 = +inf,
     * cast en int = comportement indefini (observe : INT_MIN). upload_params
     * gere deja ce cas (repli BQ_NO_FLUID_DT si n_mats==0) -- il suffit de
     * l'appeler une fois ici pour que bq_step ait toujours un dt valide, meme
     * au tout premier appel, meme sans materiau. */
    if (upload_params(s) < 0) {
        snprintf(g_error, sizeof(g_error), "upload_params: echec init");
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
    cudaFree(s->d_tri_body); cudaFree(s->d_cbody);
    cudaFree(s->d_bodies); cudaFree(s->d_body_wrench); cudaFree(s->d_body_gather);
    cudaFree(s->d_body_sdf_table);
    for (int i = 0; i < BQ_MAX_BODIES; ++i) {
        cudaFree(s->body_sdf_host[i].phi);
        cudaFree(s->d_body_samples[i]);
    }
    cudaFree(s->d_body_samples_table); cudaFree(s->d_body_samples_n_table);
    cudaFree(s->d_body_lo); cudaFree(s->d_body_hi); cudaFree(s->d_pair_valid);
    cudaFree(s->d_contacts); cudaFree(s->d_contact_count); cudaFree(s->d_contact_overflow);
    cudaFree(s->d_prev_contact_cache); cudaFree(s->d_prev_contact_count);
    cudaFree(s->d_body_pushout); cudaFree(s->d_body_sleep_timer); cudaFree(s->d_body_asleep);
    cudaFree(s->d_bucket_off); cudaFree(s->d_bucket_tri);
    cudaFree(s->d_ext); cudaFree(s->d_changed);
    cudaFree(s->d_x2);  cudaFree(s->d_v2);  cudaFree(s->d_C2);
    cudaFree(s->d_F2);  cudaFree(s->d_J2);  cudaFree(s->d_mat2);
    cudaFree(s->d_reseed_count);      cudaFree(s->d_reseed_jsum);
    cudaFree(s->d_reseed_first);      cudaFree(s->d_reseed_birth);
    cudaFree(s->d_reseed_birth_scan); cudaFree(s->d_reseed_keepprob);
    cudaFree(s->d_reseed_alive);      cudaFree(s->d_reseed_alive_scan);
    cudaFree(s->d_reseed_cub_tmp);
    cudaFree(s->d_speed); cudaFree(s->d_max_speed); cudaFree(s->d_speed_cub_tmp);
    cudaFree(s->d_sort_count); cudaFree(s->d_sort_offset); cudaFree(s->d_sort_cursor);
    cudaFree(s->d_sort_cub_tmp);
    delete s;
}

BQ_API int bq_add_material(BqSim* s, const BqMaterial* mat) {
    if (s->n_mats >= BQ_MAX_MATERIALS) {
        snprintf(g_error, sizeof(g_error), "max %d materiaux", BQ_MAX_MATERIALS);
        return -1;
    }
    /* Garde-fou d'ABI : c'est ICI, cote hote, qu'un modele est valide -- pas
     * sur le device, qui ne peut pas lever d'erreur proprement. Objectif
     * precis (cf. plan-milestone-18.md, "Le vrai risque : le else
     * silencieux") : qu'aucun modele inconnu, ni un SAND aux parametres
     * absurdes, n'atteigne jamais material_sound_speed ou k_p2g. */
    if (mat->model != BQ_MODEL_ELASTIC && mat->model != BQ_MODEL_WATER &&
        mat->model != BQ_MODEL_SAND) {
        snprintf(g_error, sizeof(g_error), "modele constitutif inconnu: %d",
                 mat->model);
        return -1;
    }
    if (mat->model == BQ_MODEL_SAND) {
        if (!(mat->E > 0.f)) {
            snprintf(g_error, sizeof(g_error), "SAND: E doit etre > 0");
            return -1;
        }
        if (!(mat->rho > 0.f)) {
            snprintf(g_error, sizeof(g_error), "SAND: rho doit etre > 0");
            return -1;
        }
        if (!(mat->friction_angle > 0.f && mat->friction_angle < 90.f)) {
            snprintf(g_error, sizeof(g_error),
                     "SAND: friction_angle doit etre dans ]0, 90[ degres");
            return -1;
        }
        if (mat->cohesion != 0.f) {
            /* Cohesion non cablee dans le return mapping (cf. D2/M18, S2) :
             * plutot que de l'accepter en silence et l'ignorer, on refuse.
             * Un champ documente comme non implemente vaut mieux qu'une
             * cohesion fantome. */
            snprintf(g_error, sizeof(g_error),
                     "SAND: cohesion non implementee dans ce build, doit etre 0");
            return -1;
        }
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

    std::vector<float> id9(count * 9, 0.f), jinit(count, 1.f);
    for (int i = 0; i < count; ++i) { id9[9 * i] = id9[9 * i + 4] = id9[9 * i + 8] = 1.f; }
    std::vector<float> zero9(count * 9, 0.f);
    std::vector<uint8_t> mid(count, (uint8_t)mat_id);

    /* Initialisation HYDROSTATIQUE de J (modele WATER, corps emis au repos).
     *
     * A J = 1 la pression de Tait est nulle : un corps pose sous gravite n'est
     * porte par RIEN a t = 0. Il tombe, comprime, depasse, et sonne autour de
     * son equilibre -- l'effet de ressort visible au demarrage d'un bloc au
     * repos. Le fluide etant inviscide, rien ne l'amortit : mesure sur une
     * colonne de 0.25 m, 3.2 mm crete-a-crete encore presents apres 20 s.
     *
     * On part donc directement de l'equilibre. La colonne est lagrangienne (la
     * masse par particule est fixe), donc la pression a la profondeur h vaut
     * exactement rho*g*h, et l'inversion de l'EOS de Tait donne
     *
     *     p = (K/gamma) * (J^-gamma - 1)   =>   J = (1 + gamma*p/K)^(-1/gamma)
     *
     * `h` est mesuree sous la surface libre du corps emis, prise a l'extremite
     * du nuage de points situee du cote oppose a la gravite.
     *
     * Restriction aux corps emis SANS VITESSE, volontaire : un jet d'inflow ou
     * un bloc lance n'est pas une colonne au repos et n'a aucune raison d'etre
     * pre-comprime. Lui appliquer ce profil le ferait se detendre au demarrage,
     * ce qui est le meme artefact qu'on corrige ici, juste en sens inverse. */
    const BqMaterial& mem = s->mats_host[mat_id];
    const float gy = s->cfg.gravity_y;
    if (mem.model == BQ_MODEL_WATER && gy != 0.f && mem.bulk > 0.f &&
        mem.gamma > 0.f) {
        bool au_repos = true;
        for (int i = 0; i < count && au_repos; ++i)
            au_repos = (pv[i].x == 0.f && pv[i].y == 0.f && pv[i].z == 0.f);

        if (au_repos) {
            /* surface libre = extremite du corps a l'oppose de la gravite */
            float y_surf = px[0].y;
            for (int i = 1; i < count; ++i)
                y_surf = (gy < 0.f) ? fmaxf(y_surf, px[i].y)
                                    : fminf(y_surf, px[i].y);

            const float rho_g = mem.rho * fabsf(gy);
            for (int i = 0; i < count; ++i) {
                float h = (gy < 0.f) ? (y_surf - px[i].y) : (px[i].y - y_surf);
                if (h < 0.f) h = 0.f;
                float J = powf(1.f + mem.gamma * rho_g * h / mem.bulk,
                               -1.f / mem.gamma);
                /* meme domaine de validite que le clamp de k_g2p */
                jinit[i] = fminf(fmaxf(J, 0.5f), 1.5f);
            }
        }
    }

    int off = s->n;
    BQ_CUDA_CHECK(cudaMemcpy(s->d_x + off, px, count * sizeof(float3),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_v + off, pv, count * sizeof(float3),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_F + 9 * off, id9.data(),
                             count * 9 * sizeof(float), cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_C + 9 * off, zero9.data(),
                             count * 9 * sizeof(float), cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_J + off, jinit.data(), count * sizeof(float),
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
                            const float* tri_friction, const int* tri_body,
                            int n_tri) {
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
        k_fill_sdf<<<gc, bp>>>(s->d_sdf, s->d_cvel, s->d_cnrm, s->d_cbody, ncell);
        BQ_CUDA_CHECK(cudaGetLastError());
        BQ_CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    }
    if (tri == NULL || tri_vel == NULL || tri_friction == NULL) {
        snprintf(g_error, sizeof(g_error),
                 "bq_set_colliders: pointeur nul (n_tri=%d)", n_tri);
        return -1;
    }

    /* reallocation seulement quand la capacite courante est depassee.
     * d_tri_body suit la meme politique que d_tri/d_trivel/d_trifric bien
     * que son upload plus bas soit conditionnel (tri_body peut etre NULL a
     * cet appel precis, cf. bq_set_colliders dans bourrasque.h). */
    if (n_tri > s->tri_cap) {
        cudaFree(s->d_tri); cudaFree(s->d_trivel); cudaFree(s->d_trifric);
        cudaFree(s->d_tri_body);
        s->d_tri = nullptr; s->d_trivel = nullptr; s->d_trifric = nullptr;
        s->d_tri_body = nullptr;
        s->tri_cap = 0;
        if (cudaMalloc(&s->d_tri, (size_t)n_tri * 3 * sizeof(float3)) != cudaSuccess ||
            cudaMalloc(&s->d_trivel, (size_t)n_tri * 3 * sizeof(float3)) != cudaSuccess ||
            cudaMalloc(&s->d_trifric, (size_t)n_tri * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&s->d_tri_body, (size_t)n_tri * sizeof(int)) != cudaSuccess) {
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
    if (tri_body != NULL)
        BQ_CUDA_CHECK(cudaMemcpy(s->d_tri_body, tri_body, (size_t)n_tri * sizeof(int),
                                 cudaMemcpyHostToDevice));
    /* default_body pour k_sdf_unsigned : utilise seulement quand tri_body ==
     * NULL a CET appel (cf. le pointeur passe au kernel plus bas, jamais
     * s->d_tri_body dans ce cas -- son contenu peut etre perime d'un appel
     * precedent, sans consequence puisqu'il n'est pas lu). */
    int default_body = (s->n_bodies > 0) ? 0 : -1;

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
    s->bucket_origin = bg.origin; s->bucket_h = bg.h; s->bucket_res = bg.res;

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
                               s->d_tri, s->d_trivel, s->d_trifric,
                               (tri_body != NULL) ? s->d_tri_body : nullptr,
                               default_body, s->d_cbody,
                               s->d_bucket_off, s->d_bucket_tri,
                               bg.origin, bg.h, bg.res, lo, hi, ncell,
                               s->prm.res, s->prm.dx);
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
        k_sdf_propagate_sign<<<gc, bp>>>(s->d_ext, ncell, s->d_changed, s->prm.res);
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

BQ_API size_t bq_rigid_body_size(void) {
    return sizeof(BqRigidBody);
}

BQ_API int bq_set_collider_bodies(BqSim* s, const BqRigidBody* bodies, int n_bodies) {
    if (n_bodies < 0) {
        snprintf(g_error, sizeof(g_error),
                 "bq_set_collider_bodies: n_bodies negatif (%d)", n_bodies);
        return -1;
    }
    if (n_bodies > BQ_MAX_BODIES) {
        snprintf(g_error, sizeof(g_error),
                 "bq_set_collider_bodies: n_bodies (%d) depasse le plafond de %d corps",
                 n_bodies, BQ_MAX_BODIES);
        return -1;
    }
    s->n_bodies = n_bodies;
    /* Reinitialise le solveur de contact (B3b) : l'identite des corps peut
     * avoir change entierement (n_bodies==0 efface tout, ou un nouveau bake
     * redeclare un jeu de corps different) -- un cache de warm start ou un
     * etat de sommeil perime associerait un lambda ou un gel a un corps qui
     * n'est plus le meme. bq_set_collider_bodies n'est appelee qu'une fois
     * au debut du bake (cf. commentaire dans bourrasque.h), ce cudaMemset
     * n'entre jamais dans la boucle de sous-pas. */
    BQ_CUDA_CHECK(cudaMemset(s->d_prev_contact_count, 0, sizeof(int)));
    BQ_CUDA_CHECK(cudaMemset(s->d_body_sleep_timer, 0, BQ_MAX_BODIES * sizeof(float)));
    BQ_CUDA_CHECK(cudaMemset(s->d_body_asleep, 0, BQ_MAX_BODIES * sizeof(uint8_t)));
    if (n_bodies == 0) return 0; /* efface tout, cf. commentaire dans bourrasque.h */
    BQ_CUDA_CHECK(cudaMemcpy(s->d_bodies, bodies, (size_t)n_bodies * sizeof(BqRigidBody),
                             cudaMemcpyHostToDevice));
    return 0;
}

BQ_API int bq_read_collider_bodies(BqSim* s, float* dst) {
    if (s->n_bodies == 0) return 0;
    /* BqRigidBody est dense (tous ses membres font 4 octets, alignes
     * naturellement -- aucun padding), donc x/q/v/w (13 floats contigus en
     * son sein) sont extractibles directement par un memcpy 2D a foulee
     * sizeof(BqRigidBody), sans kernel ni tampon intermediaire. */
    const char* src = (const char*)s->d_bodies + offsetof(BqRigidBody, x);
    BQ_CUDA_CHECK(cudaMemcpy2D(dst, 13 * sizeof(float), src, sizeof(BqRigidBody),
                               13 * sizeof(float), (size_t)s->n_bodies,
                               cudaMemcpyDeviceToHost));
    return s->n_bodies;
}

/* Met a jour la pose ET la vitesse d'un corps CINEMATIQUE (cf. contrat
   complet dans bourrasque.h). Ecrit UNIQUEMENT les 13 floats x/q/v/w du
   corps vise -- meme bloc contigu que celui lu par bq_read_collider_bodies
   (offsetof(BqRigidBody, x), 13*sizeof(float), cf. commentaire de cette
   fonction) -- donc ne touche a rien d'autre : ni l'etat des autres corps,
   ni d_body_sleep_timer/d_body_asleep (caches de sommeil), ni le warm start
   du contact (d_prev_contact_count et son tampon associe), qui ne sont
   remis a zero que par bq_set_collider_bodies. */
BQ_API int bq_set_body_pose(BqSim* s, int body, const float x[3], const float q[4],
                            const float v[3], const float w[3]) {
    if (body < 0 || body >= s->n_bodies) {
        snprintf(g_error, sizeof(g_error),
                 "bq_set_body_pose: indice de corps %d hors bornes [0, %d[",
                 body, s->n_bodies);
        return -1;
    }
    /* dynamic est le premier membre de BqRigidBody : lu isolement, sans
     * rapatrier le reste du corps. */
    int dyn = 0;
    BQ_CUDA_CHECK(cudaMemcpy(&dyn,
                             (const char*)s->d_bodies + (size_t)body * sizeof(BqRigidBody) +
                                 offsetof(BqRigidBody, dynamic),
                             sizeof(int), cudaMemcpyDeviceToHost));
    if (dyn != 0) {
        snprintf(g_error, sizeof(g_error),
                 "bq_set_body_pose: le corps %d est DYNAMIQUE (dynamic=1) -- sa pose est "
                 "calculee par le solveur (k_body_predict/k_advance_bodies), l'ecraser "
                 "depuis l'hote serait un bug silencieux",
                 body);
        return -1;
    }

    /* Normalisation defensive du quaternion (convention w, x, y, z) : une
     * entree legerement desaxee par accumulation flottante cote Python ne
     * doit pas s'introduire dans l'etat du solveur. */
    float qn[4] = {q[0], q[1], q[2], q[3]};
    float qlen = sqrtf(qn[0] * qn[0] + qn[1] * qn[1] + qn[2] * qn[2] + qn[3] * qn[3]);
    if (qlen > 1e-12f) {
        float inv = 1.f / qlen;
        qn[0] *= inv; qn[1] *= inv; qn[2] *= inv; qn[3] *= inv;
    } else {
        qn[0] = 1.f; qn[1] = qn[2] = qn[3] = 0.f;
    }

    float buf[13] = {x[0], x[1], x[2],
                     qn[0], qn[1], qn[2], qn[3],
                     v[0], v[1], v[2],
                     w[0], w[1], w[2]};
    char* dst = (char*)s->d_bodies + (size_t)body * sizeof(BqRigidBody) + offsetof(BqRigidBody, x);
    BQ_CUDA_CHECK(cudaMemcpy(dst, buf, 13 * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

BQ_API int bq_read_collider_wrench(BqSim* s, float* dst) {
    if (s->n_bodies == 0) return 0;
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_body_wrench,
                             (size_t)s->n_bodies * 7 * sizeof(float),
                             cudaMemcpyDeviceToHost));
    return s->n_bodies;
}

/* Construit le SDF local d'un corps a partir de ses triangles de repos en
   repere de CORPS (cf. bourrasque.h pour le contrat complet). Reutilise
   k_sdf_unsigned / k_sdf_propagate_sign / k_sdf_finalize_sign TELS QUELS
   (D0, plan-milestone-17.md) sur une grille INDEPENDANTE de celle du
   solveur -- exactement ce que ce decouplage rend possible. */
BQ_API int bq_build_body_sdf(BqSim* s, int body, const float* tri, int n_tri,
                             float target_cell, int max_res) {
    if (body < 0 || body >= BQ_MAX_BODIES) {
        snprintf(g_error, sizeof(g_error),
                 "bq_build_body_sdf: corps %d hors de [0, %d[", body, BQ_MAX_BODIES);
        return -1;
    }
    if (n_tri <= 0 || tri == NULL) {
        snprintf(g_error, sizeof(g_error),
                 "bq_build_body_sdf: triangles invalides (n_tri=%d)", n_tri);
        return -1;
    }
    if (!(target_cell > 0.f)) {
        snprintf(g_error, sizeof(g_error),
                 "bq_build_body_sdf: target_cell doit etre > 0 (%g)", target_cell);
        return -1;
    }
    if (max_res < 4) {
        snprintf(g_error, sizeof(g_error),
                 "bq_build_body_sdf: max_res doit etre >= 4 (%d)", max_res);
        return -1;
    }

    /* AABB des triangles de repos, dilatee d'au moins 4 voxels (cible) de
     * chaque cote -- une requete juste a l'exterieur du corps doit rendre
     * une distance positive utile, pas une sortie de grille (cf. header). */
    float3 lo = make_float3(3.4e38f, 3.4e38f, 3.4e38f);
    float3 hi = make_float3(-3.4e38f, -3.4e38f, -3.4e38f);
    for (int i = 0; i < 3 * n_tri; ++i) {
        float x = tri[3 * i], y = tri[3 * i + 1], z = tri[3 * i + 2];
        lo.x = fminf(lo.x, x); lo.y = fminf(lo.y, y); lo.z = fminf(lo.z, z);
        hi.x = fmaxf(hi.x, x); hi.y = fmaxf(hi.y, y); hi.z = fmaxf(hi.z, z);
    }
    const float pad_voxels = 4.f;
    float pad = pad_voxels * target_cell;
    lo = make_float3(lo.x - pad, lo.y - pad, lo.z - pad);
    hi = make_float3(hi.x + pad, hi.y + pad, hi.z + pad);
    float ext_x = fmaxf(hi.x - lo.x, 1e-6f);
    float ext_y = fmaxf(hi.y - lo.y, 1e-6f);
    float ext_z = fmaxf(hi.z - lo.z, 1e-6f);

    /* Taille de voxel visee = target_cell, agrandie si necessaire pour tenir
     * sous le plafond max_res par axe (D9, plan-milestone-17.md) -- meme
     * politique de degradation propre que build_bucket_grid plus haut : le
     * voxel grossit, la finesse baisse, jamais de troncature silencieuse
     * (message sur stderr, cf. plus bas). */
    float cell = target_cell;
    int3 res;
    auto compute_res = [&]() {
        res.x = (int)ceilf(ext_x / cell) + 1; if (res.x < 2) res.x = 2;
        res.y = (int)ceilf(ext_y / cell) + 1; if (res.y < 2) res.y = 2;
        res.z = (int)ceilf(ext_z / cell) + 1; if (res.z < 2) res.z = 2;
    };
    compute_res();
    bool capped = false;
    int guard = 0;
    while ((res.x > max_res || res.y > max_res || res.z > max_res) && guard++ < 64) {
        capped = true;
        float scale = fmaxf(fmaxf((float)res.x / (float)max_res,
                                  (float)res.y / (float)max_res),
                            (float)res.z / (float)max_res);
        cell *= fmaxf(scale, 1.001f);
        compute_res();
    }
    if (capped) {
        fprintf(stderr,
                "bourrasque: bq_build_body_sdf: corps %d depasse %d^3 voxels a la "
                "taille cible %g -- voxel agrandi a %g (resolution %dx%dx%d)\n",
                body, max_res, (double)target_cell, (double)cell, res.x, res.y, res.z);
    }
    int ncell = res.x * res.y * res.z;

    /* Triangles TRANSLATES pour que le coin min de la grille locale tombe sur
     * (0,0,0) -- k_sdf_unsigned n'a pas de parametre d'origine, cf. le grand
     * commentaire au-dessus de BqBodySdf plus haut dans ce fichier. */
    std::vector<float> tri_t((size_t)9 * n_tri);
    for (int i = 0; i < 3 * n_tri; ++i) {
        tri_t[3 * i + 0] = tri[3 * i + 0] - lo.x;
        tri_t[3 * i + 1] = tri[3 * i + 1] - lo.y;
        tri_t[3 * i + 2] = tri[3 * i + 2] - lo.z;
    }

    BucketGridHost bg;
    std::vector<int> bucket_off, bucket_tri;
    if (!build_bucket_grid(tri_t.data(), n_tri, BQ_BUCKET_DX_MULT * cell, bg,
                           bucket_off, bucket_tri)) {
        return -1; /* g_error deja rempli par build_bucket_grid */
    }
    int nb = bg.res.x * bg.res.y * bg.res.z;

    /* Tampons device SCRATCH : ce SDF local ne sert ni vitesse de mur ni
     * friction (trivel/trifric bidons, mis a zero) ni identite de corps par
     * cellule (tri_body/cbody sans objet ici) -- k_sdf_unsigned les exige
     * neanmoins en parametres non nuls. Duree de vie limitee a cet appel ;
     * seul d_phi_new survit, transfere a s->body_sdf_host[body]. */
    float3* d_tri_t = nullptr; float3* d_trivel0 = nullptr; float* d_trifric0 = nullptr;
    int* d_bucket_off = nullptr; int* d_bucket_tri = nullptr;
    uint8_t* d_state = nullptr; int* d_changed = nullptr;
    float4* d_cvel_tmp = nullptr; float4* d_cnrm_tmp = nullptr; int* d_cbody_tmp = nullptr;
    float* d_phi_new = nullptr;

    auto cleanup_scratch = [&]() {
        cudaFree(d_tri_t); cudaFree(d_trivel0); cudaFree(d_trifric0);
        cudaFree(d_bucket_off); cudaFree(d_bucket_tri);
        cudaFree(d_state); cudaFree(d_changed);
        cudaFree(d_cvel_tmp); cudaFree(d_cnrm_tmp); cudaFree(d_cbody_tmp);
    };

    if (cudaMalloc(&d_tri_t, (size_t)n_tri * 3 * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&d_trivel0, (size_t)n_tri * 3 * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&d_trifric0, (size_t)n_tri * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_bucket_off, (size_t)(nb + 1) * sizeof(int)) != cudaSuccess ||
        (bucket_tri.size() > 0 &&
         cudaMalloc(&d_bucket_tri, bucket_tri.size() * sizeof(int)) != cudaSuccess) ||
        cudaMalloc(&d_state, (size_t)ncell * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(&d_changed, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_cvel_tmp, (size_t)ncell * sizeof(float4)) != cudaSuccess ||
        cudaMalloc(&d_cnrm_tmp, (size_t)ncell * sizeof(float4)) != cudaSuccess ||
        cudaMalloc(&d_cbody_tmp, (size_t)ncell * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_phi_new, (size_t)ncell * sizeof(float)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error),
                 "bq_build_body_sdf: cudaMalloc echoue (corps %d, ncell=%d)", body, ncell);
        cleanup_scratch();
        cudaFree(d_phi_new);
        return -1;
    }

    cudaMemcpy(d_tri_t, tri_t.data(), (size_t)n_tri * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_trivel0, 0, (size_t)n_tri * 3 * sizeof(float3));
    cudaMemset(d_trifric0, 0, (size_t)n_tri * sizeof(float));
    cudaMemcpy(d_bucket_off, bucket_off.data(), (size_t)(nb + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    if (bucket_tri.size() > 0)
        cudaMemcpy(d_bucket_tri, bucket_tri.data(), bucket_tri.size() * sizeof(int),
                   cudaMemcpyHostToDevice);

    dim3 bp(256), gc((ncell + 255) / 256);
    /* AABB "active" = grille entiere : pas d'optimisation de bande a faire
     * ici, ncell <= max_res^3 reste petit (ce n'est PAS le champ collider
     * fusionne de bq_set_colliders, potentiellement bien plus grand). */
    float3 far_lo = make_float3(-3.4e38f, -3.4e38f, -3.4e38f);
    float3 far_hi = make_float3(3.4e38f, 3.4e38f, 3.4e38f);
    k_sdf_unsigned<<<gc, bp>>>(d_phi_new, d_cvel_tmp, d_cnrm_tmp, d_state,
                               d_tri_t, d_trivel0, d_trifric0,
                               nullptr, -1, d_cbody_tmp,
                               d_bucket_off, d_bucket_tri,
                               bg.origin, bg.h, bg.res,
                               far_lo, far_hi, ncell, res, cell);
    if (cudaGetLastError() != cudaSuccess) {
        snprintf(g_error, sizeof(g_error),
                 "bq_build_body_sdf: k_sdf_unsigned a echoue (corps %d)", body);
        cleanup_scratch(); cudaFree(d_phi_new);
        return -1;
    }

    /* Propagation du signe, meme motif que bq_set_colliders (batching des
     * lectures hote pour ne pas synchroniser a chaque passe). Grille petite
     * (<= max_res^3) donc peu de passes attendues ; troncature signalee sur
     * stderr, jamais silencieuse (regle de securite de k_sdf_finalize_sign :
     * une cellule encore indeterminee reste exterieure). */
    int max_iter = 2 * (res.x + res.y + res.z);
    const int check_every = 8;
    cudaMemset(d_changed, 0, sizeof(int));
    bool converged = false;
    for (int it = 0; it < max_iter; ++it) {
        k_sdf_propagate_sign<<<gc, bp>>>(d_state, ncell, d_changed, res);
        bool check_now = ((it + 1) % check_every == 0) || (it + 1 == max_iter);
        if (check_now) {
            int h_changed = 0;
            cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
            if (!h_changed) { converged = true; break; }
            cudaMemset(d_changed, 0, sizeof(int));
        }
    }
    if (!converged) {
        fprintf(stderr,
                "bourrasque: bq_build_body_sdf: corps %d, propagation du signe "
                "tronquee apres %d passes -- cellules encore indeterminees "
                "traitees comme exterieures (regle de securite).\n",
                body, max_iter);
    }
    k_sdf_finalize_sign<<<gc, bp>>>(d_phi_new, d_cnrm_tmp, d_state, ncell);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        snprintf(g_error, sizeof(g_error),
                 "bq_build_body_sdf: propagation/finalisation du signe a echoue (corps %d)",
                 body);
        cleanup_scratch(); cudaFree(d_phi_new);
        return -1;
    }

    cleanup_scratch();

    /* Remplace le SDF existant du corps s'il y en avait deja un (rebuild). */
    cudaFree(s->body_sdf_host[body].phi);
    BqBodySdf entry;
    entry.origin = lo;
    entry.cell = cell;
    entry.res = res;
    entry.phi = d_phi_new;
    s->body_sdf_host[body] = entry;
    BQ_CUDA_CHECK(cudaMemcpy(s->d_body_sdf_table + body, &entry, sizeof(BqBodySdf),
                             cudaMemcpyHostToDevice));
    return 0;
}

BQ_API int bq_set_body_samples(BqSim* s, int body, const float* pts, int n) {
    if (body < 0 || body >= BQ_MAX_BODIES) {
        snprintf(g_error, sizeof(g_error),
                 "bq_set_body_samples: corps %d hors de [0, %d[", body, BQ_MAX_BODIES);
        return -1;
    }
    if (n < 0) {
        snprintf(g_error, sizeof(g_error), "bq_set_body_samples: n negatif (%d)", n);
        return -1;
    }
    cudaFree(s->d_body_samples[body]);
    s->d_body_samples[body] = nullptr;
    s->body_samples_n[body] = 0;
    if (n == 0) {
        /* Miroirs device (D10, B3a) tenus a jour meme sur effacement --
         * sinon k_gen_contacts continuerait a lire un pointeur/compte
         * perimes pour ce corps. */
        BQ_CUDA_CHECK(cudaMemcpy(s->d_body_samples_table + body, &s->d_body_samples[body],
                                 sizeof(float3*), cudaMemcpyHostToDevice));
        BQ_CUDA_CHECK(cudaMemcpy(s->d_body_samples_n_table + body, &s->body_samples_n[body],
                                 sizeof(int), cudaMemcpyHostToDevice));
        return 0;
    }
    if (pts == NULL) {
        snprintf(g_error, sizeof(g_error), "bq_set_body_samples: pts nul (n=%d)", n);
        return -1;
    }
    BQ_CUDA_CHECK(cudaMalloc(&s->d_body_samples[body], (size_t)n * sizeof(float3)));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_body_samples[body], pts, (size_t)n * 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
    s->body_samples_n[body] = n;
    /* Miroirs device (D10, B3a) : un seul element mis a jour, pas de
     * retelevesement complet de la table (cf. commentaire du champ dans
     * BqSim). */
    BQ_CUDA_CHECK(cudaMemcpy(s->d_body_samples_table + body, &s->d_body_samples[body],
                             sizeof(float3*), cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_body_samples_n_table + body, &s->body_samples_n[body],
                             sizeof(int), cudaMemcpyHostToDevice));
    return 0;
}

BQ_API int bq_read_body_sdf(BqSim* s, int body, float* dst, int res_out[3],
                            float* cell, float origin[3]) {
    if (body < 0 || body >= BQ_MAX_BODIES) {
        snprintf(g_error, sizeof(g_error),
                 "bq_read_body_sdf: corps %d hors de [0, %d[", body, BQ_MAX_BODIES);
        return -1;
    }
    const BqBodySdf& b = s->body_sdf_host[body];
    if (b.phi == nullptr) {
        snprintf(g_error, sizeof(g_error),
                 "bq_read_body_sdf: corps %d sans SDF local construit "
                 "(bq_build_body_sdf non appele)", body);
        return -1;
    }
    int ncell = b.res.x * b.res.y * b.res.z;
    if (dst != NULL)
        BQ_CUDA_CHECK(cudaMemcpy(dst, b.phi, (size_t)ncell * sizeof(float),
                                 cudaMemcpyDeviceToHost));
    if (res_out != NULL) { res_out[0] = b.res.x; res_out[1] = b.res.y; res_out[2] = b.res.z; }
    if (cell != NULL) *cell = b.cell;
    if (origin != NULL) { origin[0] = b.origin.x; origin[1] = b.origin.y; origin[2] = b.origin.z; }
    return ncell;
}

/* Kernel de diagnostic : un thread par point, cf. bq_query_body_sdf. */
__global__ void k_query_body_sdf(const BqBodySdf* __restrict__ table, int body,
                                 const float3* __restrict__ pts, int n,
                                 float* __restrict__ phi_out,
                                 float3* __restrict__ grad_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 g;
    float phi = body_sdf(table[body], pts[i], grad_out ? &g : nullptr);
    phi_out[i] = phi;
    if (grad_out) grad_out[i] = g;
}

BQ_API int bq_query_body_sdf(BqSim* s, int body, const float* pts_local, int n,
                             float* phi_out, float* grad_out) {
    if (body < 0 || body >= BQ_MAX_BODIES) {
        snprintf(g_error, sizeof(g_error),
                 "bq_query_body_sdf: corps %d hors de [0, %d[", body, BQ_MAX_BODIES);
        return -1;
    }
    if (n < 0) {
        snprintf(g_error, sizeof(g_error), "bq_query_body_sdf: n negatif (%d)", n);
        return -1;
    }
    if (n == 0) return 0;
    if (pts_local == NULL || phi_out == NULL) {
        snprintf(g_error, sizeof(g_error), "bq_query_body_sdf: pointeur nul (n=%d)", n);
        return -1;
    }
    float3* d_pts = nullptr; float* d_phi = nullptr; float3* d_grad = nullptr;
    BQ_CUDA_CHECK(cudaMalloc(&d_pts, (size_t)n * sizeof(float3)));
    BQ_CUDA_CHECK(cudaMemcpy(d_pts, pts_local, (size_t)n * 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMalloc(&d_phi, (size_t)n * sizeof(float)));
    if (grad_out != NULL) BQ_CUDA_CHECK(cudaMalloc(&d_grad, (size_t)n * sizeof(float3)));

    dim3 bp(256), gp((n + 255) / 256);
    k_query_body_sdf<<<gp, bp>>>(s->d_body_sdf_table, body, d_pts, n, d_phi, d_grad);
    if (cudaGetLastError() != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "bq_query_body_sdf: kernel a echoue (corps %d)", body);
        cudaFree(d_pts); cudaFree(d_phi); cudaFree(d_grad);
        return -1;
    }
    if (cudaMemcpy(phi_out, d_phi, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "bq_query_body_sdf: lecture phi echouee");
        cudaFree(d_pts); cudaFree(d_phi); cudaFree(d_grad);
        return -1;
    }
    if (grad_out != NULL &&
        cudaMemcpy(grad_out, d_grad, (size_t)n * 3 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "bq_query_body_sdf: lecture gradient echouee");
        cudaFree(d_pts); cudaFree(d_phi); cudaFree(d_grad);
        return -1;
    }
    cudaFree(d_pts); cudaFree(d_phi); cudaFree(d_grad);
    return n;
}

/* Diagnostic (M17, phase B, B3a) : copie les contacts du DERNIER sous-pas
   execute (cf. commentaire de k_gen_contacts -- remis a zero a chaque
   sous-pas, pas une somme sur la frame, meme discipline que
   bq_read_collider_wrench). Renvoie le nombre de contacts effectivement
   detectes ce sous-pas (peut depasser `max` -- seuls min(count, max) sont
   copies dans dst, qui doit alors pointer sur au moins max*9 floats : par
   contact (corps A, corps B, point[3], normale[3], profondeur), corps en
   entier stockes comme float. dst == NULL pour ne lire que le compte.
   Renvoie -1 sur erreur. */
BQ_API int bq_read_contacts(BqSim* s, float* dst, int max) {
    int h_count = 0;
    BQ_CUDA_CHECK(cudaMemcpy(&h_count, s->d_contact_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (dst == NULL || max <= 0 || h_count == 0) return h_count;
    int n_copy = (h_count < max) ? h_count : max;
    std::vector<BqContact> tmp((size_t)n_copy);
    BQ_CUDA_CHECK(cudaMemcpy(tmp.data(), s->d_contacts, (size_t)n_copy * sizeof(BqContact),
                             cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_copy; ++i) {
        float* o = dst + 9 * i;
        o[0] = (float)tmp[i].bodyA;
        o[1] = (float)tmp[i].bodyB;
        o[2] = tmp[i].point.x;  o[3] = tmp[i].point.y;  o[4] = tmp[i].point.z;
        o[5] = tmp[i].normal.x; o[6] = tmp[i].normal.y; o[7] = tmp[i].normal.z;
        o[8] = tmp[i].depth;
    }
    return h_count;
}

/* Diagnostic (M17, phase B, B3a) : le tampon de contacts a-t-il sature au
   DERNIER sous-pas execute (cf. BQ_MAX_CONTACTS) -- 1 si au moins un contact
   a ete refuse faute de place, 0 sinon. Meme motif que
   bq_whitewater_last_refused. */
BQ_API int bq_contacts_last_overflow(BqSim* s) {
    int h_overflow = 0;
    BQ_CUDA_CHECK(cudaMemcpy(&h_overflow, s->d_contact_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    return h_overflow;
}

/* Diagnostic (M17, phase B, B3b) : copie l'etat de sommeil courant des
   corps -- cf. section "mise en sommeil" de bourrasque.h. */
BQ_API int bq_read_body_sleep(BqSim* s, uint8_t* dst) {
    if (s->n_bodies == 0) return 0;
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_body_asleep, (size_t)s->n_bodies * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost));
    return s->n_bodies;
}

/* Norme de la vitesse par particule, tampon scratch pour la reduction max
 * (garde-fou dt, cf. commentaire sur prev_frame_max_speed dans BqSim). */
__global__ void k_velocity_norm(const float3* __restrict__ v,
                                 float* __restrict__ speed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 vi = v[i];
    speed[i] = sqrtf(vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
}

/* Reseeding (M10, T1-T3) : cf. section "reseeding" plus haut (kernels) pour
 * la justification et les references de production. Sequence complete :
 *   1. comptage par cellule (k_reseed_count) sur les n particules courantes ;
 *   2. plan par cellule (k_reseed_plan) : naissances a generer, probabilite
 *      de survie a la mort ;
 *   3. scan des naissances (cub) -> offsets + total ;
 *   4. marquage de survie par particule (k_reseed_mark_alive), scan (cub)
 *      -> offsets + nombre de survivantes ;
 *   5. compaction des survivantes vers le second jeu de tampons
 *      (k_reseed_compact_survivors) ;
 *   6. emission des naissances a la suite, plafonnee a la capacite restante
 *      (k_reseed_emit_births) -- exces omis silencieusement (D5) ;
 *   7. echange des tampons (ping-pong) et mise a jour de s->n.
 * Appelee UNE FOIS par bq_step, apres le dernier sous-pas (cf. bq_step). */
static int reseed(BqSim* s) {
    /* Commutateur de diagnostic (M12, investigation de la derive volume/masse
     * mesuree sur la nappe au repos) : desactive tout le reseeding sans
     * toucher a l'ABI publique -- pas un reglage utilisateur, sert
     * uniquement a isoler la contribution du reseeding d'une eventuelle
     * derive numerique independante de J. Lu une fois, mis en cache
     * statique (getenv n'est pas cher, mais pas de raison de le refaire a
     * chaque frame). */
    static int disabled = -1;
    if (disabled < 0) disabled = (getenv("BQ_DISABLE_RESEED") != nullptr) ? 1 : 0;
    if (disabled) return 0;

    int ncell = s->prm.res.x * s->prm.res.y * s->prm.res.z;
    int target = s->cfg.ppc_axis * s->cfg.ppc_axis * s->cfg.ppc_axis;
    dim3 bp(256), gp((s->n + 255) / 256), gc((ncell + 255) / 256);

    BQ_CUDA_CHECK(cudaMemset(s->d_reseed_count, 0, (size_t)ncell * sizeof(int)));
    BQ_CUDA_CHECK(cudaMemset(s->d_reseed_jsum, 0, (size_t)ncell * sizeof(float)));
    BQ_CUDA_CHECK(cudaMemset(s->d_reseed_first, 0xFF, (size_t)ncell * sizeof(int))); /* -1 */

    k_reseed_count<<<gp, bp>>>(s->d_x, s->d_J, s->d_reseed_count,
                               s->d_reseed_jsum, s->d_reseed_first, s->n);
    BQ_CUDA_CHECK(cudaGetLastError());

    BQ_CUDA_CHECK(cudaMemset(s->d_reseed_birth + ncell, 0, sizeof(int))); /* sentinelle scan */
    k_reseed_plan<<<gc, bp>>>(s->d_reseed_count, s->d_reseed_birth,
                              s->d_reseed_keepprob, target, ncell);
    BQ_CUDA_CHECK(cudaGetLastError());

    BQ_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        s->d_reseed_cub_tmp, s->reseed_cub_tmp_bytes, s->d_reseed_birth,
        s->d_reseed_birth_scan, ncell + 1));
    int total_births = 0;
    BQ_CUDA_CHECK(cudaMemcpy(&total_births, s->d_reseed_birth_scan + ncell,
                             sizeof(int), cudaMemcpyDeviceToHost));

    BQ_CUDA_CHECK(cudaMemset(s->d_reseed_alive + s->n, 0, sizeof(int))); /* sentinelle scan */
    k_reseed_mark_alive<<<gp, bp>>>(s->d_x, s->d_reseed_keepprob,
                                    s->d_reseed_alive, s->n);
    BQ_CUDA_CHECK(cudaGetLastError());

    BQ_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        s->d_reseed_cub_tmp, s->reseed_cub_tmp_bytes, s->d_reseed_alive,
        s->d_reseed_alive_scan, s->n + 1));
    int n_survivors = 0;
    BQ_CUDA_CHECK(cudaMemcpy(&n_survivors, s->d_reseed_alive_scan + s->n,
                             sizeof(int), cudaMemcpyDeviceToHost));

    k_reseed_compact_survivors<<<gp, bp>>>(
        s->d_x, s->d_v, s->d_C, s->d_F, s->d_J, s->d_mat,
        s->d_reseed_alive, s->d_reseed_alive_scan,
        s->d_x2, s->d_v2, s->d_C2, s->d_F2, s->d_J2, s->d_mat2, s->n);
    BQ_CUDA_CHECK(cudaGetLastError());

    int room = s->cfg.max_particles - n_survivors;
    if (room < 0) room = 0;
    int actually_births = (total_births < room) ? total_births : room;

    if (actually_births > 0) {
        k_reseed_emit_births<<<gc, bp>>>(
            s->d_reseed_count, s->d_reseed_first, s->d_reseed_jsum,
            s->d_reseed_birth, s->d_reseed_birth_scan,
            s->d_mat, s->d_F, s->d_grid,
            s->d_x2, s->d_v2, s->d_C2, s->d_F2, s->d_J2, s->d_mat2,
            n_survivors, actually_births, ncell);
        BQ_CUDA_CHECK(cudaGetLastError());
    }

    std::swap(s->d_x, s->d_x2); std::swap(s->d_v, s->d_v2);
    std::swap(s->d_C, s->d_C2); std::swap(s->d_F, s->d_F2);
    std::swap(s->d_J, s->d_J2); std::swap(s->d_mat, s->d_mat2);
    s->n = n_survivors + actually_births;
    return 0;
}

/* --------------------------------------------------------- tri spatial (perf)
 * k_p2g (scatter, atomicAdd vers ~27 cellules voisines par particule) et
 * k_g2p (gather depuis ces memes cellules) tournent `substeps` fois par
 * frame -- les deux kernels les plus chauds du solveur. Si les particules
 * voisines dans l'espace sont a des indices arbitrairement eloignes dans le
 * tableau (ce qui arrive naturellement : le reseeding ajoute/retire des
 * particules par cellule, l'advection deplace les particules d'une cellule a
 * l'autre au fil des substeps), les acces GPU sur ces deux kernels ne sont
 * plus coalescents. Motif observe dans les solveurs SPH (tri spatial des
 * particules avant le calcul de voisinage, pour la coherence de cache), ici
 * adapte au motif de compaction par flux DEJA etabli dans ce fichier
 * (comptage/scan/scatter, cf. reseed() ci-dessus) plutot que copie tel quel.
 *
 * Reordonne physiquement le tableau de particules par cellule de grille
 * (meme convention d'indexation que reseed_cell_index, deja utilisee par le
 * contact collider) : deux particules voisines dans l'espace se retrouvent
 * proches en indice. Frequence : une fois par frame, juste apres reseed()
 * (meme cadence et meme justification que le reseeding lui-meme -- refaire
 * ce tri a chaque sous-pas couterait plus cher que le gain qu'il procure).
 * Passe INDEPENDANTE et SUBSEQUENTE a reseed() : ne modifie ni sa logique de
 * naissance/mort, ni les kernels physiques (k_p2g/k_g2p/k_grid_update)
 * eux-memes -- seulement l'ordre memoire des particules. */

/* Comptage par cellule : un thread par particule, atomicAdd sur le compte de
 * sa cellule (meme convention que reseed_cell_index). count doit etre remis
 * a zero avant l'appel (cf. sort_particles()). */
__global__ void k_sort_count(const float3* __restrict__ x,
                             int* __restrict__ count, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    int idx = reseed_cell_index(x[p]);
    atomicAdd(&count[idx], 1);
}

/* Scatter : un thread par particule, retrouve sa cellule (meme appel
 * reseed_cell_index que k_sort_count), obtient un slot d'ecriture via
 * offset[cellule] + rang local (atomicAdd sur un curseur par cellule,
 * initialise a zero avant l'appel -- cf. sort_particles()), puis copie tous
 * les champs par particule vers le second jeu de tampons (*2) a cet indice.
 * Meme ensemble de champs et meme motif de copie que
 * k_reseed_compact_survivors. */
__global__ void k_sort_scatter(
    const float3* __restrict__ old_x, const float3* __restrict__ old_v,
    const float* __restrict__ old_C, const float* __restrict__ old_F,
    const float* __restrict__ old_J, const uint8_t* __restrict__ old_mat,
    const int* __restrict__ offset, int* __restrict__ cursor,
    float3* __restrict__ new_x, float3* __restrict__ new_v,
    float* __restrict__ new_C, float* __restrict__ new_F,
    float* __restrict__ new_J, uint8_t* __restrict__ new_mat, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    int cell = reseed_cell_index(old_x[p]);
    int slot = offset[cell] + atomicAdd(&cursor[cell], 1);
    new_x[slot] = old_x[p];
    new_v[slot] = old_v[p];
    for (int c9 = 0; c9 < 9; ++c9) {
        new_C[9 * slot + c9] = old_C[9 * p + c9];
        new_F[9 * slot + c9] = old_F[9 * p + c9];
    }
    new_J[slot] = old_J[p];
    new_mat[slot] = old_mat[p];
}

/* Sequence complete (meme motif que reseed(), cf. commentaire de section
 * ci-dessus) :
 *   1. comptage par cellule (k_sort_count) sur les n particules courantes ;
 *   2. scan exclusif (cub) -> offset de depart par cellule ;
 *   3. scatter (k_sort_scatter) vers le second jeu de tampons, curseur par
 *      cellule remis a zero juste avant ;
 *   4. echange des tampons (ping-pong) -- s->n est INCHANGE (aucune
 *      naissance/mort ici, seulement un reordonnement).
 * Appelee UNE FOIS par bq_step, juste apres reseed() (cf. bq_step). */
static int sort_particles(BqSim* s) {
    int ncell = s->prm.res.x * s->prm.res.y * s->prm.res.z;
    dim3 bp(256), gp((s->n + 255) / 256);

    BQ_CUDA_CHECK(cudaMemset(s->d_sort_count, 0, (size_t)ncell * sizeof(int)));
    k_sort_count<<<gp, bp>>>(s->d_x, s->d_sort_count, s->n);
    BQ_CUDA_CHECK(cudaGetLastError());

    BQ_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        s->d_sort_cub_tmp, s->sort_cub_tmp_bytes, s->d_sort_count,
        s->d_sort_offset, ncell));

    BQ_CUDA_CHECK(cudaMemset(s->d_sort_cursor, 0, (size_t)ncell * sizeof(int)));
    k_sort_scatter<<<gp, bp>>>(
        s->d_x, s->d_v, s->d_C, s->d_F, s->d_J, s->d_mat,
        s->d_sort_offset, s->d_sort_cursor,
        s->d_x2, s->d_v2, s->d_C2, s->d_F2, s->d_J2, s->d_mat2, s->n);
    BQ_CUDA_CHECK(cudaGetLastError());

    std::swap(s->d_x, s->d_x2); std::swap(s->d_v, s->d_v2);
    std::swap(s->d_C, s->d_C2); std::swap(s->d_F, s->d_F2);
    std::swap(s->d_J, s->d_J2); std::swap(s->d_mat, s->d_mat2);
    return 0;
}

BQ_API int bq_step(BqSim* s, float frame_dt) {
    /* M17/B3b, point 0 de la spec : une scene de SOLIDES SEULS (sans aucune
     * particule fluide) doit quand meme simuler -- sinon les corps ne
     * tombent jamais et le contact ne tourne jamais, ce qui condamne les
     * trois verifications de la porte de phase B specifiees sans fluide
     * (repos, empilement, energie). L'ancien garde-fou "s->n == 0 ||
     * s->n_mats == 0 => sortie immediate" bloquait exactement ce cas. Seul
     * un garde-fou "rien du tout a simuler" (ni fluide, ni corps) reste
     * legitime. */
    if (s->n == 0 && s->n_bodies == 0) return 0;
    bool has_fluid = (s->n > 0 && s->n_mats > 0);
    int substeps = (int)ceilf(frame_dt / s->dt);
    int ncell = s->prm.res.x * s->prm.res.y * s->prm.res.z;
    dim3 bp(256), gp((s->n + 255) / 256), gc((ncell + 255) / 256);

    /* Grille de lancement des noyaux par corps (D5, plan M17) : BQ_MAX_BODIES
     * est petit (64), un seul bloc suffit toujours -- pas de dependance a
     * s->n_bodies dans le dimensionnement pour eviter tout recalcul de
     * grille par sous-pas quand n_bodies change (il ne change pas en cours
     * de bake de toute facon, bq_set_collider_bodies n'est appelee qu'une
     * fois). */
    dim3 bpb(64), gpb((BQ_MAX_BODIES + 63) / 64);

    /* Detection de contact corps<->corps (M17, phase B, B3a) : dimensions de
     * lancement calculees UNE FOIS avant la boucle de sous-pas -- les
     * comptes d'echantillons par corps (s->body_samples_n) ne changent pas
     * en cours de bake (bq_set_body_samples n'est appele qu'au demarrage),
     * donc aucune raison de recalculer max_samples a chaque sous-pas. Pas de
     * synchronisation device introduite : body_samples_n est un tableau
     * HOTE deja a jour (cf. bq_set_body_samples). */
    int body_max_samples = 0;
    for (int b = 0; b < s->n_bodies; ++b)
        body_max_samples = (body_max_samples > s->body_samples_n[b]) ? body_max_samples : s->body_samples_n[b];
    dim3 bp_pairs(256);
    dim3 gp_pairs((unsigned int)(((size_t)s->n_bodies * s->n_bodies + 255) / 256));
    size_t contact_threads = (size_t)s->n_bodies * (size_t)s->n_bodies * (size_t)body_max_samples;
    dim3 bp_contacts(256);
    dim3 gp_contacts((unsigned int)((contact_threads + 255) / 256));

    for (int i = 0; i < substeps; ++i) {
        /* Ordre du sous-pas (D13 du plan, REVU par M17/A5 -- couplage
         * implicite -- puis par M17/B3b -- solveur de contact + sommeil) :
         *   [fluide] k_clear_grid -> k_p2g -> k_grid_apply_gravity
         *   -> k_body_predict (gele si endormi, cf. son commentaire)
         *   -> [fluide] k_grid_gather -> k_body_solve
         *   -> [fluide] k_grid_update (contact fluide, mur vif = etat RESOLU)
         *   -> broadphase/generation de contacts corps<->corps
         *   -> k_contact_solve (impulsions sequentielles, D11)
         *   -> k_body_sleep_update (D11)
         *   -> k_advance_bodies (position/orientation, canal SEPARE inclus)
         *   -> [fluide] k_g2p.
         *
         * "[fluide]" = noyau SAUTE (pas appele avec n=0) quand has_fluid est
         * faux -- M17/B3b, point 0 : une scene de corps rigides purs, sans
         * aucune particule, n'a besoin d'AUCUN de ces noyaux (ils operent
         * sur une grille qui resterait perimee/non pertinente sans jamais
         * etre reinitialisee). k_body_predict/k_contact_solve/
         * k_body_sleep_update/k_advance_bodies, EUX, tournent des que des
         * corps sont declares, fluide present ou non -- c'est exactement ce
         * qui fait tomber une caisse sur un sol sans qu'aucun fluide ne soit
         * emis. */
        if (s->n_bodies > 0) {
            cudaMemsetAsync(s->d_body_gather, 0, (size_t)s->n_bodies * 16 * sizeof(float));
            cudaMemsetAsync(s->d_body_wrench, 0, (size_t)s->n_bodies * 7 * sizeof(float));
        }
        if (has_fluid) {
            k_clear_grid<<<gc, bp>>>(s->d_grid, ncell);
            k_p2g<<<gp, bp>>>(s->d_x, s->d_v, s->d_C, s->d_F, s->d_J, s->d_mat,
                              s->d_grid, s->n);
            /* Meme calcul (v = mv/m, +gravite) que l'ancien bloc inline de
             * k_grid_update -- aucun changement de comportement pour le cas
             * fluide (D14, non-regression). */
            k_grid_apply_gravity<<<gc, bp>>>(s->d_grid, ncell);
        }
        if (s->n_bodies > 0) {
            k_body_predict<<<gpb, bpb>>>(s->d_bodies, s->d_body_asleep, s->n_bodies);
            if (has_fluid) {
                k_grid_gather<<<gc, bp>>>(s->d_grid, s->d_sdf, s->d_cnrm, s->d_cbody, s->d_bodies,
                                          s->d_body_gather, ncell);
                k_body_solve<<<gpb, bpb>>>(s->d_bodies, s->d_body_gather, s->d_body_wrench,
                                           s->n_bodies);
            }
        }
        if (has_fluid) {
            k_grid_update<<<gc, bp>>>(s->d_grid, s->d_sdf, s->d_cvel, s->d_cnrm,
                                      s->d_cbody, s->d_bodies, ncell);
        }
        if (s->n_bodies > 0) {
            /* Detection de contact corps<->corps (D10, B3a). Compteurs remis
             * a zero a CHAQUE sous-pas -- meme politique que
             * d_body_gather/d_body_wrench, aucune synchronisation hote
             * introduite. */
            cudaMemsetAsync(s->d_contact_count, 0, sizeof(int));
            cudaMemsetAsync(s->d_contact_overflow, 0, sizeof(int));
            k_body_world_aabb<<<gpb, bpb>>>(s->d_bodies, s->d_body_sdf_table,
                                            s->d_body_lo, s->d_body_hi, s->n_bodies);
            k_broadphase_pairs<<<gp_pairs, bp_pairs>>>(s->d_bodies, s->d_body_lo, s->d_body_hi,
                                                       s->d_pair_valid, s->n_bodies);
            if (body_max_samples > 0) {
                k_gen_contacts<<<gp_contacts, bp_contacts>>>(
                    s->d_bodies, s->d_body_sdf_table, s->d_body_samples_table,
                    s->d_body_samples_n_table, s->d_pair_valid,
                    s->n_bodies, body_max_samples,
                    s->d_contacts, s->d_contact_count, BQ_MAX_CONTACTS, s->d_contact_overflow);
            }
            /* Resolution (M17, B3b, D11) : impulsions sequentielles
             * (mono-bloc, warm starting, split impulse -- cf. commentaire
             * complet du noyau), puis mise en sommeil, puis avancee. */
            k_contact_solve<<<1, BQ_CONTACT_SOLVE_CAP>>>(
                s->d_bodies, s->d_contacts, s->d_contact_count,
                s->d_prev_contact_cache, s->d_prev_contact_count,
                s->d_body_pushout, s->n_bodies);
            k_body_sleep_update<<<gpb, bpb>>>(s->d_bodies, s->d_body_sleep_timer,
                                              s->d_body_asleep, s->n_bodies);
            k_advance_bodies<<<gpb, bpb>>>(s->d_bodies, s->d_body_pushout, s->n_bodies);
        }
        if (has_fluid) {
            k_g2p<<<gp, bp>>>(s->d_x, s->d_v, s->d_C, s->d_J, s->d_mat,
                              s->d_grid, s->d_sdf, s->d_cnrm,
                              s->d_tri, s->d_bucket_off, s->d_bucket_tri,
                              s->bucket_origin, s->bucket_h, s->bucket_res,
                              s->n_tri, s->n);
        }
    }
    BQ_CUDA_CHECK(cudaGetLastError());
    BQ_CUDA_CHECK(cudaDeviceSynchronize());

    if (has_fluid) {
        /* Plancher de dt sur la vitesse reelle (garde-fou CCD, cf.
         * commentaire sur prev_frame_max_speed dans BqSim) : reduction max
         * sur |v[p]| des particules simulees cette frame, consommee par
         * upload_params pour la frame suivante. Doit s'executer avant
         * reseed() : on veut la vitesse reelle produite par la physique de
         * cette frame, pas une eventuelle vitesse heritee d'une naissance. */
        k_velocity_norm<<<gp, bp>>>(s->d_v, s->d_speed, s->n);
        BQ_CUDA_CHECK(cudaGetLastError());
        BQ_CUDA_CHECK(cub::DeviceReduce::Max(s->d_speed_cub_tmp, s->speed_cub_tmp_bytes,
                                             s->d_speed, s->d_max_speed, s->n));
        BQ_CUDA_CHECK(cudaMemcpy(&s->prev_frame_max_speed, s->d_max_speed, sizeof(float),
                                 cudaMemcpyDeviceToHost));
    }
    if (upload_params(s) < 0) return -1;

    if (has_fluid) {
        /* Reseeding (M10) : une fois par frame, apres le dernier sous-pas --
         * jamais a chaque sous-pas (cf. reseed() et plan-milestone-10.md D5).
         * Saute sans fluide (M17/B3b, point 0) : rien a re-semer. */
        if (reseed(s) < 0) return -1;
        BQ_CUDA_CHECK(cudaGetLastError());

        /* Tri spatial (optimisation perf) : une fois par frame, juste apres
         * le reseeding -- meme cadence et meme justification, cf. section
         * "tri spatial" ci-dessus et sort_particles(). Passe independante,
         * ne change que l'ordre memoire des particules (s->n inchange). */
        if (sort_particles(s) < 0) return -1;
        BQ_CUDA_CHECK(cudaGetLastError());
        BQ_CUDA_CHECK(cudaDeviceSynchronize());
    }

    return substeps;
}

BQ_API int bq_particle_count(const BqSim* s) { return s->n; }

BQ_API int bq_read_positions(BqSim* s, float* dst) {
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_x, s->n * sizeof(float3),
                             cudaMemcpyDeviceToHost));
    return s->n;
}

BQ_API int bq_read_velocities(BqSim* s, float* dst) {
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_v, s->n * sizeof(float3),
                             cudaMemcpyDeviceToHost));
    return s->n;
}

BQ_API int bq_read_J(BqSim* s, float* dst) {
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_J, s->n * sizeof(float),
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

/* Copie le champ de normale de contact courant vers dst (ncell*4 floats,
 * meme indexation aux noeuds que bq_read_sdf : x,y,z normale unitaire,
 * w distance non signee). Diagnostic et validation, et transmission vers
 * whitewater via bq_whitewater_set_collider_cnrm. */
BQ_API int bq_read_cnrm(BqSim* s, float* dst) {
    int ncell = s->prm.res.x * s->prm.res.y * s->prm.res.z;
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_cnrm, ncell * sizeof(float4),
                             cudaMemcpyDeviceToHost));
    return ncell;
}

BQ_API const char* bq_last_error(void) { return g_error; }

} /* extern "C" */

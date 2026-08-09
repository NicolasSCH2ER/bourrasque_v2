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
 * de droite au solve porte deja une composante qui devrait etre nulle. */
__global__ void k_body_predict(BqRigidBody* __restrict__ bodies, int n_bodies) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_bodies) return;
    if (!bodies[b].dynamic) return;

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

/* Avancee des corps rigides (D5, plan M17), un thread par corps -- separe
 * de k_body_solve (et non fusionne) parce que la phase B du jalon inserera
 * le solveur de contact corps-corps entre les deux (cf. D13 du plan). Ne
 * fait rien si le corps est cinematique/statique. */
__global__ void k_advance_bodies(BqRigidBody* __restrict__ bodies, int n_bodies) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_bodies) return;
    if (!bodies[b].dynamic) return;

    float dt = c_p.dt;
    bodies[b].x[0] += dt * bodies[b].v[0];
    bodies[b].x[1] += dt * bodies[b].v[1];
    bodies[b].x[2] += dt * bodies[b].v[2];

    /* q += dt * 0.5 * quat(0, w) (x) q -- produit de Hamilton, w purement
     * imaginaire A GAUCHE. Convention (w, x, y, z), IMPERATIVE : le module
     * Python cote extension utilise deja exactement celle-ci. */
    float qw = bodies[b].q[0], qx = bodies[b].q[1], qy = bodies[b].q[2], qz = bodies[b].q[3];
    float wx = bodies[b].w[0], wy = bodies[b].w[1], wz = bodies[b].w[2];
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
 * C=0, F=identite : meme initialisation par defaut qu'une particule fraiche
 * de bq_emit_box (emit_particles). Masse : pas de champ par particule dans
 * ce solveur, deja derivee du materiau (MaterialGpu.p_mass) a chaque
 * substep -- rien a initialiser ici. */
__global__ void k_reseed_emit_births(
    const int* __restrict__ cell_count, const int* __restrict__ cell_first,
    const float* __restrict__ cell_jsum, const int* __restrict__ birth,
    const int* __restrict__ birth_scan, const uint8_t* __restrict__ old_mat,
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
            new_F[9 * idx + c9] = (c9 == 0 || c9 == 4 || c9 == 8) ? 1.f : 0.f;
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

static float material_sound_speed(const BqMaterial& m) {
    float stiff = (m.model == BQ_MODEL_ELASTIC) ? m.E : m.bulk;
    return sqrtf(stiff / m.rho);
}

static int upload_params(BqSim* s) {
    float c_max = 1e-3f;
    for (int i = 0; i < s->n_mats; ++i)
        c_max = fmaxf(c_max, material_sound_speed(s->mats_host[i]));
    /* plancher de vitesse reelle (garde-fou CCD, cf. commentaire sur
     * prev_frame_max_speed dans BqSim) : vaut 0 tant qu'aucune frame n'a ete
     * simulee, donc sans effet au demarrage. */
    c_max = fmaxf(c_max, s->prev_frame_max_speed);
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
        cudaMalloc(&s->d_sort_cursor, ncell * sizeof(int)) != cudaSuccess) {
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
    /* pas de collider au depart : sdf grand partout, vitesse/friction nulles */
    dim3 bp(256), gc((ncell + 255) / 256);
    k_fill_sdf<<<gc, bp>>>(s->d_sdf, s->d_cvel, s->d_cnrm, s->d_cbody, ncell);
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
    cudaFree(s->d_tri_body); cudaFree(s->d_cbody);
    cudaFree(s->d_bodies); cudaFree(s->d_body_wrench); cudaFree(s->d_body_gather);
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

BQ_API int bq_read_collider_wrench(BqSim* s, float* dst) {
    if (s->n_bodies == 0) return 0;
    BQ_CUDA_CHECK(cudaMemcpy(dst, s->d_body_wrench,
                             (size_t)s->n_bodies * 7 * sizeof(float),
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
            s->d_mat, s->d_grid,
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
    if (s->n == 0 || s->n_mats == 0) return 0;
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

    for (int i = 0; i < substeps; ++i) {
        /* Ordre du sous-pas (D13 du plan, REVU par M17/A5 -- couplage
         * implicite) :
         *   k_clear_grid -> k_p2g -> k_grid_apply_gravity -> k_body_predict
         *   -> k_grid_gather -> k_body_solve -> k_grid_update (contact, mur
         *   vif = etat RESOLU) -> k_advance_bodies -> k_g2p.
         *
         * k_grid_gather doit lire une grille qui porte deja "vitesse apres
         * gravite" (d'ou k_grid_apply_gravity avant), et k_grid_update doit
         * lire un etat de corps deja RESOLU par k_body_solve (d'ou son
         * deplacement apres, alors qu'il portait autrefois lui-meme
         * l'application de la gravite). Les accumulateurs de corps
         * (d_body_gather : cinq sommes de la recolte ; d_body_wrench :
         * wrench EFFECTIF de diagnostic) sont remis a zero a CHAQUE sous-pas
         * -- une somme sur toute la frame melangerait des geometries de
         * corps qui ont deja bouge d'un sous-pas a l'autre. cudaMemsetAsync,
         * aucune synchronisation hote introduite (contrainte de performance
         * existante de cette boucle). */
        if (s->n_bodies > 0) {
            cudaMemsetAsync(s->d_body_gather, 0, (size_t)s->n_bodies * 16 * sizeof(float));
            cudaMemsetAsync(s->d_body_wrench, 0, (size_t)s->n_bodies * 7 * sizeof(float));
        }
        k_clear_grid<<<gc, bp>>>(s->d_grid, ncell);
        k_p2g<<<gp, bp>>>(s->d_x, s->d_v, s->d_C, s->d_F, s->d_J, s->d_mat,
                          s->d_grid, s->n);
        /* Tourne INCONDITIONNELLEMENT (meme n_bodies == 0) : c'est le meme
         * calcul (v = mv/m, +gravite) que l'ancien bloc inline de
         * k_grid_update, scinde pour que la grille porte deja "vitesse
         * apres gravite" avant la recolte du couplage implicite -- aucun
         * changement de comportement pour ce cas (D14, non-regression). */
        k_grid_apply_gravity<<<gc, bp>>>(s->d_grid, ncell);
        if (s->n_bodies > 0) {
            k_body_predict<<<gpb, bpb>>>(s->d_bodies, s->n_bodies);
            k_grid_gather<<<gc, bp>>>(s->d_grid, s->d_sdf, s->d_cnrm, s->d_cbody, s->d_bodies,
                                      s->d_body_gather, ncell);
            k_body_solve<<<gpb, bpb>>>(s->d_bodies, s->d_body_gather, s->d_body_wrench,
                                       s->n_bodies);
        }
        k_grid_update<<<gc, bp>>>(s->d_grid, s->d_sdf, s->d_cvel, s->d_cnrm,
                                  s->d_cbody, s->d_bodies, ncell);
        if (s->n_bodies > 0) {
            /* phase B (plan M17) inserera ici k_contact_solve, entre le
             * couplage fluide et l'avancee des positions. */
            k_advance_bodies<<<gpb, bpb>>>(s->d_bodies, s->n_bodies);
        }
        k_g2p<<<gp, bp>>>(s->d_x, s->d_v, s->d_C, s->d_J, s->d_mat,
                          s->d_grid, s->d_sdf, s->d_cnrm,
                          s->d_tri, s->d_bucket_off, s->d_bucket_tri,
                          s->bucket_origin, s->bucket_h, s->bucket_res,
                          s->n_tri, s->n);
    }
    BQ_CUDA_CHECK(cudaGetLastError());
    BQ_CUDA_CHECK(cudaDeviceSynchronize());

    /* Plancher de dt sur la vitesse reelle (garde-fou CCD, cf. commentaire
     * sur prev_frame_max_speed dans BqSim) : reduction max sur |v[p]| des
     * particules simulees cette frame, consommee par upload_params pour la
     * frame suivante. Doit s'executer avant reseed() : on veut la vitesse
     * reelle produite par la physique de cette frame, pas une eventuelle
     * vitesse heritee d'une naissance. */
    k_velocity_norm<<<gp, bp>>>(s->d_v, s->d_speed, s->n);
    BQ_CUDA_CHECK(cudaGetLastError());
    BQ_CUDA_CHECK(cub::DeviceReduce::Max(s->d_speed_cub_tmp, s->speed_cub_tmp_bytes,
                                         s->d_speed, s->d_max_speed, s->n));
    BQ_CUDA_CHECK(cudaMemcpy(&s->prev_frame_max_speed, s->d_max_speed, sizeof(float),
                             cudaMemcpyDeviceToHost));
    if (upload_params(s) < 0) return -1;

    /* Reseeding (M10) : une fois par frame, apres le dernier sous-pas --
     * jamais a chaque sous-pas (cf. reseed() et plan-milestone-10.md D5). */
    if (reseed(s) < 0) return -1;
    BQ_CUDA_CHECK(cudaGetLastError());

    /* Tri spatial (optimisation perf) : une fois par frame, juste apres le
     * reseeding -- meme cadence et meme justification, cf. section "tri
     * spatial" ci-dessus et sort_particles(). Passe independante, ne change
     * que l'ordre memoire des particules (s->n inchange). */
    if (sort_particles(s) < 0) return -1;
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

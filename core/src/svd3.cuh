/* svd3.cuh -- SVD 3x3 sur device (M18/S1).
 *
 * Fichier separe de mlsmpm.cu (et pas du code ajoute dedans) pour une raison
 * precise : le harnais de validation (tools/repro/svd3_harness.cu) doit
 * pouvoir compiler cette fonction SANS lier tout le solveur.
 *
 * mat3 et ses operateurs vivaient jusqu'ici dans mlsmpm.cu ("petite algebre",
 * juste apres les includes). Ils sont deplaces ICI, definition UNIQUE :
 * mlsmpm.cu n'en garde aucune copie, il inclut ce header au meme endroit
 * pour les obtenir. Le harnais inclut ce meme fichier et n'a donc jamais
 * deux definitions de mat3 a synchroniser a la main -- c'est le piege que
 * ce choix evite (cf. plan-milestone-18.md, D1).
 */
#ifndef BOURRASQUE_SVD3_CUH
#define BOURRASQUE_SVD3_CUH

#include <cuda_runtime.h>
#include <cmath>

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

/* ---------------------------------------------------- petits helpers vecteur */
/* Prefixes svd3_ pour ne jamais entrer en collision avec vsub/vcross/vdot,
 * definis plus loin dans mlsmpm.cu pour la CCD (meme fichier, portee globale
 * une fois ce header inclus). */
__device__ inline float3 svd3_vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 svd3_vscale(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ inline float svd3_vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float3 svd3_vcross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
__device__ inline float svd3_vlen(float3 a) {
    return sqrtf(svd3_vdot(a, a));
}

/* --------------------------------------------------- diagonalisation Jacobi */
/* Diagonalisation d'une matrice symetrique 3x3 par Jacobi cyclique
 * (rotations de Givens successives sur les paires (0,1), (0,2), (1,2)).
 * Golub & Van Loan, "Matrix Computations", 8.4.3.
 *
 * Sortie : d[0..2] les valeurs propres (ordre non trie -- correspond a
 * l'ordre des colonnes de V en sortie), et V la rotation des vecteurs
 * propres, V^T A V = diag(d).
 *
 * V est TOUJOURS une rotation propre (det(V) = +1) : chaque rotation de
 * Givens appliquee a un determinant de +1, et le produit de rotations de
 * determinant +1 reste de determinant +1. Ceci reste vrai meme si deux
 * valeurs propres sont egales ou si A est nulle -- Jacobi ne "s'emmele" pas
 * sur les valeurs propres degenerees, contrairement a certains algorithmes
 * bases sur le polynome caracteristique.
 *
 * Nombre de sweeps fixe (MAX_SWEEPS) plutot qu'une boucle jusqu'a
 * convergence stricte : plus previsible sur device (pas de boucle a duree
 * variable par thread au sein d'un warp), et trois a quatre sweeps suffisent
 * deja en pratique pour une matrice 3x3 en simple precision. Une sortie
 * anticipee reste faite des que la somme des elements hors-diagonale devient
 * negligeable.
 */
__device__ inline void svd3_jacobi_eigen_sym3(float a00, float a01, float a02,
                                              float a11, float a12, float a22,
                                              mat3& V, float d[3]) {
    V = mat3::identity();
    float A[3][3] = {{a00, a01, a02}, {a01, a11, a12}, {a02, a12, a22}};

    const int MAX_SWEEPS = 12;
    const int PAIRS[3][2] = {{0, 1}, {0, 2}, {1, 2}};

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        float off = fabsf(A[0][1]) + fabsf(A[0][2]) + fabsf(A[1][2]);
        if (off < 1e-13f) break;

        for (int pi = 0; pi < 3; ++pi) {
            int p = PAIRS[pi][0], q = PAIRS[pi][1];
            float apq = A[p][q];
            if (fabsf(apq) < 1e-20f) continue;

            float theta = (A[q][q] - A[p][p]) / (2.f * apq);
            float t = copysignf(1.f, theta) / (fabsf(theta) + sqrtf(1.f + theta * theta));
            float c = 1.f / sqrtf(1.f + t * t);
            float s = t * c;

            float app = A[p][p], aqq = A[q][q];
            A[p][p] = app - t * apq;
            A[q][q] = aqq + t * apq;
            A[p][q] = A[q][p] = 0.f;

            int k = 3 - p - q; /* le troisieme indice */
            float akp = A[k][p], akq = A[k][q];
            A[k][p] = A[p][k] = c * akp - s * akq;
            A[k][q] = A[q][k] = s * akp + c * akq;

            for (int i = 0; i < 3; ++i) {
                float vip = V.m[3 * i + p], viq = V.m[3 * i + q];
                V.m[3 * i + p] = c * vip - s * viq;
                V.m[3 * i + q] = s * vip + c * viq;
            }
        }
    }

    d[0] = A[0][0]; d[1] = A[1][1]; d[2] = A[2][2];
}

/* ------------------------------------------------------------------- svd3 */
/*
 * svd3 -- decomposition en valeurs singulieres d'une matrice 3x3 F,
 * F = U * diag(S) * V^T, sur device.
 *
 * METHODE : eigen-decomposition symetrique + reconstruction, pas McAdams et
 * al. 2011. Choix motive par la robustesse directe aux cas degeneres
 * (valeurs singulieres quasi egales, matrice quasi singuliere) plutot que
 * par la performance brute -- acceptable ici, svd3 n'est appelee qu'une
 * fois par particule et par sous-pas (return mapping de Drucker-Prager,
 * futur consommateur).
 *
 *   1. A = F^T F, symetrique semi-definie positive (ses valeurs propres
 *      sont les carres des valeurs singulieres de F, ses vecteurs propres
 *      sont les colonnes de V).
 *   2. Diagonalisation de A par Jacobi cyclique (svd3_jacobi_eigen_sym3) :
 *      A = V D V^T, V rotation propre, D >= 0 aux erreurs d'arrondi pres.
 *   3. Tri des colonnes de V / valeurs de D par magnitude decroissante. Un
 *      tri est une permutation, qui peut inverser le determinant de V (un
 *      echange de deux colonnes le fait) : si c'est le cas on neutralise en
 *      inversant le signe de la colonne de plus petite valeur propre --
 *      choix arbitraire et sans consequence, le signe d'un vecteur propre
 *      n'est pas observable.
 *   4. sigma_i = sqrt(max(D_i, 0)), toujours >= 0 a ce stade.
 *   5. U_i = F V_i normalise, pour les i ou sigma_i est significatif. Ces
 *      colonnes sont automatiquement orthonormees entre elles : (F V_i) .
 *      (F V_j) = V_i^T A V_j = D_j (V_i . V_j) = 0 pour i != j -- propriete
 *      des vecteurs propres, VRAIE MEME quand deux valeurs propres sont
 *      egales (rotation indeterminee : n'importe quelle base orthonormee du
 *      sous-espace propre convient, la reconstruction reste exacte). Un
 *      Gram-Schmidt est quand meme applique pour la robustesse numerique
 *      pres d'une degenerescence. Quand sigma_i est proche de 0 (F quasi
 *      singuliere sur cet axe), F V_i ne donne aucune direction fiable : la
 *      colonne manquante est completee par produit vectoriel a partir des
 *      colonnes deja connues (ou une base canonique arbitraire si F = 0).
 *   6. CONVENTION DE SIGNE : a ce stade U et V sont deux rotations propres
 *      ET S >= 0 -- mais alors U diag(S) V^T ne peut PAS reproduire un F de
 *      determinant negatif (une reflexion). On verifie donc det(U) : s'il
 *      vaut -1, on l'annule en inversant le signe de la colonne de plus
 *      petite MAGNITUDE de U (la troisieme, S est triee) et de la valeur
 *      singuliere correspondante (S.z). C'est "la reflexion absorbee par la
 *      plus petite valeur singuliere", jamais par det(U) = -1 : le
 *      consommateur (return mapping, log des valeurs singulieres) peut donc
 *      toujours supposer U et V rotations pures.
 *
 * SORTIE :
 *   - U, V : ROTATIONS (det(U) = det(V) = +1), jamais des reflexions.
 *   - S : valeurs singulieres triees par MAGNITUDE decroissante,
 *     |S.x| >= |S.y| >= |S.z|. S.z PEUT ETRE NEGATIVE OU NULLE (matrice
 *     source quasi singuliere ou reflexion) -- c'est a L'APPELANT de
 *     clamper avant un log(), cette fonction ne cache jamais une inversion
 *     reelle.
 *   - F = U * diag(S) * V^T, a la precision float32 pres (voir le harnais
 *     tools/repro/svd3_harness.cu pour les bornes chiffrees).
 */
__device__ inline void svd3(const mat3& F, mat3& U, float3& S, mat3& V) {
    mat3 A = matmul(transpose(F), F);
    float d[3];
    mat3 Vraw;
    svd3_jacobi_eigen_sym3(A.m[0], A.m[1], A.m[2], A.m[4], A.m[5], A.m[8], Vraw, d);

    /* tri par magnitude decroissante des valeurs propres (insertion, 3
     * elements) */
    int idx[3] = {0, 1, 2};
    for (int i = 1; i < 3; ++i) {
        int k = idx[i];
        float dk = d[k];
        int j = i - 1;
        while (j >= 0 && d[idx[j]] < dk) { idx[j + 1] = idx[j]; --j; }
        idx[j + 1] = k;
    }

    mat3 Vs;
    float Ds[3];
    for (int i = 0; i < 3; ++i) {
        int c = idx[i];
        Vs.m[0 * 3 + i] = Vraw.m[0 * 3 + c];
        Vs.m[1 * 3 + i] = Vraw.m[1 * 3 + c];
        Vs.m[2 * 3 + i] = Vraw.m[2 * 3 + c];
        Ds[i] = fmaxf(d[c], 0.f);
    }
    /* permutation impaire -> det(Vs) = -1 : on la restaure en inversant la
     * colonne de plus petite valeur propre (indice 2, deja triee) */
    if (det(Vs) < 0.f) {
        Vs.m[2] = -Vs.m[2]; Vs.m[5] = -Vs.m[5]; Vs.m[8] = -Vs.m[8];
    }
    V = Vs;

    float sigma[3] = { sqrtf(Ds[0]), sqrtf(Ds[1]), sqrtf(Ds[2]) };
    float thresh = fmaxf(sigma[0] * 1e-6f, 1e-12f);

    float3 v0 = make_float3(V.m[0], V.m[3], V.m[6]);
    float3 v1 = make_float3(V.m[1], V.m[4], V.m[7]);
    float3 v2 = make_float3(V.m[2], V.m[5], V.m[8]);
    float3 fv[3] = { matvec(F, v0), matvec(F, v1), matvec(F, v2) };

    float3 u[3];
    bool have[3] = { false, false, false };
    for (int i = 0; i < 3; ++i) {
        float3 c = fv[i];
        for (int j = 0; j < i; ++j) {
            if (!have[j]) continue;
            float pdot = svd3_vdot(c, u[j]);
            c = svd3_vsub(c, svd3_vscale(u[j], pdot));
        }
        float len = svd3_vlen(c);
        if (len > thresh) { u[i] = svd3_vscale(c, 1.f / len); have[i] = true; }
    }

    /* completion des colonnes non fiables (sigma quasi nulle) par produit
     * vectoriel a partir des colonnes deja connues, ou base canonique si F
     * est entierement degeneree sur cet axe. */
    if (!have[0] && !have[1] && !have[2]) {
        u[0] = make_float3(1.f, 0.f, 0.f);
        u[1] = make_float3(0.f, 1.f, 0.f);
        u[2] = make_float3(0.f, 0.f, 1.f);
    } else if (have[0] && !have[1] && !have[2]) {
        float3 seed = (fabsf(u[0].x) < 0.9f) ? make_float3(1.f, 0.f, 0.f) : make_float3(0.f, 1.f, 0.f);
        float3 t = svd3_vsub(seed, svd3_vscale(u[0], svd3_vdot(seed, u[0])));
        u[1] = svd3_vscale(t, 1.f / svd3_vlen(t));
        u[2] = svd3_vcross(u[0], u[1]);
    } else if (have[0] && have[1] && !have[2]) {
        u[2] = svd3_vcross(u[0], u[1]);
    } else if (!have[0] && have[1] && have[2]) {
        u[0] = svd3_vcross(u[1], u[2]);
    } else if (have[0] && !have[1] && have[2]) {
        u[1] = svd3_vcross(u[2], u[0]);
    } else if (!have[0] && !have[1] && have[2]) {
        float3 seed = (fabsf(u[2].x) < 0.9f) ? make_float3(1.f, 0.f, 0.f) : make_float3(0.f, 1.f, 0.f);
        float3 t = svd3_vsub(seed, svd3_vscale(u[2], svd3_vdot(seed, u[2])));
        u[0] = svd3_vscale(t, 1.f / svd3_vlen(t));
        u[1] = svd3_vcross(u[2], u[0]);
    }
    /* have[0] && have[1] && have[2] : rien a completer */

    U.m[0] = u[0].x; U.m[3] = u[0].y; U.m[6] = u[0].z;
    U.m[1] = u[1].x; U.m[4] = u[1].y; U.m[7] = u[1].z;
    U.m[2] = u[2].x; U.m[5] = u[2].y; U.m[8] = u[2].z;

    /* convention de signe : la reflexion eventuelle de F est absorbee par
     * S.z negative, jamais par det(U) = -1 (cf. commentaire de tete) */
    if (det(U) < 0.f) {
        U.m[2] = -U.m[2]; U.m[5] = -U.m[5]; U.m[8] = -U.m[8];
        sigma[2] = -sigma[2];
    }

    S = make_float3(sigma[0], sigma[1], sigma[2]);
}

#endif /* BOURRASQUE_SVD3_CUH */

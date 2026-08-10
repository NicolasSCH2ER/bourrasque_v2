/* svd3_harness.cu -- validation autonome de core/src/svd3.cuh (M18/S1).
 *
 * Compile SEUL (pas via CMake, pas de lien avec le solveur) : c'est
 * precisement pour permettre ca que svd3.cuh est un header separe de
 * mlsmpm.cu. Compilation :
 *
 *   nvcc -O2 -arch=native -std=c++17 svd3_harness.cu -o svd3_harness.exe
 *
 * Genere >= 5000 matrices 3x3 couvrant les cas connus pour casser une SVD
 * naive (quasi singuliere, valeurs singulieres quasi egales, determinant
 * negatif, matrice diagonale, identite, matrice nulle, rotation pure,
 * perturbation de l'identite), evalue svd3 sur GPU, et rapporte les
 * erreurs MAXIMALES (jamais moyennes -- c'est le pire cas qui casse une
 * simulation). Ecrit aussi un CSV (F + S) pour verification independante
 * via numpy.linalg.svd (svd3_compare_numpy.py).
 */
#include <cuda_runtime.h>
#include "../../core/src/svd3.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

/* ------------------------------------------------------- algebre 3x3 hote */
/* Volontairement independante de mat3/matmul (qui sont __device__ only) :
 * ce sont de simples tableaux de 9 float, row-major, meme convention que
 * mat3. Sert uniquement a fabriquer les matrices de test sur l'hote. */
static void h_matmul(const float a[9], const float b[9], float out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float s = 0.f;
            for (int k = 0; k < 3; ++k) s += a[3 * i + k] * b[3 * k + j];
            out[3 * i + j] = s;
        }
}
static void h_transpose(const float a[9], float out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[3 * i + j] = a[3 * j + i];
}
static void h_identity(float out[9]) {
    for (int i = 0; i < 9; ++i) out[i] = 0.f;
    out[0] = out[4] = out[8] = 1.f;
}
static float h_det(const float a[9]) {
    return a[0] * (a[4] * a[8] - a[5] * a[7])
         - a[1] * (a[3] * a[8] - a[5] * a[6])
         + a[2] * (a[3] * a[7] - a[4] * a[6]);
}

/* rotation aleatoire via Rodrigues (axe unitaire aleatoire, angle uniforme) */
static void h_random_rotation(std::mt19937& rng, float out[9]) {
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    float z = 2.f * u01(rng) - 1.f;
    float phi = 2.f * 3.14159265358979f * u01(rng);
    float r = sqrtf(fmaxf(0.f, 1.f - z * z));
    float ax = r * cosf(phi), ay = r * sinf(phi), az = z;
    float theta = 2.f * 3.14159265358979f * u01(rng);
    float c = cosf(theta), s = sinf(theta), t = 1.f - c;
    out[0] = t * ax * ax + c;       out[1] = t * ax * ay - s * az;   out[2] = t * ax * az + s * ay;
    out[3] = t * ax * ay + s * az;  out[4] = t * ay * ay + c;        out[5] = t * ay * az - s * ax;
    out[6] = t * ax * az - s * ay;  out[7] = t * ay * az + s * ax;   out[8] = t * az * az + c;
}

static void h_from_svd(const float U[9], float s0, float s1, float s2, const float V[9], float out[9]) {
    float S[9] = {0}; S[0] = s0; S[4] = s1; S[8] = s2;
    float tmp[9], Vt[9];
    h_matmul(U, S, tmp);
    h_transpose(V, Vt);
    h_matmul(tmp, Vt, out);
}

/* ------------------------------------------------------------------ kernel */
struct TestResult {
    float err_abs, err_rel;
    float ortho_u, ortho_v;
    float det_u, det_v;
    int sorted_ok;
    int has_nan;
    float sx, sy, sz;
};

__device__ inline float frob9(const float a[9]) {
    float s = 0.f; for (int i = 0; i < 9; ++i) s += a[i] * a[i]; return sqrtf(s);
}

__global__ void k_test_svd3(const float* __restrict__ Fin, int n, TestResult* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    mat3 F; memcpy(F.m, Fin + 9 * i, 9 * sizeof(float));
    mat3 U, V; float3 S;
    svd3(F, U, S, V);

    /* reconstruction : U * diag(S) * V^T */
    mat3 diagS = mat3::zero();
    diagS.m[0] = S.x; diagS.m[4] = S.y; diagS.m[8] = S.z;
    mat3 recon = matmul(matmul(U, diagS), transpose(V));
    float diff[9];
    for (int k = 0; k < 9; ++k) diff[k] = recon.m[k] - F.m[k];
    float err_abs = frob9(diff);
    float fF = frob9(F.m);
    float err_rel = err_abs / fmaxf(fF, 1e-8f);

    mat3 UtU = matmul(transpose(U), U);
    mat3 VtV = matmul(transpose(V), V);
    float du[9], dv[9];
    float ident[9] = {1,0,0, 0,1,0, 0,0,1};
    for (int k = 0; k < 9; ++k) { du[k] = UtU.m[k] - ident[k]; dv[k] = VtV.m[k] - ident[k]; }
    float ortho_u = frob9(du);
    float ortho_v = frob9(dv);

    float det_u = det(U);
    float det_v = det(V);

    float ax = fabsf(S.x), ay = fabsf(S.y), az = fabsf(S.z);
    float tol = 1e-5f * fmaxf(ax, 1e-6f);
    int sorted_ok = (ax + tol >= ay) && (ay + tol >= az) ? 1 : 0;

    int has_nan = 0;
    float allv[15] = { U.m[0],U.m[1],U.m[2],U.m[3],U.m[4],U.m[5],U.m[6],U.m[7],U.m[8],
                        V.m[0], S.x, S.y, S.z, det_u, det_v };
    for (int k = 0; k < 15; ++k) if (!isfinite(allv[k])) has_nan = 1;

    TestResult r;
    r.err_abs = err_abs; r.err_rel = err_rel;
    r.ortho_u = ortho_u; r.ortho_v = ortho_v;
    r.det_u = det_u; r.det_v = det_v;
    r.sorted_ok = sorted_ok; r.has_nan = has_nan;
    r.sx = S.x; r.sy = S.y; r.sz = S.z;
    out[i] = r;
}

/* --------------------------------------------------------------- scenario */
struct Case { std::string label; std::vector<float> F; }; /* F : n*9 floats */

static void push_mat(std::vector<float>& buf, const float m[9]) {
    for (int i = 0; i < 9; ++i) buf.push_back(m[i]);
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u2(-2.f, 2.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    std::vector<float> F;   /* n*9, matrices de test */
    std::vector<int> cat;   /* categorie de chaque matrice, pour le rapport */
    std::vector<std::string> cat_names;
    auto add_cat = [&](const char* name) { cat_names.push_back(name); return (int)cat_names.size() - 1; };

    int C_GENERAL   = add_cat("general_aleatoire");
    int C_QSING     = add_cat("quasi_singuliere");
    int C_TWOEQ     = add_cat("deux_valeurs_quasi_egales");
    int C_THREEEQ   = add_cat("trois_valeurs_quasi_egales");
    int C_NEGDET    = add_cat("determinant_negatif");
    int C_DIAGPOS   = add_cat("diagonale_positive");
    int C_DIAGNEG   = add_cat("diagonale_negative");
    int C_IDENT     = add_cat("identite");
    int C_ZERO      = add_cat("nulle");
    int C_ROT       = add_cat("rotation_pure");
    int C_NEARI     = add_cat("proche_identite");

    auto add_matrix = [&](const float m[9], int c) { push_mat(F, m); cat.push_back(c); };

    /* general aleatoire */
    for (int i = 0; i < 4000; ++i) {
        float m[9]; for (int k = 0; k < 9; ++k) m[k] = u2(rng);
        add_matrix(m, C_GENERAL);
    }
    /* quasi singuliere : une valeur singuliere ~1e-7 */
    for (int i = 0; i < 200; ++i) {
        float U[9], V[9], m[9];
        h_random_rotation(rng, U); h_random_rotation(rng, V);
        float s0 = 0.5f + u01(rng), s1 = 0.3f + 0.5f * u01(rng), s2 = 1e-7f * (0.5f + u01(rng));
        h_from_svd(U, s0, s1, s2, V, m);
        add_matrix(m, C_QSING);
    }
    /* deux valeurs singulieres quasi egales : rotation indeterminee */
    for (int i = 0; i < 200; ++i) {
        float U[9], V[9], m[9];
        h_random_rotation(rng, U); h_random_rotation(rng, V);
        float s0 = 1.f, s1 = 1.f + 1e-6f * (u01(rng) - 0.5f), s2 = 0.2f + 0.3f * u01(rng);
        h_from_svd(U, s0, s1, s2, V, m);
        add_matrix(m, C_TWOEQ);
    }
    /* trois valeurs quasi egales : quasi-homothetie */
    for (int i = 0; i < 200; ++i) {
        float U[9], V[9], m[9];
        h_random_rotation(rng, U); h_random_rotation(rng, V);
        float base = 0.5f + u01(rng);
        float s0 = base, s1 = base + 1e-6f * (u01(rng) - 0.5f), s2 = base + 1e-6f * (u01(rng) - 0.5f);
        h_from_svd(U, s0, s1, s2, V, m);
        add_matrix(m, C_THREEEQ);
    }
    /* determinant negatif : reflexion */
    for (int i = 0; i < 200; ++i) {
        float m[9]; for (int k = 0; k < 9; ++k) m[k] = u2(rng);
        if (h_det(m) > 0.f) { m[6] = -m[6]; m[7] = -m[7]; m[8] = -m[8]; } /* force det < 0 */
        add_matrix(m, C_NEGDET);
    }
    /* diagonale, entrees positives */
    for (int i = 0; i < 50; ++i) {
        float m[9] = {0}; m[0] = 0.1f + 3.f * u01(rng); m[4] = 0.1f + 3.f * u01(rng); m[8] = 0.1f + 3.f * u01(rng);
        add_matrix(m, C_DIAGPOS);
    }
    /* diagonale, entrees negatives */
    for (int i = 0; i < 50; ++i) {
        float m[9] = {0}; m[0] = -(0.1f + 3.f * u01(rng)); m[4] = -(0.1f + 3.f * u01(rng)); m[8] = (0.1f + 3.f * u01(rng));
        add_matrix(m, C_DIAGNEG);
    }
    /* identite */
    { float m[9]; h_identity(m); add_matrix(m, C_IDENT); }
    /* nulle */
    { float m[9] = {0,0,0,0,0,0,0,0,0}; add_matrix(m, C_ZERO); }
    /* rotation pure : S = (1,1,1) */
    for (int i = 0; i < 200; ++i) {
        float m[9]; h_random_rotation(rng, m);
        add_matrix(m, C_ROT);
    }
    /* proche de l'identite : F = I + petite perturbation */
    for (int i = 0; i < 300; ++i) {
        float m[9]; h_identity(m);
        float eps = 1e-4f + 1e-2f * u01(rng);
        for (int k = 0; k < 9; ++k) m[k] += eps * (u2(rng) * 0.5f);
        add_matrix(m, C_NEARI);
    }

    int n = (int)cat.size();
    std::printf("svd3_harness : %d matrices de test\n", n);

    float* d_F; TestResult* d_out;
    cudaMalloc(&d_F, sizeof(float) * 9 * n);
    cudaMalloc(&d_out, sizeof(TestResult) * n);
    cudaMemcpy(d_F, F.data(), sizeof(float) * 9 * n, cudaMemcpyHostToDevice);

    int threads = 128, blocks = (n + threads - 1) / threads;
    k_test_svd3<<<blocks, threads>>>(d_F, n, d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "erreur cuda : %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<TestResult> res(n);
    cudaMemcpy(res.data(), d_out, sizeof(TestResult) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_F); cudaFree(d_out);

    /* -------------------------------------------------------- agregation */
    float max_err_abs = 0.f, max_err_rel = 0.f, max_ortho_u = 0.f, max_ortho_v = 0.f;
    float min_det_u = 1e30f, min_det_v = 1e30f;
    int n_sorted_bad = 0, n_nan = 0;
    for (int i = 0; i < n; ++i) {
        const TestResult& r = res[i];
        if (r.err_abs > max_err_abs) max_err_abs = r.err_abs;
        if (r.err_rel > max_err_rel) max_err_rel = r.err_rel;
        if (r.ortho_u > max_ortho_u) max_ortho_u = r.ortho_u;
        if (r.ortho_v > max_ortho_v) max_ortho_v = r.ortho_v;
        if (r.det_u < min_det_u) min_det_u = r.det_u;
        if (r.det_v < min_det_v) min_det_v = r.det_v;
        if (!r.sorted_ok) n_sorted_bad++;
        if (r.has_nan) n_nan++;
    }

    std::printf("\n--- criteres d'acceptation (S1, plan-milestone-18.md D1) ---\n");
    std::printf("1. max||U S V^T - F||_F (Frobenius)      : abs = %.6e   relatif a ||F|| = %.6e\n", max_err_abs, max_err_rel);
    std::printf("2. orthogonalite                          : max||U^T U - I|| = %.6e   max||V^T V - I|| = %.6e\n", max_ortho_u, max_ortho_v);
    std::printf("3. determinant (doit valoir +1, jamais -1): min det(U) = %.6f   min det(V) = %.6f\n", min_det_u, min_det_v);
    std::printf("4. tri |Sx|>=|Sy|>=|Sz|                    : %d/%d cas en echec\n", n_sorted_bad, n);
    std::printf("5. NaN/Inf                                 : %d/%d cas\n", n_nan, n);

    /* pire cas par categorie, pour le rapport */
    std::printf("\n--- pire err_abs par categorie ---\n");
    for (size_t c = 0; c < cat_names.size(); ++c) {
        float worst = 0.f; int worst_i = -1;
        for (int i = 0; i < n; ++i) if (cat[i] == (int)c && res[i].err_abs > worst) { worst = res[i].err_abs; worst_i = i; }
        std::printf("  %-28s : max err_abs = %.6e (n=%d)\n", cat_names[c].c_str(), worst,
                    (int)std::count(cat.begin(), cat.end(), (int)c));
        (void)worst_i;
    }

    /* ------------------------------------------------------------- CSV */
    const char* csv_path = "svd3_harness_out.csv";
    FILE* f = std::fopen(csv_path, "w");
    if (!f) { std::fprintf(stderr, "impossible d'ecrire %s\n", csv_path); return 1; }
    std::fprintf(f, "cat,F0,F1,F2,F3,F4,F5,F6,F7,F8,Sx,Sy,Sz\n");
    for (int i = 0; i < n; ++i) {
        std::fprintf(f, "%s", cat_names[cat[i]].c_str());
        for (int k = 0; k < 9; ++k) std::fprintf(f, ",%.9g", F[9 * i + k]);
        std::fprintf(f, ",%.9g,%.9g,%.9g\n", res[i].sx, res[i].sy, res[i].sz);
    }
    std::fclose(f);
    std::printf("\nCSV ecrit : %s (%d lignes)\n", csv_path, n);

    return 0;
}

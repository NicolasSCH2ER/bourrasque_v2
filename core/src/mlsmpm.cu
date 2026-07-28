/* mlsmpm.cu — solveur MLS-MPM 3D (Hu et al. 2018), B-splines quadratiques.
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
    int   res;
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

    /* affine = (-dt vol 4/dx^2) sigma + m C  — cf. reference NumPy */
    float coeff = -c_p.dt * c_p.p_vol * 4.f * c_p.inv_dx * c_p.inv_dx;
    mat3 affine = coeff * stress + m.p_mass * C;

    float3 vp = v[p];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                float3 dpos = make_float3((i - fx.x) * c_p.dx,
                                          (j - fx.y) * c_p.dx,
                                          (k - fx.z) * c_p.dx);
                float weight = w[i][0] * w[j][1] * w[k][2];
                float3 mom = matvec(affine, dpos);
                mom.x = weight * (m.p_mass * vp.x + mom.x);
                mom.y = weight * (m.p_mass * vp.y + mom.y);
                mom.z = weight * (m.p_mass * vp.z + mom.z);
                int idx = ((base.x + i) * c_p.res + (base.y + j)) * c_p.res
                          + (base.z + k);
                atomicAdd(&grid[idx].x, mom.x);
                atomicAdd(&grid[idx].y, mom.y);
                atomicAdd(&grid[idx].z, mom.z);
                atomicAdd(&grid[idx].w, weight * m.p_mass);
            }
}

__global__ void k_grid_update(float4* grid, int ncell) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ncell) return;
    float4 g = grid[id];
    if (g.w <= 0.f) return;

    float3 v = make_float3(g.x / g.w, g.y / g.w, g.z / g.w);
    v.y += c_p.dt * c_p.gravity_y;

    int res = c_p.res, b = c_p.bound;
    int i = id / (res * res);
    int j = (id / res) % res;
    int k = id % res;
    /* conditions separantes : composante normale annulee vers la paroi */
    if (i < b && v.x < 0.f) v.x = 0.f;
    if (i >= res - b && v.x > 0.f) v.x = 0.f;
    if (j < b && v.y < 0.f) v.y = 0.f;
    if (j >= res - b && v.y > 0.f) v.y = 0.f;
    if (k < b && v.z < 0.f) v.z = 0.f;
    if (k >= res - b && v.z > 0.f) v.z = 0.f;

    grid[id] = make_float4(v.x, v.y, v.z, g.w);
}

__global__ void k_g2p(float3* __restrict__ x,
                      float3* __restrict__ v,
                      float* __restrict__ Cbuf,
                      float* __restrict__ Jw,
                      const uint8_t* __restrict__ mat,
                      const float4* __restrict__ grid, int n) {
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
                float3 dpos = make_float3((i - fx.x) * c_p.dx,
                                          (j - fx.y) * c_p.dx,
                                          (k - fx.z) * c_p.dx);
                float weight = w[i][0] * w[j][1] * w[k][2];
                int idx = ((base.x + i) * c_p.res + (base.y + j)) * c_p.res
                          + (base.z + k);
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
    float hi = c_p.res * c_p.dx - lo;
    x[p] = make_float3(fminf(fmaxf(xp.x + c_p.dt * nv.x, lo), hi),
                       fminf(fmaxf(xp.y + c_p.dt * nv.y, lo), hi),
                       fminf(fmaxf(xp.z + c_p.dt * nv.z, lo), hi));

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
};

static float material_sound_speed(const BqMaterial& m) {
    float stiff = (m.model == BQ_MODEL_ELASTIC) ? m.E : m.bulk;
    return sqrtf(stiff / m.rho);
}

static int upload_params(BqSim* s) {
    float c_max = 1e-3f;
    for (int i = 0; i < s->n_mats; ++i)
        c_max = fmaxf(c_max, material_sound_speed(s->mats_host[i]));
    float dx = s->cfg.domain / s->cfg.grid_res;
    s->dt = s->cfg.cfl * dx / c_max;

    SimParamsGpu& p = s->prm;
    p.res = s->cfg.grid_res;
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

BQ_API void bq_default_config(BqConfig* cfg) {
    cfg->grid_res = 64;
    cfg->domain = 1.f;
    cfg->gravity_y = -9.8f;
    cfg->cfl = 0.3f;
    cfg->ppc_axis = 2;
    cfg->max_particles = 2000000;
}

BQ_API BqSim* bq_create(const BqConfig* cfg) {
    BqSim* s = new BqSim();
    s->cfg = cfg ? *cfg : (bq_default_config(&s->cfg), s->cfg);
    int cap = s->cfg.max_particles;
    int ncell = s->cfg.grid_res * s->cfg.grid_res * s->cfg.grid_res;
    if (cudaMalloc(&s->d_x, cap * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&s->d_v, cap * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&s->d_C, cap * 9 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_F, cap * 9 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_J, cap * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&s->d_mat, cap * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(&s->d_grid, ncell * sizeof(float4)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "cudaMalloc: memoire insuffisante");
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

BQ_API int bq_emit_box(BqSim* s, int mat_id, const float lo[3],
                       const float hi[3], const float vel[3]) {
    if (mat_id < 0 || mat_id >= s->n_mats) {
        snprintf(g_error, sizeof(g_error), "mat_id %d invalide", mat_id);
        return -1;
    }
    float spacing = s->prm.dx / s->cfg.ppc_axis;
    std::vector<float3> px, pv;
    for (float x = lo[0] + spacing / 2; x < hi[0]; x += spacing)
        for (float y = lo[1] + spacing / 2; y < hi[1]; y += spacing)
            for (float z = lo[2] + spacing / 2; z < hi[2]; z += spacing) {
                px.push_back(make_float3(x, y, z));
                pv.push_back(make_float3(vel[0], vel[1], vel[2]));
            }
    int add = (int)px.size();
    if (s->n + add > s->cfg.max_particles) {
        snprintf(g_error, sizeof(g_error), "capacite depassee (%d + %d > %d)",
                 s->n, add, s->cfg.max_particles);
        return -1;
    }
    std::vector<float> id9(add * 9, 0.f), ones(add, 1.f);
    for (int i = 0; i < add; ++i) { id9[9 * i] = id9[9 * i + 4] = id9[9 * i + 8] = 1.f; }
    std::vector<float> zero9(add * 9, 0.f);
    std::vector<uint8_t> mid(add, (uint8_t)mat_id);

    int off = s->n;
    BQ_CUDA_CHECK(cudaMemcpy(s->d_x + off, px.data(), add * sizeof(float3),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_v + off, pv.data(), add * sizeof(float3),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_F + 9 * off, id9.data(),
                             add * 9 * sizeof(float), cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_C + 9 * off, zero9.data(),
                             add * 9 * sizeof(float), cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_J + off, ones.data(), add * sizeof(float),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(s->d_mat + off, mid.data(), add * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));
    s->n += add;
    return add;
}

BQ_API int bq_step(BqSim* s, float frame_dt) {
    if (s->n == 0 || s->n_mats == 0) return 0;
    int substeps = (int)ceilf(frame_dt / s->dt);
    int ncell = s->prm.res * s->prm.res * s->prm.res;
    dim3 bp(256), gp((s->n + 255) / 256), gc((ncell + 255) / 256);

    for (int i = 0; i < substeps; ++i) {
        k_clear_grid<<<gc, bp>>>(s->d_grid, ncell);
        k_p2g<<<gp, bp>>>(s->d_x, s->d_v, s->d_C, s->d_F, s->d_J, s->d_mat,
                          s->d_grid, s->n);
        k_grid_update<<<gc, bp>>>(s->d_grid, ncell);
        k_g2p<<<gp, bp>>>(s->d_x, s->d_v, s->d_C, s->d_J, s->d_mat,
                          s->d_grid, s->n);
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

BQ_API const char* bq_last_error(void) { return g_error; }

} /* extern "C" */

#include "bourrasque.h"
#include <cstdio>
#include <cmath>
#include <vector>

static const float PI = 3.14159265358979323846f;

static int make_sphere(float cx, float cy, float cz, float r, int rings, int segs,
                        float vx, float vy, float vz, float friction,
                        std::vector<float>& tri, std::vector<float>& trivel,
                        std::vector<float>& trifric) {
    std::vector<float> px, py, pz;
    for (int i = 0; i <= rings; ++i) {
        float theta = PI * i / rings;
        for (int j = 0; j <= segs; ++j) {
            float phi = 2.f * PI * j / segs;
            px.push_back(cx + r * sinf(theta) * cosf(phi));
            py.push_back(cy + r * cosf(theta));
            pz.push_back(cz + r * sinf(theta) * sinf(phi));
        }
    }
    auto idx = [&](int i, int j) { return i * (segs + 1) + j; };
    int n_tri = 0;
    for (int i = 0; i < rings; ++i)
        for (int j = 0; j < segs; ++j) {
            int a = idx(i, j), b = idx(i, j + 1), c = idx(i + 1, j), d = idx(i + 1, j + 1);
            int tris[2][3] = { {a, c, b}, {b, c, d} };
            for (auto& t : tris) {
                for (int k = 0; k < 3; ++k) {
                    tri.push_back(px[t[k]]); tri.push_back(py[t[k]]); tri.push_back(pz[t[k]]);
                    trivel.push_back(vx); trivel.push_back(vy); trivel.push_back(vz);
                }
                trifric.push_back(friction);
                ++n_tri;
            }
        }
    return n_tri;
}

static float run_once(float vc) {
    BqConfig cfg; bq_default_config(&cfg);
    cfg.grid_res[0]=cfg.grid_res[1]=cfg.grid_res[2]=64;
    cfg.cell_size = 1.f/64.f;
    cfg.gravity_y = 0.f;
    BqSim* sim = bq_create(&cfg);
    BqMaterial water{}; water.model=BQ_MODEL_WATER; water.rho=1000.f; water.bulk=4e4f; water.gamma=3.f;
    int mw = bq_add_material(sim, &water);
    float lo[3]={0.30f,0.30f,0.30f}, hi[3]={0.70f,0.70f,0.70f};
    float v0[3]={0,0,0};
    bq_emit_box(sim, mw, lo, hi, v0);
    int n = bq_particle_count(sim);
    printf("n=%d\n", n);

    std::vector<float> tri, trivel, trifric;
    /* sphere fully overlapping the water block, centre commun */
    int n_tri = make_sphere(0.5f, 0.5f, 0.5f, 0.15f, 20, 20, vc, 0, 0, 0.0f, tri, trivel, trifric);
    if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), n_tri) < 0) {
        printf("set_colliders err: %s\n", bq_last_error());
    }
    bq_step(sim, 1.f/240.f); /* frame courte */
    std::vector<float> pos(3*(size_t)n);
    bq_read_positions(sim, pos.data());
    float bx=0; for(int i=0;i<n;++i) bx+=pos[3*i];
    bx/=n;
    bq_destroy(sim);
    return bx;
}

int main() {
    float b0 = run_once(0.f);
    float b1 = run_once(3.f);
    printf("barycentre x, vc=0 : %.8f\n", b0);
    printf("barycentre x, vc=3 : %.8f\n", b1);
    printf("ecart : %.8f\n", b1-b0);
    return 0;
}

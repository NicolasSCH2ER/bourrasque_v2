/* Verif 3 (rigoureuse) : plusieurs sondes ponctuelles a distances connues du
 * centre d'une sphere, comparees a l'estimation de phi obtenue indirectement
 * via le couplage (vitesse normale bloquee ssi phi<0). Comme d_sdf n'est pas
 * expose par l'API publique, on verifie plutot le comportement dynamique a
 * plusieurs profondeurs : une particule loin sous la surface doit etre
 * bloquee (delta_y ~ 0), une particule loin a l'exterieur doit tomber
 * librement (delta_y ~ v*dt). */
#include "bourrasque.h"
#include <cstdio>
#include <cmath>
#include <vector>

static const float PI = 3.14159265358979323846f;

static int make_sphere(float cx, float cy, float cz, float r, int rings, int segs,
                        float friction,
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
                    trivel.push_back(0.f); trivel.push_back(0.f); trivel.push_back(0.f);
                }
                trifric.push_back(friction);
                ++n_tri;
            }
        }
    return n_tri;
}

int main() {
    BqConfig cfg; bq_default_config(&cfg);
    cfg.grid_res[0]=cfg.grid_res[1]=cfg.grid_res[2]=64;
    cfg.cell_size = 1.f/64.f;
    cfg.gravity_y = 0.f;
    float dx = cfg.cell_size;
    float cx=0.5f, cy=0.5f, cz=0.5f, r=0.15f;

    /* echantillons a distance connue du centre, le long de +y */
    float depths[] = { -5*dx, -2*dx, 0.5f*dx, 2*dx, 5*dx }; /* offset relatif a r : negatif = interieur */
    for (float off : depths) {
        BqSim* sim = bq_create(&cfg);
        BqMaterial water{}; water.model=BQ_MODEL_WATER; water.rho=1000.f; water.bulk=4e4f; water.gamma=3.f;
        int mw = bq_add_material(sim, &water);
        std::vector<float> tri, trivel, trifric;
        int n_tri = make_sphere(cx, cy, cz, r, 32, 32, 0.3f, tri, trivel, trifric);
        bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), n_tri);

        float dist_from_center = r + off; /* off<0 => interieur, off>0 => exterieur */
        float px[3] = { cx, cy + dist_from_center, cz };
        float pv[3] = { 0.f, -1.f, 0.f };
        bq_emit_points_vel(sim, mw, px, pv, 1);
        float pos0[3]; bq_read_positions(sim, pos0);
        bq_step(sim, 1.f/240.f);
        float pos1[3]; bq_read_positions(sim, pos1);
        float dy = pos1[1] - pos0[1];
        float free_fall = -1.f * (1.f/240.f);
        printf("offset=%+.4f (dist_center=%.4f, r=%.4f, %s) : dy=%.6f (chute libre attendue=%.6f) -> %s\n",
               off, dist_from_center, r, off < 0 ? "interieur" : "exterieur",
               dy, free_fall,
               (off < 0) ? (fabsf(dy) < fabsf(free_fall) * 0.5f ? "bloque (coherent, phi<0)" : "PAS bloque -- probleme")
                         : (fabsf(dy - free_fall) < 1e-4f ? "chute libre (coherent, phi>=0)" : "chute anormale -- probleme"));
        bq_destroy(sim);
    }
    return 0;
}

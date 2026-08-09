/* Repro minimal pour isoler un crash sans corps rigide (aucun bq_set_colliders
 * ni bq_set_collider_bodies appele) -- meme scenario que bourrasque_headless
 * "dam", avec fflush apres chaque etape pour localiser. */
#include "bourrasque.h"
#include <cstdio>
#include <vector>

int main() {
    BqConfig cfg; bq_default_config(&cfg);
    fprintf(stderr, "[1] config ok\n"); fflush(stderr);
    BqSim* sim = bq_create(&cfg);
    fprintf(stderr, "[2] create ok: %p\n", (void*)sim); fflush(stderr);
    BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho=1000.f; water.bulk=4e4f; water.gamma=3.f;
    int mw = bq_add_material(sim, &water);
    fprintf(stderr, "[3] add_material ok: %d\n", mw); fflush(stderr);
    float lo[3]={0.10f,0.10f,0.10f}, hi[3]={0.35f,0.60f,0.90f}, v0[3]={0,0,0};
    int n0 = bq_emit_box(sim, mw, lo, hi, v0);
    fprintf(stderr, "[4] emit_box ok: %d\n", n0); fflush(stderr);
    for (int fr = 0; fr < 10; ++fr) {
        int sub = bq_step(sim, 1.f/24.f);
        fprintf(stderr, "[5.%d] step ok: %d substeps\n", fr, sub); fflush(stderr);
        int n = bq_particle_count(sim);
        std::vector<float> pos(3*(size_t)n);
        bq_read_positions(sim, pos.data());
        fprintf(stderr, "[5.%d] read_positions ok: n=%d\n", fr, n); fflush(stderr);
    }
    fprintf(stderr, "[6] loop done, destroying\n"); fflush(stderr);
    bq_destroy(sim);
    fprintf(stderr, "[7] destroy ok\n"); fflush(stderr);
    return 0;
}

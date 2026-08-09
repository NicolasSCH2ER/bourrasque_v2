/* bourrasque_headless — test du core sans Blender.
 *
 * N'utilise QUE l'API C publique (bourrasque.h) : c'est volontaire,
 * l'extension Blender passera par exactement la meme surface via ctypes.
 *
 * Usage : bourrasque_headless <scene> <frames> <out.bqd>
 *   scenes : jelly (M1) | dam (M2) | splash (M1+M2)
 *
 * Formats :
 *   out.bqd : int32 n, int32 frames, puis frames*n*3 float32 (positions)
 *   out.mat : n uint8 (id materiau par particule, pour le viewer)
 */
#include "bourrasque.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const float V0[3] = {0.f, 0.f, 0.f};

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <jelly|dam|splash> <frames> <out.bqd>\n",
                argv[0]);
        return 1;
    }
    std::string scene = argv[1];
    int frames = atoi(argv[2]);
    std::string out = argv[3];

    BqConfig cfg;
    bq_default_config(&cfg);
    BqSim* sim = bq_create(&cfg);
    if (!sim) { fprintf(stderr, "create: %s\n", bq_last_error()); return 1; }

    BqMaterial jelly{};
    jelly.model = BQ_MODEL_ELASTIC;
    jelly.rho = 1000.f; jelly.E = 5.0e4f; jelly.nu = 0.2f;

    BqMaterial water{};
    water.model = BQ_MODEL_WATER;
    water.rho = 1000.f; water.bulk = 4.0e4f; water.gamma = 3.f;

    int m_jelly = bq_add_material(sim, &jelly);
    if (m_jelly < 0) { fprintf(stderr, "add_material(jelly): %s\n", bq_last_error()); return 1; }
    int m_water = bq_add_material(sim, &water);
    if (m_water < 0) { fprintf(stderr, "add_material(water): %s\n", bq_last_error()); return 1; }

    if (scene == "jelly") {
        float lo[3] = {0.35f, 0.55f, 0.35f}, hi[3] = {0.65f, 0.85f, 0.65f};
        if (bq_emit_box(sim, m_jelly, lo, hi, V0) < 0) {
            fprintf(stderr, "emit_box: %s\n", bq_last_error());
            return 1;
        }
    } else if (scene == "dam") {
        float lo[3] = {0.10f, 0.10f, 0.10f}, hi[3] = {0.35f, 0.60f, 0.90f};
        if (bq_emit_box(sim, m_water, lo, hi, V0) < 0) {
            fprintf(stderr, "emit_box: %s\n", bq_last_error());
            return 1;
        }
    } else if (scene == "splash") {
        float wlo[3] = {0.10f, 0.10f, 0.10f}, whi[3] = {0.90f, 0.30f, 0.90f};
        float jlo[3] = {0.40f, 0.60f, 0.40f}, jhi[3] = {0.60f, 0.80f, 0.60f};
        if (bq_emit_box(sim, m_water, wlo, whi, V0) < 0) {
            fprintf(stderr, "emit_box: %s\n", bq_last_error());
            return 1;
        }
        if (bq_emit_box(sim, m_jelly, jlo, jhi, V0) < 0) {
            fprintf(stderr, "emit_box: %s\n", bq_last_error());
            return 1;
        }
    } else {
        fprintf(stderr, "scene inconnue: %s\n", scene.c_str());
        return 1;
    }

    int n = bq_particle_count(sim);
    printf("%d particules, %d frames, grille %dx%dx%d\n", n, frames,
           cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2]);

    FILE* f = fopen(out.c_str(), "wb");
    if (!f) { fprintf(stderr, "impossible d'ouvrir %s\n", out.c_str()); return 1; }
    int32_t hdr[2] = {n, frames};
    fwrite(hdr, sizeof(int32_t), 2, f);

    /* sidecar materiaux pour la coloration du viewer */
    std::vector<uint8_t> mats(n);
    if (bq_read_materials(sim, mats.data()) < 0) {
        fprintf(stderr, "read_materials: %s\n", bq_last_error());
        return 1;
    }
    std::string matpath = out.substr(0, out.find_last_of('.')) + ".mat";
    FILE* fm = fopen(matpath.c_str(), "wb");
    fwrite(mats.data(), 1, n, fm);
    fclose(fm);

    std::vector<float> pos(3 * (size_t)n);
    auto t0 = std::chrono::steady_clock::now();
    for (int fr = 0; fr < frames; ++fr) {
        int sub = bq_step(sim, 1.f / 24.f);
        if (sub < 0) { fprintf(stderr, "step: %s\n", bq_last_error()); return 1; }
        /* le reseeding (M10) fait grandir le nombre de particules frame apres
         * frame -- redimensionner le tampon avant lecture, sinon
         * bq_read_positions ecrit au-dela de sa taille (depassement de tas
         * silencieux jusqu'au crash suivant). */
        int n_now = bq_particle_count(sim);
        if (n_now > n) { n = n_now; pos.resize(3 * (size_t)n); }
        if (bq_read_positions(sim, pos.data()) < 0) {
            fprintf(stderr, "read_positions: %s\n", bq_last_error());
            return 1;
        }
        fwrite(pos.data(), sizeof(float), pos.size(), f);
        if (fr % 24 == 0) printf("frame %d/%d (%d substeps)\n", fr, frames, sub);
    }
    auto dt = std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t0).count();
    printf("%.1f s au total (%.1f ms/frame)\n", dt, 1000.0 * dt / frames);

    fclose(f);
    bq_destroy(sim);
    printf("ecrit : %s + %s\n", out.c_str(), matpath.c_str());
    return 0;
}

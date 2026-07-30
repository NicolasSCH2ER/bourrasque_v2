/* Test manuel de bq_emit_points_vel : verifie que des vitesses par
 * particule produisent des deplacements differencies, et que
 * bq_emit_points (vitesse uniforme) reste inchange. */
#include "bourrasque.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static BqSim* make_sim(int* m_water) {
    BqConfig cfg;
    bq_default_config(&cfg);
    BqSim* sim = bq_create(&cfg);
    if (!sim) { fprintf(stderr, "create: %s\n", bq_last_error()); return nullptr; }

    BqMaterial water{};
    water.model = BQ_MODEL_WATER;
    water.rho = 1000.f; water.bulk = 4.0e4f; water.gamma = 3.f;
    *m_water = bq_add_material(sim, &water);
    return sim;
}

static const int N = 8;

/* 8 positions distinctes, bien a l'interieur du domaine [0,1]^3. */
static const float POS[N * 3] = {
    0.30f, 0.50f, 0.30f,
    0.70f, 0.50f, 0.30f,
    0.30f, 0.50f, 0.70f,
    0.70f, 0.50f, 0.70f,
    0.40f, 0.30f, 0.40f,
    0.60f, 0.30f, 0.40f,
    0.40f, 0.70f, 0.60f,
    0.60f, 0.70f, 0.60f,
};

/* 8 vitesses tres differentes et bien separees (2 m/s, directions variees,
 * bien plus grandes que l'impulsion de gravite sur un pas tres court). */
static const float VEL[N * 3] = {
     2.f,  0.f,  0.f,
    -2.f,  0.f,  0.f,
     0.f,  2.f,  0.f,
     0.f, -2.f,  0.f,
     0.f,  0.f,  2.f,
     0.f,  0.f, -2.f,
     2.f,  2.f,  2.f,
    -2.f, -2.f, -2.f,
};

int test_diff_velocities() {
    printf("=== test 1 : bq_emit_points_vel, 8 vitesses differentes ===\n");
    int m_water;
    BqSim* sim = make_sim(&m_water);
    if (!sim) return 1;

    int emitted = bq_emit_points_vel(sim, m_water, POS, VEL, N);
    if (emitted != N) {
        fprintf(stderr, "bq_emit_points_vel: %s\n", bq_last_error());
        return 1;
    }

    std::vector<float> before(N * 3);
    bq_read_positions(sim, before.data());

    int sub = bq_step(sim, 1.0e-3f);
    if (sub < 0) { fprintf(stderr, "step: %s\n", bq_last_error()); return 1; }

    std::vector<float> after(N * 3);
    bq_read_positions(sim, after.data());

    printf("substeps effectues: %d\n", sub);
    bool all_different = true;
    bool consistent_with_vel = true;
    for (int i = 0; i < N; ++i) {
        float dx = after[3*i] - before[3*i];
        float dy = after[3*i+1] - before[3*i+1];
        float dz = after[3*i+2] - before[3*i+2];
        printf("p%d: deplacement = (% .6f, % .6f, % .6f)  vel_emission = (% .1f, % .1f, % .1f)\n",
               i, dx, dy, dz, VEL[3*i], VEL[3*i+1], VEL[3*i+2]);
        /* le signe du deplacement doit correspondre au signe de la vitesse
           d'emission (composante par composante, en ignorant les
           composantes nulles). */
        for (int a = 0; a < 3; ++a) {
            float v = VEL[3*i+a];
            float d = (a==0)?dx:(a==1)?dy:dz;
            if (v > 0 && d <= 0.f) consistent_with_vel = false;
            if (v < 0 && d >= 0.f) consistent_with_vel = false;
        }
    }
    /* verifie qu'au moins deux particules ont des deplacements distincts
       (une vitesse uniforme donnerait un deplacement identique partout). */
    for (int i = 1; i < N; ++i) {
        float ddx = fabsf((after[3*i]-before[3*i]) - (after[0]-before[0]));
        float ddy = fabsf((after[3*i+1]-before[3*i+1]) - (after[1]-before[1]));
        float ddz = fabsf((after[3*i+2]-before[3*i+2]) - (after[2]-before[2]));
        if (ddx < 1e-8f && ddy < 1e-8f && ddz < 1e-8f) all_different = false;
    }

    bq_destroy(sim);

    printf("deplacements tous differents: %s\n", all_different ? "OUI" : "NON");
    printf("deplacements coherents avec le signe de la vitesse d'emission: %s\n",
           consistent_with_vel ? "OUI" : "NON");
    return (all_different && consistent_with_vel) ? 0 : 1;
}

int test_non_regression() {
    printf("\n=== test 2 : non-regression bq_emit_points, vitesse uniforme ===\n");
    int m_water;
    BqSim* sim = make_sim(&m_water);
    if (!sim) return 1;

    const float vel_uniform[3] = {0.5f, -0.25f, 0.1f};
    int emitted = bq_emit_points(sim, m_water, POS, N, vel_uniform);
    if (emitted != N) {
        fprintf(stderr, "bq_emit_points: %s\n", bq_last_error());
        return 1;
    }

    std::vector<float> before(N * 3);
    bq_read_positions(sim, before.data());

    int sub = bq_step(sim, 1.0e-3f);
    if (sub < 0) { fprintf(stderr, "step: %s\n", bq_last_error()); return 1; }

    std::vector<float> after(N * 3);
    bq_read_positions(sim, after.data());

    printf("substeps effectues: %d\n", sub);
    bool identical = true;
    for (int i = 0; i < N; ++i) {
        float dx = after[3*i] - before[3*i];
        float dy = after[3*i+1] - before[3*i+1];
        float dz = after[3*i+2] - before[3*i+2];
        printf("p%d: deplacement = (% .6f, % .6f, % .6f)\n", i, dx, dy, dz);
        float ddx = fabsf(dx - (after[0]-before[0]));
        float ddy = fabsf(dy - (after[1]-before[1]));
        float ddz = fabsf(dz - (after[2]-before[2]));
        if (ddx > 1e-6f || ddy > 1e-6f || ddz > 1e-6f) identical = false;
    }
    bq_destroy(sim);
    printf("deplacements identiques entre particules (vitesse uniforme): %s\n",
           identical ? "OUI" : "NON");
    return identical ? 0 : 1;
}

int main() {
    int r1 = test_diff_velocities();
    int r2 = test_non_regression();
    printf("\n=== resultat global: %s ===\n", (r1==0 && r2==0) ? "OK" : "ECHEC");
    return (r1==0 && r2==0) ? 0 : 1;
}

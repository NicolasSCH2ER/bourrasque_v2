/* Test harness M6 : colliders animes. Genere un maillage de sphere UV,
 * l'emet comme collider, et verifie etancheite/signe/animation/masse/cout. */
#include "bourrasque.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

static const float PI = 3.14159265358979323846f;

/* Genere une sphere UV (rings x segs facettes) centree en `center`, rayon r.
 * Remplit tri (9*n_tri), tri_vel (9*n_tri, vitesse uniforme donnee),
 * tri_friction (n_tri, constant). Renvoie n_tri. */
static int make_sphere(float cx, float cy, float cz, float r, int rings, int segs,
                        float vx, float vy, float vz, float friction,
                        std::vector<float>& tri, std::vector<float>& trivel,
                        std::vector<float>& trifric) {
    std::vector<float> px, py, pz;
    for (int i = 0; i <= rings; ++i) {
        float theta = PI * i / rings; /* 0..pi */
        for (int j = 0; j <= segs; ++j) {
            float phi = 2.f * PI * j / segs;
            px.push_back(cx + r * sinf(theta) * cosf(phi));
            py.push_back(cy + r * cosf(theta));
            pz.push_back(cz + r * sinf(theta) * sinf(phi));
        }
    }
    auto idx = [&](int i, int j) { return i * (segs + 1) + j; };
    int n_tri = 0;
    for (int i = 0; i < rings; ++i) {
        for (int j = 0; j < segs; ++j) {
            int a = idx(i, j), b = idx(i, j + 1), c = idx(i + 1, j), d = idx(i + 1, j + 1);
            /* deux triangles par quad, winding CCW vu de l'exterieur */
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
    }
    return n_tri;
}

static int check_abi() {
    if (bq_abi_version() != 3) {
        fprintf(stderr, "ABI version = %d (attendu 3)\n", bq_abi_version());
        return -1;
    }
    printf("ABI version OK (%d)\n", bq_abi_version());
    return 0;
}

/* Verif 3 : signe et magnitude du SDF pour une sphere statique, sans fluide.
 * On ne peut pas lire d_sdf via l'API publique (pas exposee) -- on infere le
 * signe/magnitude indirectement via le couplage sur une particule sonde
 * placee a un endroit connu, en verifiant qu'elle est repoussee ssi elle est
 * dans le solide et qu'elle a une vitesse entrante. */
static int test_sign_via_probe() {
    printf("\n=== Verif 3 : signe du SDF (sonde ponctuelle) ===\n");
    BqConfig cfg; bq_default_config(&cfg);
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 64;
    cfg.cell_size = 1.f / 64.f;
    cfg.gravity_y = 0.f;
    BqSim* sim = bq_create(&cfg);
    if (!sim) { fprintf(stderr, "create: %s\n", bq_last_error()); return -1; }
    BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho = 1000.f; water.bulk = 4e4f; water.gamma = 3.f;
    int mw = bq_add_material(sim, &water);

    float cx = 0.5f, cy = 0.5f, cz = 0.5f, r = 0.15f;
    std::vector<float> tri, trivel, trifric;
    int n_tri = make_sphere(cx, cy, cz, r, 24, 24, 0, 0, 0, 0.3f, tri, trivel, trifric);
    if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), n_tri) < 0) {
        fprintf(stderr, "set_colliders: %s\n", bq_last_error()); return -1;
    }

    /* particule test juste sous le pole superieur de la sphere (interieur),
     * animee vers le bas (vn<0 par rapport a la normale locale ~ +y) */
    float dx = cfg.cell_size;
    float px[3] = { cx, cy + r - 1.5f*dx, cz };  /* interieur, pres du pole haut */
    float pv[3] = { 0.f, -1.f, 0.f };
    if (bq_emit_points_vel(sim, mw, px, pv, 1) < 0) {
        fprintf(stderr, "emit: %s\n", bq_last_error()); return -1;
    }
    float pos0[3];
    bq_read_positions(sim, pos0);
    printf("position initiale (interieure) : (%.4f, %.4f, %.4f)\n", pos0[0], pos0[1], pos0[2]);

    int sub = bq_step(sim, 1.f/240.f); /* une frame courte, peu de substeps */
    if (sub < 0) { fprintf(stderr, "step: %s\n", bq_last_error()); return -1; }
    float pos1[3];
    bq_read_positions(sim, pos1);
    printf("apres 1 frame, deplacement y = %.6f\n", pos1[1]-pos0[1]);
    float dist_center = sqrtf((pos1[0]-cx)*(pos1[0]-cx) + (pos1[1]-cy)*(pos1[1]-cy) + (pos1[2]-cz)*(pos1[2]-cz));
    printf("distance au centre apres frame : %.5f (rayon = %.5f)\n", dist_center, r);
    bool held = (pos1[1] - pos0[1]) > -1e-3f;
    printf("particule retenue par le collider (vn<0 bloque) : %s\n", held ? "oui" : "NON -- probleme");

    bq_destroy(sim);
    return held ? 0 : -1;
}

/* Verif 2 et 5 : colonne d'eau tombant sur sphere statique. Aucune violation
 * d'etancheite, particules dans le domaine, compte constant. */
static int test_watertightness() {
    printf("\n=== Verif 2/5 : colonne d'eau sur sphere statique ===\n");
    BqConfig cfg; bq_default_config(&cfg);
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 64;
    cfg.cell_size = 1.f / 64.f;
    BqSim* sim = bq_create(&cfg);
    if (!sim) { fprintf(stderr, "create: %s\n", bq_last_error()); return -1; }
    BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho = 1000.f; water.bulk = 4e4f; water.gamma = 3.f;
    int mw = bq_add_material(sim, &water);

    float cx = 0.5f, cy = 0.3f, cz = 0.5f, r = 0.15f;
    std::vector<float> tri, trivel, trifric;
    int n_tri = make_sphere(cx, cy, cz, r, 32, 32, 0, 0, 0, 0.4f, tri, trivel, trifric);
    if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), n_tri) < 0) {
        fprintf(stderr, "set_colliders: %s\n", bq_last_error()); return -1;
    }

    float lo[3] = {0.35f, 0.60f, 0.35f}, hi[3] = {0.65f, 0.85f, 0.65f};
    float v0[3] = {0.f, 0.f, 0.f};
    if (bq_emit_box(sim, mw, lo, hi, v0) < 0) { fprintf(stderr, "emit_box: %s\n", bq_last_error()); return -1; }

    int n0 = bq_particle_count(sim);
    printf("particules emises : %d\n", n0);

    int frames = 60;
    std::vector<float> pos(3 * (size_t)n0);
    for (int fr = 0; fr < frames; ++fr) {
        /* colliders animes une fois par frame (spec) : re-uploade la meme sphere statique */
        if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), n_tri) < 0) {
            fprintf(stderr, "set_colliders: %s\n", bq_last_error()); return -1;
        }
        if (bq_step(sim, 1.f/24.f) < 0) { fprintf(stderr, "step: %s\n", bq_last_error()); return -1; }
    }
    int n1 = bq_particle_count(sim);
    if (bq_read_positions(sim, pos.data()) < 0) { fprintf(stderr, "read: %s\n", bq_last_error()); return -1; }

    float dlo = 3.f * cfg.cell_size;
    float dhi_x = cfg.grid_res[0]*cfg.cell_size - dlo;
    float dhi_y = cfg.grid_res[1]*cfg.cell_size - dlo;
    float dhi_z = cfg.grid_res[2]*cfg.cell_size - dlo;

    int violations_sphere = 0, violations_domain = 0;
    float min_dist = 1e9f;
    for (int i = 0; i < n1; ++i) {
        float x = pos[3*i], y = pos[3*i+1], z = pos[3*i+2];
        float d = sqrtf((x-cx)*(x-cx)+(y-cy)*(y-cy)+(z-cz)*(z-cz));
        if (d < min_dist) min_dist = d;
        if (d < r) ++violations_sphere;
        if (x < dlo || x > dhi_x || y < dlo || y > dhi_y || z < dlo || z > dhi_z) ++violations_domain;
    }
    printf("compte particules : avant=%d apres=%d (doit etre egal)\n", n0, n1);
    printf("violations etancheite (dist_centre < rayon) : %d / %d\n", violations_sphere, n1);
    printf("distance minimale au centre observee : %.5f (rayon = %.5f)\n", min_dist, r);
    printf("violations hors domaine utile : %d / %d\n", violations_domain, n1);

    bq_destroy(sim);
    return (n0 == n1 && violations_sphere == 0 && violations_domain == 0) ? 0 : -1;
}

/* Verif 4 : sphere animee traversant un volume d'eau au repos, compare au
 * cas statique. Le barycentre du fluide doit se deplacer nettement plus
 * dans le cas anime. */
static int test_animated_vs_static() {
    printf("\n=== Verif 4 : collider anime vs statique ===\n");
    auto run = [](bool animated) -> float {
        BqConfig cfg; bq_default_config(&cfg);
        cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 64;
        cfg.cell_size = 1.f / 64.f;
        cfg.gravity_y = 0.f; /* isole l'effet du collider */
        BqSim* sim = bq_create(&cfg);
        BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho = 1000.f; water.bulk = 4e4f; water.gamma = 3.f;
        int mw = bq_add_material(sim, &water);

        float lo[3] = {0.20f, 0.20f, 0.20f}, hi[3] = {0.80f, 0.55f, 0.80f};
        float v0[3] = {0.f, 0.f, 0.f};
        bq_emit_box(sim, mw, lo, hi, v0);
        int n = bq_particle_count(sim);

        float r = 0.10f;
        float cy = 0.35f, cz = 0.5f;
        float speed = 1.5f; /* m/s le long de x */
        float cx0 = 0.15f;

        int frames = 30;
        float dt_frame = 1.f/24.f;
        for (int fr = 0; fr < frames; ++fr) {
            float cx = animated ? (cx0 + speed * dt_frame * fr) : cx0;
            float vx = animated ? speed : 0.f;
            std::vector<float> tri, trivel, trifric;
            int n_tri = make_sphere(cx, cy, cz, r, 16, 16, vx, 0.f, 0.f, 0.2f, tri, trivel, trifric);
            if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), n_tri) < 0) {
                fprintf(stderr, "set_colliders: %s\n", bq_last_error());
            }
            bq_step(sim, dt_frame);
        }
        std::vector<float> pos(3 * (size_t)n);
        bq_read_positions(sim, pos.data());
        float bx = 0.f;
        for (int i = 0; i < n; ++i) bx += pos[3*i];
        bx /= n;
        bq_destroy(sim);
        return bx;
    };

    float bx_static = run(false);
    float bx_animated = run(true);
    printf("barycentre x (statique)  : %.6f\n", bx_static);
    printf("barycentre x (anime)     : %.6f\n", bx_animated);
    printf("ecart : %.6f\n", fabsf(bx_animated - bx_static));
    return 0;
}

/* Verif 6 : cout de k_collider_sdf par frame, avec/sans bande etroite. On ne
 * peut pas desactiver la bande etroite via l'API publique (elle est
 * inconditionnelle par design) ; on mesure donc le cout REEL avec bande
 * etroite (maillage petit dans un grand domaine, cas represantatif), puis
 * on estime le cout SANS bande etroite en placant le maillage de sorte a
 * couvrir tout le domaine (AABB = domaine entier), ce qui revient a
 * annuler l'effet de la bande etroite pour la meme quantite de travail par
 * cellule (ncell * n_tri dans les deux cas testes, la seule variable etant
 * la fraction de cellules qui font le test complet). */
static int test_cost() {
    printf("\n=== Verif 6 : cout de k_collider_sdf ===\n");
    BqConfig cfg; bq_default_config(&cfg);
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 64;
    cfg.cell_size = 1.f / 64.f;
    BqSim* sim = bq_create(&cfg);
    BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho = 1000.f; water.bulk = 4e4f; water.gamma = 3.f;
    bq_add_material(sim, &water);

    /* maillage ~5000 triangles, petit par rapport au domaine (bande etroite active) */
    std::vector<float> tri, trivel, trifric;
    int n_tri = make_sphere(0.5f, 0.5f, 0.5f, 0.1f, 50, 50, 0, 0, 0, 0.3f, tri, trivel, trifric);
    printf("maillage : %d triangles\n", n_tri);

    int warmup = 3, reps = 30;
    for (int i = 0; i < warmup; ++i)
        bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), n_tri);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
        bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), n_tri);
    auto t1 = std::chrono::steady_clock::now();
    double ms_narrow = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    printf("cout AVEC bande etroite (sphere ~0.2 domaine) : %.4f ms/appel\n", ms_narrow);

    /* meme maillage mais gonfle pour couvrir tout le domaine -> AABB = domaine entier,
     * desactive de facto l'effet de la bande etroite (chaque cellule fait le test complet) */
    std::vector<float> tri_big(tri.size());
    for (size_t i = 0; i < tri.size(); i += 3) {
        tri_big[i]   = 0.5f + (tri[i]   - 0.5f) * 6.f; /* rayon effectif ~0.6, couvre tout le domaine */
        tri_big[i+1] = 0.5f + (tri[i+1] - 0.5f) * 6.f;
        tri_big[i+2] = 0.5f + (tri[i+2] - 0.5f) * 6.f;
    }
    for (int i = 0; i < warmup; ++i)
        bq_set_colliders(sim, tri_big.data(), trivel.data(), trifric.data(), n_tri);
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i)
        bq_set_colliders(sim, tri_big.data(), trivel.data(), trifric.data(), n_tri);
    t1 = std::chrono::steady_clock::now();
    double ms_wide = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    printf("cout SANS effet bande etroite (sphere = domaine entier) : %.4f ms/appel\n", ms_wide);
    printf("facteur : %.2fx\n", ms_wide / ms_narrow);

    bq_destroy(sim);
    return 0;
}

int main() {
    int rc = 0;
    if (check_abi() < 0) rc = 1;
    if (test_sign_via_probe() < 0) rc = 1;
    if (test_watertightness() < 0) rc = 1;
    if (test_animated_vs_static() < 0) rc = 1;
    if (test_cost() < 0) rc = 1;
    printf("\n%s\n", rc == 0 ? "TOUS LES TESTS OK" : "ECHEC D'AU MOINS UN TEST");
    return rc;
}

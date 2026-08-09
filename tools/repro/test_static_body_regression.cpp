/* Harnais de mesure M17/A1b (priorite ajoutee par l'orchestrateur) :
 * garantir qu'un collider FIXE reste parfaitement supporte une fois que
 * TOUS les colliders seront declares comme corps (dynamic=0), pas
 * seulement le chemin n_bodies==0 deja verifie par
 * tools/repro/test_nonregression_a1.cpp.
 *
 * Trois configurations sur la meme scene fixe (colonne d'eau tombant sur
 * une boite collider immobile) :
 *   A : aucun corps declare (bq_set_collider_bodies jamais appele),
 *       tri_body == NULL -- comportement d'aujourd'hui.
 *   B : un corps declare, dynamic=0, tri_body pointant tous les triangles
 *       vers lui -- chemin JAMAIS exerce avant cette tache.
 *   C : comme B, mais collider ANIME (translation par frame via tri_vel)
 *       -- verifie que la vitesse de mur par difference finie continue
 *       d'entrainer le fluide quand dynamic=0, ET que l'etat du corps
 *       declare (bq_read_collider_bodies) ne bouge pas d'un iota (aucune
 *       recolte d'impulsion, aucune vitesse vive appliquee a un corps
 *       kinematique).
 *
 * Usage : test_static_body_regression <ab|c> [n_runs]
 *   ab : verif A vs B, n_runs executions de chaque cote (defaut 5)
 *   c  : verif C (collider anime, dynamic=0)
 */
#include "bourrasque.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

static float V0[3] = {0.f, 0.f, 0.f};

static int make_box(float cx, float cy, float cz, float hx, float hy, float hz,
                    float friction, std::vector<float>& tri,
                    std::vector<float>& trivel, std::vector<float>& trifric,
                    float vx=0.f, float vy=0.f, float vz=0.f) {
    float v[8][3] = {
        {cx-hx,cy-hy,cz-hz}, {cx+hx,cy-hy,cz-hz}, {cx+hx,cy+hy,cz-hz}, {cx-hx,cy+hy,cz-hz},
        {cx-hx,cy-hy,cz+hz}, {cx+hx,cy-hy,cz+hz}, {cx+hx,cy+hy,cz+hz}, {cx-hx,cy+hy,cz+hz},
    };
    int faces[6][4] = {
        {0,3,2,1}, {4,5,6,7},
        {0,1,5,4}, {3,7,6,2},
        {0,4,7,3}, {1,2,6,5},
    };
    int n_tri = 0;
    for (auto& f : faces) {
        int tris[2][3] = { {f[0],f[1],f[2]}, {f[0],f[2],f[3]} };
        for (auto& t : tris) {
            for (int k = 0; k < 3; ++k) {
                tri.push_back(v[t[k]][0]); tri.push_back(v[t[k]][1]); tri.push_back(v[t[k]][2]);
                trivel.push_back(vx); trivel.push_back(vy); trivel.push_back(vz);
            }
            trifric.push_back(friction);
            ++n_tri;
        }
    }
    return n_tri;
}

static void barycenter(const std::vector<float>& pos, double b[3]) {
    int n = (int)pos.size() / 3;
    b[0]=b[1]=b[2]=0.0;
    for (int i = 0; i < n; ++i) { b[0]+=pos[3*i]; b[1]+=pos[3*i+1]; b[2]+=pos[3*i+2]; }
    b[0]/=n; b[1]/=n; b[2]/=n;
}

/* scene commune A/B : colonne d'eau tombant sur boite collider immobile
 * (h=0.10 a cy=0.30, meme scene que test_heavy de test_rigidbody_a1.cpp,
 * pour rester dans un regime deja connu -- contact fluide/collider franc,
 * pas un cas degenere). declare_body : appelle bq_set_collider_bodies
 * (dynamic=0) et passe tri_body si vrai ; sinon chemin n_bodies==0 actuel. */
static std::vector<float> run_static_scene(bool declare_body, int frames = 60) {
    BqConfig cfg; bq_default_config(&cfg);
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 64;
    cfg.cell_size = 1.f/64.f;
    cfg.gravity_y = -9.8f;
    BqSim* sim = bq_create(&cfg);
    BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho = 1000.f; water.bulk = 4e4f; water.gamma = 3.f;
    int mw = bq_add_material(sim, &water);

    float cx=0.5f, cy=0.30f, cz=0.5f, h=0.10f;
    std::vector<float> tri, trivel, trifric;
    int n_tri = make_box(cx, cy, cz, h, h, h, 0.3f, tri, trivel, trifric);
    std::vector<int> tribody(n_tri, 0);

    if (declare_body) {
        BqRigidBody body{};
        body.dynamic = 0; /* le chemin sous test : declare mais kinematique */
        body.mass = 0.f;
        body.q[0] = 1.f;
        body.x[0]=cx; body.x[1]=cy; body.x[2]=cz;
        if (bq_set_collider_bodies(sim, &body, 1) < 0) {
            fprintf(stderr, "set_collider_bodies: %s\n", bq_last_error());
        }
    }
    if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(),
                         declare_body ? tribody.data() : nullptr, n_tri) < 0) {
        fprintf(stderr, "set_colliders: %s\n", bq_last_error());
    }

    float lo[3] = {0.30f, 0.55f, 0.30f}, hi[3] = {0.70f, 0.85f, 0.70f};
    bq_emit_box(sim, mw, lo, hi, V0);

    for (int fr = 0; fr < frames; ++fr) {
        if (bq_step(sim, 1.f/24.f) < 0) { fprintf(stderr, "step: %s\n", bq_last_error()); break; }
    }
    int n = bq_particle_count(sim); /* APRES la boucle -- reseeding M10 */
    std::vector<float> pos(3*(size_t)n);
    bq_read_positions(sim, pos.data());
    bq_destroy(sim);
    return pos;
}

static int test_ab(int n_runs) {
    printf("=== A1b priorite : corps FIXE declare (dynamic=0) vs non declare ===\n");
    printf("(%d executions de chaque cote)\n", n_runs);

    std::vector<double> bxA(n_runs), byA(n_runs), bzA(n_runs);
    std::vector<double> bxB(n_runs), byB(n_runs), bzB(n_runs);

    for (int i = 0; i < n_runs; ++i) {
        auto posA = run_static_scene(false);
        double b[3]; barycenter(posA, b);
        bxA[i]=b[0]; byA[i]=b[1]; bzA[i]=b[2];
        printf("  A run %d : n=%zu barycentre=(%.8f,%.8f,%.8f)\n", i, posA.size()/3, b[0],b[1],b[2]);
    }
    for (int i = 0; i < n_runs; ++i) {
        auto posB = run_static_scene(true);
        double b[3]; barycenter(posB, b);
        bxB[i]=b[0]; byB[i]=b[1]; bzB[i]=b[2];
        printf("  B run %d : n=%zu barycentre=(%.8f,%.8f,%.8f)\n", i, posB.size()/3, b[0],b[1],b[2]);
    }

    /* enveloppe de bruit intra-configuration : ecart max entre deux
     * executions de la MEME configuration (norme du vecteur barycentre) */
    auto intra_envelope = [](const std::vector<double>& bx, const std::vector<double>& by,
                              const std::vector<double>& bz) -> double {
        double mx = 0.0;
        int n = (int)bx.size();
        for (int i = 0; i < n; ++i)
            for (int j = i+1; j < n; ++j) {
                double dx=bx[i]-bx[j], dy=by[i]-by[j], dz=bz[i]-bz[j];
                double d = std::sqrt(dx*dx+dy*dy+dz*dz);
                if (d > mx) mx = d;
            }
        return mx;
    };
    double noiseA = intra_envelope(bxA, byA, bzA);
    double noiseB = intra_envelope(bxB, byB, bzB);
    double noise_env = std::max(noiseA, noiseB); /* enveloppe conservatrice */

    /* moyenne des barycentres de chaque configuration, ecart inter-config */
    auto mean = [](const std::vector<double>& v) {
        double s=0; for (double x : v) s+=x; return s/v.size();
    };
    double mAx=mean(bxA), mAy=mean(byA), mAz=mean(bzA);
    double mBx=mean(bxB), mBy=mean(byB), mBz=mean(bzB);
    double ddx=mAx-mBx, ddy=mAy-mBy, ddz=mAz-mBz;
    double dev = std::sqrt(ddx*ddx+ddy*ddy+ddz*ddz);

    printf("\n  enveloppe de bruit intra-A (max sur %d paires) : %.8f m\n", n_runs*(n_runs-1)/2, noiseA);
    printf("  enveloppe de bruit intra-B (max sur %d paires) : %.8f m\n", n_runs*(n_runs-1)/2, noiseB);
    printf("  enveloppe retenue (max des deux)                : %.8f m\n", noise_env);
    printf("  barycentre moyen A (non declare)  = (%.8f, %.8f, %.8f)\n", mAx, mAy, mAz);
    printf("  barycentre moyen B (dynamic=0)    = (%.8f, %.8f, %.8f)\n", mBx, mBy, mBz);
    printf("  ecart inter-configuration (A vs B) : %.8f m (%.2fx l'enveloppe)\n",
           dev, noise_env > 1e-12 ? dev/noise_env : -1.0);
    printf("  VERDICT : %s\n", dev <= 2.0*noise_env ? "DANS LE BRUIT" : "HORS BRUIT -- A INVESTIGUER");
    return 0;
}

static int test_c(int frames) {
    printf("=== A1b priorite verif C : collider ANIME, dynamic=0 ===\n");
    auto run = [](bool animated, float* out_final_body /* 13 floats ou nullptr */) -> float {
        BqConfig cfg; bq_default_config(&cfg);
        cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 64;
        cfg.cell_size = 1.f/64.f;
        cfg.gravity_y = 0.f; /* isole l'effet du collider anime, comme test_colliders.cpp verif4 */
        BqSim* sim = bq_create(&cfg);
        BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho = 1000.f; water.bulk = 4e4f; water.gamma = 3.f;
        int mw = bq_add_material(sim, &water);

        float lo[3] = {0.20f, 0.20f, 0.20f}, hi[3] = {0.80f, 0.55f, 0.80f};
        bq_emit_box(sim, mw, lo, hi, V0);
        int n = bq_particle_count(sim);

        float r = 0.10f;
        float cy = 0.35f, cz = 0.5f;
        float speed = 1.5f;
        float cx0 = 0.15f;

        BqRigidBody body{};
        body.dynamic = 0;
        body.mass = 0.f;
        body.q[0] = 1.f;
        body.x[0]=cx0; body.x[1]=cy; body.x[2]=cz;
        bq_set_collider_bodies(sim, &body, 1);

        int frames_local = 30;
        float dt_frame = 1.f/24.f;
        for (int fr = 0; fr < frames_local; ++fr) {
            float cx = animated ? (cx0 + speed * dt_frame * fr) : cx0;
            float vx = animated ? speed : 0.f;
            std::vector<float> tri, trivel, trifric;
            int n_tri = make_box(cx, cy, cz, r, r, r, 0.2f, tri, trivel, trifric, vx, 0.f, 0.f);
            std::vector<int> tribody(n_tri, 0);
            if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(),
                                 tribody.data(), n_tri) < 0) {
                fprintf(stderr, "set_colliders: %s\n", bq_last_error());
            }
            bq_step(sim, dt_frame);
        }
        std::vector<float> pos(3 * (size_t)n);
        bq_read_positions(sim, pos.data());
        float bx = 0.f;
        for (int i = 0; i < n; ++i) bx += pos[3*i];
        bx /= n;

        if (out_final_body) bq_read_collider_bodies(sim, out_final_body);
        bq_destroy(sim);
        return bx;
    };

    float bs_final[13];
    float bx_static = run(false, nullptr);
    float bx_animated = run(true, bs_final);

    printf("  barycentre x fluide (collider statique, dynamic=0) : %.6f\n", bx_static);
    printf("  barycentre x fluide (collider anime,    dynamic=0) : %.6f\n", bx_animated);
    printf("  ecart (le fluide doit etre nettement entraine)     : %.6f\n", fabsf(bx_animated - bx_static));

    float x0=0.15f, y0=0.35f, z0=0.5f;
    float dx = bs_final[0]-x0, dy = bs_final[1]-y0, dz = bs_final[2]-z0;
    float dq = fabsf(bs_final[3]-1.f)+fabsf(bs_final[4])+fabsf(bs_final[5])+fabsf(bs_final[6]);
    float dv = fabsf(bs_final[7])+fabsf(bs_final[8])+fabsf(bs_final[9]);
    float dw = fabsf(bs_final[10])+fabsf(bs_final[11])+fabsf(bs_final[12]);
    printf("  etat du corps declare (dynamic=0) APRES le bake, ecart a l'etat initial :\n");
    printf("    x : (%.8f, %.8f, %.8f) -- attendu (0,0,0)\n", dx, dy, dz);
    printf("    q : ecart somme |dq| = %.8f -- attendu 0\n", dq);
    printf("    v : somme |v| = %.8f -- attendu 0\n", dv);
    printf("    w : somme |w| = %.8f -- attendu 0\n", dw);
    bool untouched = (fabsf(dx) < 1e-9f && fabsf(dy) < 1e-9f && fabsf(dz) < 1e-9f &&
                       dq < 1e-9f && dv < 1e-9f && dw < 1e-9f);
    printf("  VERDICT : corps dynamic=0 %s par le bake\n",
           untouched ? "EXACTEMENT INCHANGE (attendu)" : "MODIFIE -- PROBLEME");
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "ab";
    int n_runs = (argc > 2) ? atoi(argv[2]) : 5;
    if (mode == "ab") test_ab(n_runs);
    else if (mode == "c") test_c(0);
    else { fprintf(stderr, "mode inconnu: %s (ab|c)\n", mode.c_str()); return 1; }
    return 0;
}

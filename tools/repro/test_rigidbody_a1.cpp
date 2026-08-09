/* Harnais de verification M17/A1 : couplage fluide -> solide (recolte
 * d'impulsion, identite par cellule, vitesse de mur vive, integration 6 DDL).
 * N'utilise QUE l'API C publique (bourrasque.h), meme discipline que
 * tools/repro/test_colliders.cpp -- pas de dependance a extension/lib.py
 * (qui n'a pas encore les nouveaux prototypes, cf. spec de la tache A1).
 *
 * Modes (argv[1]) :
 *   heavy     -- verif 2 : corps tres lourd, ne doit quasiment pas bouger,
 *                le fluide doit se comporter comme face au collider statique
 *                d'aujourd'hui (dynamic=0).
 *   momentum  -- verif 3 : domaine ferme, gravite nulle, jet sur un corps
 *                libre -- conservation de la quantite de mouvement totale.
 *   rotation  -- verif 4 : impulsion decentree -> rotation dans le sens
 *                attendu ; impulsion centree -> pas de rotation.
 *   abi       -- affiche bq_abi_version() et bq_rigid_body_size().
 */
#include "bourrasque.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <array>

static float V0[3] = {0.f, 0.f, 0.f};

/* Boite triangulee (12 triangles), CCW vu de l'exterieur, demi-etendue
 * (hx,hy,hz) autour de (cx,cy,cz). Vitesse par sommet nulle (la vitesse de
 * mur vive, si dynamique, est recalculee par k_grid_update -- cf. D4). */
static int make_box(float cx, float cy, float cz, float hx, float hy, float hz,
                    float friction, std::vector<float>& tri,
                    std::vector<float>& trivel, std::vector<float>& trifric) {
    float v[8][3] = {
        {cx-hx,cy-hy,cz-hz}, {cx+hx,cy-hy,cz-hz}, {cx+hx,cy+hy,cz-hz}, {cx-hx,cy+hy,cz-hz},
        {cx-hx,cy-hy,cz+hz}, {cx+hx,cy-hy,cz+hz}, {cx+hx,cy+hy,cz+hz}, {cx-hx,cy+hy,cz+hz},
    };
    /* 6 faces, 2 tris chacune, CCW vu de l'exterieur */
    int faces[6][4] = {
        {0,3,2,1}, /* -z */ {4,5,6,7}, /* +z */
        {0,1,5,4}, /* -y */ {3,7,6,2}, /* +y */
        {0,4,7,3}, /* -x */ {1,2,6,5}, /* +x */
    };
    int n_tri = 0;
    for (auto& f : faces) {
        int tris[2][3] = { {f[0],f[1],f[2]}, {f[0],f[2],f[3]} };
        for (auto& t : tris) {
            for (int k = 0; k < 3; ++k) {
                tri.push_back(v[t[k]][0]); tri.push_back(v[t[k]][1]); tri.push_back(v[t[k]][2]);
                trivel.push_back(0.f); trivel.push_back(0.f); trivel.push_back(0.f);
            }
            trifric.push_back(friction);
            ++n_tri;
        }
    }
    return n_tri;
}

static BqRigidBody default_body() {
    BqRigidBody b{};
    b.dynamic = 1;
    b.mass = 1.f;
    /* boite 2h=0.2 cube, densite ~1000 -> I = m/6 * (2h)^2 par axe (cube) */
    b.inv_inertia[0] = b.inv_inertia[4] = b.inv_inertia[8] = 1.f;
    b.q[0] = 1.f; /* identite (w,x,y,z) */
    b.use_gravity = 0;
    b.added_mass = 1.0f;
    b.restitution = 0.f;
    return b;
}

/* ---------------------------------------------------------- verif "heavy" */
/* Barycentre du fluide -- metrique agregee, PAS une comparaison par indice
 * de particule : le solveur est non deterministe (atomicAdd flottant, cf.
 * memoire projet "verification-par-bruit-run-a-run") et le reseeding (M10)
 * fait varier le nombre de particules et leur ordre d'une execution a
 * l'autre, meme a scenario strictement identique. Deux executions du MEME
 * binaire sur la MEME scene donnent des comptes de particules differents
 * (mesure : 214671 vs 214714 sur ce test) -- comparer position[i] entre deux
 * runs n'a donc aucun sens (les deux indices i ne designent pas la meme
 * particule physique). Le barycentre, lui, converge vers la meme valeur a
 * peu pres quel que soit le detail de quelle particule est ou. */
static void barycenter(const std::vector<float>& pos, double b[3]) {
    int n = (int)pos.size() / 3;
    b[0]=b[1]=b[2]=0.0;
    for (int i = 0; i < n; ++i) { b[0]+=pos[3*i]; b[1]+=pos[3*i+1]; b[2]+=pos[3*i+2]; }
    b[0]/=n; b[1]/=n; b[2]/=n;
}

static int test_heavy() {
    printf("=== A1 verif 2 : corps tres lourd vs collider statique ===\n");
    auto run = [](bool dynamic_heavy) -> std::vector<float> {
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

        BqRigidBody body = default_body();
        body.mass = 1.0e6f; /* tres lourd */
        body.inv_inertia[0]=body.inv_inertia[4]=body.inv_inertia[8]=1e-6f;
        body.x[0]=cx; body.x[1]=cy; body.x[2]=cz;
        body.use_gravity = 0; /* isole l'effet de contact fluide, cf. rapport */
        body.dynamic = dynamic_heavy ? 1 : 0;
        bq_set_collider_bodies(sim, &body, 1);

        if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(),
                             dynamic_heavy ? tribody.data() : nullptr, n_tri) < 0) {
            fprintf(stderr, "set_colliders: %s\n", bq_last_error());
        }

        float lo[3] = {0.30f, 0.55f, 0.30f}, hi[3] = {0.70f, 0.85f, 0.70f};
        bq_emit_box(sim, mw, lo, hi, V0);

        int frames = 60;
        for (int fr = 0; fr < frames; ++fr) {
            if (bq_step(sim, 1.f/24.f) < 0) { fprintf(stderr, "step: %s\n", bq_last_error()); break; }
        }
        /* IMPORTANT : le reseeding (M10) fait varier s->n au fil des frames --
         * requeter bq_particle_count APRES la boucle, jamais avant (sinon
         * bq_read_positions ecrit au-dela d'un tampon dimensionne pour
         * l'ancien compte -- corruption tas silencieuse jusqu'au crash
         * suivant, tel que rencontre lors de la mise au point de ce
         * harnais). */
        int n = bq_particle_count(sim);
        std::vector<float> pos(3*(size_t)n);
        bq_read_positions(sim, pos.data());

        if (dynamic_heavy) {
            float bx[13];
            int nb = bq_read_collider_bodies(sim, bx);
            printf("  corps lourd dynamic=1 : deplacement = (%.6f, %.6f, %.6f) m, nb=%d\n",
                   bx[0]-cx, bx[1]-cy, bx[2]-cz, nb);
        }
        bq_destroy(sim);
        return pos;
    };

    /* A1b : l'enveloppe de bruit d'A1 etait estimee sur DEUX executions --
     * pas exploitable. Au moins 5 executions de CHAQUE cote (statique
     * dynamic=0, et dynamique-lourd dynamic=1 masse 1e6), enveloppe de
     * bruit intra-configuration mesuree DES DEUX cotes (max sur toutes les
     * paires), comparee a l'ecart inter-configuration des moyennes. */
    const int N_RUNS = 5;
    std::vector<std::array<double,3>> statB(N_RUNS), heavyB(N_RUNS);
    for (int i = 0; i < N_RUNS; ++i) {
        auto pos = run(false);
        double b[3]; barycenter(pos, b);
        statB[i] = {b[0], b[1], b[2]};
        printf("  statique   run %d : n=%zu barycentre=(%.8f,%.8f,%.8f)\n",
               i, pos.size()/3, b[0], b[1], b[2]);
    }
    for (int i = 0; i < N_RUNS; ++i) {
        auto pos = run(true);
        double b[3]; barycenter(pos, b);
        heavyB[i] = {b[0], b[1], b[2]};
        printf("  dyn.lourd  run %d : n=%zu barycentre=(%.8f,%.8f,%.8f)\n",
               i, pos.size()/3, b[0], b[1], b[2]);
    }

    auto intra_envelope = [](const std::vector<std::array<double,3>>& v) -> double {
        double mx = 0.0;
        for (size_t i = 0; i < v.size(); ++i)
            for (size_t j = i+1; j < v.size(); ++j) {
                double dx=v[i][0]-v[j][0], dy=v[i][1]-v[j][1], dz=v[i][2]-v[j][2];
                double d = sqrt(dx*dx+dy*dy+dz*dz);
                if (d > mx) mx = d;
            }
        return mx;
    };
    double noiseStat = intra_envelope(statB);
    double noiseHeavy = intra_envelope(heavyB);
    double noise_env = std::max(noiseStat, noiseHeavy);

    auto mean3 = [](const std::vector<std::array<double,3>>& v) -> std::array<double,3> {
        std::array<double,3> m{0,0,0};
        for (auto& b : v) { m[0]+=b[0]; m[1]+=b[1]; m[2]+=b[2]; }
        m[0]/=v.size(); m[1]/=v.size(); m[2]/=v.size();
        return m;
    };
    auto mStat = mean3(statB), mHeavy = mean3(heavyB);
    double dx = mStat[0]-mHeavy[0], dy = mStat[1]-mHeavy[1], dz = mStat[2]-mHeavy[2];
    double dev = sqrt(dx*dx+dy*dy+dz*dz);

    printf("\n  enveloppe de bruit intra-statique  (max sur %d paires) : %.8f m\n",
           N_RUNS*(N_RUNS-1)/2, noiseStat);
    printf("  enveloppe de bruit intra-dyn.lourd (max sur %d paires) : %.8f m\n",
           N_RUNS*(N_RUNS-1)/2, noiseHeavy);
    printf("  enveloppe retenue (max des deux)                        : %.8f m\n", noise_env);
    printf("  barycentre moyen statique  (dynamic=0) = (%.8f, %.8f, %.8f)\n", mStat[0], mStat[1], mStat[2]);
    printf("  barycentre moyen dyn.lourd (dynamic=1) = (%.8f, %.8f, %.8f)\n", mHeavy[0], mHeavy[1], mHeavy[2]);
    printf("  ecart inter-configuration (statique vs dyn.lourd)      : %.8f m (%.2fx l'enveloppe)\n",
           dev, noise_env > 1e-12 ? dev/noise_env : -1.0);
    printf("  VERDICT : %s\n", dev <= 2.0*noise_env ? "DANS LE BRUIT" : "HORS BRUIT -- A INVESTIGUER");
    return 0;
}

/* ------------------------------------------------------- verif "momentum" */
static int test_momentum() {
    printf("=== A1 verif 3 : conservation de la quantite de mouvement ===\n");
    /* Domaine large (2 m) et jet lent : le clamp de paroi (k_grid_update,
     * conditions separantes) ABSORBE la quantite de mouvement normale --
     * n'est PAS une paroi reflechissante. "Domaine ferme" au sens de cette
     * verification signifie donc : aucune particule n'atteint la bande de
     * paroi pendant la fenetre de mesure, pas que les murs conservent P.
     * On le verifie explicitement (min_wall_dist rapporte a chaque frame). */
    BqConfig cfg; bq_default_config(&cfg);
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 64;
    cfg.cell_size = 2.0f/64.f; /* domaine 2m cube */
    cfg.gravity_y = 0.f; /* verif 3 : gravite nulle, imperatif */
    BqSim* sim = bq_create(&cfg);
    BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho = 1000.f; water.bulk = 4e4f; water.gamma = 3.f;
    int mw = bq_add_material(sim, &water);
    float spacing = cfg.cell_size / cfg.ppc_axis;
    float p_mass = water.rho * spacing*spacing*spacing;
    float domain = cfg.grid_res[0] * cfg.cell_size;
    float wall_lo = 3.f * cfg.cell_size, wall_hi = domain - wall_lo; /* bound=3, cf. upload_params */

    /* corps libre au centre du domaine, sans gravite, sans verrou */
    float cx=1.0f, cy=1.0f, cz=1.0f, h=0.08f;
    std::vector<float> tri, trivel, trifric;
    int n_tri = make_box(cx, cy, cz, h, h, h, 0.2f, tri, trivel, trifric);
    std::vector<int> tribody(n_tri, 0);

    BqRigidBody body = default_body();
    body.mass = 4.0f; /* comparable a la masse de fluide en jeu */
    /* added_mass = 0 pour CE test : D5 (masse ajoutee) integre volontairement
     * le corps avec m_eff = mass + added_mass*m_contact a la place de mass,
     * ce qui rend la reponse du corps plus molle que l'impulsion recoltee ne
     * le voudrait -- un compromis de stabilite assume (cf. plan-milestone-17
     * D5, "a alpha=1 un bouchon tres leger repond un peu mollement"), PAS un
     * defaut de signe/facteur de la recolte elle-meme. Le laisser a sa valeur
     * par defaut (1.0) ici masquerait ce que ce test doit isoler -- la
     * recolte D1 est testee separement de l'amortissement D5, cf. rapport. */
    body.added_mass = 0.f;
    float Ibox = body.mass/6.f * (2*h)*(2*h);
    body.inv_inertia[0]=body.inv_inertia[4]=body.inv_inertia[8]=1.f/Ibox;
    body.x[0]=cx; body.x[1]=cy; body.x[2]=cz;
    bq_set_collider_bodies(sim, &body, 1);
    bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), tribody.data(), n_tri);

    /* jet de fluide, vitesse initiale non nulle vers +x, vise le centre de
     * masse (pas de couple attendu, seule la conservation lineaire est
     * testee ici -- verif 4 couvre la rotation separement) */
    float lo[3] = {0.55f, 0.92f, 0.92f}, hi[3] = {0.70f, 1.08f, 1.08f};
    float vjet[3] = {1.0f, 0.f, 0.f};
    bq_emit_box(sim, mw, lo, hi, vjet);
    int n = bq_particle_count(sim);
    printf("  domaine %.2fm cube, mur utile [%.4f, %.4f]\n", domain, wall_lo, wall_hi);
    printf("  %d particules de fluide, p_mass=%.6e kg, corps mass=%.3f kg\n", n, p_mass, body.mass);

    std::vector<float> pos(3*(size_t)n), vel(3*(size_t)n);
    auto total_p_and_wall = [&](double p[3], float* min_wall) {
        bq_read_velocities(sim, vel.data());
        bq_read_positions(sim, pos.data());
        double px=0,py=0,pz=0;
        float mw_dist = 1e9f;
        for (int i = 0; i < n; ++i) {
            px += vel[3*i]; py += vel[3*i+1]; pz += vel[3*i+2];
            for (int a = 0; a < 3; ++a) {
                float d = std::min(pos[3*i+a]-wall_lo, wall_hi-pos[3*i+a]);
                if (d < mw_dist) mw_dist = d;
            }
        }
        px *= p_mass; py *= p_mass; pz *= p_mass;
        double fluid_p[3] = {px, py, pz};
        float bs[13]; bq_read_collider_bodies(sim, bs);
        fprintf(stderr, "    [dbg] fluid_P=(%.6f,%.6f,%.6f) body_v=(%.6f,%.6f,%.6f) body_P=(%.6f,%.6f,%.6f)\n",
                fluid_p[0], fluid_p[1], fluid_p[2], bs[7], bs[8], bs[9],
                body.mass*bs[7], body.mass*bs[8], body.mass*bs[9]);
        px += body.mass * bs[7]; py += body.mass * bs[8]; pz += body.mass * bs[9];
        p[0]=px; p[1]=py; p[2]=pz;
        *min_wall = mw_dist;
    };

    double p0[3]; float w0; total_p_and_wall(p0, &w0);
    double p0norm = sqrt(p0[0]*p0[0]+p0[1]*p0[1]+p0[2]*p0[2]);
    printf("  P(0) = (%.6f, %.6f, %.6f), |P0|=%.6f\n", p0[0],p0[1],p0[2], p0norm);

    int frames = 14;
    double max_rel_drift = 0.0;
    float min_wall_seen = w0;
    for (int fr = 0; fr < frames; ++fr) {
        if (bq_step(sim, 1.f/24.f) < 0) { fprintf(stderr, "step: %s\n", bq_last_error()); break; }
        double p[3]; float wd; total_p_and_wall(p, &wd);
        if (wd < min_wall_seen) min_wall_seen = wd;
        double dx=p[0]-p0[0], dy=p[1]-p0[1], dz=p[2]-p0[2];
        double drift = sqrt(dx*dx+dy*dy+dz*dz);
        double rel = drift / p0norm;
        if (rel > max_rel_drift) max_rel_drift = rel;
        printf("  frame %2d : P=(%.6f,%.6f,%.6f) derive relative=%.6f  min_dist_mur=%.4f\n",
               fr, p[0],p[1],p[2], rel, wd);
    }
    printf("  derive relative MAX sur %d frames : %.6f (%.4f %%)\n", frames, max_rel_drift, 100.0*max_rel_drift);
    printf("  distance minimale au mur observee : %.4f (domaine reste ferme si > 0)\n", min_wall_seen);
    bq_destroy(sim);
    return 0;
}

/* ------------------------------------------------------- verif "rotation" */
static int test_rotation() {
    printf("=== A1 verif 4 : rotation (poussee decentree / centree) ===\n");
    auto run = [](bool offset) -> void {
        BqConfig cfg; bq_default_config(&cfg);
        cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 48;
        cfg.cell_size = 1.f/48.f;
        cfg.gravity_y = 0.f;
        BqSim* sim = bq_create(&cfg);
        BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho = 1000.f; water.bulk = 4e4f; water.gamma = 3.f;
        int mw = bq_add_material(sim, &water);

        float cx=0.5f, cy=0.5f, cz=0.5f, h=0.08f;
        std::vector<float> tri, trivel, trifric;
        int n_tri = make_box(cx, cy, cz, h, h, h, 0.2f, tri, trivel, trifric);
        std::vector<int> tribody(n_tri, 0);

        BqRigidBody body = default_body();
        body.mass = 4.0f;
        float Ibox = body.mass/6.f * (2*h)*(2*h);
        body.inv_inertia[0]=body.inv_inertia[4]=body.inv_inertia[8]=1.f/Ibox;
        body.x[0]=cx; body.x[1]=cy; body.x[2]=cz;
        bq_set_collider_bodies(sim, &body, 1);
        bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(), tribody.data(), n_tri);

        /* jet vers +x, vise soit le centre (cy), soit decale vers +y (cy+0.5h)
         * pour produire un couple autour de +z (r=(0,+dy,0) x F=(+Fx,0,0) =
         * (0,0,-dy*Fx) -- couple negatif en z attendu si offset > 0, cf.
         * commentaire plus bas au moment de l'evaluation). */
        float oy = offset ? 0.5f*h : 0.f;
        float lo[3] = {0.10f, cy+oy-0.03f, cz-0.03f}, hi[3] = {0.30f, cy+oy+0.03f, cz+0.03f};
        float vjet[3] = {2.5f, 0.f, 0.f};
        bq_emit_box(sim, mw, lo, hi, vjet);

        int frames = 10;
        for (int fr = 0; fr < frames; ++fr) bq_step(sim, 1.f/24.f);

        float bs[13]; bq_read_collider_bodies(sim, bs);
        float wx=bs[10], wy=bs[11], wz=bs[12];
        printf("  offset=%d : v=(%.5f,%.5f,%.5f) w=(%.5f,%.5f,%.5f) |w|=%.6f\n",
               offset, bs[7],bs[8],bs[9], wx,wy,wz, sqrt((double)wx*wx+wy*wy+wz*wz));
        bq_destroy(sim);
    };
    printf("  jet vise le centre de masse (attendu : |w| proche de 0) :\n");
    run(false);
    printf("  jet decale de +0.5h en y (attendu : rotation autour de -z, wz < 0) :\n");
    run(true);
    return 0;
}

static void print_abi() {
    printf("bq_abi_version() = %d\n", bq_abi_version());
    printf("bq_rigid_body_size() = %zu (sizeof(BqRigidBody) cote appelant = %zu)\n",
           bq_rigid_body_size(), sizeof(BqRigidBody));
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "all";
    if (mode == "abi") { print_abi(); return 0; }
    print_abi();
    if (mode == "heavy" || mode == "all") test_heavy();
    if (mode == "momentum" || mode == "all") test_momentum();
    if (mode == "rotation" || mode == "all") test_rotation();
    return 0;
}

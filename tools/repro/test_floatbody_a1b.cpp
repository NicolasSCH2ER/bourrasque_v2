/* Harnais de mesure M17/A1b (voir spec de la tache) : stabilite d'un corps
 * FLOTTANT lache dans une colonne d'eau au repos, gravite normale, en
 * fonction de la densite du corps et de added_mass (alpha).
 *
 * N'utilise QUE l'API C publique (bourrasque.h), meme discipline que
 * test_rigidbody_a1.cpp -- pas de dependance a extension/lib.py.
 *
 * IMPORTANT (D4 du plan-milestone-17) : la geometrie du champ collider vue
 * par le fluide n'est PAS recalculee depuis l'etat du corps a chaque
 * sous-pas -- seule la VITESSE de mur est vive. La geometrie doit donc etre
 * reuploadee UNE FOIS PAR FRAME via bq_set_colliders, avec les triangles
 * repositionnes depuis l'etat courant (x, q) lu par bq_read_collider_bodies
 * -- exactement le role que jouera extension/ops.py (_collect_collider_frame)
 * en production. Omettre ce reupload (comme la premiere version de ce
 * harnais) fige la geometrie a sa position initiale : le corps qui tombe
 * s'en detache en quelques frames, m_contact retombe a zero, et on observe
 * une chute libre qui n'a RIEN a voir avec le couplage -- piege verifie et
 * documente ici pour ne pas le retomber.
 *
 * Usage : test_floatbody_a1b <density> <added_mass> [frames]
 *   density    : kg/m3 du corps (eau = 1000)
 *   added_mass : alpha (cf. D5 du plan-milestone-17)
 *   frames     : nombre de frames de MESURE a 24 Hz (defaut 72 = 3s),
 *                APRES 1s de pre-etablissement de la colonne (cf. plus bas)
 *
 * Sortie : une ligne CSV sur stdout, plus le detail par frame sur stderr
 * (pour diagnostic, pas parse) :
 *   density,alpha,diverged,diverge_frame,exited_water,exited_frame,
 *   eq_frac,archimede_frac,amp_frac,amp_trend
 *
 * exited_water : le corps a touche le fond de la colonne d'eau -- PAS une
 * instabilite du couplage : aucun sol rigide n'existe en phase A (le
 * contact corps-corps est hors perimetre de ce jalon), donc un corps plus
 * dense que l'eau finit toujours par sortir par le fond puis tomber en
 * espace vide. La mesure s'arrete a cet instant (eq_frac etc. restent a 0).
 *
 * eq_frac       : fraction immergee moyenne sur la derniere seconde simulee
 * archimede_frac: densite/1000 (valeur analytique attendue, flottant, coule
 *                 au-dela de 1.0)
 * amp_frac      : amplitude d'oscillation (max-min)/2 sur la derniere
 *                 seconde simulee de la fraction immergee
 * amp_trend     : rapport amplitude(derniere seconde)/amplitude(seconde
 *                 precedente) -- <1 = oscillation qui decroit (stable),
 *                 >~1 = oscillation entretenue ou qui croit
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

/* 8 sommets d'un cube en repere LOCAL (centre au centre de masse), et
 * topologie fixe (12 triangles, CCW vu de l'exterieur) -- reutilises a
 * chaque frame, seule la transformation (x, q) change. */
static void local_box_verts(float h, float out[8][3]) {
    float v[8][3] = {
        {-h,-h,-h}, {+h,-h,-h}, {+h,+h,-h}, {-h,+h,-h},
        {-h,-h,+h}, {+h,-h,+h}, {+h,+h,+h}, {-h,+h,+h},
    };
    memcpy(out, v, sizeof(v));
}
static const int FACE_QUADS[6][4] = {
    {0,3,2,1}, {4,5,6,7},
    {0,1,5,4}, {3,7,6,2},
    {0,4,7,3}, {1,2,6,5},
};

/* rotation d'un vecteur par un quaternion (w,x,y,z) suppose unitaire */
static void quat_rotate(const float q[4], const float v[3], float out[3]) {
    float qw=q[0], qx=q[1], qy=q[2], qz=q[3];
    float tx = 2.f*(qy*v[2]-qz*v[1]);
    float ty = 2.f*(qz*v[0]-qx*v[2]);
    float tz = 2.f*(qx*v[1]-qy*v[0]);
    out[0] = v[0] + qw*tx + (qy*tz - qz*ty);
    out[1] = v[1] + qw*ty + (qz*tx - qx*tz);
    out[2] = v[2] + qw*tz + (qx*ty - qy*tx);
}

/* reconstruit le tampon de triangles monde depuis l'etat courant du corps */
static int build_tri(float h, const float x[3], const float q[4], float friction,
                     std::vector<float>& tri, std::vector<float>& trivel,
                     std::vector<float>& trifric) {
    float lv[8][3]; local_box_verts(h, lv);
    float wv[8][3];
    for (int i = 0; i < 8; ++i) {
        float r[3]; quat_rotate(q, lv[i], r);
        wv[i][0] = r[0]+x[0]; wv[i][1] = r[1]+x[1]; wv[i][2] = r[2]+x[2];
    }
    tri.clear(); trivel.clear(); trifric.clear();
    int n_tri = 0;
    for (auto& f : FACE_QUADS) {
        int tris[2][3] = { {f[0],f[1],f[2]}, {f[0],f[2],f[3]} };
        for (auto& t : tris) {
            for (int k = 0; k < 3; ++k) {
                tri.push_back(wv[t[k]][0]); tri.push_back(wv[t[k]][1]); tri.push_back(wv[t[k]][2]);
                /* dynamique : vitesse ignoree par k_grid_update (D4, vitesse
                   vive recalculee depuis l'etat du corps) -- laissee a 0. */
                trivel.push_back(0.f); trivel.push_back(0.f); trivel.push_back(0.f);
            }
            trifric.push_back(friction);
            ++n_tri;
        }
    }
    return n_tri;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <density> <added_mass> [frames]\n", argv[0]);
        return 1;
    }
    float density = (float)atof(argv[1]);
    float alpha = (float)atof(argv[2]);
    int frames = (argc > 3) ? atoi(argv[3]) : 72; /* 3s a 24Hz */

    BqConfig cfg; bq_default_config(&cfg);
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 48;
    cfg.cell_size = 1.f/48.f;
    cfg.gravity_y = -9.8f; /* gravite normale, imperatif du scenario */
    BqSim* sim = bq_create(&cfg);
    if (!sim) { fprintf(stderr, "create: %s\n", bq_last_error()); return 1; }

    BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho = 1000.f;
    water.bulk = 4e4f; water.gamma = 3.f;
    int mw = bq_add_material(sim, &water);
    if (mw < 0) { fprintf(stderr, "add_material: %s\n", bq_last_error()); return 1; }

    /* colonne d'eau au repos, PROFONDE : domaine [0,1]^3, bande de mur
     * absorbante ~3 cellules (0.0625 a grille 48). Une colonne peu profonde
     * laisse un corps dense la traverser tout entiere avant qu'un equilibre
     * ne puisse s'etablir -- il "tombe" alors en espace vide sous l'eau
     * (pas de sol rigide en phase A). Colonne a 0.77m ici. */
    float surface_y = 0.85f;
    float water_bottom = 0.08f;
    float wlo[3] = {0.15f, water_bottom, 0.15f}, whi[3] = {0.85f, surface_y, 0.85f};
    if (bq_emit_box(sim, mw, wlo, whi, V0) < 0) {
        fprintf(stderr, "emit_box: %s\n", bq_last_error()); return 1;
    }

    /* colonne d'eau "au repos" au sens de l'enonce : la pression
     * hydrostatique doit avoir eu le temps de s'etablir AVANT qu'on y
     * lache le corps -- a l'instant t=0 le materiau demarre non comprime
     * (J=1). 1s de pre-etablissement, sans corps, avant la mesure. */
    for (int fr = 0; fr < 24; ++fr) {
        if (bq_step(sim, 1.f/24.f) < 0) {
            fprintf(stderr, "step (pre-etablissement): %s\n", bq_last_error()); return 1;
        }
    }

    /* corps flottant : cube 0.12m de cote, centre initial AU NIVEAU de la
     * surface (fraction immergee initiale = 0.5, point de depart commun a
     * toutes les densites -- le systeme converge ensuite vers l'equilibre
     * propre a chaque densite). */
    float h = 0.06f;
    float side = 2.f*h;
    float x0[3] = {0.5f, surface_y, 0.5f};
    float q0[4] = {1.f, 0.f, 0.f, 0.f};
    std::vector<float> tri, trivel, trifric;
    int n_tri = build_tri(h, x0, q0, 0.3f, tri, trivel, trifric);
    std::vector<int> tribody(n_tri, 0);

    BqRigidBody body{};
    body.dynamic = 1;
    body.mass = density * side*side*side;
    float Ibox = body.mass/6.f * side*side;
    body.inv_inertia[0] = body.inv_inertia[4] = body.inv_inertia[8] = 1.f/Ibox;
    body.x[0]=x0[0]; body.x[1]=x0[1]; body.x[2]=x0[2];
    body.q[0]=1.f;
    body.use_gravity = 1; /* corps flottant : soumis a son propre poids */
    body.added_mass = alpha;
    body.restitution = 0.f;
    if (bq_set_collider_bodies(sim, &body, 1) < 0) {
        fprintf(stderr, "set_collider_bodies: %s\n", bq_last_error()); return 1;
    }
    if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(),
                         tribody.data(), n_tri) < 0) {
        fprintf(stderr, "set_colliders: %s\n", bq_last_error()); return 1;
    }

    std::vector<float> frac(frames, 0.f);
    bool diverged = false;
    int diverge_frame = -1;
    bool exited_water = false; /* touche le fond de la colonne -- pas une
                                   instabilite du couplage : aucun sol rigide
                                   en phase A (contact corps-corps hors
                                   perimetre), le corps tombe en espace vide
                                   une fois sorti de l'eau. */
    int exited_frame = -1;
    float bs[13] = {0};
    bs[0]=x0[0]; bs[1]=x0[1]; bs[2]=x0[2]; bs[3]=1.f;

    for (int fr = 0; fr < frames; ++fr) {
        /* D4 : reupload de la geometrie collider depuis l'etat de FIN de
         * frame precedente (frame 0 : etat initial deja uploade ci-dessus). */
        if (fr > 0) {
            float xcur[3] = {bs[0], bs[1], bs[2]};
            float qcur[4] = {bs[3], bs[4], bs[5], bs[6]};
            build_tri(h, xcur, qcur, 0.3f, tri, trivel, trifric);
            if (bq_set_colliders(sim, tri.data(), trivel.data(), trifric.data(),
                                 tribody.data(), n_tri) < 0) {
                fprintf(stderr, "set_colliders (refresh): %s\n", bq_last_error());
                return 1;
            }
        }
        if (bq_step(sim, 1.f/24.f) < 0) {
            fprintf(stderr, "step: %s\n", bq_last_error());
            diverged = true; diverge_frame = fr; break;
        }
        bq_read_collider_bodies(sim, bs);
        float by = bs[1];
        float vy = bs[8];
        bool bad = std::isnan(by) || std::isnan(vy) ||
                   std::fabs(vy) > 50.f || by < -1.f || by > 2.f;
        if (bad) {
            diverged = true; diverge_frame = fr;
            fprintf(stderr, "  DIVERGENCE frame %d : y=%.6f vy=%.6f\n", fr, by, vy);
            break;
        }
        if ((by - h) < water_bottom + 0.02f) {
            exited_water = true; exited_frame = fr;
            fprintf(stderr, "  SORTIE DU FOND DE LA COLONNE frame %d : y=%.6f vy=%.6f "
                            "(pas de sol rigide en phase A -- arret de la mesure)\n",
                    fr, by, vy);
            break;
        }
        /* fraction immergee approximee : surface d'eau supposee proche de
         * sa hauteur initiale (bassin large devant la section du corps).
         * Convention : bas du corps = by - h, haut = by + h. */
        float submerged = surface_y - (by - h);
        float f = submerged / side;
        if (f < 0.f) f = 0.f;
        if (f > 1.f) f = 1.f;
        frac[fr] = f;
        fprintf(stderr, "  frame %2d : y=%.6f vy=%.6f frac_immergee=%.4f\n", fr, by, vy, f);
    }

    float eq_frac = 0.f, amp_frac = 0.f, amp_prev = 0.f, trend = -1.f;
    float archimede = density / 1000.f;
    if (archimede > 1.f) archimede = 1.f; /* coule, pas de flottaison analytique */

    if (!diverged && !exited_water) {
        int last_n = std::min(24, frames); /* derniere seconde simulee */
        int start = frames - last_n;
        float mn = 1e9f, mx = -1e9f, sum = 0.f;
        for (int i = start; i < frames; ++i) {
            mn = std::min(mn, frac[i]); mx = std::max(mx, frac[i]); sum += frac[i];
        }
        eq_frac = sum / last_n;
        amp_frac = (mx - mn) * 0.5f;

        int prev_n = std::min(24, start);
        if (prev_n > 0) {
            int pstart = start - prev_n;
            float pmn = 1e9f, pmx = -1e9f;
            for (int i = pstart; i < start; ++i) {
                pmn = std::min(pmn, frac[i]); pmx = std::max(pmx, frac[i]);
            }
            amp_prev = (pmx - pmn) * 0.5f;
            trend = (amp_prev > 1e-6f) ? (amp_frac / amp_prev) : -1.f;
        }
    }

    printf("%.1f,%.2f,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f\n",
           density, alpha, diverged ? 1 : 0, diverge_frame,
           exited_water ? 1 : 0, exited_frame,
           eq_frac, archimede, amp_frac, trend);

    bq_destroy(sim);
    return 0;
}

/* Verif 1 (decisive) de la tache A1 (M17) : aucun corps declare, aucun
 * collider -- le bake doit etre INCHANGE par rapport au binaire d'avant A1.
 * Reprend le scenario "dam" de bourrasque_headless (meme boite, meme
 * cadence), mais NE REUTILISE PAS son buffer de positions de taille fixe :
 * le reseeding (M10) fait grandir le nombre de particules frame apres
 * frame, et bourrasque_headless (core/headless/main.cpp, hors perimetre de
 * cette tache) dimensionne son tampon UNE FOIS avant la boucle -- bug
 * latent preexistant qui debordait silencieusement le tas jusqu'a cette
 * tache (le nouvel etat BqSim, plus gros, a change assez la disposition du
 * tas pour transformer la corruption silencieuse en crash observable). Non
 * introduit par A1, non corrige ici (core/headless/main.cpp hors perimetre),
 * contourne dans CE harnais en requetant bq_particle_count() a CHAQUE frame.
 *
 * Compile deux fois (cf. tools/repro/build_nr.bat) : une fois contre le
 * bourrasque.lib COURANT (post-A1), une fois contre une copie de reference
 * (pre-A1, cf. C:\tmp\bq_ref) -- meme source, deux .lib/.dll differents.
 * Aucun appel a bq_set_colliders ni bq_set_collider_bodies : exercice exact
 * du chemin n_bodies == 0 / n_tri == 0. */
#include "bourrasque.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    int frames = (argc > 1) ? atoi(argv[1]) : 40;
    BqConfig cfg; bq_default_config(&cfg);
    BqSim* sim = bq_create(&cfg);
    if (!sim) { fprintf(stderr, "create: %s\n", bq_last_error()); return 1; }
    BqMaterial water{}; water.model = BQ_MODEL_WATER; water.rho=1000.f; water.bulk=4e4f; water.gamma=3.f;
    int mw = bq_add_material(sim, &water);
    float lo[3]={0.10f,0.10f,0.10f}, hi[3]={0.35f,0.60f,0.90f}, v0[3]={0,0,0};
    bq_emit_box(sim, mw, lo, hi, v0);

    for (int fr = 0; fr < frames; ++fr) {
        if (bq_step(sim, 1.f/24.f) < 0) { fprintf(stderr, "step: %s\n", bq_last_error()); return 1; }
    }
    int n = bq_particle_count(sim); /* APRES la boucle -- cf. commentaire en tete */
    std::vector<float> pos(3*(size_t)n);
    bq_read_positions(sim, pos.data());

    double bx=0, by=0, bz=0;
    for (int i = 0; i < n; ++i) { bx += pos[3*i]; by += pos[3*i+1]; bz += pos[3*i+2]; }
    bx/=n; by/=n; bz/=n;
    printf("n=%d barycentre=(%.8f,%.8f,%.8f)\n", n, bx, by, bz);
    bq_destroy(sim);
    return 0;
}

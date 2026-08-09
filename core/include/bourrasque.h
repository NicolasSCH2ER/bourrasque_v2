/* bourrasque.h — API C plate du solveur MLS-MPM GPU.
 *
 * Zero dependance : consommable par ctypes (extension Blender), par le
 * headless, ou par n'importe quel hote C/C++. Toutes les fonctions
 * retournent un code >= 0 en cas de succes, < 0 en cas d'erreur
 * (bq_last_error() donne le detail).
 */
#ifndef BOURRASQUE_H
#define BOURRASQUE_H

#include <stdint.h>
#include <stddef.h> /* size_t, pour bq_rigid_body_size */

#if defined(_WIN32)
#  ifdef BQ_BUILD
#    define BQ_API __declspec(dllexport)
#  else
#    define BQ_API __declspec(dllimport)
#  endif
#else
#  define BQ_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BqSim BqSim; /* handle opaque */

/* Version de l'ABI : a incrementer des que la disposition d'une struct
   publique ou la signature d'une fonction exportee change. */
#define BQ_ABI_VERSION 11

/* Modeles constitutifs. D'autres viendront (sable, neige) sans changer
 * l'API : c'est tout l'interet du pipeline MPM unifie. */
enum BqModel {
    BQ_MODEL_ELASTIC = 0, /* corotationnel fixe (E, nu)        */
    BQ_MODEL_WATER   = 1  /* EOS de Tait J-based (bulk, gamma) */
};

typedef struct BqConfig {
    int   grid_res[3];   /* nombre de cellules par axe (x, y, z) (defaut 64,64,64) */
    float cell_size;     /* taille de maille, uniforme sur les trois axes, en metres (defaut 1/64) */
    float gravity_y;     /* defaut -9.8                                 */
    float cfl;           /* defaut 0.3 (CFL acoustique)                 */
    int   ppc_axis;      /* particules/cellule/axe (defaut 2 -> 8/cell) */
    int   max_particles; /* capacite allouee (defaut 2'000'000)         */
} BqConfig;

typedef struct BqMaterial {
    int   model;  /* BqModel                                   */
    float rho;    /* densite kg/m^3                            */
    float E;      /* module de Young (ELASTIC)                 */
    float nu;     /* coefficient de Poisson (ELASTIC)          */
    float bulk;   /* module de compressibilite k (WATER)       */
    float gamma;  /* exposant de Tait (WATER, typ. 3-7)        */
} BqMaterial;

/* Version d'ABI de la bibliotheque compilee. A comparer a BQ_ABI_VERSION
   par l'appelant avant tout autre appel. */
BQ_API int bq_abi_version(void);

/* sizeof(BqConfig) tel que la bibliotheque le voit. Detecte un decalage de
   struct meme si la version d'ABI a ete oubliee. */
BQ_API int bq_config_size(void);

/* Remplit cfg avec les valeurs par defaut. */
BQ_API void bq_default_config(BqConfig* cfg);

/* Cree/detruit une simulation. bq_create retourne NULL en cas d'echec. */
BQ_API BqSim* bq_create(const BqConfig* cfg);
BQ_API void   bq_destroy(BqSim* sim);

/* Enregistre un materiau, retourne son id (max 8 materiaux). */
BQ_API int bq_add_material(BqSim* sim, const BqMaterial* mat);

/* Emet un bloc AABB de particules du materiau donne.
 * Retourne le nombre de particules emises. */
BQ_API int bq_emit_box(BqSim* sim, int mat_id,
                       const float lo[3], const float hi[3],
                       const float vel[3]);

/* Emet des particules a des positions explicites (count triplets xyz).
 * Meme initialisation que bq_emit_box. Retourne le nombre emis, ou < 0. */
BQ_API int bq_emit_points(BqSim* sim, int mat_id,
                          const float* pos, int count,
                          const float vel[3]);

/* Comme bq_emit_points, mais avec une vitesse propre a chaque particule.
   `vel` pointe sur count*3 floats (vx,vy,vz entrelaces), meme indexation
   que `pos`. Renvoie le nombre de particules emises, ou -1 sur erreur. */
BQ_API int bq_emit_points_vel(BqSim* sim, int mat_id,
                              const float* pos, const float* vel, int count);

/* Avance d'une frame ; le solveur decoupe en substeps via la CFL.
 * Retourne le nombre de substeps effectues. */
BQ_API int bq_step(BqSim* sim, float frame_dt);

/* Remplace l'ensemble des colliders. Les triangles de tous les objets sont
   concatenes. `tri` : 9*n_tri floats (3 sommets xyz, espace solveur).
   `tri_vel` : 9*n_tri floats, vitesse par SOMMET. `tri_friction` : n_tri
   floats, coefficient par triangle. `tri_body` : n_tri int32, indice du
   corps rigide (cf. bq_set_collider_bodies) proprietaire de chaque triangle,
   ou NULL -- dans ce cas tous les triangles sont attribues au corps 0 si des
   corps ont ete declares, ou a aucun corps (-1) sinon. Sert a attribuer
   l'impulsion recoltee par cellule (k_grid_update) au bon corps : le
   triangle le plus proche d'une cellule (deja calcule pour la distance)
   donne son corps a la cellule, sans recherche supplementaire. A appeler une
   fois par frame, avant bq_step. n_tri == 0 efface les colliders. Renvoie 0,
   ou -1 sur erreur. */
BQ_API int bq_set_colliders(BqSim* sim, const float* tri, const float* tri_vel,
                            const float* tri_friction, const int* tri_body,
                            int n_tri);

BQ_API int bq_particle_count(const BqSim* sim);

/* Copie device -> hote. dst doit contenir n*3 floats / n octets. */
BQ_API int bq_read_positions(BqSim* sim, float* dst);
BQ_API int bq_read_materials(BqSim* sim, uint8_t* dst);

/* Copie les vitesses courantes vers dst (n*3 floats), meme convention que
   bq_read_positions. Retourne le nombre de particules copiees. */
BQ_API int bq_read_velocities(BqSim* sim, float* dst);

/* Copie les `n` valeurs de J (etat de volume, modele WATER) vers dst, qui doit
   pointer sur au moins bq_particle_count(sim) floats. Retourne le nombre de
   valeurs copiees, ou -1 (bq_last_error rempli). Pour un materiau ELASTIC, la
   valeur retournee n'a pas de sens : J y est porte par det(F), pas par ce
   tampon. */
BQ_API int bq_read_J(BqSim* sim, float* dst);

/* Copie le champ de distance signee courant vers `dst`, qui doit pouvoir
   contenir grid_res[0]*grid_res[1]*grid_res[2] floats. Diagnostic et
   validation. Renvoie le nombre de cellules copiees, ou -1 sur erreur. */
BQ_API int bq_read_sdf(BqSim* sim, float* dst);

/* Copie le champ de normale de contact courant vers `dst`, qui doit pouvoir
   contenir grid_res[0]*grid_res[1]*grid_res[2]*4 floats. Meme indexation AUX
   NOEUDS que bq_read_sdf (idx = (i*res[1]+j)*res[2]+k, point echantillonne a
   (i*cell_size, j*cell_size, k*cell_size)). Chaque cellule fournit 4 floats :
   x,y,z la normale de contact unitaire (direction point-le-plus-proche-sur-
   triangle -> cellule, exacte, calculee sur le maillage collider brut, pas un
   gradient de champ rasterise), w la distance NON signee jusqu'au triangle le
   plus proche. Diagnostic et validation, et source destinee a etre retransmise
   telle quelle a bq_whitewater_set_collider_cnrm. Renvoie le nombre de
   cellules copiees, ou -1 sur erreur (bq_last_error rempli). */
BQ_API int bq_read_cnrm(BqSim* sim, float* dst);

/* ------------------------------------------------------- corps rigides
 *
 * Couplage fluide -> solide (M17, phase A) : le solveur porte l'etat de
 * chaque corps (position, orientation, vitesses) et l'integre a la cadence
 * du sous-pas -- l'impulsion que le fluide donne a un corps est recoltee
 * gratuitement la ou la condition de contact la retire au fluide
 * (troisieme loi de Newton, cf. k_grid_update dans mlsmpm.cu), aucune
 * integrale de pression a reconstruire. Python possede la geometrie
 * (maillage, proprietes massiques) et relit l'etat apres chaque bq_step ;
 * le solveur ne connait que la boite englobante physique du corps (masse,
 * inertie) et son etat cinematique.
 */
typedef struct BqRigidBody {
    int   dynamic;          /* 0 = collider cinematique/statique, comportement actuel inchange */
    float mass;             /* kg, > 0 si dynamic */
    float inv_inertia[9];   /* inverse du tenseur d'inertie AU CENTRE DE MASSE, repere de
                               corps, rangee-majeur */
    float x[3];             /* position initiale du centre de masse, espace solveur */
    float q[4];             /* orientation initiale, (w, x, y, z) */
    float v[3];             /* vitesse lineaire initiale */
    float w[3];             /* vitesse angulaire initiale */
    int   use_gravity;
    float added_mass;       /* alpha, cf. D5 du plan */
    int   lock_lin[3];      /* 1 = axe monde bloque en translation */
    int   lock_ang[3];
    float restitution;      /* reserve pour la phase B (contact corps-corps), non lu ici */
} BqRigidBody;

/* sizeof(BqRigidBody) tel que la bibliotheque le voit -- meme motif que
   bq_config_size, verifie cote extension avant tout appel. */
BQ_API size_t bq_rigid_body_size(void);

/* Declare l'ensemble des corps rigides et REINITIALISE leur etat (x, q, v,
   w) aux valeurs fournies dans `bodies`. A appeler une fois au debut du
   bake -- pas par frame : l'etat evolue ensuite en interne (integre par
   bq_step), rappeler cette fonction l'ecraserait. n_bodies == 0 efface tous
   les corps (equivalent a n'en avoir jamais declare -- k_grid_update retombe
   alors integralement sur le chemin cinematique actuel). Plafond de 64
   corps ; le depasser est une erreur explicite (bq_last_error rempli),
   jamais une troncature silencieuse. Renvoie 0, ou -1 sur erreur. */
BQ_API int bq_set_collider_bodies(BqSim* sim, const BqRigidBody* bodies, int n_bodies);

/* Copie l'etat courant des corps vers `dst` (n_bodies*13 floats :
   x[3], q[4], v[3], w[3] par corps, meme ordre que dans BqRigidBody).
   Diagnostic et retour vers Blender (keyframes). Renvoie le nombre de corps
   copies (peut etre 0 si aucun corps declare), ou -1 sur erreur. */
BQ_API int bq_read_collider_bodies(BqSim* sim, float* dst);

/* Copie vers `dst` (n_bodies*7 floats par corps : impulsion lineaire [3],
   impulsion angulaire (couple) [3], masse de fluide en contact [1]) le
   wrench recolte par le DERNIER sous-pas effectue -- diagnostic, et utilise
   par les tests de validation de conservation de quantite de mouvement.
   N'est PAS une somme sur la frame : chaque sous-pas remet l'accumulateur a
   zero (cf. bq_step). Renvoie le nombre de corps copies (peut etre 0), ou
   -1 sur erreur. */
BQ_API int bq_read_collider_wrench(BqSim* sim, float* dst);

BQ_API const char* bq_last_error(void);

/* ---------------------------------------------------------------- mailleur
 *
 * Champ de distance signee de Zhu-Bridson (2005), independant de tout
 * BqSim : consomme un nuage de positions, d'ou qu'il vienne (etat vivant du
 * solveur ou cache .bqd relu). Pas de marching cubes ici -- seulement le
 * champ, verifiable independamment de la polygonisation.
 */
typedef struct BqMesherConfig {
    int   grid_res[3];        /* resolution du champ de maillage */
    float cell_size;           /* taille de cellule du champ, unites monde */
    float influence_radius;    /* R de Zhu-Bridson */
    float particle_radius;     /* r_i de Zhu-Bridson */
    float collider_offset;     /* cf. bq_mesher_set_collider_sdf : decale le rognage */
    int   smoothing_iters;     /* nombre de paires de passes Taubin (lambda puis mu, non-retrecissantes) sur le champ avant rognage collider (0 = desactive) */
    int   min_component_tris;  /* composantes connexes du maillage de moins de N triangles supprimees apres le marching cubes (0 = desactive) */
    int   channels;            /* inutilise dans T1, prevoir le champ */
} BqMesherConfig;

typedef struct BqMesher BqMesher; /* handle opaque */

/* sizeof(BqMesherConfig) tel que la bibliotheque le voit. */
BQ_API int bq_mesher_config_size(void);

/* Remplit cfg avec les valeurs par defaut, coherentes avec la config de
   simulation par defaut (grille 64^3, cell_size 1/64, ppc_axis 2, donc un
   pas inter-particules de 1/128) :
     grid_res         = {128, 128, 128}
     cell_size        = 1/128
     influence_radius = 3/128
     particle_radius  = 1/128
     collider_offset  = 0
     smoothing_iters  = 0
     min_component_tris = 50
     channels         = 0 */
BQ_API void bq_mesher_default_config(BqMesherConfig* cfg);

/* Empreinte VRAM du champ (sans le champ collider ni le marching cubes),
   sans rien allouer : bytes = n_cells*4 (champ) + n_cells*4 (tampon de
   lissage) + n_buckets*4 + n_particles*4 + n_particles*12. n_particles peut
   valoir 0 (part independante des particules seule, cf. bq_mesher_create).
   Renvoie 0, ou -1 sur erreur. */
BQ_API int bq_mesher_vram_estimate(const BqMesherConfig* cfg, int n_particles,
                                   int64_t* bytes);

/* Cree/detruit un mailleur. bq_mesher_create verifie la VRAM disponible
   (part independante des particules) avant toute allocation et renvoie NULL
   en cas de refus (bq_last_error rempli, avec la resolution cubique maximale
   qui tiendrait sur la carte presente). */
BQ_API BqMesher* bq_mesher_create(const BqMesherConfig* cfg);
BQ_API void      bq_mesher_destroy(BqMesher* m);

/* Reconstruit le champ a partir de n positions (3*n floats). Reverifie la
   VRAM complete (champ + buckets + particules) avant toute (re)allocation
   liee aux particules. Renvoie 0, ou -1 sur erreur (bq_last_error rempli). */
BQ_API int bq_mesher_run(BqMesher* m, const float* pos, int n);

/* Fournit le champ de distance signee des colliders, echantillonne sur SA
   PROPRE grille (celle du solveur en general, de resolution differente de
   celle du mailleur) : idx = (i*res[1]+j)*res[2]+k, valeur a (i,j,k) prise
   AUX NOEUDS, c'est-a-dire au point (i*cell_size, j*cell_size, k*cell_size).

   ATTENTION, deux conventions coexistent et c'est voulu. Le champ collider est
   aux noeuds parce qu'il vient du solveur (cf. k_sdf_unsigned dans mlsmpm.cu,
   qui le justifie : c'est k_grid_update qui le consomme, indexe comme la
   grille MPM). Le champ du MAILLEUR, lui, est au CENTRE des cellules. La
   conversion se fait ici, au seul point de contact entre les deux. Fournir un
   champ echantillonne au centre de cellule decalerait le rognage d'un demi-pas
   par axe -- 0.87 pas en diagonale -- et le fluide serait rogne a cote de la
   surface du collider. Le passer tel quel depuis bq_read_sdf est correct.

   Convention de signe identique a celle du solveur :
   negatif a l'interieur du solide. Le champ est persistant entre les appels
   a bq_mesher_run (reutilise tel quel tant qu'il n'est pas remplace), et
   consomme au rognage (avant polygonisation) : phi_final = max(phi_fluide,
   -phi_solide_interpole + collider_offset), phi_solide_interpole obtenu par
   interpolation trilineaire ; hors du domaine du champ fourni, une grande
   valeur positive est utilisee (jamais 0 ni une extrapolation). Passer
   sdf = NULL efface le collider courant. Renvoie 0, ou -1 sur erreur. */
BQ_API int bq_mesher_set_collider_sdf(BqMesher* m, const float* sdf,
                                      const int res[3], float cell_size);

/* Copie le champ phi courant vers dst (grid_res[0]*grid_res[1]*grid_res[2]
   floats). Diagnostic et validation. Renvoie 0, ou -1 sur erreur. */
BQ_API int bq_mesher_read_field(BqMesher* m, float* dst);

/* Comptes du maillage produit par le dernier bq_mesher_run, pour que
   l'appelant dimensionne ses tampons avant bq_mesher_read. Un maillage vide
   (0 sommet, 0 triangle) est un resultat NORMAL : le champ peut ne contenir
   aucun fluide. Renvoie 0, ou -1 sur erreur. */
BQ_API int bq_mesher_counts(const BqMesher* m, int* n_verts, int* n_tris);

/* Copie le maillage vers l'hote : `verts` recoit n_verts*3 float32 (x, y, z),
   `tris` recoit n_tris*3 int32 (indices de sommets). Les sommets sont
   DEDUPLIQUES : un par arete de grille traversee, partage entre les triangles
   voisins, ce qui permet a Blender de lisser les normales.
   `vel` est reserve au canal de vitesse par sommet (motion blur) et reste
   ignore tant qu'il n'est pas implemente ; passer NULL.
   Chacun des trois pointeurs peut etre NULL pour ne pas lire ce canal.
   Renvoie 0, ou -1 sur erreur. */
BQ_API int bq_mesher_read(BqMesher* m, float* verts, int* tris, float* vel);

/* ------------------------------------------------------------ whitewater
 *
 * Particules secondaires (ecume, bulles, embruns), simulation a etat
 * persistant entre appels -- PAS une reconstruction sans memoire comme le
 * mailleur ci-dessus. bq_whitewater_step doit etre appele UNE FOIS PAR
 * FRAME, dans l'ordre croissant, sans saut : l'etat interne (particules
 * actives, position/vitesse/age/regime) est celui laisse par l'appel
 * precedent. Aucun numero de frame n'est verifie par l'API -- un appel hors
 * sequence n'est pas detecte (cf. plan-milestone-8.md, D5 et risque 3).
 *
 * Trois regimes, reevalues a chaque pas depuis la distance signee au champ
 * fluide (pas figes a la naissance) : SPRAY (au-dessus de la surface,
 * balistique pur), FOAM (a la surface, suit la vitesse ambiante), BUBBLE
 * (sous la surface, suit la vitesse ambiante + flottabilite).
 */
enum BqWhitewaterType {
    BQ_WW_SPRAY  = 0,
    BQ_WW_FOAM   = 1,
    BQ_WW_BUBBLE = 2
};

typedef struct BqWhitewaterConfig {
    int   max_particles;      /* capacite active maximale, garde-fou VRAM */
    float influence_radius;   /* meme grandeur que BqMesherConfig.influence_radius */
    float gravity_y;          /* defaut -9.8, meme convention que BqConfig */
    /* seuils (bas/haut, normalisation [0,1]) et poids des trois potentiels */
    float ta_min, ta_max, ta_weight;   /* air piege */
    float wc_min, wc_max, wc_weight;   /* crete de vague */
    float ke_min, ke_max, ke_weight;   /* energie cinetique */
    float spawn_rate;         /* particules/s au potentiel combine = 1 */
    float life_spray, life_foam, life_bubble;   /* duree de vie moyenne, secondes */
    float drag_spray;         /* trainee quadratique (spray, balistique) */
    float drag_foam;          /* relaxation vers la vitesse ambiante (foam ET bubble) */
    float buoyancy_bubble;    /* acceleration verticale supplementaire (bubble) */
    int   grid_res[3];   /* resolution de la grille de simulation -- domaine = [0, grid_res[i]*cell_size] par axe, meme convention que BqConfig */
    float cell_size;     /* meme grandeur que BqConfig.cell_size */
} BqWhitewaterConfig;

typedef struct BqWhitewater BqWhitewater; /* handle opaque, etat persistant */

BQ_API int  bq_whitewater_config_size(void);
BQ_API void bq_whitewater_default_config(BqWhitewaterConfig* cfg);

/* Empreinte VRAM a max_particles et n_fluid_hint donnes, sans rien allouer.
   n_fluid_hint peut valoir 0 (parts dependantes du nombre de particules
   fluides non comptees, cf. bq_whitewater_create). Renvoie 0, ou -1. */
BQ_API int bq_whitewater_vram_estimate(const BqWhitewaterConfig* cfg,
                                       int n_fluid_hint, int64_t* bytes);

/* Verifie la VRAM (part independante du nombre de particules fluides,
   c'est-a-dire la capacite active max_particles) avant toute allocation.
   Renvoie NULL en cas de refus, bq_last_error rempli. */
BQ_API BqWhitewater* bq_whitewater_create(const BqWhitewaterConfig* cfg);
BQ_API void          bq_whitewater_destroy(BqWhitewater* w);

/* Avance d'UNE frame (cf. contrat de sequentialite en tete de section).
   pos/vel : n particules fluides de CETTE frame (espace solveur, memoire
   HOTE comme bq_mesher_run). Renvoie 0, ou -1 sur erreur. */
BQ_API int bq_whitewater_step(BqWhitewater* w, const float* pos,
                              const float* vel, int n, float dt);

BQ_API int bq_whitewater_count(const BqWhitewater* w);

/* Copie l'etat courant vers l'hote. type : n int32 (BqWhitewaterType).
   size/age : n float32. vel : n*3 float32, vitesse courante de chaque
   particule active, espace solveur, memes unites que pos/dt (meme
   convention que pos). Chaque pointeur peut etre NULL. Renvoie le nombre
   de particules copiees, ou -1. */
BQ_API int bq_whitewater_read(const BqWhitewater* w, float* pos, int* type,
                              float* size, float* age, float* vel);

/* Nombre de particules secondaires que le dernier bq_whitewater_step aurait
   generees si la capacite le permettait, et qui ont ete refusees (cf.
   BqWhitewaterConfig.max_particles). 0 tant qu'aucun step n'a sature la
   capacite. Sert a informer l'artiste plutot qu'a laisser une perte de
   densite invisible (cf. plan-milestone-8.md, risque 4). */
BQ_API int bq_whitewater_last_refused(const BqWhitewater* w);

/* Fournit le champ de distance signee des colliders, MEME CONVENTION EXACTE
   que bq_mesher_set_collider_sdf (cf. son commentaire complet dans ce
   fichier, section mailleur) : echantillonne AUX NOEUDS, idx =
   (i*res[1]+j)*res[2]+k, negatif = interieur du solide, grande valeur
   positive hors du domaine fourni (jamais une extrapolation). Persistant
   entre les appels a bq_whitewater_step (reutilise tel quel tant qu'il
   n'est pas remplace). sdf = NULL efface le collider courant -- dans ce cas,
   seule la borne de domaine (grid_res/cell_size de BqWhitewaterConfig)
   contraint les particules. Renvoie 0, ou -1 sur erreur. */
BQ_API int bq_whitewater_set_collider_sdf(BqWhitewater* w, const float* sdf,
                                          const int res[3], float cell_size);

/* Fournit le champ de normale de contact des colliders, MEME CONVENTION
   EXACTE que bq_read_cnrm (cf. son commentaire complet dans ce fichier,
   section solveur principal) : echantillonne AUX NOEUDS, idx =
   (i*res[1]+j)*res[2]+k, 4 floats par cellule (x,y,z normale unitaire, w
   distance non signee jusqu'au triangle le plus proche). Utilise, quand
   fourni, a la place d'un gradient par differences finies recalcule sur le
   SDF scalaire pour detecter le franchissement d'un collider pendant
   l'advection whitewater -- plus precis pres des aretes/sommets, moins
   couteux par sous-pas. Persistant entre les appels a bq_whitewater_step
   (reutilise tel quel tant qu'il n'est pas remplace). cnrm = NULL efface le
   champ courant -- dans ce cas, bq_whitewater_step retombe sur le calcul par
   differences finies a partir du seul collider_sdf. Renvoie 0, ou -1 sur
   erreur. */
BQ_API int bq_whitewater_set_collider_cnrm(BqWhitewater* w, const float* cnrm,
                                           const int res[3], float cell_size);

#ifdef __cplusplus
}
#endif
#endif /* BOURRASQUE_H */

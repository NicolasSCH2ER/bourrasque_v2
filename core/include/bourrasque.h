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
#define BQ_ABI_VERSION 4

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
   floats, coefficient par triangle. A appeler une fois par frame, avant
   bq_step. n_tri == 0 efface les colliders. Renvoie 0, ou -1 sur erreur. */
BQ_API int bq_set_colliders(BqSim* sim, const float* tri, const float* tri_vel,
                            const float* tri_friction, int n_tri);

BQ_API int bq_particle_count(const BqSim* sim);

/* Copie device -> hote. dst doit contenir n*3 floats / n octets. */
BQ_API int bq_read_positions(BqSim* sim, float* dst);
BQ_API int bq_read_materials(BqSim* sim, uint8_t* dst);

/* Copie le champ de distance signee courant vers `dst`, qui doit pouvoir
   contenir grid_res[0]*grid_res[1]*grid_res[2] floats. Diagnostic et
   validation. Renvoie le nombre de cellules copiees, ou -1 sur erreur. */
BQ_API int bq_read_sdf(BqSim* sim, float* dst);

BQ_API const char* bq_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* BOURRASQUE_H */

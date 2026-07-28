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

/* Modeles constitutifs. D'autres viendront (sable, neige) sans changer
 * l'API : c'est tout l'interet du pipeline MPM unifie. */
enum BqModel {
    BQ_MODEL_ELASTIC = 0, /* corotationnel fixe (E, nu)        */
    BQ_MODEL_WATER   = 1  /* EOS de Tait J-based (bulk, gamma) */
};

typedef struct BqConfig {
    int   grid_res;      /* noeuds par axe (defaut 64)                  */
    float domain;        /* cote du domaine cubique en metres (defaut 1)*/
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

/* Avance d'une frame ; le solveur decoupe en substeps via la CFL.
 * Retourne le nombre de substeps effectues. */
BQ_API int bq_step(BqSim* sim, float frame_dt);

BQ_API int bq_particle_count(const BqSim* sim);

/* Copie device -> hote. dst doit contenir n*3 floats / n octets. */
BQ_API int bq_read_positions(BqSim* sim, float* dst);
BQ_API int bq_read_materials(BqSim* sim, uint8_t* dst);

BQ_API const char* bq_last_error(void);

#ifdef __cplusplus
}
#endif
#endif /* BOURRASQUE_H */

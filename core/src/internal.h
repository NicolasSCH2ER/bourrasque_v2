/* internal.h -- surface partagee minimale entre unites de compilation du
 * coeur (mlsmpm.cu, mesher.cu). N'expose rien de plus : chaque .cu reste
 * autonome (CUDA_SEPARABLE_COMPILATION OFF), ce fichier ne fait que donner
 * acces au buffer d'erreur commun et a la macro de verification CUDA.
 */
#ifndef BQ_INTERNAL_H
#define BQ_INTERNAL_H

#include <cuda_runtime.h>
#include <cstdio>

/* Definie (une seule fois) dans mlsmpm.cu. */
extern char g_error[512];

#define BQ_CUDA_CHECK(call)                                                  \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            snprintf(g_error, sizeof(g_error), "%s:%d CUDA: %s", __FILE__,   \
                     __LINE__, cudaGetErrorString(err_));                    \
            return -1;                                                       \
        }                                                                    \
    } while (0)

#endif /* BQ_INTERNAL_H */

/* Test minimal : verifie bq_abi_version() et bq_config_size() AVANT tout
   bq_create, sans GPU requis. */
#include <stdio.h>
#include "bourrasque.h"

int main(void) {
    int abi = bq_abi_version();
    int sz  = bq_config_size();
    printf("bq_abi_version() = %d (attendu BQ_ABI_VERSION=%d)\n", abi, BQ_ABI_VERSION);
    printf("bq_config_size() = %d (attendu sizeof(BqConfig)=%d)\n", sz, (int)sizeof(BqConfig));
    if (abi != BQ_ABI_VERSION) { printf("ECHEC: version ABI inattendue\n"); return 1; }
    if (sz != (int)sizeof(BqConfig)) { printf("ECHEC: taille de struct incoherente\n"); return 1; }
    printf("OK\n");
    return 0;
}

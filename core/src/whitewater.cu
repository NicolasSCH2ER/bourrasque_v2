/* whitewater.cu -- particules secondaires (ecume/spray, bulles) generees a
 * partir du champ fluide primaire (M8/T1+T2). Reference : Ihmsen, Akinci,
 * Teschner, Erleben, "Unified Spray, Foam and Bubbles for Particle-Based
 * Fluids" (2012), pour les trois potentiels de generation (air piege, crete
 * de vague, energie cinetique) -- mais ce fichier en adopte une version
 * SIMPLIFIEE : un SEUL noyau de ponderation k(s) = (1-s^2)^3 partout (le
 * papier original utilise des noyaux differents par potentiel), et un
 * facteur d'attenuation continu plutot que le test booleen "crete" du
 * papier (cf. commentaire au point d'appel, D3 du plan). Ces ecarts sont
 * assumes pour ce premier jalon, pas des oublis.
 *
 * PORTEE DE CE FICHIER (cf. plan-milestone-8.md, decoupage T1/T2/T3) :
 * generation ET dynamique. bq_whitewater_step construit la grille de
 * buckets sur les particules fluides de la frame, evalue les trois
 * potentiels de generation (T2), avance les particules deja actives
 * (advection multi-regime en sous-pas internes, vieillissement, mort par
 * age/capacite, compaction des survivantes -- T3, D4/D5/D6 du plan), puis
 * emet les nouvelles particules dans la liste compactee jusqu'a la
 * capacite.
 *
 * Contrairement au mailleur (bq_mesher_run, fonction pure d'une frame),
 * bq_whitewater_step porte un ETAT PERSISTANT entre appels (les particules
 * actives) -- c'est un second solveur, pas un second mailleur (D1 du plan).
 *
 * Unite de compilation AUTONOME : CUDA_SEPARABLE_COMPILATION est OFF pour
 * la cible (voir CMakeLists.txt), donc aucun symbole __device__ ne peut
 * traverser mlsmpm.cu, mesher.cu et ce fichier. Les quelques helpers
 * vectoriels dont ce fichier a besoin sont donc redefinis ici plutot que
 * partages -- meme politique que mesher.cu (cf. sa note d'autonomie).
 */
#define BQ_BUILD
#include "bourrasque.h"
#include "internal.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

/* ------------------------------------------------------------- petite algebre
 * (dupliquee depuis mlsmpm.cu/mesher.cu, cf. note d'autonomie en tete de
 * fichier) */
__host__ __device__ inline float3 vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float vlen(float3 a) {
    return sqrtf(vdot(a, a));
}

/* Fraction de influence_radius definissant la bande de surface : une
 * particule fluide est candidate a la generation si son phi (distance
 * signee, cf. ww_sample_fluid) depasse -SURFACE_BAND_FRAC*R -- pas trop
 * profondement dans le volume (cf. section 5 de la spec T2). Reutilisee
 * aussi (avec un facteur 0.5 supplementaire) pour la classification
 * grossiere spray/foam/bubble a l'emission. */
static constexpr float SURFACE_BAND_FRAC = 0.5f;

/* Classification de regime partagee, seuil identique a celui deja en place
 * dans k_ww_emit (bande de surface = SURFACE_BAND_FRAC*0.5*R) -- factorisee
 * ici pour etre reutilisee telle quelle par k_ww_emit (classification
 * initiale) ET k_ww_advect (reevaluation a chaque pas, cf. D4 du plan). */
__device__ inline int ww_classify(float phi, float R) {
    const float band = SURFACE_BAND_FRAC * 0.5f * R;
    if (phi > 0.f) return BQ_WW_SPRAY;
    if (phi > -band) return BQ_WW_FOAM;
    return BQ_WW_BUBBLE;
}

/* --------------------------------------------------------- grille de buckets
 * Contrairement au mailleur, le whitewater n'a pas de domaine de champ fixe
 * dans sa config (pas de grid_res/cell_size) : la grille de buckets est
 * reconstruite CHAQUE frame sur l'AABB des positions fluides recues par
 * bq_whitewater_step, avec une marge de +2*influence_radius de chaque cote
 * pour que les requetes ponctuelles pres du bord de l'AABB trouvent quand
 * meme leurs voisins (cf. spec T1). */
struct BucketGrid {
    float3 origin;
    float  h;
    int3   res;
};

/* Hash entier bon marche (variante Wang hash), duplique depuis mc_hash01 de
 * mesher.cu sous un nom distinct -- meme raison d'autonomie de compilation
 * que le reste de ce fichier (aucun symbole partage entre unites). */
__device__ inline float ww_hash01(unsigned int x) {
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x ^= x >> 4;
    x *= 0x27d4eb2du;
    x ^= x >> 15;
    return (float)(x & 0x00FFFFFFu) * (1.f / 16777216.f); /* [0,1) */
}

/* --------------------------------------------------------- echantillonnage
 * Champ fluide en un point de requete arbitraire, meme bucketing gather que
 * k_zhu_bridson_field (mesher.cu) : 27 buckets voisins, meme noyau
 * k(s)=(1-s^2)^3, meme convention "aucune particule dans le rayon -> phi =
 * R, jamais NaN". Le rayon de particule (r_i de Zhu-Bridson) n'est pas un
 * champ de BqWhitewaterConfig : le whitewater n'a pas besoin de la meme
 * precision de surface que le mailleur, seulement de savoir si un point est
 * proche de l'interface -- valeur fixe 0.3*R. */
struct WwSample {
    float  phi;   /* distance signee Zhu-Bridson au point de requete */
    float3 v_moy; /* vitesse moyenne ponderee des voisins, (0,0,0) si aucun */
};

__device__ inline WwSample ww_sample_fluid(float3 q, const float3* __restrict__ pos,
                                           const float3* __restrict__ vel,
                                           const int* __restrict__ bucket_off,
                                           const int* __restrict__ bucket_idx,
                                           int3 bucket_res, float bucket_h,
                                           float3 bucket_origin, float R) {
    float particle_radius = 0.3f * R;
    int3 bc = make_int3((int)floorf((q.x - bucket_origin.x) / bucket_h),
                        (int)floorf((q.y - bucket_origin.y) / bucket_h),
                        (int)floorf((q.z - bucket_origin.z) / bucket_h));

    float sum_w = 0.f;
    float3 sum_wx = make_float3(0.f, 0.f, 0.f);
    float sum_wr = 0.f;
    float3 sum_wv = make_float3(0.f, 0.f, 0.f);
    float inv_R = 1.f / R;
    float R2 = R * R;

    for (int di = -1; di <= 1; ++di) {
        int bi = bc.x + di;
        if (bi < 0 || bi >= bucket_res.x) continue;
        for (int dj = -1; dj <= 1; ++dj) {
            int bj = bc.y + dj;
            if (bj < 0 || bj >= bucket_res.y) continue;
            for (int dk = -1; dk <= 1; ++dk) {
                int bk = bc.z + dk;
                if (bk < 0 || bk >= bucket_res.z) continue;
                int bidx = (bi * bucket_res.y + bj) * bucket_res.z + bk;
                int off0 = bucket_off[bidx], off1 = bucket_off[bidx + 1];
                for (int e = off0; e < off1; ++e) {
                    int pidx = bucket_idx[e];
                    float3 xp = pos[pidx];
                    float3 d = vsub(q, xp);
                    float d2 = vdot(d, d);
                    if (d2 >= R2) continue;
                    float s = sqrtf(d2) * inv_R;
                    float t = 1.f - s * s;
                    float w = t * t * t;
                    sum_w += w;
                    sum_wx.x += w * xp.x; sum_wx.y += w * xp.y; sum_wx.z += w * xp.z;
                    sum_wr += w * particle_radius;
                    float3 vp = vel[pidx];
                    sum_wv.x += w * vp.x; sum_wv.y += w * vp.y; sum_wv.z += w * vp.z;
                }
            }
        }
    }

    WwSample out;
    if (sum_w <= 0.f) {
        /* aucune particule dans le rayon : grande valeur positive pour phi
         * (jamais zero ni NaN), vitesse ambiante nulle par defaut (defaut
         * sain contrairement a phi, cf. spec T1). */
        out.phi = R;
        out.v_moy = make_float3(0.f, 0.f, 0.f);
        return out;
    }
    float inv_sum = 1.f / sum_w;
    float3 xmean = make_float3(sum_wx.x * inv_sum, sum_wx.y * inv_sum, sum_wx.z * inv_sum);
    float rmean = sum_wr * inv_sum;
    out.phi = vlen(vsub(q, xmean)) - rmean;
    out.v_moy = make_float3(sum_wv.x * inv_sum, sum_wv.y * inv_sum, sum_wv.z * inv_sum);
    return out;
}

/* Normale sortante approchee (gradient de densite SPH standard),
 * n_hat(q) = normalize(somme_j w_ij * (q - x_j)), meme noyau que
 * ww_sample_fluid. (0,1,0) par defaut si la somme est quasi nulle
 * (voisinage parfaitement symetrique -- evite une normale degeneree).
 *
 * Helper reutilise par k_ww_compute_normals (une seule evaluation par
 * particule fluide, cf. commentaire au-dessus de ce kernel) -- ne plus
 * appeler cette fonction depuis l'interieur de la boucle de voisinage de
 * k_ww_generate_potentials, qui lit desormais les normales precalculees. */
__device__ inline float3 ww_neighbor_normal(float3 q, const float3* __restrict__ pos,
                                            const int* __restrict__ bucket_off,
                                            const int* __restrict__ bucket_idx,
                                            int3 bucket_res, float bucket_h,
                                            float3 bucket_origin, float R) {
    int3 bc = make_int3((int)floorf((q.x - bucket_origin.x) / bucket_h),
                        (int)floorf((q.y - bucket_origin.y) / bucket_h),
                        (int)floorf((q.z - bucket_origin.z) / bucket_h));
    float3 sum = make_float3(0.f, 0.f, 0.f);
    float inv_R = 1.f / R;
    float R2 = R * R;

    for (int di = -1; di <= 1; ++di) {
        int bi = bc.x + di;
        if (bi < 0 || bi >= bucket_res.x) continue;
        for (int dj = -1; dj <= 1; ++dj) {
            int bj = bc.y + dj;
            if (bj < 0 || bj >= bucket_res.y) continue;
            for (int dk = -1; dk <= 1; ++dk) {
                int bk = bc.z + dk;
                if (bk < 0 || bk >= bucket_res.z) continue;
                int bidx = (bi * bucket_res.y + bj) * bucket_res.z + bk;
                int off0 = bucket_off[bidx], off1 = bucket_off[bidx + 1];
                for (int e = off0; e < off1; ++e) {
                    int pidx = bucket_idx[e];
                    float3 xp = pos[pidx];
                    float3 d = vsub(q, xp);
                    float d2 = vdot(d, d);
                    if (d2 >= R2) continue;
                    float s = sqrtf(d2) * inv_R;
                    float t = 1.f - s * s;
                    float w = t * t * t;
                    sum.x += w * d.x; sum.y += w * d.y; sum.z += w * d.z;
                }
            }
        }
    }
    float len = vlen(sum);
    if (len < 1e-8f) return make_float3(0.f, 1.f, 0.f);
    float inv = 1.f / len;
    return make_float3(sum.x * inv, sum.y * inv, sum.z * inv);
}

/* ------------------------------------------------- precalcul des normales
 * Passe separee, PAS un appel imbrique par voisin : mesure sur une nappe au
 * repos de 430k particules, 7 s pour un pas (13 particules generees) avant
 * ce correctif -- I_wc lisait la normale d'un voisin en refaisant sa
 * recherche de voisinage complete, cout O(K^2) par candidat. Precalculer
 * une fois O(n) puis relire O(1) par voisin ramene le cout a O(K) par
 * candidat.
 *
 * Un thread par particule fluide (les n recues par bq_whitewater_step, pas
 * seulement les candidates pres de la surface -- une normale peut etre lue
 * par un candidat meme si le voisin lui-meme n'est pas candidat). */
__global__ void k_ww_compute_normals(float3* __restrict__ out_normal,
                                     const float3* __restrict__ pos,
                                     const int* __restrict__ bucket_off,
                                     const int* __restrict__ bucket_idx,
                                     int3 bucket_res, float bucket_h,
                                     float3 bucket_origin, float R, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_normal[i] = ww_neighbor_normal(pos[i], pos, bucket_off, bucket_idx,
                                       bucket_res, bucket_h, bucket_origin, R);
}

/* ---------------------------------------------------- potentiels (D3/T2)
 * Un thread par particule fluide i. Ecrit gen_count[i] (nombre de
 * particules secondaires a generer depuis i cette frame, pas de plafond de
 * capacite ici -- cf. spec, l'emission gere le plafond) et met a jour
 * carry[i] (reste fractionnaire reporte a l'appel suivant, cf. section 5
 * point 4 de la spec T2 : sans lui une turbulence faible mais soutenue ne
 * genererait jamais rien). */
__global__ void k_ww_generate_potentials(
    int* __restrict__ gen_count, float* __restrict__ carry,
    float* __restrict__ isolated_time,
    const float3* __restrict__ pos, const float3* __restrict__ vel,
    const float3* __restrict__ normal,
    const int* __restrict__ bucket_off, const int* __restrict__ bucket_idx,
    int3 bucket_res, float bucket_h, float3 bucket_origin, float R,
    float ta_min, float ta_max, float ta_weight,
    float wc_min, float wc_max, float wc_weight,
    float ke_min, float ke_max, float ke_weight,
    float spawn_rate, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3 xi = pos[i];
    float3 vi = vel[i];

    WwSample s = ww_sample_fluid(xi, pos, vel, bucket_off, bucket_idx, bucket_res,
                                 bucket_h, bucket_origin, R);
    if (s.phi <= -SURFACE_BAND_FRAC * R) {
        /* trop profondement dans le volume : pas candidate, pas de seconde
         * boucle de voisinage inutile. */
        gen_count[i] = 0;
        return;
    }

    float3 v_hat_i = make_float3(0.f, 0.f, 0.f);
    if (vdot(vi, vi) > 1e-12f) {
        float inv = 1.f / vlen(vi);
        v_hat_i = make_float3(vi.x * inv, vi.y * inv, vi.z * inv);
    }
    float3 n_hat_i = normal[i];

    int3 bc = make_int3((int)floorf((xi.x - bucket_origin.x) / bucket_h),
                        (int)floorf((xi.y - bucket_origin.y) / bucket_h),
                        (int)floorf((xi.z - bucket_origin.z) / bucket_h));
    float inv_R = 1.f / R;
    float R2 = R * R;

    float I_ta = 0.f;
    float wc_sum = 0.f;
    int neighbor_count = 0;

    for (int di = -1; di <= 1; ++di) {
        int bi = bc.x + di;
        if (bi < 0 || bi >= bucket_res.x) continue;
        for (int dj = -1; dj <= 1; ++dj) {
            int bj = bc.y + dj;
            if (bj < 0 || bj >= bucket_res.y) continue;
            for (int dk = -1; dk <= 1; ++dk) {
                int bk = bc.z + dk;
                if (bk < 0 || bk >= bucket_res.z) continue;
                int bidx = (bi * bucket_res.y + bj) * bucket_res.z + bk;
                int off0 = bucket_off[bidx], off1 = bucket_off[bidx + 1];
                for (int e = off0; e < off1; ++e) {
                    int j = bucket_idx[e];
                    if (j == i) continue;
                    float3 xj = pos[j];
                    float3 d = vsub(xi, xj);
                    float d2 = vdot(d, d);
                    if (d2 >= R2) continue;
                    float sN = sqrtf(d2) * inv_R;
                    float t = 1.f - sN * sN;
                    float w_ij = t * t * t;
                    neighbor_count++;

                    float3 vj = vel[j];

                    /* Fix A (ecart papier eq. 2) : v_diff_i doit comparer la
                     * direction de la vitesse RELATIVE (vi-vj) a la direction
                     * de la position RELATIVE (xi-xj), pas les directions de
                     * vitesse ABSOLUES de i et j -- ce que faisait l'ancien
                     * code (vdot(v_hat_i, v_hat_j)), qui ne mesure pas du
                     * tout une convergence entre particules. d est deja
                     * xi-xj (calcule plus haut pour le test d2>=R2), donc
                     * x_hat_ij = normalize(d). Garde-fou vij_len quasi nul :
                     * contribue 0 (coherent avec le prefacteur ||vij|| du
                     * papier, qui annulerait le terme de toute facon). */
                    float3 vij = vsub(vi, vj);
                    float vij_len = vlen(vij);
                    if (vij_len > 1e-8f) {
                        float3 v_hat_ij = make_float3(vij.x / vij_len, vij.y / vij_len, vij.z / vij_len);
                        float inv_dlen = 1.f / sqrtf(d2);
                        float3 x_hat_ij = make_float3(d.x * inv_dlen, d.y * inv_dlen, d.z * inv_dlen);
                        I_ta += w_ij * vij_len * (1.f - vdot(v_hat_ij, x_hat_ij));
                    }

                    /* n_hat_j : normale au voisin j, lue depuis le tampon
                     * precalcule par k_ww_compute_normals (cf. commentaire
                     * au-dessus de ce kernel) au lieu d'une recherche de
                     * voisinage imbriquee. */
                    float3 n_hat_j = normal[j];

                    /* Fix B (ecart papier eq. 4-6) : ne compter que les
                     * voisins du cote CONVEXE (filtre de convexite manquant
                     * dans l'ancien code, qui sommait sur tous les voisins
                     * sans condition). x_hat_ji = -x_hat_ij = -d/||d|| ;
                     * condition d'inclusion vdot(x_hat_ji, n_hat_i) < 0,
                     * equivalent (sans division) a vdot(d, n_hat_i) > 0 --
                     * test de signe sur d brut, pas besoin de normaliser. */
                    if (vdot(d, n_hat_i) > 0.f) {
                        wc_sum += w_ij * (1.f - vdot(n_hat_i, n_hat_j));
                    }
                }
            }
        }
    }

    /* Le premier facteur (vitesse sortante de i) remplace le test booleen
     * "crete" du papier original par une attenuation continue -- deviation
     * assumee (cf. D3 du plan). */
    float I_wc = fmaxf(0.f, vdot(v_hat_i, n_hat_i)) * wc_sum * vlen(vi);
    float I_k = 0.5f * vdot(vi, vi);

    float norm_ta = (ta_max > ta_min)
        ? fminf(fmaxf((I_ta - ta_min) / (ta_max - ta_min), 0.f), 1.f) : 0.f;
    float norm_wc = (wc_max > wc_min)
        ? fminf(fmaxf((I_wc - wc_min) / (wc_max - wc_min), 0.f), 1.f) : 0.f;
    float norm_k = (ke_max > ke_min)
        ? fminf(fmaxf((I_k - ke_min) / (ke_max - ke_min), 0.f), 1.f) : 0.f;

    float I = norm_ta * ta_weight + norm_wc * wc_weight + norm_k * ke_weight;

    /* Garantie pour les particules quasi isolees (D4, plan-milestone-9.md) :
     * le mailleur exclut ces particules de la reconstruction de surface
     * (meme seuil, duplique independamment dans ce fichier -- aucune donnee
     * partagee entre mesher.cu et whitewater.cu, cf. M8/D2). Sans cette
     * garantie, une particule isolee mais lente (potentiel de generation
     * faible) pourrait ne produire ni surface ni whitewater -- un vide, pas
     * seulement un defaut esthetique.
     *
     * Bug mesure (pas suppose) : la version initiale forcait cnt=1 A CHAQUE
     * FRAME des que neighbor_count<=k_n_cull, sans tenir compte de carry[i]
     * ni consommer aucun budget -- correct pour un evenement TRANSITOIRE
     * (une goutte isolee qui n'existe que 1-2 frames), mais une region
     * STRUCTURELLEMENT quasi-isolee en continu (un film d'eau coince dans un
     * coin de domaine entre deux parois, presque sans voisins a CHAQUE frame
     * tant qu'il grimpe) redeclenchait la garantie sans relache -- une
     * colonne de particules generee en continu au meme endroit plutot qu'un
     * correctif ponctuel (observe sur un dam-break reel du repo). Le
     * plancher ci-dessous remplace le forcage direct par un plancher sur le
     * potentiel I, qui transite ensuite par le MEME pipeline raw/carry que
     * la generation normale : le budget carry[i] agit alors comme un
     * cooldown naturel (une generation consomme sa part de carry, la
     * suivante doit se re-accumuler), sans etat supplementaire. Valeur
     * choisie (0.05) pour garantir une generation en au plus ~0.4s a
     * spawn_rate par defaut (50/s) -- largement sous life_spray/foam/bubble
     * (>=1s par defaut), donc la garantie anti-trou reste tenue pour une
     * goutte transitoire, tout en divisant par plus de 10x le debit d'une
     * source structurellement isolee en continu (colonne).
     *
     * Insuffisant en pratique (mesure, pas suppose, sur le dam-break reel de
     * l'utilisateur) : un plancher CONSTANT continue de forcer une
     * generation a CHAQUE frame tant que neighbor_count<=k_n_cull reste vrai,
     * or un film d'eau plaque en continu contre une paroi du domaine reste
     * structurellement sous ce seuil pendant toute la duree du contact (pas
     * 1-2 frames comme une vraie goutte isolee transitoire) -- d'ou une
     * colonne visible sur un bake de plusieurs secondes, seulement moins
     * dense qu'avant. Le plancher decroit desormais avec isolated_time[i],
     * la duree en continu (en secondes) passee par CETTE particule fluide
     * sous le seuil d'isolement : une goutte transitoire (isolated_time
     * proche de 0) garde un plancher quasi plein, un film structurellement
     * isole pendant plusieurs secondes voit le sien diviser par 10-50x en
     * plus du facteur deja applique par le plancher lui-meme. */
    const int k_n_cull = 1;
    const float k_isolated_floor = 0.05f;
    if (neighbor_count <= k_n_cull) {
        isolated_time[i] += dt;
    } else {
        isolated_time[i] = 0.f;
    }
    const float k_isolated_decay_tau = 0.3f; /* secondes : au-dela de ~0.3-0.5s d'isolement
        continu, le plancher a deja perdu la majeure partie de son poids -- une vraie goutte
        transitoire (1-2 frames, largement sous ce seuil) garde un plancher quasi plein, un
        film structurellement isole pendant plusieurs secondes (le cas mesure sur le dam-break)
        voit le sien diviser par 10-50x en plus du facteur deja applique par le plancher lui-meme. */
    if (neighbor_count <= k_n_cull) {
        float decay = 1.f / (1.f + isolated_time[i] / k_isolated_decay_tau);
        I = fmaxf(I, k_isolated_floor * decay);
    }

    float raw = I * spawn_rate * dt + carry[i];
    int cnt = (int)floorf(raw);
    carry[i] = raw - cnt;

    gen_count[i] = cnt; /* pas de plafond ici, cf. bq_whitewater_step */
}

/* ---------------------------------------------------- emission (T2)
 * Un thread par particule fluide candidate. Pour chaque particule
 * secondaire k dans [0, gen_count[i]), si son offset global depasse la
 * capacite restante (actually_gen), elle est silencieusement ignoree --
 * pas d'eviction dans ce jalon (cf. D6 du plan). */
__global__ void k_ww_emit(
    float3* __restrict__ active_pos, float3* __restrict__ active_vel,
    int* __restrict__ active_type, float* __restrict__ active_size,
    float* __restrict__ active_age,
    const float3* __restrict__ pos, const float3* __restrict__ vel,
    const int* __restrict__ gen_count, const int* __restrict__ gen_scan,
    const int* __restrict__ bucket_off, const int* __restrict__ bucket_idx,
    int3 bucket_res, float bucket_h, float3 bucket_origin, float R,
    int n_active_base, int actually_gen, float influence_radius, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cnt = gen_count[i];
    if (cnt <= 0) return;

    float3 xi = pos[i];
    float3 vi = vel[i];

    /* Classification initiale grossiere depuis le phi de la particule
     * fluide source -- reevaluee a chaque pas par k_ww_advect (D4 du plan),
     * ici c'est juste un point de depart, meme helper partage (ww_classify). */
    WwSample s = ww_sample_fluid(xi, pos, vel, bucket_off, bucket_idx, bucket_res,
                                 bucket_h, bucket_origin, R);
    int type = ww_classify(s.phi, R);

    float vi_len = vlen(vi);

    /* Fix C (ecart papier, section Sampling / Fig. 3) : echantillonnage dans
     * un CYLINDRE le long de la trajectoire de la particule fluide, pas un
     * jitter isotrope. Base = xi, axe = v_hat_i, hauteur = ||dt*vi||
     * (distance parcourue ce pas de temps), rayon rV = 0.1*influence_radius
     * (meme echelle que l'ancien jitter isotrope, pas de nouveau parametre
     * expose, cf. spec Fix C point 1). Base orthonormee (e1,e2) perpendiculaire
     * a v_hat_i par la technique standard (vecteur de reference non parallele,
     * deux produits vectoriels). Repli sur jitter isotrope si vi_len est
     * quasi nul : une base construite sur une direction degeneree n'a pas de
     * sens. */
    const float rV = 0.1f * influence_radius;
    bool has_dir = vi_len > 1e-6f;
    float3 v_hat_i = has_dir
        ? make_float3(vi.x / vi_len, vi.y / vi_len, vi.z / vi_len)
        : make_float3(0.f, 0.f, 0.f);
    float3 e1 = make_float3(0.f, 0.f, 0.f);
    float3 e2 = make_float3(0.f, 0.f, 0.f);
    if (has_dir) {
        float3 ref = (fabsf(v_hat_i.y) > 0.99f) ? make_float3(1.f, 0.f, 0.f)
                                                 : make_float3(0.f, 1.f, 0.f);
        /* e1 = normalize(v_hat_i x ref) */
        float3 c1 = make_float3(v_hat_i.y * ref.z - v_hat_i.z * ref.y,
                                v_hat_i.z * ref.x - v_hat_i.x * ref.z,
                                v_hat_i.x * ref.y - v_hat_i.y * ref.x);
        float c1len = vlen(c1);
        e1 = make_float3(c1.x / c1len, c1.y / c1len, c1.z / c1len);
        /* e2 = v_hat_i x e1, deja unitaire (v_hat_i et e1 orthonormes) */
        e2 = make_float3(v_hat_i.y * e1.z - v_hat_i.z * e1.y,
                         v_hat_i.z * e1.x - v_hat_i.x * e1.z,
                         v_hat_i.x * e1.y - v_hat_i.y * e1.x);
    }

    int base = gen_scan[i];
    for (int k = 0; k < cnt; ++k) {
        int slot = base + k;
        if (slot >= actually_gen) break; /* candidate excedentaire, ignoree (D6) */

        unsigned seed = (unsigned)i * 0x9E3779B1u + (unsigned)k * 0x85EBCA6Bu;

        float3 xd, vd;
        if (has_dir) {
            float Xr = ww_hash01(seed + 1u);
            float Xtheta = ww_hash01(seed + 2u);
            float Xh = ww_hash01(seed + 3u);
            float r = rV * sqrtf(Xr);
            float theta = Xtheta * 6.28318530718f;
            float h_dist = Xh * dt * vi_len;
            float ct = cosf(theta), st = sinf(theta);

            xd = make_float3(
                xi.x + r * ct * e1.x + r * st * e2.x + h_dist * v_hat_i.x,
                xi.y + r * ct * e1.y + r * st * e2.y + h_dist * v_hat_i.y,
                xi.z + r * ct * e1.z + r * st * e2.z + h_dist * v_hat_i.z);
            /* La base (e1,e2) sert AUSSI de perturbation de vitesse, a la
             * meme echelle r -- intentionnel dans le papier (section
             * Sampling, dernier paragraphe), pas une echelle vel_amp
             * separee comme l'ancien code. */
            vd = make_float3(
                vi.x + r * ct * e1.x + r * st * e2.x,
                vi.y + r * ct * e1.y + r * st * e2.y,
                vi.z + r * ct * e1.z + r * st * e2.z);
        } else {
            /* repli jitter isotrope, particule fluide quasi immobile :
             * meme forme que l'ancien code, seule echelle rV reutilisee. */
            float3 jp = make_float3(
                rV * (2.f * ww_hash01(seed + 1u) - 1.f),
                rV * (2.f * ww_hash01(seed + 2u) - 1.f),
                rV * (2.f * ww_hash01(seed + 3u) - 1.f));
            float3 jv = make_float3(
                rV * (2.f * ww_hash01(seed + 4u) - 1.f),
                rV * (2.f * ww_hash01(seed + 5u) - 1.f),
                rV * (2.f * ww_hash01(seed + 6u) - 1.f));
            xd = make_float3(xi.x + jp.x, xi.y + jp.y, xi.z + jp.z);
            vd = make_float3(vi.x + jv.x, vi.y + jv.y, vi.z + jv.z);
        }

        int out = n_active_base + slot;
        active_pos[out] = xd;
        active_vel[out] = vd;
        active_type[out] = type;
        active_size[out] = 0.3f * influence_radius; /* cosmetique, pas critique */
        active_age[out] = 0.f;
    }
}

/* -------------------------------------------------------- champ collider
 * Duplique depuis mc_sample_collider_sdf (mesher.cu), MEME formule, MEME
 * convention AUX NOEUDS -- pas au centre de cellule (cf. commentaire de
 * bq_mesher_set_collider_sdf dans bourrasque.h, qui justifie ce choix : le
 * champ vient du solveur, indexe comme sa grille). Adapte ici uniquement
 * pour l'autonomie de compilation de ce fichier (cf. note en tete de
 * fichier, aucun symbole __device__ ne traverse les unites). */
__device__ inline float ww_sample_collider_sdf(const float* __restrict__ csdf,
                                                int3 cres, float ccell, float3 p) {
    float gx = p.x / ccell;
    float gy = p.y / ccell;
    float gz = p.z / ccell;
    int i0 = (int)floorf(gx), j0 = (int)floorf(gy), k0 = (int)floorf(gz);
    if (i0 < 0 || i0 + 1 >= cres.x || j0 < 0 || j0 + 1 >= cres.y ||
        k0 < 0 || k0 + 1 >= cres.z) {
        return 1e6f;
    }
    float tx = gx - i0, ty = gy - j0, tz = gz - k0;
    int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
    float c000 = csdf[(i0 * cres.y + j0) * cres.z + k0];
    float c100 = csdf[(i1 * cres.y + j0) * cres.z + k0];
    float c010 = csdf[(i0 * cres.y + j1) * cres.z + k0];
    float c110 = csdf[(i1 * cres.y + j1) * cres.z + k0];
    float c001 = csdf[(i0 * cres.y + j0) * cres.z + k1];
    float c101 = csdf[(i1 * cres.y + j0) * cres.z + k1];
    float c011 = csdf[(i0 * cres.y + j1) * cres.z + k1];
    float c111 = csdf[(i1 * cres.y + j1) * cres.z + k1];
    float c00 = c000 * (1.f - tx) + c100 * tx;
    float c10 = c010 * (1.f - tx) + c110 * tx;
    float c01 = c001 * (1.f - tx) + c101 * tx;
    float c11 = c011 * (1.f - tx) + c111 * tx;
    float c0 = c00 * (1.f - ty) + c10 * ty;
    float c1 = c01 * (1.f - ty) + c11 * ty;
    return c0 * (1.f - tz) + c1 * tz;
}

/* Champ de normale de contact, MEME convention AUX NOEUDS que
 * ww_sample_collider_sdf ci-dessus, mais lookup au NOEUD LE PLUS PROCHE --
 * PAS d'interpolation trilineaire, contrairement au SDF scalaire.
 * Interpoler des directions de normale entre deux triangles differents n'a
 * pas de sens geometrique (une moyenne de deux normales peut pointer vers
 * un troisieme triangle qui n'existe pas), alors qu'un scalaire s'y prete
 * naturellement. Meme convention que k_grid_update (mlsmpm.cu), qui lit
 * cnrm[id] sans interpolation pour la meme raison. Index clampe a [0,res)
 * par axe : hors du domaine du champ fourni, renvoie la valeur du noeud le
 * plus proche du bord plutot qu'une extrapolation. */
__device__ inline float4 ww_sample_collider_cnrm(const float4* __restrict__ ccnrm,
                                                  int3 cres, float ccell, float3 p) {
    int i = (int)lroundf(p.x / ccell);
    int j = (int)lroundf(p.y / ccell);
    int k = (int)lroundf(p.z / ccell);
    i = max(0, min(i, cres.x - 1));
    j = max(0, min(j, cres.y - 1));
    k = max(0, min(k, cres.z - 1));
    return ccnrm[(i * cres.y + j) * cres.z + k];
}

/* ---------------------------------------------------- advection (D4/D5)
 * Un thread par particule secondaire ACTIVE (les n_active d'avant cet
 * appel, cf. bq_whitewater_step). Tampons actifs lus ET ecrits EN PLACE --
 * pas de ping-pong ici, c'est la compaction qui suit (k_ww_compact) qui
 * s'en charge. Reutilise la MEME grille de buckets fluide que la
 * generation, construite une seule fois par bq_whitewater_step (cf. D5 :
 * champ fluide fige sur la frame, pas d'etat intermediaire interpole).
 *
 * Sous-pas CFL-like (D5) : le nombre de sous-pas est calcule UNE FOIS en
 * debut de frame depuis la vitesse de depart, pas recalcule a chaque
 * sous-pas -- un nombre de sous-pas qui changerait en cours de route
 * compliquerait la garantie h*n_sub == dt. */
__global__ void k_ww_advect(
    float3* __restrict__ pos, float3* __restrict__ vel,
    int* __restrict__ type, float* __restrict__ age,
    int* __restrict__ out_alive,
    const float3* __restrict__ fluid_pos, const float3* __restrict__ fluid_vel,
    const int* __restrict__ bucket_off, const int* __restrict__ bucket_idx,
    int3 bucket_res, float bucket_h, float3 bucket_origin,
    float3 domain_hi,
    const float* __restrict__ collider_sdf, int3 collider_res,
    float collider_cell_size, bool has_collider,
    const float4* __restrict__ collider_cnrm, int3 collider_cnrm_res,
    float collider_cnrm_cell_size, bool has_collider_cnrm,
    float gravity_y, float drag_spray, float drag_foam, float buoyancy_bubble,
    float influence_radius, float dt,
    float life_spray, float life_foam, float life_bubble, int n_active) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_active) return;

    float R = influence_radius;
    float3 x = pos[p];
    float3 v = vel[p];
    float  a = age[p];

    /* Second critere de sous-pas, base sur la finesse du collider (mesure
     * dynamique, jet rapide contre la paroi reelle d'un verre ~4mm) : malgre
     * la bande de marge eps_contact deja en place plus bas dans ce kernel,
     * jusqu'a 55% des particules whitewater traversaient encore la paroi au
     * moment de l'impact -- tunneling. En cause : le nombre de sous-pas
     * n'etait dimensionne QUE sur l'echelle fluide (influence_radius, R),
     * sans rapport avec la finesse d'un obstacle. Une particule rapide peut
     * franchir toute l'epaisseur d'un collider fin (+ sa marge de contact)
     * en un seul sous-pas, quelle que soit cette marge, puisqu'elle n'est
     * verifiee qu'APRES l'integration de position dans la boucle ci-dessous,
     * jamais PENDANT. D'ou un second critere CFL-like sur collider_cell_size,
     * et n_sub = max des deux.
     *
     * Vitesse MAJOREE sur toute la frame (D2, plan-milestone-13.md), pas
     * seulement la vitesse de depart : n_sub est calcule UNE FOIS en debut de
     * frame (cf. commentaire du kernel), mais la vitesse peut CROITRE en
     * cours de sous-boucle (la gravite est le terme dominant de croissance,
     * la trainee ne fait que decelerer) -- une borne construite sur |v0| seul
     * peut devenir invalide en cours de frame. v_major = |v0| +
     * |gravity_y|*dt majore la vitesse atteignable sur TOUTE la frame par la
     * seule gravite, sans avoir a recalculer n_sub en cours de route (ce qui
     * casserait la garantie h*n_sub == dt, cf. commentaire du kernel). Le
     * plafond passe de 32 a 256 : garde-fou anti-emballement, plus le
     * mecanisme de correction primaire -- c'est desormais la normale de
     * contact precalculee (cnrm, D1) qui porte la garantie geometrique dure
     * quand elle est fournie (cf. plus bas), le sous-pas n'etant plus qu'une
     * mitigation complementaire. */
    float v_major = vlen(v) + fabsf(gravity_y) * dt;
    int n_sub_fluid = (int)ceilf(v_major * dt / (0.5f * R));
    int n_sub = n_sub_fluid;
    if (has_collider) {
        int n_sub_collider = (int)ceilf(v_major * dt / (0.5f * collider_cell_size));
        n_sub = max(n_sub, n_sub_collider);
    }
    n_sub = min(max(n_sub, 1), 256);
    float h = dt / n_sub;

    /* Detection de franchissement par comparaison de NORMALES (pas de signe
     * seul), meme principe que D8/D9 du solveur principal (mlsmpm.cu,
     * k_g2p, cf. normal_corroborated / dot_side / franchi), adapte ici a un
     * gradient recalcule a la volee (ww_sample_collider_sdf) plutot qu'un
     * tampon de normales cnrm precalcule sur la grille : un champ de
     * distance SIGNE seul ne peut pas distinguer "dehors du cote proche" de
     * "dehors du cote oppose" d'une paroi fine -- une particule qui saute
     * proprement d'un cote a l'autre en un seul sous-pas ne declenche jamais
     * phi_s < eps_contact, puisque les DEUX extremites du saut sont hors de
     * la bande. Mesure (pas suppose) sur un jet reel a 1 m/s contre la paroi
     * fine (~4mm) d'un verre du repo : jusqu'a 29% des particules
     * whitewater generees (58358 sur 200000) traversaient quand meme la
     * paroi malgre la bande de marge eps_contact ET les sous-pas augmentes
     * ci-dessus -- ni l'un ni l'autre ne suffit seul.
     *
     * n_old est le gradient normalise du collider a la position de DEPART de
     * la particule pour cette frame (avant tout sous-pas) ; n_new (recalcule
     * ci-dessous a CHAQUE sous-pas, avant le test de reponse plutot que
     * seulement quand phi_s < eps_contact) est le meme gradient a la
     * position COURANTE. Si les deux normales pointent en sens opposes
     * (vdot < 0), une surface fine a ete traversee entre les deux positions,
     * meme si aucune des deux n'est proche du signe negatif -- la reponse
     * collider se declenche alors aussi, en plus du test phi_s < eps_contact
     * existant.
     *
     * Cout : quand has_collider_cnrm est vrai (D1, plan-milestone-13.md), ce
     * gradient n'est plus recalcule DU TOUT -- remplace par un lookup direct
     * de la normale de contact EXACTE precalculee sur le maillage collider
     * brut (cf. commentaire de ww_sample_collider_cnrm), moins couteux qu'un
     * gradient a 6 echantillons trilineaires ET plus precis pres des
     * aretes/sommets. Repli sur le gradient par differences centrees
     * ci-dessous UNIQUEMENT si l'appelant n'a fourni que collider_sdf, sans
     * collider_cnrm (chemin de compatibilite, pas une branche morte) -- cout
     * permanent dans ce cas quand has_collider est vrai, nul sinon (toute
     * cette section est gardee par has_collider). */
    float3 n_old = make_float3(0.f, 0.f, 0.f);
    if (has_collider) {
        if (has_collider_cnrm) {
            float4 cn0 = ww_sample_collider_cnrm(collider_cnrm, collider_cnrm_res,
                                                 collider_cnrm_cell_size, x);
            n_old = make_float3(cn0.x, cn0.y, cn0.z);
        } else {
            float eps0 = 0.5f * collider_cell_size;
            float gx0 = ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                            make_float3(x.x + eps0, x.y, x.z))
                      - ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                            make_float3(x.x - eps0, x.y, x.z));
            float gy0 = ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                            make_float3(x.x, x.y + eps0, x.z))
                      - ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                            make_float3(x.x, x.y - eps0, x.z));
            float gz0 = ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                            make_float3(x.x, x.y, x.z + eps0))
                      - ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                            make_float3(x.x, x.y, x.z - eps0));
            float3 grad0 = make_float3(gx0, gy0, gz0);
            float glen0 = vlen(grad0);
            if (glen0 > 1e-8f) {
                n_old = make_float3(grad0.x / glen0, grad0.y / glen0, grad0.z / glen0);
            }
        }
    }

    for (int step = 0; step < n_sub; ++step) {
        WwSample s = ww_sample_fluid(x, fluid_pos, fluid_vel, bucket_off, bucket_idx,
                                     bucket_res, bucket_h, bucket_origin, R);
        int regime = ww_classify(s.phi, R);

        if (regime == BQ_WW_SPRAY) {
            float vl = vlen(v);
            v.x += (0.f     - drag_spray * v.x * vl) * h;
            v.y += (gravity_y - drag_spray * v.y * vl) * h;
            v.z += (0.f     - drag_spray * v.z * vl) * h;
        } else if (regime == BQ_WW_FOAM) {
            /* Fix D (ecart papier, x_foam(t+dt) = x_foam(t) + dt*v_tilde_f) :
             * la mousse n'a pas d'inertie propre, elle suit directement la
             * vitesse locale moyenne du fluide -- affectation, pas de
             * relaxation (pas de "+="). La position avance ensuite
             * normalement plus bas (x += v*h) comme le reste du code. */
            v = s.v_moy;
        } else { /* BQ_WW_BUBBLE */
            v.x += drag_foam * (s.v_moy.x - v.x) * h;
            v.y += (drag_foam * (s.v_moy.y - v.y) + buoyancy_bubble) * h;
            v.z += drag_foam * (s.v_moy.z - v.z) * h;
        }
        x.x += v.x * h; x.y += v.y * h; x.z += v.z * h;

        /* Borne de domaine, absorbante, 6 faces -- meme principe que les
         * conditions separantes de k_grid_update (mlsmpm.cu, "le mur
         * absorbe") : on annule la composante de vitesse qui pousse HORS du
         * domaine et on clampe la position, jamais un rebond elastique.
         * Appliquee a CHAQUE sous-pas (pas seulement en fin de frame),
         * meme raison d'etre que les sous-pas eux-memes : eviter qu'un
         * grand deplacement en un seul saut traverse une face fine. */
        if (x.x < 0.f)         { x.x = 0.f;         if (v.x < 0.f) v.x = 0.f; }
        if (x.x > domain_hi.x) { x.x = domain_hi.x; if (v.x > 0.f) v.x = 0.f; }
        if (x.y < 0.f)         { x.y = 0.f;         if (v.y < 0.f) v.y = 0.f; }
        if (x.y > domain_hi.y) { x.y = domain_hi.y; if (v.y > 0.f) v.y = 0.f; }
        if (x.z < 0.f)         { x.z = 0.f;         if (v.z < 0.f) v.z = 0.f; }
        if (x.z > domain_hi.z) { x.z = domain_hi.z; if (v.z > 0.f) v.z = 0.f; }

        /* Reponse collider, meme principe "le mur absorbe" que ci-dessus
         * (et que k_grid_update, mlsmpm.cu) : pas une redecouverte
         * independante, une reutilisation consciente de ce principe deja
         * valide mesure a l'appui dans ce projet. */
        if (has_collider) {
            float phi_s = ww_sample_collider_sdf(collider_sdf, collider_res,
                                                 collider_cell_size, x);
            /* Bug mesure (pas suppose) : sur la paroi REELLE d'un verre du
             * repo (Bourrasque_test_verre.glb, ~4mm, 1020 triangles),
             * l'echantillonnage du SDF collider EXACTEMENT SUR LA SURFACE
             * reste POSITIF a 63-91% des points testes selon la resolution
             * de grille (min=-0.00041 max=0.00208 a res 64, min=-0.00030
             * max=0.00105 a res 128) -- une paroi plus fine que dx est
             * quasi invisible au seul signe de phi_s. Meme nature de bug que
             * D7 en M6 pour le solveur principal (bande de contact sur la
             * distance non signee), mais applique ici a whitewater qui
             * n'avait recu aucune marge equivalente : une particule qui
             * approche cette paroi depuis l'exterieur ne declenchait quasiment
             * jamais phi_s < 0, elle traversait sans etre repoussee.
             * Correctif : bande de marge eps_contact = 0.5*collider_cell_size
             * plutot qu'un test de signe strict. Ce facteur est choisi car
             * les depassements positifs mesures sur la paroi reelle
             * (jusqu'a ~2mm) restent nettement sous cette marge aux
             * resolutions testees (64/96/128). La formule de reponse
             * ci-dessous (x -= phi_s * n, absorption de la composante
             * normale de vitesse) reste mathematiquement correcte pour
             * phi_s legerement positif : c'est un pas de Newton vers le
             * niveau zero de l'implicite, valide des deux cotes de la
             * surface -- pas une extension bancale, la meme formule
             * fonctionne nativement sur une petite marge positive. Limite
             * partiellement corrigee ci-dessous par la detection de
             * franchissement par comparaison de normales (cf. commentaire
             * avant la boucle de sous-pas) : une particule TRES rapide dont
             * le deplacement d'un seul sous-pas depasse 2*eps_contact peut
             * quand meme etre rattrapee si son gradient collider a change de
             * sens entre les deux positions -- ce n'est toujours pas une
             * garantie geometrique dure comme D9 du solveur principal
             * (tampon cnrm precalcule sur la grille), mais reduit tres
             * fortement le tunneling residuel mesure (29%). */
            float eps_contact = 0.5f * collider_cell_size;

            /* Normale de contact courante : lookup direct de cnrm (D1) si
             * fourni, sinon repli sur le gradient par differences centrees
             * recalcule a CHAQUE sous-pas comme avant (cf. commentaire avant
             * la boucle de sous-pas pour la justification complete des deux
             * chemins). cn_w n'a de sens que dans le chemin cnrm : distance
             * NON signee au triangle le plus proche, utilisee ci-dessous en
             * plus du signe de phi_s pour rattraper les parois plus fines que
             * la bande signee (meme principe que k_grid_update, mlsmpm.cu,
             * "solid_here || cn.w < h"). */
            float3 n_new = make_float3(0.f, 0.f, 0.f);
            float glen = 0.f;
            float cn_w = 1e6f;
            if (has_collider_cnrm) {
                float4 cn = ww_sample_collider_cnrm(collider_cnrm, collider_cnrm_res,
                                                    collider_cnrm_cell_size, x);
                n_new = make_float3(cn.x, cn.y, cn.z);
                cn_w = cn.w;
                glen = vlen(n_new);
            } else {
                float eps = 0.5f * collider_cell_size;
                float gx = ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                               make_float3(x.x + eps, x.y, x.z))
                         - ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                               make_float3(x.x - eps, x.y, x.z));
                float gy = ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                               make_float3(x.x, x.y + eps, x.z))
                         - ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                               make_float3(x.x, x.y - eps, x.z));
                float gz = ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                               make_float3(x.x, x.y, x.z + eps))
                         - ww_sample_collider_sdf(collider_sdf, collider_res, collider_cell_size,
                               make_float3(x.x, x.y, x.z - eps));
                float3 grad = make_float3(gx, gy, gz);
                glen = vlen(grad);
                if (glen > 1e-8f) {
                    n_new = make_float3(grad.x / glen, grad.y / glen, grad.z / glen);
                }
            }

            bool crossed = has_collider && (vlen(n_old) > 1e-6f) &&
                           (vlen(n_new) > 1e-6f) && (vdot(n_old, n_new) < 0.f);

            /* Garde contre la sentinelle "hors domaine" du SDF collider.
             * Bug mesure : explosion numerique a partir d'environ la frame
             * 28 sur le test dynamique jet/verre, particules whitewater
             * projetees a des coordonnees de l'ordre de 1e6. Mecanisme :
             * une particule clampee exactement sur la borne de domaine par
             * le clamp de bordure absorbant (juste au-dessus dans ce
             * kernel) peut tomber hors du domaine de validite de
             * l'interpolation trilineaire du SDF collider (decalage
             * d'indice de 1 a la frontiere exacte) ; ww_sample_collider_sdf
             * retourne alors sa valeur sentinelle 1e6f ("hors domaine",
             * cf. sa definition). L'ancienne condition phi_s < eps_contact
             * excluait deja naturellement ce cas (1e6 n'est jamais sous une
             * marge de quelques millimetres), mais la condition crossed
             * (comparaison de normales n_old/n_new) peut se declencher
             * independamment de la magnitude de phi_s -- si crossed est
             * vrai alors que phi_s vaut 1e6, la formule de reponse
             * ci-dessous (x -= phi_s * n) teleporte la particule a des
             * coordonnees ~1e6, exactement le bug mesure. phi_s < 1e5f
             * exclut la sentinelle avec une marge de securite claire, sans
             * affecter aucun cas legitime : aucune distance geometrique
             * reelle dans ce projet n'approche cette magnitude (domaines
             * de simulation de quelques metres au plus).
             *
             * Condition additionnelle cn_w < eps_contact (uniquement quand
             * has_collider_cnrm) : meme principe que k_grid_update
             * (mlsmpm.cu, "solid_here || cn.w < h") -- rattrape les parois
             * plus fines que la bande signee ne peut detecter par le seul
             * signe de phi_s, en utilisant la distance non signee EXACTE du
             * champ cnrm plutot que le SDF rasterise dont le signe peut
             * rester positif des deux cotes d'une paroi fine (cf. bug decrit
             * plus haut dans ce commentaire). */
            if ((phi_s < eps_contact || crossed ||
                 (has_collider_cnrm && cn_w < eps_contact)) && phi_s < 1e5f) {
                if (glen > 1e-8f) {
                    float3 n = n_new;
                    /* pousse hors du solide : phi_s < 0 donc -phi_s > 0,
                     * avance selon +n. */
                    x.x -= phi_s * n.x; x.y -= phi_s * n.y; x.z -= phi_s * n.z;
                    /* absorbe la composante de vitesse qui pousse VERS le
                     * solide, garde la tangentielle (glisse le long de la
                     * surface) -- PAS un rebond elastique. */
                    float vn = v.x * n.x + v.y * n.y + v.z * n.z;
                    if (vn < 0.f) { v.x -= vn * n.x; v.y -= vn * n.y; v.z -= vn * n.z; }
                }
            }

            /* report pour le sous-pas suivant : reutilise n_new tel quel
             * (pas de recalcul de gradient a la position corrigee -- ecart
             * negligeable, cf. commentaire avant la boucle), degenere a zero
             * si le gradient etait invalide pour ne pas propager une normale
             * fausse. */
            n_old = (glen > 1e-8f) ? n_new : make_float3(0.f, 0.f, 0.f);
        }
    }

    a += dt; /* une seule fois par frame, cf. commentaire du kernel */

    /* regime final depuis la DERNIERE position -- x a bouge apres le
     * dernier echantillonnage de la boucle (le v*h final est applique
     * apres), donc reechantillonnage necessaire, pas de reutilisation du
     * s.phi du dernier tour. */
    WwSample s_final = ww_sample_fluid(x, fluid_pos, fluid_vel, bucket_off, bucket_idx,
                                       bucket_res, bucket_h, bucket_origin, R);
    int regime_final = ww_classify(s_final.phi, R);

    type[p] = regime_final;
    pos[p] = x;
    vel[p] = v;
    age[p] = a;

    float life = (regime_final == BQ_WW_SPRAY) ? life_spray
               : (regime_final == BQ_WW_FOAM)  ? life_foam
                                                 : life_bubble;
    out_alive[p] = (a < life) ? 1 : 0;
}

/* ---------------------------------------------------- compaction (D6)
 * Un thread par particule active d'origine (les n_active d'AVANT cet
 * appel). Copie les survivantes (alive_flag[p]==1) vers le second jeu de
 * tampons actifs, a l'index donne par le scan exclusif de alive_flag --
 * meme motif que k_mc_emit_vertices (mesher.cu) applique a un flag de vie
 * plutot qu'un flag d'arete. */
__global__ void k_ww_compact(
    const float3* __restrict__ in_pos, const float3* __restrict__ in_vel,
    const int* __restrict__ in_type, const float* __restrict__ in_size,
    const float* __restrict__ in_age,
    float3* __restrict__ out_pos, float3* __restrict__ out_vel,
    int* __restrict__ out_type, float* __restrict__ out_size,
    float* __restrict__ out_age,
    const int* __restrict__ alive_flag, const int* __restrict__ alive_scan,
    int n_active) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_active) return;
    if (!alive_flag[p]) return;
    int dst = alive_scan[p];
    out_pos[dst]  = in_pos[p];
    out_vel[dst]  = in_vel[p];
    out_type[dst] = in_type[p];
    out_size[dst] = in_size[p];
    out_age[dst]  = in_age[p];
}

/* ------------------------------------------------------------------ BqWhitewater */
struct BqWhitewater {
    BqWhitewaterConfig cfg;
    /* particules actives, capacite fixe = cfg.max_particles, allouees une
     * fois a la creation (jamais reallouees -- contrairement aux tampons
     * dependants des particules fluides ci-dessous). */
    float3* d_active_pos = nullptr;
    float3* d_active_vel = nullptr;
    int*    d_active_type = nullptr;
    float*  d_active_size = nullptr;
    float*  d_active_age = nullptr;
    int     n_active = 0;

    /* second jeu, meme capacite fixe -- cible de la compaction des
     * survivantes puis de l'emission des nouvelles particules (k_ww_emit),
     * echange par pointeurs avec le jeu ci-dessus a chaque bq_whitewater_step
     * (ping-pong, meme motif que d_field/d_field_tmp de mesher.cu). Prefere
     * a une compaction en place : capacite fixe donc cout memoire modeste et
     * connu a l'avance, contre une gestion d'index plus complexe en place. */
    float3* d_active2_pos = nullptr;
    float3* d_active2_vel = nullptr;
    int*    d_active2_type = nullptr;
    float*  d_active2_size = nullptr;
    float*  d_active2_age = nullptr;

    /* drapeau de survie (1/0) et scan exclusif associe, dimensionnes
     * max_particles+1 (sentinelle a zero, meme motif que d_gen_count/
     * d_gen_scan), capacite fixe elle aussi (dimensionnee sur le nombre de
     * particules actives, jamais sur le nombre de particules fluides). */
    int*    d_alive_flag = nullptr;
    int*    d_alive_scan = nullptr;

    /* candidates refusees au dernier appel faute de place (cf.
     * bq_whitewater_last_refused, D6/risque 4 du plan). */
    int     last_refused = 0;

    /* dependants du nombre de particules FLUIDES de la frame : realloues
     * (taille exacte) quand la capacite courante est depassee, jamais
     * retrecis -- meme politique que le mailleur (bq_mesher_run). d_carry
     * est le seul tampon dont le CONTENU doit survivre a une croissance
     * (cf. bq_whitewater_step) : les autres sont entierement recalcules a
     * chaque appel. */
    float3* d_fluid_pos = nullptr;
    float3* d_fluid_vel = nullptr;
    float3* d_fluid_normal = nullptr;   /* normale locale par particule fluide, precalculee (cf. k_ww_compute_normals) */
    int*    d_bucket_idx = nullptr;
    float*  d_carry = nullptr;      /* reste fractionnaire par particule fluide */
    float*  d_isolated_time = nullptr; /* duree (s) en continu sous neighbor_count<=k_n_cull
        par particule fluide, cf. k_ww_generate_potentials -- meme cycle de vie que d_carry */
    int*    d_gen_count = nullptr;  /* fluid_cap+1 (case n = sentinelle du scan) */
    int*    d_gen_scan = nullptr;   /* fluid_cap+1, sentinelle a zero */
    int     fluid_cap = 0;

    int*    d_bucket_off = nullptr; /* n_buckets+1, realloue si la resolution
                                        de la grille change (l'AABB bouge) */
    int     bucket_cap = 0;         /* n_buckets alloues */

    void*   d_cub_tmp = nullptr;
    size_t  cub_tmp_bytes = 0;

    /* champ collider, fourni par l'appelant (jamais compte dans les
     * garde-fous VRAM de ce fichier, meme traitement que d_collider_sdf de
     * BqMesher, cf. bq_whitewater_set_collider_sdf) : persistant, realloue
     * seulement si sa resolution change. */
    float*  d_collider_sdf = nullptr;
    int3    collider_res = make_int3(0, 0, 0);
    float   collider_cell_size = 0.f;
    bool    has_collider = false;

    /* champ de normale de contact, MEME cycle de vie que d_collider_sdf
     * ci-dessus (fourni par l'appelant, persistant, realloue seulement si sa
     * resolution change) -- cf. bq_whitewater_set_collider_cnrm. Resolution
     * independante de collider_res : rien n'impose que les deux champs
     * partagent la meme grille, meme si l'appelant les fournit typiquement
     * ensemble depuis le meme d_cnrm/d_sdf du solveur (cf. bourrasque.h). */
    float4* d_collider_cnrm = nullptr;
    int3    collider_cnrm_res = make_int3(0, 0, 0);
    float   collider_cnrm_cell_size = 0.f;
    bool    has_collider_cnrm = false;
};

/* Empreinte d'UN jeu de tampons actifs (independante des particules
 * fluides) : pos(12)+vel(12)+type(4)+size(4)+age(4) par particule active. */
static int64_t ww_active_bytes(int max_particles) {
    return (int64_t)max_particles * (12 + 12 + 4 + 4 + 4);
}

/* Empreinte TOTALE de la capacite active a max_particles fixe (independante
 * des particules fluides) : DEUX jeux complets de tampons actifs (ping-pong
 * de compaction, cf. struct BqWhitewater) + les deux tampons de scan
 * d_alive_flag/d_alive_scan (max_particles+1 int chacun). Avant la tache
 * d'advection/compaction, ce total valait ww_active_bytes(max_particles)
 * seul (un jeu, pas de scan de survie) -- double desormais la part active
 * et ajoute 2*(max_particles+1)*4 octets. */
static int64_t ww_active_bytes_total(int max_particles) {
    int64_t one_set = ww_active_bytes(max_particles);
    int64_t scan_bytes = 2 * (int64_t)(max_particles + 1) * (int64_t)sizeof(int);
    return 2 * one_set + scan_bytes;
}

/* Capacite active lineaire maximale tenant dans budget octets -- recherche
 * binaire, meme motif que max_cubic_res_for_budget (mesher.cu), mais sur
 * une grandeur lineaire (particules) plutot que cubique (resolution). */
static int max_linear_particles_for_budget(size_t budget) {
    int lo = 1, hi = 1 << 30, best = 0;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int64_t b = ww_active_bytes_total(mid);
        if ((uint64_t)b <= (uint64_t)budget) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    return best;
}

/* -------------------------------------------------------------------- API */
extern "C" {

BQ_API int bq_whitewater_config_size(void) {
    return (int)sizeof(BqWhitewaterConfig);
}

BQ_API void bq_whitewater_default_config(BqWhitewaterConfig* cfg) {
    if (!cfg) return;
    cfg->max_particles = 200000;
    cfg->influence_radius = 3.f / 128.f;
    cfg->gravity_y = -9.8f;
    /* valeurs de depart raisonnables, PAS mesurees -- a calibrer sur une
     * scene reelle avant un premier passage de production (cf. plan
     * milestone 8, risque 1). */
    cfg->ta_min = 2.f;  cfg->ta_max = 8.f;  cfg->ta_weight = 1.f;
    cfg->wc_min = 1.f;  cfg->wc_max = 5.f;  cfg->wc_weight = 1.f;
    cfg->ke_min = 1.f;  cfg->ke_max = 10.f; cfg->ke_weight = 1.f;
    cfg->spawn_rate = 50.f;
    cfg->life_spray = 1.f;
    cfg->life_foam = 2.f;
    cfg->life_bubble = 1.5f;
    cfg->drag_spray = 0.1f;
    cfg->drag_foam = 3.f;
    cfg->buoyancy_bubble = 2.f;
    cfg->grid_res[0] = cfg->grid_res[1] = cfg->grid_res[2] = 64;
    cfg->cell_size = 1.f / 64.f;
}

BQ_API int bq_whitewater_vram_estimate(const BqWhitewaterConfig* cfg,
                                       int n_fluid_hint, int64_t* bytes) {
    if (!cfg || !bytes) {
        snprintf(g_error, sizeof(g_error), "bq_whitewater_vram_estimate: pointeur nul");
        return -1;
    }
    if (cfg->max_particles <= 0) {
        snprintf(g_error, sizeof(g_error),
                 "bq_whitewater_vram_estimate: max_particles invalide (%d)",
                 cfg->max_particles);
        return -1;
    }
    int64_t indep = ww_active_bytes_total(cfg->max_particles);
    int64_t nf = n_fluid_hint > 0 ? (int64_t)n_fluid_hint : 0;
    /* Estimation grossiere de la part dependante des particules fluides :
     * position+vitesse+bucket_idx+carry+gen_count+gen_scan. La grille de
     * buckets reelle depend de l'AABB fluide au moment de l'appel, pas
     * connue a l'avance -- non comptee ici. */
    int64_t fluid_part = nf * (12 + 12 + 4 + 4 + 4 + 4);
    *bytes = indep + fluid_part;
    return 0;
}

BQ_API BqWhitewater* bq_whitewater_create(const BqWhitewaterConfig* cfg) {
    BqWhitewaterConfig c;
    if (cfg) c = *cfg; else bq_whitewater_default_config(&c);

    if (c.max_particles <= 0) {
        snprintf(g_error, sizeof(g_error),
                 "bq_whitewater_create: max_particles invalide (%d)", c.max_particles);
        return nullptr;
    }

    /* garde-fou VRAM : part independante des particules fluides (capacite
     * active), le nombre de particules fluides n'est pas encore connu (cf.
     * spec T2, meme contrat que bq_mesher_create). Jamais un cudaMalloc qui
     * echoue en silence. */
    int64_t bytes_indep = ww_active_bytes_total(c.max_particles);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "bq_whitewater_create: cudaMemGetInfo echoue");
        return nullptr;
    }
    if ((uint64_t)bytes_indep > (uint64_t)free_b) {
        int max_mp = max_linear_particles_for_budget(free_b);
        snprintf(g_error, sizeof(g_error),
                 "bq_whitewater_create: VRAM insuffisante (demande %.1f Mo, libre "
                 "%.1f Mo) -- capacite maximale estimee sur cette carte : %d "
                 "particules",
                 bytes_indep / 1e6, (double)free_b / 1e6, max_mp);
        return nullptr;
    }

    BqWhitewater* w = new BqWhitewater();
    w->cfg = c;

    if (cudaMalloc(&w->d_active_pos, (size_t)c.max_particles * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&w->d_active_vel, (size_t)c.max_particles * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&w->d_active_type, (size_t)c.max_particles * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&w->d_active_size, (size_t)c.max_particles * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&w->d_active_age, (size_t)c.max_particles * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&w->d_active2_pos, (size_t)c.max_particles * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&w->d_active2_vel, (size_t)c.max_particles * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&w->d_active2_type, (size_t)c.max_particles * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&w->d_active2_size, (size_t)c.max_particles * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&w->d_active2_age, (size_t)c.max_particles * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&w->d_alive_flag, (size_t)(c.max_particles + 1) * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&w->d_alive_scan, (size_t)(c.max_particles + 1) * sizeof(int)) != cudaSuccess) {
        snprintf(g_error, sizeof(g_error), "bq_whitewater_create: cudaMalloc echoue");
        bq_whitewater_destroy(w);
        return nullptr;
    }
    return w;
}

BQ_API void bq_whitewater_destroy(BqWhitewater* w) {
    if (!w) return;
    cudaFree(w->d_active_pos);
    cudaFree(w->d_active_vel);
    cudaFree(w->d_active_type);
    cudaFree(w->d_active_size);
    cudaFree(w->d_active_age);
    cudaFree(w->d_active2_pos);
    cudaFree(w->d_active2_vel);
    cudaFree(w->d_active2_type);
    cudaFree(w->d_active2_size);
    cudaFree(w->d_active2_age);
    cudaFree(w->d_alive_flag);
    cudaFree(w->d_alive_scan);
    cudaFree(w->d_fluid_pos);
    cudaFree(w->d_fluid_vel);
    cudaFree(w->d_fluid_normal);
    cudaFree(w->d_bucket_idx);
    cudaFree(w->d_carry);
    cudaFree(w->d_isolated_time);
    cudaFree(w->d_gen_count);
    cudaFree(w->d_gen_scan);
    cudaFree(w->d_bucket_off);
    cudaFree(w->d_cub_tmp);
    cudaFree(w->d_collider_sdf);
    cudaFree(w->d_collider_cnrm);
    delete w;
}

BQ_API int bq_whitewater_step(BqWhitewater* w, const float* pos,
                              const float* vel, int n, float dt) {
    if (!w) {
        snprintf(g_error, sizeof(g_error), "bq_whitewater_step: whitewater nul");
        return -1;
    }
    if (n < 0) {
        snprintf(g_error, sizeof(g_error), "bq_whitewater_step: n negatif (%d)", n);
        return -1;
    }
    if (n > 0 && (!pos || !vel)) {
        snprintf(g_error, sizeof(g_error),
                 "bq_whitewater_step: pos/vel nul (n=%d)", n);
        return -1;
    }
    if (n == 0) {
        /* Pas de particules fluides cette frame : aucune grille de buckets
         * ne peut etre construite (l'AABB serait degenere), donc ni
         * generation ni avancement des particules deja actives -- la liste
         * active reste entierement inchangee pour cet appel. Limite
         * assumee, pas couverte par la tache d'advection (non mentionnee
         * dans sa spec) : a signaler si une scene produit reellement des
         * frames fluides vides en cours de bake. */
        return 0;
    }

    float R = w->cfg.influence_radius > 1e-8f ? w->cfg.influence_radius : 1e-8f;

    /* AABB hote des positions fluides de cette frame. */
    float3 mn = make_float3(pos[0], pos[1], pos[2]);
    float3 mx = mn;
    for (int p = 1; p < n; ++p) {
        float x = pos[3 * p + 0], y = pos[3 * p + 1], z = pos[3 * p + 2];
        mn.x = std::min(mn.x, x); mx.x = std::max(mx.x, x);
        mn.y = std::min(mn.y, y); mx.y = std::max(mx.y, y);
        mn.z = std::min(mn.z, z); mx.z = std::max(mx.z, z);
    }

    BucketGrid bg;
    float margin = 2.f * R;
    bg.origin = make_float3(mn.x - margin, mn.y - margin, mn.z - margin);
    bg.h = R;
    float dom_x = (mx.x - mn.x) + 2.f * margin;
    float dom_y = (mx.y - mn.y) + 2.f * margin;
    float dom_z = (mx.z - mn.z) + 2.f * margin;
    bg.res.x = std::max(1, (int)ceilf(dom_x / R));
    bg.res.y = std::max(1, (int)ceilf(dom_y / R));
    bg.res.z = std::max(1, (int)ceilf(dom_z / R));
    int64_t n_buckets = (int64_t)bg.res.x * bg.res.y * bg.res.z;

    /* reallocation grow-only des tampons dependants des particules fluides,
     * taille exacte. d_carry doit preserver son contenu existant (l'indice
     * d'une particule fluide dans `pos` est suppose stable d'une frame a
     * l'autre, meme hypothese que le jitter de mesher.cu) : les nouveaux
     * elements sont mis a zero, les anciens ne sont PAS touches. */
    if (n > w->fluid_cap) {
        float3* new_fluid_pos = nullptr;
        float3* new_fluid_vel = nullptr;
        float3* new_fluid_normal = nullptr;
        int*    new_bucket_idx = nullptr;
        float*  new_carry = nullptr;
        float*  new_isolated_time = nullptr;
        int*    new_gen_count = nullptr;
        int*    new_gen_scan = nullptr;

        bool ok = cudaMalloc(&new_fluid_pos, (size_t)n * sizeof(float3)) == cudaSuccess &&
                  cudaMalloc(&new_fluid_vel, (size_t)n * sizeof(float3)) == cudaSuccess &&
                  cudaMalloc(&new_fluid_normal, (size_t)n * sizeof(float3)) == cudaSuccess &&
                  cudaMalloc(&new_bucket_idx, (size_t)n * sizeof(int)) == cudaSuccess &&
                  cudaMalloc(&new_carry, (size_t)n * sizeof(float)) == cudaSuccess &&
                  cudaMalloc(&new_isolated_time, (size_t)n * sizeof(float)) == cudaSuccess &&
                  cudaMalloc(&new_gen_count, (size_t)(n + 1) * sizeof(int)) == cudaSuccess &&
                  cudaMalloc(&new_gen_scan, (size_t)(n + 1) * sizeof(int)) == cudaSuccess;
        if (!ok) {
            cudaFree(new_fluid_pos); cudaFree(new_fluid_vel); cudaFree(new_fluid_normal);
            cudaFree(new_bucket_idx);
            cudaFree(new_carry); cudaFree(new_isolated_time);
            cudaFree(new_gen_count); cudaFree(new_gen_scan);
            snprintf(g_error, sizeof(g_error),
                     "bq_whitewater_step: cudaMalloc echoue (n=%d)", n);
            return -1;
        }

        if (w->d_carry && w->fluid_cap > 0) {
            BQ_CUDA_CHECK(cudaMemcpy(new_carry, w->d_carry,
                                     (size_t)w->fluid_cap * sizeof(float),
                                     cudaMemcpyDeviceToDevice));
        }
        BQ_CUDA_CHECK(cudaMemset(new_carry + w->fluid_cap, 0,
                                 (size_t)(n - w->fluid_cap) * sizeof(float)));

        if (w->d_isolated_time && w->fluid_cap > 0) {
            BQ_CUDA_CHECK(cudaMemcpy(new_isolated_time, w->d_isolated_time,
                                     (size_t)w->fluid_cap * sizeof(float),
                                     cudaMemcpyDeviceToDevice));
        }
        BQ_CUDA_CHECK(cudaMemset(new_isolated_time + w->fluid_cap, 0,
                                 (size_t)(n - w->fluid_cap) * sizeof(float)));

        cudaFree(w->d_fluid_pos); cudaFree(w->d_fluid_vel); cudaFree(w->d_fluid_normal);
        cudaFree(w->d_bucket_idx);
        cudaFree(w->d_carry); cudaFree(w->d_isolated_time);
        cudaFree(w->d_gen_count); cudaFree(w->d_gen_scan);

        w->d_fluid_pos = new_fluid_pos;
        w->d_fluid_vel = new_fluid_vel;
        w->d_fluid_normal = new_fluid_normal;
        w->d_bucket_idx = new_bucket_idx;
        w->d_carry = new_carry;
        w->d_isolated_time = new_isolated_time;
        w->d_gen_count = new_gen_count;
        w->d_gen_scan = new_gen_scan;
        w->fluid_cap = n;
    }

    /* d_bucket_off : realloue des que la resolution de la grille change
     * (l'AABB fluide bouge d'une frame a l'autre), pas de donnees a
     * preserver (recalcule entierement chaque appel). */
    if ((int64_t)w->bucket_cap != n_buckets) {
        cudaFree(w->d_bucket_off);
        w->d_bucket_off = nullptr;
        w->bucket_cap = 0;
        if (cudaMalloc(&w->d_bucket_off, (size_t)(n_buckets + 1) * sizeof(int)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_whitewater_step: cudaMalloc echoue (buckets=%lld)",
                     (long long)n_buckets);
            return -1;
        }
        w->bucket_cap = (int)n_buckets;
    }

    /* espace de travail CUB, grow-only, dimensionne pour le PLUS GRAND des
     * deux scans de cet appel : celui de gen_count (n+1, particules
     * fluides) et celui de d_alive_flag (w->n_active+1, particules actives
     * courantes) -- meme tampon d_cub_tmp reutilise pour les deux, jamais
     * simultanement. */
    size_t tmp_needed = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_needed, (int*)nullptr, (int*)nullptr, n + 1);
    size_t tmp_needed_alive = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_needed_alive, (int*)nullptr, (int*)nullptr,
                                   w->n_active + 1);
    tmp_needed = std::max(tmp_needed, tmp_needed_alive);
    if (tmp_needed > w->cub_tmp_bytes) {
        cudaFree(w->d_cub_tmp);
        w->d_cub_tmp = nullptr;
        w->cub_tmp_bytes = 0;
        if (cudaMalloc(&w->d_cub_tmp, tmp_needed) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_whitewater_step: cudaMalloc echoue (espace de travail CUB, "
                     "%.1f Mo)", tmp_needed / 1e6);
            return -1;
        }
        w->cub_tmp_bytes = tmp_needed;
    }

    /* bucketing CSR cote hote (pos/vel en memoire hote, meme motif que
     * bq_mesher_run), puis televersement. */
    int nb = (int)n_buckets;
    std::vector<int> off((size_t)nb + 1, 0);
    std::vector<int> bidx_of_p((size_t)n);
    for (int p = 0; p < n; ++p) {
        float x = pos[3 * p + 0], y = pos[3 * p + 1], z = pos[3 * p + 2];
        int bi = (int)floorf((x - bg.origin.x) / bg.h);
        int bj = (int)floorf((y - bg.origin.y) / bg.h);
        int bk = (int)floorf((z - bg.origin.z) / bg.h);
        bi = std::min(std::max(bi, 0), bg.res.x - 1);
        bj = std::min(std::max(bj, 0), bg.res.y - 1);
        bk = std::min(std::max(bk, 0), bg.res.z - 1);
        int b = (bi * bg.res.y + bj) * bg.res.z + bk;
        bidx_of_p[p] = b;
        off[b + 1]++;
    }
    for (int b = 0; b < nb; ++b) off[b + 1] += off[b];

    std::vector<int> idx((size_t)n);
    std::vector<int> cursor(off.begin(), off.end());
    for (int p = 0; p < n; ++p) {
        int b = bidx_of_p[p];
        idx[cursor[b]++] = p;
    }

    BQ_CUDA_CHECK(cudaMemcpy(w->d_bucket_off, off.data(), (size_t)(nb + 1) * sizeof(int),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(w->d_fluid_pos, pos, (size_t)n * sizeof(float3),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(w->d_fluid_vel, vel, (size_t)n * sizeof(float3),
                             cudaMemcpyHostToDevice));
    BQ_CUDA_CHECK(cudaMemcpy(w->d_bucket_idx, idx.data(), (size_t)n * sizeof(int),
                             cudaMemcpyHostToDevice));

    /* sentinelle du scan : la case n de d_gen_count doit valoir 0 a CHAQUE
     * appel (elle peut avoir ete ecrasee par le kernel de generation si n a
     * grandi puis retreci entre deux appels). */
    BQ_CUDA_CHECK(cudaMemset(w->d_gen_count + n, 0, sizeof(int)));

    dim3 bs(256), gs((unsigned)((n + 255) / 256));
    k_ww_compute_normals<<<gs, bs>>>(
        w->d_fluid_normal, w->d_fluid_pos, w->d_bucket_off, w->d_bucket_idx,
        bg.res, bg.h, bg.origin, R, n);
    BQ_CUDA_CHECK(cudaGetLastError());

    k_ww_generate_potentials<<<gs, bs>>>(
        w->d_gen_count, w->d_carry, w->d_isolated_time, w->d_fluid_pos, w->d_fluid_vel,
        w->d_fluid_normal,
        w->d_bucket_off, w->d_bucket_idx, bg.res, bg.h, bg.origin, R,
        w->cfg.ta_min, w->cfg.ta_max, w->cfg.ta_weight,
        w->cfg.wc_min, w->cfg.wc_max, w->cfg.wc_weight,
        w->cfg.ke_min, w->cfg.ke_max, w->cfg.ke_weight,
        w->cfg.spawn_rate, dt, n);
    BQ_CUDA_CHECK(cudaGetLastError());

    BQ_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        w->d_cub_tmp, w->cub_tmp_bytes, w->d_gen_count, w->d_gen_scan, n + 1));

    int total_gen = 0;
    BQ_CUDA_CHECK(cudaMemcpy(&total_gen, w->d_gen_scan + n, sizeof(int),
                             cudaMemcpyDeviceToHost));

    /* Avancement des particules secondaires DEJA actives (celles d'avant cet
     * appel) : advection multi-regime, sous-pas internes, vieillissement,
     * puis compaction des survivantes. Independant de la generation
     * ci-dessus (etapes 2-3), converge seulement au point de jonction
     * generation/compaction plus bas (cf. D4/D5/D6 du plan). Reutilise la
     * MEME grille de buckets fluide (bg) que la generation -- construite
     * une seule fois en debut d'appel. */
    int n_active_cur = w->n_active;
    dim3 bs_a(256), gs_a((unsigned)((n_active_cur + 255) / 256));

    /* domaine de simulation, calcule UNE FOIS par appel (pas par sous-pas,
     * cf. commentaire de k_ww_advect) -- meme convention [0, res*cell_size]
     * que BqConfig. */
    float3 domain_hi = make_float3(w->cfg.grid_res[0] * w->cfg.cell_size,
                                   w->cfg.grid_res[1] * w->cfg.cell_size,
                                   w->cfg.grid_res[2] * w->cfg.cell_size);

    if (n_active_cur > 0) {
        k_ww_advect<<<gs_a, bs_a>>>(
            w->d_active_pos, w->d_active_vel, w->d_active_type, w->d_active_age,
            w->d_alive_flag, w->d_fluid_pos, w->d_fluid_vel, w->d_bucket_off,
            w->d_bucket_idx, bg.res, bg.h, bg.origin,
            domain_hi, w->d_collider_sdf, w->collider_res, w->collider_cell_size,
            w->has_collider,
            w->d_collider_cnrm, w->collider_cnrm_res, w->collider_cnrm_cell_size,
            w->has_collider_cnrm,
            w->cfg.gravity_y, w->cfg.drag_spray, w->cfg.drag_foam,
            w->cfg.buoyancy_bubble, w->cfg.influence_radius, dt,
            w->cfg.life_spray, w->cfg.life_foam, w->cfg.life_bubble, n_active_cur);
        BQ_CUDA_CHECK(cudaGetLastError());
    }

    /* sentinelle du scan de survie : la case n_active_cur de d_alive_flag
     * doit valoir 0 a CHAQUE appel (n_active_cur change d'un appel a
     * l'autre, meme prudence que pour d_gen_count plus haut). */
    BQ_CUDA_CHECK(cudaMemset(w->d_alive_flag + n_active_cur, 0, sizeof(int)));

    BQ_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        w->d_cub_tmp, w->cub_tmp_bytes, w->d_alive_flag, w->d_alive_scan,
        n_active_cur + 1));

    int n_survivants = 0;
    BQ_CUDA_CHECK(cudaMemcpy(&n_survivants, w->d_alive_scan + n_active_cur, sizeof(int),
                             cudaMemcpyDeviceToHost));

    if (n_active_cur > 0) {
        k_ww_compact<<<gs_a, bs_a>>>(
            w->d_active_pos, w->d_active_vel, w->d_active_type, w->d_active_size,
            w->d_active_age,
            w->d_active2_pos, w->d_active2_vel, w->d_active2_type, w->d_active2_size,
            w->d_active2_age,
            w->d_alive_flag, w->d_alive_scan, n_active_cur);
        BQ_CUDA_CHECK(cudaGetLastError());
    }

    int room = std::max(0, w->cfg.max_particles - n_survivants);
    int actually_gen = std::min(total_gen, room);

    if (actually_gen > 0) {
        k_ww_emit<<<gs, bs>>>(
            w->d_active2_pos, w->d_active2_vel, w->d_active2_type, w->d_active2_size,
            w->d_active2_age, w->d_fluid_pos, w->d_fluid_vel, w->d_gen_count,
            w->d_gen_scan, w->d_bucket_off, w->d_bucket_idx, bg.res, bg.h,
            bg.origin, R, n_survivants, actually_gen, w->cfg.influence_radius, dt, n);
        BQ_CUDA_CHECK(cudaGetLastError());
    }

    /* echange des pointeurs : le second jeu (compaction + emission) devient
     * le jeu actif, l'ancien devient la cible de la prochaine compaction. */
    std::swap(w->d_active_pos, w->d_active2_pos);
    std::swap(w->d_active_vel, w->d_active2_vel);
    std::swap(w->d_active_type, w->d_active2_type);
    std::swap(w->d_active_size, w->d_active2_size);
    std::swap(w->d_active_age, w->d_active2_age);

    w->n_active = n_survivants + actually_gen;
    w->last_refused = total_gen - actually_gen;

    BQ_CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

BQ_API int bq_whitewater_count(const BqWhitewater* w) {
    return w ? w->n_active : 0;
}

BQ_API int bq_whitewater_read(const BqWhitewater* w, float* pos, int* type,
                              float* size, float* age, float* vel) {
    if (!w) {
        snprintf(g_error, sizeof(g_error), "bq_whitewater_read: whitewater nul");
        return -1;
    }
    int n = w->n_active;
    if (n > 0 && pos) {
        BQ_CUDA_CHECK(cudaMemcpy(pos, w->d_active_pos, (size_t)n * sizeof(float3),
                                 cudaMemcpyDeviceToHost));
    }
    if (n > 0 && type) {
        BQ_CUDA_CHECK(cudaMemcpy(type, w->d_active_type, (size_t)n * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    }
    if (n > 0 && size) {
        BQ_CUDA_CHECK(cudaMemcpy(size, w->d_active_size, (size_t)n * sizeof(float),
                                 cudaMemcpyDeviceToHost));
    }
    if (n > 0 && age) {
        BQ_CUDA_CHECK(cudaMemcpy(age, w->d_active_age, (size_t)n * sizeof(float),
                                 cudaMemcpyDeviceToHost));
    }
    if (n > 0 && vel) {
        BQ_CUDA_CHECK(cudaMemcpy(vel, w->d_active_vel, (size_t)n * sizeof(float3),
                                 cudaMemcpyDeviceToHost));
    }
    return n;
}

BQ_API int bq_whitewater_last_refused(const BqWhitewater* w) {
    if (!w) {
        snprintf(g_error, sizeof(g_error), "bq_whitewater_last_refused: whitewater nul");
        return -1;
    }
    return w->last_refused;
}

BQ_API int bq_whitewater_set_collider_sdf(BqWhitewater* w, const float* sdf,
                                          const int res[3], float cell_size) {
    if (!w) {
        snprintf(g_error, sizeof(g_error), "bq_whitewater_set_collider_sdf: whitewater nul");
        return -1;
    }
    if (!sdf) {
        /* efface le collider courant (cf. bq_mesher_set_collider_sdf) */
        cudaFree(w->d_collider_sdf);
        w->d_collider_sdf = nullptr;
        w->collider_res = make_int3(0, 0, 0);
        w->collider_cell_size = 0.f;
        w->has_collider = false;
        return 0;
    }
    if (!res || res[0] <= 0 || res[1] <= 0 || res[2] <= 0 || !(cell_size > 0.f)) {
        snprintf(g_error, sizeof(g_error),
                 "bq_whitewater_set_collider_sdf: parametres invalides (res=(%d,%d,%d), "
                 "cell_size=%f)", res ? res[0] : -1, res ? res[1] : -1,
                 res ? res[2] : -1, cell_size);
        return -1;
    }

    int3 new_res = make_int3(res[0], res[1], res[2]);
    if (new_res.x != w->collider_res.x || new_res.y != w->collider_res.y ||
        new_res.z != w->collider_res.z) {
        cudaFree(w->d_collider_sdf);
        w->d_collider_sdf = nullptr;
        w->collider_res = make_int3(0, 0, 0);
        int64_t n = (int64_t)new_res.x * new_res.y * new_res.z;
        if (cudaMalloc(&w->d_collider_sdf, (size_t)n * sizeof(float)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_whitewater_set_collider_sdf: cudaMalloc echoue (%.1f Mo)",
                     n * sizeof(float) / 1e6);
            return -1;
        }
    }

    int64_t n = (int64_t)new_res.x * new_res.y * new_res.z;
    BQ_CUDA_CHECK(cudaMemcpy(w->d_collider_sdf, sdf, (size_t)n * sizeof(float),
                             cudaMemcpyHostToDevice));
    w->collider_res = new_res;
    w->collider_cell_size = cell_size;
    w->has_collider = true;
    return 0;
}

BQ_API int bq_whitewater_set_collider_cnrm(BqWhitewater* w, const float* cnrm,
                                           const int res[3], float cell_size) {
    if (!w) {
        snprintf(g_error, sizeof(g_error), "bq_whitewater_set_collider_cnrm: whitewater nul");
        return -1;
    }
    if (!cnrm) {
        /* efface le champ courant (cf. bq_whitewater_set_collider_sdf) --
         * k_ww_advect retombe alors sur le calcul par differences finies a
         * partir du seul collider_sdf, tant qu'il reste fourni. */
        cudaFree(w->d_collider_cnrm);
        w->d_collider_cnrm = nullptr;
        w->collider_cnrm_res = make_int3(0, 0, 0);
        w->collider_cnrm_cell_size = 0.f;
        w->has_collider_cnrm = false;
        return 0;
    }
    if (!res || res[0] <= 0 || res[1] <= 0 || res[2] <= 0 || !(cell_size > 0.f)) {
        snprintf(g_error, sizeof(g_error),
                 "bq_whitewater_set_collider_cnrm: parametres invalides (res=(%d,%d,%d), "
                 "cell_size=%f)", res ? res[0] : -1, res ? res[1] : -1,
                 res ? res[2] : -1, cell_size);
        return -1;
    }

    int3 new_res = make_int3(res[0], res[1], res[2]);
    if (new_res.x != w->collider_cnrm_res.x || new_res.y != w->collider_cnrm_res.y ||
        new_res.z != w->collider_cnrm_res.z) {
        cudaFree(w->d_collider_cnrm);
        w->d_collider_cnrm = nullptr;
        w->collider_cnrm_res = make_int3(0, 0, 0);
        int64_t n = (int64_t)new_res.x * new_res.y * new_res.z;
        if (cudaMalloc(&w->d_collider_cnrm, (size_t)n * sizeof(float4)) != cudaSuccess) {
            snprintf(g_error, sizeof(g_error),
                     "bq_whitewater_set_collider_cnrm: cudaMalloc echoue (%.1f Mo)",
                     n * sizeof(float4) / 1e6);
            return -1;
        }
    }

    int64_t n = (int64_t)new_res.x * new_res.y * new_res.z;
    BQ_CUDA_CHECK(cudaMemcpy(w->d_collider_cnrm, cnrm, (size_t)n * sizeof(float4),
                             cudaMemcpyHostToDevice));
    w->collider_cnrm_res = new_res;
    w->collider_cnrm_cell_size = cell_size;
    w->has_collider_cnrm = true;
    return 0;
}

} /* extern "C" */

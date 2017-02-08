#ifndef COUNTERS_H
#define COUNTERS_H

#include <stdint.h>
#define NMC 6
#define NEDC 8
struct ctrs_s
{
    uint64_t mcrd[NMC];
    uint64_t mcwr[NMC];
    uint64_t edcrd[NEDC]; 
    uint64_t edcwr[NEDC];
    uint64_t edchite[NEDC];
    uint64_t edchitm[NEDC];
    uint64_t edcmisse[NEDC];
    uint64_t edcmissm[NEDC];
};
typedef struct ctrs_s ctrs;

#ifdef __cplusplus
  extern "C"{
#endif
void readctrs(ctrs *c);
void setup();
#ifdef __cplusplus
}
#endif
#endif/*COUNTERS_H*/

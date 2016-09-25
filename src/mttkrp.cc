
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"
#include "thd_info.h"
#include "tile.h"
#include "util.h"
#include "simd_utils.h"
#include <limits.h>
#include <omp.h>
#include <time.h>
#include <algorithm>
#ifdef __INTEL_COMPILER
#include <immintrin.h>
#define SPLATT_INTRINSIC // use intrinsic
#endif
#include <unistd.h>

//#define BW_MEASUREMENT
#ifdef BW_MEASUREMENT
#include "perf/counters.h"
#include "perf/counters.cpp"
#include "perf/counters.h"
#endif

#define NLOCKS 1024
static int locks_initialized = 0;

#define SPLATT_LOCK_PAD (8)
static volatile long ttas_locks[NLOCKS*SPLATT_LOCK_PAD];
static omp_lock_t omp_locks[NLOCKS*SPLATT_LOCK_PAD];

//#define SPLATT_COUNT_FLOP
#ifdef SPLATT_COUNT_FLOP
static long mttkrp_flops = 0;
#endif

//#define SPLATT_COUNT_REDUCTION
#ifdef SPLATT_COUNT_REDUCTION
static long mttkrp_reduction = 0;
#endif

static void p_init_locks()
{
  if (!locks_initialized) {
    for(int i=0; i < NLOCKS; ++i) {
      ttas_locks[i*SPLATT_LOCK_PAD] = 0;
      omp_init_lock(omp_locks+i*SPLATT_LOCK_PAD);
    }
    locks_initialized = 1;
  }
}

timespec req = { 0, 1000 };


template<splatt_sync_type SYNC_TYPE>
static inline void splatt_set_lock(int id)
{
  int i = (id%NLOCKS)*SPLATT_LOCK_PAD;
  if(SPLATT_SYNC_TTAS == SYNC_TYPE) {
    //while (0 != ttas_locks[i] || 0 != __sync_lock_test_and_set(ttas_locks + i, 0xff)); // TTAS
    while (0 != ttas_locks[i] || 0 != __sync_lock_test_and_set(ttas_locks + i, 1)) { _mm_pause(); }; // TAS with backoff
    //while (0 != ttas_locks[i] || 0 != __sync_lock_test_and_set(ttas_locks + i, 1)) { nanosleep(&req, NULL); }; // TAS with backoff
    /*while (true) {
      while (ttas_locks[i]) _mm_pause();
      if (0 == __sync_lock_test_and_set(ttas_locks + i, 1)) break;
      else {
        for (int i = 0; i < 512; ++i) {
          _mm_pause();
        }
      }
    }*/
  }
  else if(SPLATT_SYNC_NOSYNC != SYNC_TYPE) {
    omp_set_lock(omp_locks + i);
  }
}


template<splatt_sync_type SYNC_TYPE>
static inline void splatt_unset_lock(int id)
{
  int i = (id%NLOCKS)*SPLATT_LOCK_PAD;
  if(SPLATT_SYNC_TTAS == SYNC_TYPE) {
    ttas_locks[i] = 0;
    asm volatile("" ::: "memory");
    //__sync_lock_release(ttas_locks + i); // icc generates mfence instruction for this
  }
  else if(SPLATT_SYNC_NOSYNC != SYNC_TYPE) {
    omp_unset_lock(omp_locks + i);
  }
}



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
int splatt_mttkrp(
    splatt_idx_t const mode,
    splatt_idx_t const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const matout,
    double const * const options)
{
  idx_t const nmodes = tensors->nmodes;

  /* fill matrix pointers  */
  matrix_t * mats[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (matrix_t *) splatt_malloc(sizeof(matrix_t));
    mats[m]->I = tensors->dims[m];
    mats[m]->J = ncolumns,
    mats[m]->rowmajor = 1;
    mats[m]->vals = matrices[m];
  }
  mats[MAX_NMODES] = (matrix_t *) splatt_malloc(sizeof(matrix_t));
  mats[MAX_NMODES]->I = tensors->dims[mode];
  mats[MAX_NMODES]->J = ncolumns;
  mats[MAX_NMODES]->rowmajor = 1;
  mats[MAX_NMODES]->vals = matout;

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  idx_t const nthreads = (idx_t) options[SPLATT_OPTION_NTHREADS];
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 3,
    (ncolumns * ncolumns * sizeof(val_t)) + 64,
    0,
    (nmodes * ncolumns * sizeof(val_t)) + 64);

  /* do the MTTKRP */
  mttkrp_csf(tensors, mats, mode, thds, options);

  /* cleanup */
  thd_free(thds, nthreads);
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]);
  }
  free(mats[MAX_NMODES]);

  return SPLATT_SUCCESS;
}



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static inline void p_add_hada(
  val_t * const restrict out,
  val_t const * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] += a[f] * b[f];
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*nfactors;
#endif
}


template<int NFACTORS>
static inline void p_add_hada_(
  SPLATT_SIMDFPTYPE * const restrict out,
  SPLATT_SIMDFPTYPE const * const restrict a,
  val_t const * const restrict b)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
    out[f] = _MM_FMADD(a[f], _MM_LOAD(b + f*SPLATT_VLEN), out[f]);
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*NFACTORS;
#endif
}


template<int NFACTORS>
static inline void p_add_hada_(
  val_t * const restrict out,
  SPLATT_SIMDFPTYPE const * const restrict a,
  SPLATT_SIMDFPTYPE const * const restrict b)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
    _MM_STORE(out + f*SPLATT_VLEN, _MM_FMADD(a[f], b[f], _MM_LOAD(out + f*SPLATT_VLEN)));
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*NFACTORS;
#endif
}


template<int NFACTORS, splatt_sync_type SYNC_TYPE>
static inline void p_add_hada_lock_(
  val_t * const restrict out,
  SPLATT_SIMDFPTYPE const * const restrict a,
  SPLATT_SIMDFPTYPE const * const restrict b,
  idx_t oid)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

  if(SPLATT_SYNC_RTM == SYNC_TYPE && _XBEGIN_STARTED == _xbegin()) {
    p_add_hada_<NFACTORS>(out, a, b);
    _xend();
    return;
  }

  if(SPLATT_SYNC_CAS == SYNC_TYPE) {
    __m128d old_ov, new_ov, a_v, b_v;

    do {
      old_ov = _mm_load_pd(out);
#ifdef __AVX512F__
      a_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a[0]), 0));
      b_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(b[0]), 0));
#else
      a_v = _mm256_extractf128_pd(a[0], 0);
      b_v = _mm256_extractf128_pd(b[0], 0);
#endif
      new_ov = _mm_fmadd_pd(a_v, b_v, old_ov);
    } while (!__sync_bool_compare_and_swap((__int128 *)out, *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
    do {
      old_ov = _mm_load_pd(out + 2);
#ifdef __AVX512F__
      a_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a[0]), 1));
      b_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(b[0]), 1));
#else
      a_v = _mm256_extractf128_pd(a[0], 1);
      b_v = _mm256_extractf128_pd(b[0], 1);
#endif
      new_ov = _mm_fmadd_pd(a_v, b_v, old_ov);
    } while (!__sync_bool_compare_and_swap((__int128 *)(out + 2), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

    do {
      old_ov = _mm_load_pd(out + 4);
#ifdef __AVX512F__
      a_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a[0]), 2));
      b_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(b[0]), 2));
#else
      a_v = _mm256_extractf128_pd(a[1], 0);
      b_v = _mm256_extractf128_pd(b[1], 0);
#endif
      new_ov = _mm_fmadd_pd(a_v, b_v, old_ov);
    } while (!__sync_bool_compare_and_swap((__int128 *)(out + 4), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
    do {
      old_ov = _mm_load_pd(out + 6);
#ifdef __AVX512F__
      a_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a[0]), 3));
      b_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(b[0]), 3));
#else
      a_v = _mm256_extractf128_pd(a[1], 1);
      b_v = _mm256_extractf128_pd(b[1], 1);
#endif
      new_ov = _mm_fmadd_pd(a_v, b_v, old_ov);
    } while (!__sync_bool_compare_and_swap((__int128 *)(out + 6), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

    do {
      old_ov = _mm_load_pd(out + 8);
#ifdef __AVX512F__
      a_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a[1]), 0));
      b_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(b[1]), 0));
#else
      a_v = _mm256_extractf128_pd(a[2], 0);
      b_v = _mm256_extractf128_pd(b[2], 0);
#endif
      new_ov = _mm_fmadd_pd(a_v, b_v, old_ov);
    } while (!__sync_bool_compare_and_swap((__int128 *)(out + 8), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
    do {
      old_ov = _mm_load_pd(out + 10);
#ifdef __AVX512F__
      a_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a[1]), 1));
      b_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(b[1]), 1));
#else
      a_v = _mm256_extractf128_pd(a[2], 1);
      b_v = _mm256_extractf128_pd(b[2], 1);
#endif
      new_ov = _mm_fmadd_pd(a_v, b_v, old_ov);
    } while (!__sync_bool_compare_and_swap((__int128 *)(out + 10), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

    do {
      old_ov = _mm_load_pd(out + 12);
#ifdef __AVX512F__
      a_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a[1]), 2));
      b_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(b[1]), 2));
#else
      a_v = _mm256_extractf128_pd(a[3], 0);
      b_v = _mm256_extractf128_pd(b[3], 0);
#endif
      new_ov = _mm_fmadd_pd(a_v, b_v, old_ov);
    } while (!__sync_bool_compare_and_swap((__int128 *)(out + 12), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
    do {
      old_ov = _mm_load_pd(out + 14);
#ifdef __AVX512F__
      a_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a[1]), 3));
      b_v = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(b[1]), 3));
#else
      a_v = _mm256_extractf128_pd(a[3], 1);
      b_v = _mm256_extractf128_pd(b[3], 1);
#endif
      new_ov = _mm_fmadd_pd(a_v, b_v, old_ov);
    } while (!__sync_bool_compare_and_swap((__int128 *)(out + 14), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
    mttkrp_flops += 2*NFACTORS;
#endif
  } // SPLATT_SYNC_CAS
  else if(SPLATT_SYNC_VCAS == SYNC_TYPE) {
    __m128d old_ov, new_ov;

#ifdef __AVX512F__
    __m512d acc;
    do {
      acc = _mm512_load_pd(out);
      old_ov = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(acc), 0));

      acc = _mm512_fmadd_pd(a[0], b[0], acc);
      new_ov = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(acc), 0));
    } while (!__sync_bool_compare_and_swap((__int128 *)out, *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

    do {
      acc = _mm512_load_pd(out + 8);
      old_ov = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(acc), 0));

      acc = _mm512_fmadd_pd(a[1], b[1], acc);
      new_ov = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(acc), 0));
    } while (!__sync_bool_compare_and_swap((__int128 *)(out + 8), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#else
    __m256d acc;
    do {
      acc = _mm256_load_pd(out);
      old_ov = _mm256_extractf128_pd(acc, 0);

      acc = _mm256_fmadd_pd(a[0], b[0], acc);
      new_ov = _mm256_extractf128_pd(acc, 0);
    } while (!__sync_bool_compare_and_swap((__int128 *)out, *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

    do {
      acc = _mm256_load_pd(out + 8);
      old_ov = _mm256_extractf128_pd(acc, 0);

      acc = _mm256_fmadd_pd(a[2], b[2], acc);
      new_ov = _mm256_extractf128_pd(acc, 0);
    } while (!__sync_bool_compare_and_swap((__int128 *)(out + 8), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

    _mm256_store_pd(out + 4, _mm256_fmadd_pd(a[1], b[1], _mm256_load_pd(out + 4)));
    _mm256_store_pd(out + 12, _mm256_fmadd_pd(a[3], b[3], _mm256_load_pd(out + 12)));
#endif

#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
    mttkrp_flops += 2*NFACTORS;
#endif
  } // SPLATT_SYNC_VCAS
  else {
    splatt_set_lock<SYNC_TYPE>(oid);
    p_add_hada_<NFACTORS>(out, a, b);
    splatt_unset_lock<SYNC_TYPE>(oid);
  }

#ifdef SPLATT_COUNT_REDUCTION
#pragma omp atomic
  ++mttkrp_reduction;
#endif
}


static inline void p_add_hada_clear(
  val_t * const restrict out,
  val_t * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] += a[f] * b[f];
    a[f] = 0;
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*nfactors;
#endif
}


template<int NFACTORS>
static inline void p_add_hada_clear_(
  SPLATT_SIMDFPTYPE * const restrict out,
  SPLATT_SIMDFPTYPE * const restrict a,
  val_t const * const restrict b)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);
#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
    out[f] = _MM_FMADD(a[f], _MM_LOAD(b + f*SPLATT_VLEN), out[f]);
    a[f] = _MM_SETZERO();
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*NFACTORS;
#endif
}


template<int NFACTORS>
static inline void p_add_hada_clear_(
  val_t * const restrict out,
  SPLATT_SIMDFPTYPE * const restrict a,
  SPLATT_SIMDFPTYPE const * const restrict b)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);
#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
    _MM_STORE(out + f*SPLATT_VLEN, _MM_FMADD(a[f], b[f], _MM_LOAD(out + f*SPLATT_VLEN)));
    a[f] = _MM_SETZERO();
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*NFACTORS;
#endif
}


static inline void p_assign_hada(
  val_t * const restrict out,
  val_t const * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] = a[f] * b[f];
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += nfactors;
#endif
}


template<int NFACTORS>
static inline void p_assign_hada_(
  SPLATT_SIMDFPTYPE * const restrict out,
  SPLATT_SIMDFPTYPE const * const restrict a,
  val_t const * const restrict b)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);
#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
    out[f] = _MM_MUL(a[f], _MM_LOAD(b + f*SPLATT_VLEN));
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += NFACTORS;
#endif
}


template<splatt_sync_type SYNC_TYPE>
static inline void p_csf_process_fiber_lock(
  val_t * const leafmat,
  val_t const * const restrict accumbuf,
  idx_t const nfactors,
  idx_t const start,
  idx_t const end,
  fidx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * nfactors);
    val_t const v = vals[jj];

    if(SPLATT_SYNC_RTM == SYNC_TYPE && _XBEGIN_STARTED == _xbegin()) {
      for(idx_t f=0; f < nfactors; ++f) {
        leafrow[f] += v * accumbuf[f];
      }
      _xend();
      continue;
    }

    if(SPLATT_SYNC_CAS == SYNC_TYPE) {
      for(idx_t f=0; f < nfactors; ++f) {
        double old_leafrow, new_leafrow;
        do {
          old_leafrow = leafrow[f];
          new_leafrow = old_leafrow + v * accumbuf[f];
        } while (!__sync_bool_compare_and_swap((long long *)(leafrow + f), *((long long *)(&old_leafrow)), *((long long *)(&new_leafrow))));
      }
    }
    else {
      splatt_set_lock<SYNC_TYPE>(inds[jj]);
      for(idx_t f=0; f < nfactors; ++f) {
        leafrow[f] += v * accumbuf[f];
      }
      splatt_unset_lock<SYNC_TYPE>(inds[jj]);
    }
  }

#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*nfactors*(end - start);
#endif

#ifdef SPLATT_COUNT_REDUCTION
#pragma omp atomic
  mttkrp_reduction += end - start;
#endif
}


template<int NFACTORS, splatt_sync_type SYNC_TYPE>
static inline void p_csf_process_fiber_lock_(
  val_t * const leafmat,
  SPLATT_SIMDFPTYPE const * const restrict accumbuf,
  idx_t const start,
  idx_t const end,
  fidx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  assert(NFACTORS == 16 && sizeof(val_t) == 8);

  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * NFACTORS);
    val_t const v = vals[jj];

    if(SPLATT_SYNC_RTM == SYNC_TYPE && _XBEGIN_STARTED == _xbegin()) {
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        _MM_STORE(leafrow + f*SPLATT_VLEN, _MM_FMADD(_MM_SET1(v), accumbuf[f], _MM_LOAD(leafrow + f*SPLATT_VLEN)));
      }
      _xend();
      continue;
    }

    if(SPLATT_SYNC_CAS == SYNC_TYPE) {
      __m128d old_ov, new_ov, acc;

      do {
        old_ov = _mm_load_pd(leafrow);
#ifdef __AVX512F__
        acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumbuf[0]), 0));
#else
        acc = _mm256_extractf128_pd(accumbuf[0], 0);
#endif
        new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
      } while (!__sync_bool_compare_and_swap((__int128 *)leafrow, *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
      do {
        old_ov = _mm_load_pd(leafrow + 2);
#ifdef __AVX512F__
        acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumbuf[0]), 1));
#else
        acc = _mm256_extractf128_pd(accumbuf[0], 1);
#endif
        new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
      } while (!__sync_bool_compare_and_swap((__int128 *)(leafrow + 2), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

      do {
        old_ov = _mm_load_pd(leafrow + 4);
#ifdef __AVX512F__
        acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumbuf[0]), 2));
#else
        acc = _mm256_extractf128_pd(accumbuf[1], 0);
#endif
        new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
      } while (!__sync_bool_compare_and_swap((__int128 *)(leafrow + 4), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
      do {
        old_ov = _mm_load_pd(leafrow + 6);
#ifdef __AVX512F__
        acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumbuf[0]), 3));
#else
        acc = _mm256_extractf128_pd(accumbuf[1], 1);
#endif
        new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
      } while (!__sync_bool_compare_and_swap((__int128 *)(leafrow + 6), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

      do {
        old_ov = _mm_load_pd(leafrow + 8);
#ifdef __AVX512F__
        acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumbuf[1]), 0));
#else
        acc = _mm256_extractf128_pd(accumbuf[2], 0);
#endif
        new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
      } while (!__sync_bool_compare_and_swap((__int128 *)(leafrow + 8), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
      do {
        old_ov = _mm_load_pd(leafrow + 10);
#ifdef __AVX512F__
        acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumbuf[1]), 1));
#else
        acc = _mm256_extractf128_pd(accumbuf[2], 1);
#endif
        new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
      } while (!__sync_bool_compare_and_swap((__int128 *)(leafrow + 10), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

      do {
        old_ov = _mm_load_pd(leafrow + 12);
#ifdef __AVX512F__
        acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumbuf[1]), 2));
#else
        acc = _mm256_extractf128_pd(accumbuf[3], 0);
#endif
        new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
      } while (!__sync_bool_compare_and_swap((__int128 *)(leafrow + 12), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
      do {
        old_ov = _mm_load_pd(leafrow + 14);
#ifdef __AVX512F__
        acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumbuf[1]), 3));
#else
        acc = _mm256_extractf128_pd(accumbuf[3], 1);
#endif
        new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
      } while (!__sync_bool_compare_and_swap((__int128 *)(leafrow + 14), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
    }
    else if(SPLATT_SYNC_VCAS == SYNC_TYPE) {
      __m128d old_ov, new_ov;

#ifdef __AVX512F__
      __m512d acc;
      do {
        acc = _mm512_load_pd(leafrow);
        old_ov = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(acc), 0));

        acc = _mm512_fmadd_pd(_mm512_set1_pd(v), accumbuf[0], acc);
        new_ov = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(acc), 0));
      } while (!__sync_bool_compare_and_swap((__int128 *)leafrow, *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

      do {
        acc = _mm512_load_pd(leafrow + 8);
        old_ov = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(acc), 0));

        acc = _mm512_fmadd_pd(_mm512_set1_pd(v), accumbuf[1], acc);
        new_ov = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(acc), 0));
      } while (!__sync_bool_compare_and_swap((__int128 *)(leafrow + 8), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#else
      __m256d acc;
      do {
        acc = _mm256_load_pd(leafrow);
        old_ov = _mm256_extractf128_pd(acc, 0);

        acc = _mm256_fmadd_pd(_mm256_set1_pd(v), accumbuf[0], acc);
        new_ov = _mm256_extractf128_pd(acc, 0);
      } while (!__sync_bool_compare_and_swap((__int128 *)leafrow, *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

      do {
        acc = _mm256_load_pd(leafrow + 8);
        old_ov = _mm256_extractf128_pd(acc, 0);

        acc = _mm256_fmadd_pd(_mm256_set1_pd(v), accumbuf[2], acc);
        new_ov = _mm256_extractf128_pd(acc, 0);
      } while (!__sync_bool_compare_and_swap((__int128 *)(leafrow + 8), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

      _mm256_store_pd(leafrow + 4, _mm256_fmadd_pd(_mm256_set1_pd(v), accumbuf[1], _mm256_load_pd(leafrow + 4)));
      _mm256_store_pd(leafrow + 12, _mm256_fmadd_pd(_mm256_set1_pd(v), accumbuf[3], _mm256_load_pd(leafrow + 12)));
#endif
    } // SPLATT_SYNC_VCAS
    else {
      splatt_set_lock<SYNC_TYPE>(inds[jj]);
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        _MM_STORE(leafrow + f*SPLATT_VLEN, _MM_FMADD(_MM_SET1(v), accumbuf[f], _MM_LOAD(leafrow + f*SPLATT_VLEN)));
      }
      splatt_unset_lock<SYNC_TYPE>(inds[jj]);
    }
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*NFACTORS*(end - start);
#endif

#ifdef SPLATT_COUNT_REDUCTION
#pragma omp atomic
  mttkrp_reduction += end - start;
#endif
}


static inline void p_csf_process_fiber_nolock(
  val_t * const leafmat,
  val_t const * const restrict accumbuf,
  idx_t const nfactors,
  idx_t const start,
  idx_t const end,
  fidx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * nfactors);
    val_t const v = vals[jj];
    for(idx_t f=0; f < nfactors; ++f) {
      leafrow[f] += v * accumbuf[f];
    }
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*nfactors*(end - start);
#endif
}


template<int NFACTORS>
static inline void p_csf_process_fiber_nolock_(
  val_t * const leafmat,
  SPLATT_SIMDFPTYPE const * const restrict accumbuf,
  idx_t const start,
  idx_t const end,
  fidx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * NFACTORS);
    val_t const v = vals[jj];

#pragma unroll(NFACTORS/SPLATT_VLEN)
    for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
      _MM_STORE(leafrow + f*SPLATT_VLEN, _MM_FMADD(_MM_SET1(v), accumbuf[f], _MM_LOAD(leafrow + f*SPLATT_VLEN)));
    }
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*NFACTORS*(end - start);
#endif
}


static inline void p_csf_process_fiber(
  val_t * const restrict accumbuf,
  idx_t const nfactors,
  val_t const * const leafmat,
  idx_t const start,
  idx_t const end,
  fidx_t const * const inds,
  val_t const * const vals)
{
  /* foreach nnz in fiber */
  for(idx_t j=start; j < end; ++j) {
    val_t const v = vals[j] ;
    val_t const * const restrict row = leafmat + (nfactors * inds[j]);
    for(idx_t f=0; f < nfactors; ++f) {
      accumbuf[f] += v * row[f];
    }
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*nfactors*(end - start);
#endif
}


template<int NFACTORS>
static inline void p_csf_process_fiber_(
  SPLATT_SIMDFPTYPE * const restrict accumbuf,
  val_t const * const leafmat,
  idx_t const start,
  idx_t const end,
  fidx_t const * const inds,
  val_t const * const vals)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

  /* foreach nnz in fiber */
  for(idx_t j=start; j < end; ++j) {
    val_t const v = vals[j] ;
    val_t const * const restrict row = leafmat + (NFACTORS * inds[j]);

#pragma unroll(NFACTORS/SPLATT_VLEN)
    for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
      accumbuf[f] = _MM_FMADD(_MM_SET1(v), _MM_LOAD(row + f*SPLATT_VLEN), accumbuf[f]);
    }
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*NFACTORS*(end - start);
#endif
}


static inline void p_propagate_up(
  val_t * const out,
  val_t * const * const buf,
  idx_t * const restrict idxstack,
  idx_t const init_depth,
  idx_t const init_idx,
  idx_t const * const * const fp,
  fidx_t const * const * const fids,
  val_t const * const restrict vals,
  val_t ** mvals,
  idx_t const nmodes,
  idx_t const nfactors)
{
  /* push initial idx initialize idxstack */
  idxstack[init_depth] = init_idx;
  for(idx_t m=init_depth+1; m < nmodes; ++m) {
    idxstack[m] = fp[m-1][idxstack[m-1]];
  }

  assert(init_depth < nmodes-1);

  /* clear out accumulation buffer */
  for(idx_t f=0; f < nfactors; ++f) {
    buf[init_depth+1][f] = 0;
  }

  while(idxstack[init_depth+1] < fp[init_depth][init_idx+1]) {
    /* skip to last internal mode */
    idx_t depth = nmodes - 2;

    /* process all nonzeros [start, end) into buf[depth]*/
    idx_t const start = fp[depth][idxstack[depth]];
    idx_t const end   = fp[depth][idxstack[depth]+1];
    p_csf_process_fiber(buf[depth+1], nfactors, mvals[depth+1],
        start, end, fids[depth+1], vals);

    idxstack[depth+1] = end;

    /* exit early if there is no propagation to do... */
    if(init_depth == nmodes-2) {
      for(idx_t f=0; f < nfactors; ++f) {
        out[f] = buf[depth+1][f];
      }
      return;
    }

    /* Propagate up until we reach a node with more children to process */
    do {
      /* propagate result up and clear buffer for next sibling */
      val_t const * const restrict fibrow
          = mvals[depth] + (fids[depth][idxstack[depth]] * nfactors);
      p_add_hada_clear(buf[depth], buf[depth+1], fibrow, nfactors);

      ++idxstack[depth];
      --depth;
    } while(depth > init_depth &&
        idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
  } /* end DFS */

  /* copy to out */
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] = buf[init_depth+1][f];
  }
}


template<int NFACTORS>
static inline void p_propagate_up_(
  SPLATT_SIMDFPTYPE * const out,
  SPLATT_SIMDFPTYPE * const * const buf,
  idx_t * const restrict idxstack,
  idx_t const init_depth,
  idx_t const init_idx,
  idx_t const * const * const fp,
  fidx_t const * const * const fids,
  val_t const * const restrict vals,
  val_t ** mvals,
  idx_t const nmodes)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

  /* push initial idx initialize idxstack */
  idxstack[init_depth] = init_idx;
  for(idx_t m=init_depth+1; m < nmodes; ++m) {
    idxstack[m] = fp[m-1][idxstack[m-1]];
  }

  assert(init_depth < nmodes-1);

  /* clear out accumulation buffer */
#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
    buf[init_depth+1][f] = _MM_SETZERO();
  }

  while(idxstack[init_depth+1] < fp[init_depth][init_idx+1]) {
    /* skip to last internal mode */
    idx_t depth = nmodes - 2;

    /* process all nonzeros [start, end) into buf[depth]*/
    idx_t const start = fp[depth][idxstack[depth]];
    idx_t const end   = fp[depth][idxstack[depth]+1];
    p_csf_process_fiber_<NFACTORS>(buf[depth+1], mvals[depth+1],
        start, end, fids[depth+1], vals);

    idxstack[depth+1] = end;

    /* exit early if there is no propagation to do... */
    if(init_depth == nmodes-2) {
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        out[f] = buf[depth+1][f];
      }
      return;
    }

    /* Propagate up until we reach a node with more children to process */
    do {
      /* propagate result up and clear buffer for next sibling */
      val_t const * const restrict fibrow
          = mvals[depth] + (fids[depth][idxstack[depth]] * NFACTORS);
      p_add_hada_clear_<NFACTORS>(buf[depth], buf[depth+1], fibrow);

      ++idxstack[depth];
      --depth;
    } while(depth > init_depth &&
        idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
  } /* end DFS */

  /* copy to out */
#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
    out[f] = buf[init_depth+1][f];
  }
}


template<int NFACTORS, bool TILED = false>
static void p_csf_mttkrp_root3_kernel_(
  const val_t *vals,
  const idx_t *sptr, const idx_t *fptr,
  const fidx_t *fids, const fidx_t *inds,
  const val_t *avals, const val_t *bvals, val_t *mv,
  idx_t s)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

  __declspec(aligned(64)) SPLATT_SIMDFPTYPE accumF[NFACTORS/SPLATT_VLEN], accumO[NFACTORS/SPLATT_VLEN];

#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(int i=0; i < NFACTORS/SPLATT_VLEN; ++i) {
    accumO[i] = TILED ? _MM_LOAD(mv + i*SPLATT_VLEN) : _MM_SETZERO();
  }

  /* foreach fiber in slice */
  for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
    /* first entry of the fiber is used to initialize accumF */
    idx_t const jjfirst  = fptr[f];
    val_t const vfirst   = vals[jjfirst];
    val_t const * const restrict bv = bvals + (inds[jjfirst] * NFACTORS);

#pragma unroll(NFACTORS/SPLATT_VLEN)
    for(int i=0; i < NFACTORS/SPLATT_VLEN; ++i) {
      accumF[i] = _MM_MUL(_MM_SET1(vfirst), _MM_LOAD(bv + i*SPLATT_VLEN));
    }

    p_csf_process_fiber_<NFACTORS>(accumF, bvals, fptr[f]+1, fptr[f+1], inds, vals);

    /* scale inner products by row of A and update to M */
    val_t const * const restrict av = avals  + (fids[f] * NFACTORS);
    p_add_hada_<NFACTORS>(accumO, accumF, av);
  }

#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(int i=0; i < NFACTORS/SPLATT_VLEN; ++i) {
    _MM_STREAM(mv + i*SPLATT_VLEN, accumO[i]);
  }
}

static void p_csf_mttkrp_root_tiled3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  fidx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  fidx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  fidx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[1]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  int tid = omp_get_thread_num();

  val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

  if(16 == nfactors) {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      val_t * const restrict mv = ovals + (fid * nfactors);

      p_csf_mttkrp_root3_kernel_<16, true>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);
    }
  }
  else if(32 == nfactors) {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      val_t * const restrict mv = ovals + (fid * nfactors);

      p_csf_mttkrp_root3_kernel_<32, true>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);
    }
  }
  else if(64 == nfactors) {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      val_t * const restrict mv = ovals + (fid * nfactors);

      p_csf_mttkrp_root3_kernel_<64, true>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);
    }
  }
  else if(128 == nfactors) {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      val_t * const restrict mv = ovals + (fid * nfactors);

      p_csf_mttkrp_root3_kernel_<128, true>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);
    }
  }
  else if(256 == nfactors) {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      val_t * const restrict mv = ovals + (fid * nfactors);

      p_csf_mttkrp_root3_kernel_<256, true>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);
    }
  }
  else
  {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      val_t * const restrict mv = ovals + (fid * nfactors);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t const * const restrict av = avals  + (fids[f] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }
    }
  }
}

//#define SPLATT_MEASURE_LOAD_BALANCE
#ifdef SPLATT_MEASURE_LOAD_BALANCE
#include "SpMP/Utils.hpp"
double per_thread_times[1024];
#endif

//static bool printed = false;

template<int NFACTORS, splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_root3_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  fidx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  fidx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  fidx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[1]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;

  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

  if (sids == NULL) {
    // When we're working on all slices, use static schedule
#define SPLATT_MTTKRP_USE_STATIC_SCHEDULE
#ifdef SPLATT_MTTKRP_USE_STATIC_SCHEDULE
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    idx_t nnz_per_thread = (ct->nnz + nthreads - 1)/nthreads;
    idx_t count = nslices;
    idx_t first = 0;
    idx_t val = nnz_per_thread*tid;
    while (count > 0) {
      idx_t it = first; idx_t step = count/2; it += step;
      if (fptr[sptr[it]] < val) {
        first = it + 1;
        count -= step + 1;
      }
      else count = step;
    }
    idx_t s_begin = tid == 0 ? 0 : first;

    count = nslices - first;
    val += nnz_per_thread;
    while (count > 0) {
      idx_t it = first; idx_t step = count/2; it += step;
      if (fptr[sptr[it]] < val) {
        first = it + 1;
        count -= step + 1;
      }
      else count = step;
    }
    idx_t s_end = tid == nthreads - 1 ? nslices : first;
    //for (int i = 0; i < omp_get_num_threads(); ++i) {
//#pragma omp barrier
      //if (i == omp_get_thread_num()) printf("[%d] %ld-%ld %ld\n", tid, s_begin, s_end, fptr[sptr[s_end]] - fptr[sptr[s_begin]]);
//#pragma omp barrier
    //}
    /*if(8 == tid && s_begin == 841960) {
      for (int s=s_begin; s < s_end; ++s) {
        idx_t cnt = fptr[sptr[s + 1]] - fptr[sptr[s]];
        if(cnt > 100) {
          printf("%d %ld\n", s, cnt);
        }
      }
    }*/

    for(idx_t s=s_begin; s < s_end; ++s) {
      idx_t fid = s;
      val_t *mv = ovals + (fid * NFACTORS);
      p_csf_mttkrp_root3_kernel_<NFACTORS>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);
    }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
#else
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif

      idx_t fid = s;
      val_t *mv = ovals + (fid * NFACTORS);
      p_csf_mttkrp_root3_kernel_<NFACTORS>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    }
#endif
  }
  else if(ct->nslice_hubs > 0) {
    idx_t hub_nz_begin = fptr[sptr[nslices - ct->nslice_hubs]];
    idx_t nnz_hubs = ct->nnz - hub_nz_begin;

    // process non-hub slices as usual
#ifdef SPLATT_MTTKRP_USE_STATIC_SCHEDULE
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    idx_t nnz_per_thread = (hub_nz_begin + nthreads - 1)/nthreads;
    idx_t count = nslices - ct->nslice_hubs;
    idx_t first = 0;
    idx_t val = nnz_per_thread*tid;
    while (count > 0) {
      idx_t it = first; idx_t step = count/2; it += step;
      if (fptr[sptr[it]] < val) {
        first = it + 1;
        count -= step + 1;
      }
      else count = step;
    }
    idx_t s_begin = tid == 0 ? 0 : first;

    count = nslices - ct->nslice_hubs - first;
    val += nnz_per_thread;
    while (count > 0) {
      idx_t it = first; idx_t step = count/2; it += step;
      if (fptr[sptr[it]] < val) {
        first = it + 1;
        count -= step + 1;
      }
      else count = step;
    }
    idx_t s_end = tid == nthreads - 1 ? nslices - ct->nslice_hubs : first;

    for(idx_t s=s_begin; s < s_end; ++s) {
      idx_t fid = sids[s];
      val_t *mv = ovals + (fid * NFACTORS);
      p_csf_mttkrp_root3_kernel_<NFACTORS>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);
    }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
#else /* !SPLATT_MTTKRP_USE_STATIC_SCHEDULE */
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices - ct->nslice_hubs; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif

      idx_t fid = sids[s];
      val_t *mv = ovals + (fid * NFACTORS);
      p_csf_mttkrp_root3_kernel_<NFACTORS>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    }
#endif /* !SPLATT_MTTKRP_USE_STATIC_SCHEDULE */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    idx_t hub_nnz_per_thread = (nnz_hubs + nthreads - 1)/nthreads;
    idx_t nz_begin = SS_MIN(hub_nz_begin + hub_nnz_per_thread*tid, ct->nnz);
    idx_t nz_end = SS_MIN(nz_begin + hub_nnz_per_thread, ct->nnz);

    int fbegin =
      std::lower_bound(fptr + sptr[nslices - ct->nslice_hubs], fptr + sptr[nslices], hub_nz_begin + hub_nnz_per_thread*tid) -
      fptr;
    int fend =
      std::lower_bound(fptr + sptr[nslices - ct->nslice_hubs], fptr + sptr[nslices], hub_nz_begin + hub_nnz_per_thread*(tid + 1)) -
      fptr;

    int sbegin =
      std::upper_bound(sptr + nslices - ct->nslice_hubs, sptr + nslices, fbegin) - sptr - 1;
    int send =
      std::lower_bound(sptr + nslices - ct->nslice_hubs, sptr + nslices, fend) - sptr;

    /* debug partitioning of hub slices */
    /*if(!printed) {
      if(0 == tid) {
        printf("nnz_hubs %ld sync_type %d\n", nnz_hubs, SYNC_TYPE);
      }
      for(int i=0; i < nthreads; ++i) {
#pragma omp barrier
        if(i == tid) printf("[%d] %d:%d-%d:%d %ld\n", tid, sbegin, fbegin, send, fend, fptr[fend] - fptr[fbegin]);
#pragma omp barrier
      }
      if(0 == tid) {
        printed = true;
      }
    }*/

    __declspec(aligned(64)) SPLATT_SIMDFPTYPE
      accumF[NFACTORS/SPLATT_VLEN], accumO[NFACTORS/SPLATT_VLEN];

    for(idx_t s=sbegin; s < send; ++s) {
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(int i = 0; i < NFACTORS/SPLATT_VLEN; ++i) {
        accumO[i] = _MM_SETZERO();
      }

      idx_t begin = s == sbegin ? fbegin : sptr[s];
      idx_t end = s == send - 1 ? fend : sptr[s + 1];

      for(idx_t f = begin; f < end; ++f) {
        idx_t const jjfirst = fptr[f];
        val_t const vfirst = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * NFACTORS);

#pragma unroll(NFACTORS/SPLATT_VLEN)
        for(int i=0; i < NFACTORS/SPLATT_VLEN; ++i) {
          accumF[i] = _MM_MUL(_MM_SET1(vfirst), _MM_LOAD(bv + i*SPLATT_VLEN));
        }

        p_csf_process_fiber_<NFACTORS>(accumF, bvals, fptr[f]+1, fptr[f+1], inds, vals);

        /* scale inner products by row of A and update to M */
        val_t const * const restrict av = avals  + (fids[f] * NFACTORS);
        p_add_hada_<NFACTORS>(accumO, accumF, av);
      }

      idx_t fid = sids[s];
      val_t *mv = ovals + (fid * NFACTORS);

      splatt_set_lock<SYNC_TYPE>(fid);
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(int i=0; i < NFACTORS/SPLATT_VLEN; ++i) {
        _MM_STORE(mv + i*SPLATT_VLEN, _MM_ADD(_MM_LOAD(mv + i*SPLATT_VLEN), accumO[i]));
      }
      splatt_unset_lock<SYNC_TYPE>(fid);
    }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  }
  else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    idx_t const nrows = mats[MAX_NMODES]->I;
    idx_t rows_per_thread = (nrows + nthreads - 1)/nthreads;
    idx_t row_begin = SS_MIN(rows_per_thread*tid, nrows);
    idx_t row_end = SS_MIN(row_begin + rows_per_thread, nrows);

    idx_t sbegin = std::lower_bound(sids, sids + nslices, row_begin) - sids;

    for(idx_t s=sbegin; s < nslices && sids[s] < row_end; ++s) {
      idx_t fid = sids[s];
      val_t *mv = ovals + (fid * NFACTORS);
      p_csf_mttkrp_root3_kernel_<NFACTORS, true>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, s);
    }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  }
}


template<splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_root3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  idx_t const nfactors = mats[MAX_NMODES]->J;

  if(16 == nfactors) {
    p_csf_mttkrp_root3_<16, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(32 == nfactors) {
    p_csf_mttkrp_root3_<32, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(64 == nfactors) {
    p_csf_mttkrp_root3_<64, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(128 == nfactors) {
    p_csf_mttkrp_root3_<128, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(256 == nfactors) {
    p_csf_mttkrp_root3_<256, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else
  {
    val_t const * const vals = ct->pt[tile_id].vals;

    idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
    idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

    fidx_t const * const restrict sids = ct->pt[tile_id].fids[0];
    fidx_t const * const restrict fids = ct->pt[tile_id].fids[1];
    fidx_t const * const restrict inds = ct->pt[tile_id].fids[2];

    val_t const * const avals = mats[ct->dim_perm[1]]->vals;
    val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
    val_t * const ovals = mats[MAX_NMODES]->vals;

    val_t * const restrict accumF
        = (val_t *) thds[omp_get_thread_num()].scratch[0];

    idx_t const nslices = ct->pt[tile_id].nfibs[0];
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      idx_t const fid = (sids == NULL) ? s : sids[s];

      val_t * const restrict mv = ovals + (fid * nfactors);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t const * const restrict av = avals  + (fids[f] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    }
  }
}


template<int NFACTORS, splatt_sync_type SYNC_TYPE, bool TILED = false>
static void p_csf_mttkrp_internal3_kernel_(
  const val_t *vals,
  const idx_t *sptr, const idx_t *fptr,
  const fidx_t *fids, const fidx_t *inds,
  const val_t *rv, const val_t *bvals, val_t *ovals,
  idx_t s)
{
  assert(NFACTORS == 16 && sizeof(val_t) == 8);

  __declspec(aligned(64)) SPLATT_SIMDFPTYPE accumF[NFACTORS/SPLATT_VLEN], rv_v[NFACTORS/SPLATT_VLEN];

#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(int i=0; i < NFACTORS/SPLATT_VLEN; ++i) {
    rv_v[i] = _MM_LOAD(rv + i*SPLATT_VLEN);
  }

  /* foreach fiber in slice */
  for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
    /* first entry of the fiber is used to initialize accumF */
    idx_t const jjfirst  = fptr[f];
    val_t const vfirst   = vals[jjfirst];
    val_t const * const restrict bv = bvals + (inds[jjfirst] * NFACTORS);

#pragma unroll(NFACTORS/SPLATT_VLEN)
    for(int i=0; i < NFACTORS/SPLATT_VLEN; ++i) {
      accumF[i] = _MM_MUL(_MM_SET1(vfirst), _MM_LOAD(bv + i*SPLATT_VLEN));
    }

    p_csf_process_fiber_<NFACTORS>(accumF, bvals, fptr[f]+1, fptr[f+1], inds, vals);

    /* scale inner products by row of A and update to M */
    fidx_t oid = fids[f];
    val_t * const restrict ov = ovals  + (oid * NFACTORS);
    if (TILED) {
      p_add_hada_<NFACTORS>(ov, accumF, rv_v);
    }
    else {
      p_add_hada_lock_<NFACTORS, SYNC_TYPE>(ov, accumF, rv_v, oid);
    }
  } /* for each slice */
}

template<int NFACTORS, splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_internal3_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
#ifdef SPLATT_MEASURE_LOAD_BALANCE
  per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

  assert(NFACTORS*sizeof(val_t)%64 == 0);

  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  fidx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  fidx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  fidx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;

  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();
  
  idx_t const nslices = ct->pt[tile_id].nfibs[0];

  if (sids == NULL) {
    // When we're working on all slices, use static schedule
    int nnz_per_thread = (ct->nnz + nthreads - 1)/nthreads;
    int count = nslices;
    int first = 0;
    int val = nnz_per_thread*tid;
    while (count > 0) {
      int it = first; int step = count/2; it += step;
      if (fptr[sptr[it]] < val) {
        first = it + 1;
        count -= step + 1;
      }
      else count = step;
    }
    int s_begin = tid == 0 ? 0 : first;

    count = nslices - first;
    val += nnz_per_thread;
    while (count > 0) {
      int it = first; int step = count/2; it += step;
      if (fptr[sptr[it]] < val) {
        first = it + 1;
        count -= step + 1;
      }
      else count = step;
    }
    int s_end = tid == nthreads - 1 ? nslices : first;

    for(idx_t s=s_begin; s < s_end; ++s) {
      idx_t fid = s;

      /* root row */
      val_t const * const restrict rv = avals + (fid * NFACTORS);

      p_csf_mttkrp_internal3_kernel_<NFACTORS, SYNC_TYPE>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
    }
  }
  else {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t fid = sids[s];

      /* root row */
      val_t const * const restrict rv = avals + (fid * NFACTORS);

      p_csf_mttkrp_internal3_kernel_<NFACTORS, SYNC_TYPE>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
    }
  }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
  per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
}


template<splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_internal3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  idx_t const nfactors = mats[MAX_NMODES]->J;

  if(16 == nfactors) {
    p_csf_mttkrp_internal3_<16, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(32 == nfactors) {
    p_csf_mttkrp_internal3_<32, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(64 == nfactors) {
    p_csf_mttkrp_internal3_<64, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(128 == nfactors) {
    p_csf_mttkrp_internal3_<128, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(256 == nfactors) {
    p_csf_mttkrp_internal3_<256, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else
  {
    val_t const * const vals = ct->pt[tile_id].vals;

    idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
    idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

    fidx_t const * const restrict sids = ct->pt[tile_id].fids[0];
    fidx_t const * const restrict fids = ct->pt[tile_id].fids[1];
    fidx_t const * const restrict inds = ct->pt[tile_id].fids[2];

    val_t const * const avals = mats[ct->dim_perm[0]]->vals;
    val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
    val_t * const ovals = mats[MAX_NMODES]->vals;

    val_t * const restrict accumF
        = (val_t *) thds[omp_get_thread_num()].scratch[0];

    idx_t const nslices = ct->pt[tile_id].nfibs[0];
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      idx_t const fid = (sids == NULL) ? s : sids[s];

      /* root row */
      val_t const * const restrict rv = avals + (fid * nfactors);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* write to fiber row */
        val_t * const restrict ov = ovals  + (fids[f] * nfactors);
        splatt_set_lock<SYNC_TYPE>(fids[f]);
        for(idx_t r=0; r < nfactors; ++r) {
          ov[r] += rv[r] * accumF[r];
        }
        splatt_unset_lock<SYNC_TYPE>(fids[f]);
      }

#ifdef SPLATT_COUNT_REDUCTION
#pragma omp atomic
      mttkrp_reduction += sptr[s+1] - sptr[s];
#endif

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    }
  }
}

template<int NFACTORS, splatt_sync_type SYNC_TYPE, bool TILED = false>
static void p_csf_mttkrp_leaf3_kernel_(
  const val_t *vals,
  const idx_t *sptr, const idx_t *fptr,
  const fidx_t *fids, const fidx_t *inds,
  const val_t *rv, const val_t *bvals, val_t *ovals,
  idx_t s)
{
  assert(NFACTORS == 16 && sizeof(val_t) == 8);

  __declspec(aligned(64)) SPLATT_SIMDFPTYPE accumF[NFACTORS/SPLATT_VLEN], rv_v[NFACTORS/SPLATT_VLEN];

#pragma unroll(NFACTORS/SPLATT_VLEN)
  for(int i=0; i < NFACTORS/SPLATT_VLEN; ++i) {
    rv_v[i] = _MM_LOAD(rv + i*SPLATT_VLEN);
  }

  /* foreach fiber in slice */
  for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
    /* fill fiber with hada */
    val_t const * const restrict av = bvals  + (fids[f] * NFACTORS);

#pragma unroll(NFACTORS/SPLATT_VLEN)
    for(int i=0; i < NFACTORS/SPLATT_VLEN; ++i) {
      accumF[i] = _MM_MUL(rv_v[i], _MM_LOAD(av + i*SPLATT_VLEN));
    }

    /* foreach nnz in fiber, scale with hada and write to ovals */
    p_csf_process_fiber_lock_<NFACTORS, SYNC_TYPE>(ovals, accumF, fptr[f], fptr[f+1], inds, vals);
  }
}

template<int NFACTORS, splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_leaf3_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  fidx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  fidx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  fidx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[1]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

  if (sids == NULL) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    // When we're working on all slices, use static schedule
    int nnz_per_thread = (ct->nnz + nthreads - 1)/nthreads;
    int count = nslices;
    int first = 0;
    int val = nnz_per_thread*tid;
    while (count > 0) {
      int it = first; int step = count/2; it += step;
      if (fptr[sptr[it]] < val) {
        first = it + 1;
        count -= step + 1;
      }
      else count = step;
    }
    int s_begin = tid == 0 ? 0 : first;

    count = nslices - first;
    val += nnz_per_thread;
    while (count > 0) {
      int it = first; int step = count/2; it += step;
      if (fptr[sptr[it]] < val) {
        first = it + 1;
        count -= step + 1;
      }
      else count = step;
    }
    int s_end = tid == nthreads - 1 ? nslices : first;

    for(idx_t s=s_begin; s < s_end; ++s) {
      idx_t const fid = s;

      /* root row */
      val_t const * const restrict rv = avals + (fid * NFACTORS);

      p_csf_mttkrp_leaf3_kernel_<NFACTORS, SYNC_TYPE>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
    }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  }
  else {
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif

      idx_t const fid = sids[s];

      /* root row */
      val_t const * const restrict rv = avals + (fid * NFACTORS);

      p_csf_mttkrp_leaf3_kernel_<NFACTORS, SYNC_TYPE>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    }
  }
}


template<splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_leaf3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  fidx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  fidx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  fidx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[1]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  if(16 == nfactors) {
    p_csf_mttkrp_leaf3_<16, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(32 == nfactors) {
    p_csf_mttkrp_leaf3_<32, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(64 == nfactors) {
    p_csf_mttkrp_leaf3_<64, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(128 == nfactors) {
    p_csf_mttkrp_leaf3_<128, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else if(256 == nfactors) {
    p_csf_mttkrp_leaf3_<256, SYNC_TYPE>(ct, tile_id, mats, thds);
  }
  else
  {
    val_t * const restrict accumF
        = (val_t *) thds[omp_get_thread_num()].scratch[0];

    idx_t const nslices = ct->pt[tile_id].nfibs[0];
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      idx_t const fid = (sids == NULL) ? s : sids[s];

      /* root row */
      val_t const * const restrict rv = avals + (fid * nfactors);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        /* fill fiber with hada */
        val_t const * const restrict av = bvals  + (fids[f] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] = rv[r] * av[r];
        }

        /* foreach nnz in fiber, scale with hada and write to ovals */
        for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t * const restrict ov = ovals + (inds[jj] * nfactors);
          splatt_set_lock<SYNC_TYPE>(inds[jj]);
          for(idx_t r=0; r < nfactors; ++r) {
            ov[r] += v * accumF[r];
          }
          splatt_unset_lock<SYNC_TYPE>(inds[jj]);
        }
      }

#ifdef SPLATT_COUNT_REDUCTION
#pragma omp atomic
      mttkrp_reduction += sptr[s+1] - sptr[s];
#endif

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    }
  }
}


static void p_csf_mttkrp_root_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
#ifdef SPLATT_MEASURE_LOAD_BALANCE
  per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }

  if(nmodes == 3) {
    p_csf_mttkrp_root_tiled3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  __declspec(aligned(64)) val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }

  val_t * const ovals = mats[MAX_NMODES]->vals;

  idx_t const nfibs = ct->pt[tile_id].nfibs[0];
  assert(nfibs <= mats[MAX_NMODES]->I);

  for(idx_t s=0; s < nfibs; ++s) {
    fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    assert(fid < mats[MAX_NMODES]->I);

    p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids,
        vals, mvals, nmodes, nfactors);

    val_t * const restrict orow = ovals + (fid * nfactors);
    val_t const * const restrict obuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      orow[f] += obuf[f];
    }
  } /* end foreach outer slice */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
  per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
}


template<splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_root(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }

  if(nmodes == 3) {
    p_csf_mttkrp_root3<SYNC_TYPE>(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }

  val_t * const ovals = mats[MAX_NMODES]->vals;

  idx_t const nfibs = ct->pt[tile_id].nfibs[0];
  assert(nfibs <= mats[MAX_NMODES]->I);

  if(NULL == fids[0]) {
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nfibs; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      fidx_t const fid = s;

      assert(fid < mats[MAX_NMODES]->I);

      p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids,
          vals, mvals, nmodes, nfactors);

      val_t * const restrict orow = ovals + (fid * nfactors);
      val_t const * const restrict obuf = buf[0];
      for(idx_t f=0; f < nfactors; ++f) {
        orow[f] += obuf[f];
      }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
      mttkrp_flops += nfactors;
#endif

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    } /* end foreach outer slice */
  }
  else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    idx_t const nrows = mats[MAX_NMODES]->I;
    idx_t rows_per_thread = (nrows + nthreads - 1)/nthreads;
    idx_t row_begin = SS_MIN(rows_per_thread*tid, nrows);
    idx_t row_end = SS_MIN(row_begin + rows_per_thread, nrows);

    idx_t sbegin = std::lower_bound(fids[0], fids[0] + nfibs, row_begin) - fids[0];

    for(idx_t s=sbegin; s < nfibs && fids[0][s] < row_end; ++s) {
      fidx_t const fid = fids[0][s];

      assert(fid < mats[MAX_NMODES]->I);

      p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids,
          vals, mvals, nmodes, nfactors);

      val_t * const restrict orow = ovals + (fid * nfactors);
      val_t const * const restrict obuf = buf[0];
      for(idx_t f=0; f < nfactors; ++f) {
        orow[f] += obuf[f];
      }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
      mttkrp_flops += nfactors;
#endif
    } /* end foreach outer slice */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
  }
}


template<int NFACTORS, splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_root_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }

  if(nmodes == 3) {
    p_csf_mttkrp_root3_<NFACTORS, SYNC_TYPE>(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  val_t * mvals[MAX_NMODES];
  SPLATT_SIMDFPTYPE * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = (SPLATT_SIMDFPTYPE *)(((val_t *) thds[tid].scratch[2]) + (NFACTORS * m));
    memset(buf[m], 0, NFACTORS * sizeof(val_t));
  }

  val_t * const ovals = mats[MAX_NMODES]->vals;

  idx_t const nfibs = ct->pt[tile_id].nfibs[0];
  assert(nfibs <= mats[MAX_NMODES]->I);

  if(NULL == fids[0]) {
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nfibs; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      fidx_t const fid = s;

      assert(fid < mats[MAX_NMODES]->I);

      p_propagate_up_<NFACTORS>(buf[0], buf, idxstack, 0, s, fp, fids,
          vals, mvals, nmodes);

      val_t * const restrict orow = ovals + (fid * NFACTORS);
      SPLATT_SIMDFPTYPE const * const restrict obuf = buf[0];
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        _MM_STORE(orow + f*SPLATT_VLEN, _MM_ADD(_MM_LOAD(orow + f*SPLATT_VLEN), obuf[f]));
      }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
      mttkrp_flops += NFACTORS;
#endif

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    } /* end foreach outer slice */
  }
  else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    idx_t const nrows = mats[MAX_NMODES]->I;
    idx_t rows_per_thread = (nrows + nthreads - 1)/nthreads;
    idx_t row_begin = SS_MIN(rows_per_thread*tid, nrows);
    idx_t row_end = SS_MIN(row_begin + rows_per_thread, nrows);

    idx_t sbegin = std::lower_bound(fids[0], fids[0] + nfibs, row_begin) - fids[0];

    for(idx_t s=sbegin; s < nfibs && fids[0][s] < row_end; ++s) {
      fidx_t const fid = fids[0][s];

      assert(fid < mats[MAX_NMODES]->I);

      p_propagate_up_<NFACTORS>(buf[0], buf, idxstack, 0, s, fp, fids,
          vals, mvals, nmodes);

      val_t * const restrict orow = ovals + (fid * NFACTORS);
      SPLATT_SIMDFPTYPE const * const restrict obuf = buf[0];
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        _MM_STORE(orow + f*SPLATT_VLEN, _MM_ADD(_MM_LOAD(orow + f*SPLATT_VLEN), obuf[f]));
      }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
      mttkrp_flops += NFACTORS;
#endif
    } /* end foreach outer slice */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
  }
}


template<bool PARALLELIZE_EACH_TILE=false>
static void p_csf_mttkrp_leaf_tiled3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds,
  double const * const opts)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  fidx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  fidx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  fidx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[1]]->vals;

  int const tid = omp_get_thread_num();
  bool use_privatization = PARALLELIZE_EACH_TILE && mttkrp_use_privatization(ct->nnz, mats[MAX_NMODES]->I, opts);
  val_t * const ovals = use_privatization && tid > 0 ? (val_t *)thds[tid].scratch[1] : mats[MAX_NMODES]->vals;

  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

  if(16 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<16, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<16, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    }
  }
  else if(32 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<32, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<32, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    }
  }
  else if(64 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<64, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<64, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    }
  }
  else if(128 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<128, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<128, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    }
  }
  else if(256 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<256, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_leaf3_kernel_<256, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    }
  }
  else
  {
    if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        /* foreach fiber in slice */
        for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
          /* fill fiber with hada */
          val_t const * const restrict av = bvals  + (fids[f] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            accumF[r] = rv[r] * av[r];
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += nfactors;
#endif

          /* foreach nnz in fiber, scale with hada and write to ovals */
          for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
            val_t const v = vals[jj];
            val_t * const restrict ov = ovals + (inds[jj] * nfactors);
            for(idx_t r=0; r < nfactors; ++r) {
              ov[r] += v * accumF[r];
            }
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += 2*nfactors*(fptr[f+1] - fptr[f]);
#endif
        }
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] -= omp_get_wtime();
#endif

      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        /* foreach fiber in slice */
        for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
          /* fill fiber with hada */
          val_t const * const restrict av = bvals  + (fids[f] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            accumF[r] = rv[r] * av[r];
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += nfactors;
#endif

          /* foreach nnz in fiber, scale with hada and write to ovals */
          for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
            val_t const v = vals[jj];
            val_t * const restrict ov = ovals + (inds[jj] * nfactors);
            for(idx_t r=0; r < nfactors; ++r) {
              ov[r] += v * accumF[r];
            }
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += 2*nfactors*(fptr[f+1] - fptr[f]);
#endif
        }
      }
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[omp_get_thread_num()*8] += omp_get_wtime();
#endif
    }
  } // nfactors != 16
}


template<bool PARALLELIZE_EACH_TILE=false>
static void p_csf_mttkrp_leaf_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds,
  double const * const opts)
{
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf_tiled3<PARALLELIZE_EACH_TILE>(ct, tile_id, mats, thds, opts);
    return;
  }

  /* extract tensor structures */
  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
  }
  bool use_privatization = PARALLELIZE_EACH_TILE && mttkrp_use_privatization(ct->nnz, mats[MAX_NMODES]->I, opts);
  val_t * const ovals = use_privatization && tid > 0 ? (val_t *)thds[tid].scratch[1] : mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nouter = ct->pt[tile_id].nfibs[0];
  if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nouter; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif

      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
      idxstack[0] = s;

      /* clear out stale data */
      for(idx_t m=1; m < nmodes-1; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* first buf will always just be a matrix row */
      val_t const * const rootrow = mvals[0] + (fid*nfactors);
      val_t * const rootbuf = buf[0];
      for(idx_t f=0; f < nfactors; ++f) {
        rootbuf[f] = rootrow[f];
      }

      idx_t depth = 0;

      idx_t const outer_end = fp[0][s+1];
      while(idxstack[1] < outer_end) {
        /* move down to an nnz node */
        for(; depth < nmodes-2; ++depth) {
          /* propogate buf down */
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
          p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
        }

        /* process all nonzeros [start, end) */
        idx_t const start = fp[depth][idxstack[depth]];
        idx_t const end   = fp[depth][idxstack[depth]+1];
        p_csf_process_fiber_nolock(ovals, buf[depth],
            nfactors, start, end, fids[depth+1], vals);

        /* now move back up to the next unprocessed child */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    } /* end outer slice loop */
  }
  else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    for(idx_t s=0; s < nouter; ++s) {
      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
      idxstack[0] = s;

      /* clear out stale data */
      for(idx_t m=1; m < nmodes-1; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* first buf will always just be a matrix row */
      val_t const * const rootrow = mvals[0] + (fid*nfactors);
      val_t * const rootbuf = buf[0];
      for(idx_t f=0; f < nfactors; ++f) {
        rootbuf[f] = rootrow[f];
      }

      idx_t depth = 0;

      idx_t const outer_end = fp[0][s+1];
      while(idxstack[1] < outer_end) {
        /* move down to an nnz node */
        for(; depth < nmodes-2; ++depth) {
          /* propogate buf down */
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
          p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
        }

        /* process all nonzeros [start, end) */
        idx_t const start = fp[depth][idxstack[depth]];
        idx_t const end   = fp[depth][idxstack[depth]+1];
        p_csf_process_fiber_nolock(ovals, buf[depth],
            nfactors, start, end, fids[depth+1], vals);

        /* now move back up to the next unprocessed child */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
    } /* end outer slice loop */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  }
}


int mttkrp_use_privatization(
  idx_t nnz, idx_t dim, double const * const opts)
{
  return dim*omp_get_max_threads() < nnz*opts[SPLATT_OPTION_PRIVATIZATION_THREASHOLD];
}


template<int NFACTORS, bool PARALLELIZE_EACH_TILE=false>
static void p_csf_mttkrp_leaf_tiled_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds,
  double const * const opts)
{
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf_tiled3<PARALLELIZE_EACH_TILE>(ct, tile_id, mats, thds, opts);
    return;
  }

  /* extract tensor structures */
  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  val_t * mvals[MAX_NMODES];
  SPLATT_SIMDFPTYPE * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = (SPLATT_SIMDFPTYPE *)(((val_t *) thds[tid].scratch[2]) + (NFACTORS * m));
  }
  bool use_privatization = PARALLELIZE_EACH_TILE && mttkrp_use_privatization(ct->nnz, mats[MAX_NMODES]->I, opts);
  val_t * const ovals = use_privatization && tid > 0 ? (val_t *)thds[tid].scratch[1] : mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nouter = ct->pt[tile_id].nfibs[0];
  if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nouter; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif

      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
      idxstack[0] = s;

      /* clear out stale data */
      for(idx_t m=1; m < nmodes-1; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* first buf will always just be a matrix row */
      val_t const * const rootrow = mvals[0] + (fid*NFACTORS);
      SPLATT_SIMDFPTYPE * const rootbuf = buf[0];
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        rootbuf[f] = _MM_LOAD(rootrow + f*SPLATT_VLEN);
      }

      idx_t depth = 0;

      idx_t const outer_end = fp[0][s+1];
      while(idxstack[1] < outer_end) {
        /* move down to an nnz node */
        for(; depth < nmodes-2; ++depth) {
          /* propogate buf down */
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * NFACTORS);
          p_assign_hada_<NFACTORS>(buf[depth+1], buf[depth], drow);
        }

        /* process all nonzeros [start, end) */
        idx_t const start = fp[depth][idxstack[depth]];
        idx_t const end   = fp[depth][idxstack[depth]+1];
        p_csf_process_fiber_nolock_<NFACTORS>(ovals, buf[depth],
            start, end, fids[depth+1], vals);

        /* now move back up to the next unprocessed child */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    } /* end outer slice loop */
  }
  else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    for(idx_t s=0; s < nouter; ++s) {
      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
      idxstack[0] = s;

      /* clear out stale data */
      for(idx_t m=1; m < nmodes-1; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* first buf will always just be a matrix row */
      val_t const * const rootrow = mvals[0] + (fid*NFACTORS);
      SPLATT_SIMDFPTYPE * const rootbuf = buf[0];
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        rootbuf[f] = _MM_LOAD(rootrow + f*SPLATT_VLEN);
      }

      idx_t depth = 0;

      idx_t const outer_end = fp[0][s+1];
      while(idxstack[1] < outer_end) {
        /* move down to an nnz node */
        for(; depth < nmodes-2; ++depth) {
          /* propogate buf down */
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * NFACTORS);
          p_assign_hada_<NFACTORS>(buf[depth+1], buf[depth], drow);
        }

        /* process all nonzeros [start, end) */
        idx_t const start = fp[depth][idxstack[depth]];
        idx_t const end   = fp[depth][idxstack[depth]+1];
        p_csf_process_fiber_nolock_<NFACTORS>(ovals, buf[depth],
            start, end, fids[depth+1], vals);

        /* now move back up to the next unprocessed child */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
    } /* end outer slice loop */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  }
}


template<splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_leaf(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  /* extract tensor structures */
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;

  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf3<SYNC_TYPE>(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
  }

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    idxstack[0] = s;

    /* clear out stale data */
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
    val_t * const rootbuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      rootbuf[f] = rootrow[f];
    }

    idx_t depth = 0;

    idx_t const outer_end = fp[0][s+1];
    while(idxstack[1] < outer_end) {
      /* move down to an nnz node */
      for(; depth < nmodes-2; ++depth) {
        /* propogate buf down */
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* process all nonzeros [start, end) */
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      p_csf_process_fiber_lock<SYNC_TYPE>(mats[MAX_NMODES]->vals, buf[depth],
          nfactors, start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  } /* end outer slice loop */
}


template<int NFACTORS, splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_leaf_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(NFACTORS*sizeof(val_t)%64 == 0);

  /* extract tensor structures */
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;

  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf3<SYNC_TYPE>(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  val_t * mvals[MAX_NMODES];
  SPLATT_SIMDFPTYPE * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = (SPLATT_SIMDFPTYPE *)(((val_t *) thds[tid].scratch[2]) + (NFACTORS * m));
  }

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    idxstack[0] = s;

    /* clear out stale data */
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const restrict rootrow = mvals[0] + (fid*NFACTORS);
    SPLATT_SIMDFPTYPE * const rootbuf = buf[0];
#pragma unroll(NFACTORS/SPLATT_VLEN)
    for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
      rootbuf[f] = _MM_LOAD(rootrow + f*SPLATT_VLEN);
    }

    idx_t depth = 0;

    idx_t const outer_end = fp[0][s+1];
    while(idxstack[1] < outer_end) {
      /* move down to an nnz node */
      for(; depth < nmodes-2; ++depth) {
        /* propogate buf down */
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * NFACTORS);
        p_assign_hada_<NFACTORS>(buf[depth+1], buf[depth], drow);
      }

      /* process all nonzeros [start, end) */
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      p_csf_process_fiber_lock_<NFACTORS, SYNC_TYPE>(mats[MAX_NMODES]->vals, buf[depth],
          start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  } /* end outer slice loop */
}


template<bool PARALLELIZE_EACH_TILE=false>
static void p_csf_mttkrp_internal_tiled3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds,
  double const * const opts)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  fidx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  fidx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  fidx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;

  int const tid = omp_get_thread_num();
  bool use_privatization =
    (SPLATT_NOTILE == ct->which_tile || opts[SPLATT_OPTION_TILEDEPTH] > 1) &&
    mttkrp_use_privatization(ct->nnz, mats[MAX_NMODES]->I, opts);
  val_t * const ovals = use_privatization && tid > 0 ? (val_t *)thds[tid].scratch[1] : mats[MAX_NMODES]->vals;

  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

  if(16 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {

#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] -= omp_get_wtime();
#endif
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<16, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif
      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<16, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    }
  }
  else if(32 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {

#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] -= omp_get_wtime();
#endif
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<32, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif
      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<32, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    }
  }
  else if(64 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {

#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] -= omp_get_wtime();
#endif
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<64, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif
      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<64, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    }
  }
  else if(128 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {

#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] -= omp_get_wtime();
#endif
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<128, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif
      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<128, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    }
  }
  else if(256 == nfactors) {
    if(PARALLELIZE_EACH_TILE) {

#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] -= omp_get_wtime();
#endif
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<256, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif
      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        p_csf_mttkrp_internal3_kernel_<256, SPLATT_SYNC_NOSYNC, true>(
          vals, sptr, fptr, fids, inds, rv, bvals, ovals, s);
      }
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    }
  }
  else
  {
    if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
      for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] -= omp_get_wtime();
#endif

        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        /* foreach fiber in slice */
        for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
          /* first entry of the fiber is used to initialize accumF */
          idx_t const jjfirst  = fptr[f];
          val_t const vfirst   = vals[jjfirst];
          val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            accumF[r] = vfirst * bv[r];
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += nfactors;
#endif

          /* foreach nnz in fiber */
          for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
            val_t const v = vals[jj];
            val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
            for(idx_t r=0; r < nfactors; ++r) {
              accumF[r] += v * bv[r];
            }
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += 2*nfactors*(fptr[f+1] - fptr[f] - 1);
#endif

          /* write to fiber row */
          val_t * const restrict ov = ovals  + (fids[f] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            ov[r] += rv[r] * accumF[r];
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += 2*nfactors;
#endif
        }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
        per_thread_times[tid*8] += omp_get_wtime();
#endif
      }
    }
    else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif
      for(idx_t s=0; s < nslices; ++s) {
        idx_t const fid = (sids == NULL) ? s : sids[s];

        /* root row */
        val_t const * const restrict rv = avals + (fid * nfactors);

        /* foreach fiber in slice */
        for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
          /* first entry of the fiber is used to initialize accumF */
          idx_t const jjfirst  = fptr[f];
          val_t const vfirst   = vals[jjfirst];
          val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            accumF[r] = vfirst * bv[r];
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += nfactors;
#endif

          /* foreach nnz in fiber */
          for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
            val_t const v = vals[jj];
            val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
            for(idx_t r=0; r < nfactors; ++r) {
              accumF[r] += v * bv[r];
            }
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += 2*nfactors*(fptr[f+1] - fptr[f] - 1);
#endif

          /* write to fiber row */
          val_t * const restrict ov = ovals  + (fids[f] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            ov[r] += rv[r] * accumF[r];
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += 2*nfactors;
#endif
        }
      }
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    } /* !PARALLELIZE_EACH_TILE */
  }
}


template<bool PARALLELIZE_EACH_TILE=false>
static void p_csf_mttkrp_internal_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  double const * const opts)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_internal_tiled3<PARALLELIZE_EACH_TILE>(ct, tile_id, mats, thds, opts);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  idx_t const nfactors = mats[0]->J;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }
  bool use_privatization =
    (SPLATT_NOTILE == ct->which_tile || opts[SPLATT_OPTION_TILEDEPTH] > outdepth) &&
    mttkrp_use_privatization(ct->nnz, mats[MAX_NMODES]->I, opts);
  val_t * const ovals = use_privatization && tid > 0 ? (val_t *)thds[tid].scratch[1] : mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif

      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

      /* push outer slice and fill stack */
      idxstack[0] = s;
      for(idx_t m=1; m <= outdepth; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first buf */
      val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        buf[0][f] = rootrow[f];
      }

      /* process entire subtree */
      idx_t depth = 0;
      while(idxstack[1] < fp[0][s+1]) {
        /* propagate values down to outdepth-1 */
        for(; depth < outdepth; ++depth) {
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
          p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
        }

        /* write to output and clear buf[outdepth] for next subtree */
        fidx_t const noderow = fids[outdepth][idxstack[outdepth]];

        /* propagate value up to buf[outdepth] */
        p_propagate_up(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
            fp, fids, vals, mvals, nmodes, nfactors);

        val_t * const restrict outbuf = ovals + (noderow * nfactors);
        p_add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);

        /* backtrack to next unfinished node */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    } /* end foreach outer slice */
  }
  else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    for(idx_t s=0; s < nslices; ++s) {
      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

      /* push outer slice and fill stack */
      idxstack[0] = s;
      for(idx_t m=1; m <= outdepth; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first buf */
      val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        buf[0][f] = rootrow[f];
      }

      /* process entire subtree */
      idx_t depth = 0;
      while(idxstack[1] < fp[0][s+1]) {
        /* propagate values down to outdepth-1 */
        for(; depth < outdepth; ++depth) {
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
          p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
        }

        /* write to output and clear buf[outdepth] for next subtree */
        fidx_t const noderow = fids[outdepth][idxstack[outdepth]];

        /* propagate value up to buf[outdepth] */
        p_propagate_up(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
            fp, fids, vals, mvals, nmodes, nfactors);

        val_t * const restrict outbuf = ovals + (noderow * nfactors);
        p_add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);

        /* backtrack to next unfinished node */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
    } /* end foreach outer slice */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  }
}


template<int NFACTORS, bool PARALLELIZE_EACH_TILE=false>
static void p_csf_mttkrp_internal_tiled_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  double const * const opts)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_internal_tiled3<PARALLELIZE_EACH_TILE>(ct, tile_id, mats, thds, opts);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  SPLATT_SIMDFPTYPE * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = (SPLATT_SIMDFPTYPE *)(((val_t *) thds[tid].scratch[2]) + (NFACTORS * m));
    memset(buf[m], 0, NFACTORS * sizeof(val_t));
  }
  bool use_privatization =
    (SPLATT_NOTILE == ct->which_tile || opts[SPLATT_OPTION_TILEDEPTH] > outdepth) &&
    mttkrp_use_privatization(ct->nnz, mats[MAX_NMODES]->I, opts);
  val_t * const ovals = use_privatization && tid > 0 ? (val_t *)thds[tid].scratch[1] : mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  if(PARALLELIZE_EACH_TILE) {
#pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif

      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

      /* push outer slice and fill stack */
      idxstack[0] = s;
      for(idx_t m=1; m <= outdepth; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first buf */
      val_t const * const restrict rootrow = mvals[0] + (fid*NFACTORS);
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        buf[0][f] = _MM_LOAD(rootrow + f*SPLATT_VLEN);
      }

      /* process entire subtree */
      idx_t depth = 0;
      while(idxstack[1] < fp[0][s+1]) {
        /* propagate values down to outdepth-1 */
        for(; depth < outdepth; ++depth) {
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * NFACTORS);
          p_assign_hada_<NFACTORS>(buf[depth+1], buf[depth], drow);
        }

        /* write to output and clear buf[outdepth] for next subtree */
        fidx_t const noderow = fids[outdepth][idxstack[outdepth]];

        /* propagate value up to buf[outdepth] */
        p_propagate_up_<NFACTORS>(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
            fp, fids, vals, mvals, nmodes);

        val_t * const restrict outbuf = ovals + (noderow * NFACTORS);
        p_add_hada_clear_<NFACTORS>(outbuf, buf[outdepth], buf[outdepth-1]);

        /* backtrack to next unfinished node */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    } /* end foreach outer slice */
  }
  else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    for(idx_t s=0; s < nslices; ++s) {
      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

      /* push outer slice and fill stack */
      idxstack[0] = s;
      for(idx_t m=1; m <= outdepth; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first buf */
      val_t const * const restrict rootrow = mvals[0] + (fid*NFACTORS);
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        buf[0][f] = _MM_LOAD(rootrow + f*SPLATT_VLEN);
      }

      /* process entire subtree */
      idx_t depth = 0;
      while(idxstack[1] < fp[0][s+1]) {
        /* propagate values down to outdepth-1 */
        for(; depth < outdepth; ++depth) {
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * NFACTORS);
          p_assign_hada_<NFACTORS>(buf[depth+1], buf[depth], drow);
        }

        /* write to output and clear buf[outdepth] for next subtree */
        fidx_t const noderow = fids[outdepth][idxstack[outdepth]];

        /* propagate value up to buf[outdepth] */
        p_propagate_up_<NFACTORS>(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
            fp, fids, vals, mvals, nmodes);

        val_t * const restrict outbuf = ovals + (noderow * NFACTORS);
        p_add_hada_clear_<NFACTORS>(outbuf, buf[outdepth], buf[outdepth-1]);

        /* backtrack to next unfinished node */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
    } /* end foreach outer slice */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  }
}


template<splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_internal(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_internal3<SYNC_TYPE>(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }
  val_t * const ovals = mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    /* push outer slice and fill stack */
    idxstack[0] = s;
    for(idx_t m=1; m <= outdepth; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* fill first buf */
    val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buf[0][f] = rootrow[f];
    }

    /* process entire subtree */
    idx_t depth = 0;
    while(idxstack[1] < fp[0][s+1]) {
      /* propagate values down to outdepth-1 */
      for(; depth < outdepth; ++depth) {
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* write to output and clear buf[outdepth] for next subtree */
      fidx_t const noderow = fids[outdepth][idxstack[outdepth]];

      /* propagate value up to buf[outdepth] */
      p_propagate_up(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
          fp, fids, vals, mvals, nmodes, nfactors);

      val_t * const restrict outbuf = ovals + (noderow * nfactors);
      splatt_set_lock<SYNC_TYPE>(noderow);
      p_add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);
      splatt_unset_lock<SYNC_TYPE>(noderow);

#ifdef SPLATT_COUNT_REDUCTION
      ++mttkrp_reduction;
#endif

      /* backtrack to next unfinished node */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  } /* end foreach outer slice */
}


#ifdef SPLATT_MEASURE_LOAD_BALANCE
unsigned long long assignHadaTimes[1024];
unsigned long long propagateUpTimes[1024];
unsigned long long hadaClearTimes[1024];
unsigned long long acquireTimes[1024];
#endif


template<int NFACTORS, splatt_sync_type SYNC_TYPE>
static void p_csf_mttkrp_internal_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_internal3<SYNC_TYPE>(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  SPLATT_SIMDFPTYPE * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = (SPLATT_SIMDFPTYPE *)(((val_t *) thds[tid].scratch[2]) + (NFACTORS * m));
    memset(buf[m], 0, NFACTORS * sizeof(val_t));
  }
  val_t * const ovals = mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  if(NULL == fids[0]) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    assignHadaTimes[tid*8] = 0;
    propagateUpTimes[tid*8] = 0;
    hadaClearTimes[tid*8] = 0;
    acquireTimes[tid*8] = 0;
#endif

    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] -= omp_get_wtime();
#endif
    
      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

      /* push outer slice and fill stack */
      idxstack[0] = s;
      for(idx_t m=1; m <= outdepth; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first buf */
      val_t const * const restrict rootrow = mvals[0] + (fid*NFACTORS);
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        buf[0][f] = _MM_LOAD(rootrow + f*SPLATT_VLEN);
      }

      /* process entire subtree */
      idx_t depth = 0;
      while(idxstack[1] < fp[0][s+1]) {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        unsigned long long temp_timer = __rdtsc();
#endif
        /* propagate values down to outdepth-1 */
        for(; depth < outdepth; ++depth) {
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * NFACTORS);
          p_assign_hada_<NFACTORS>(buf[depth+1], buf[depth], drow);
        }
#ifdef SPLATT_MEASURE_LOAD_BALANCE
        assignHadaTimes[tid*8] += __rdtsc() - temp_timer;
        temp_timer = __rdtsc();
#endif

        /* write to output and clear buf[outdepth] for next subtree */
        fidx_t const noderow = fids[outdepth][idxstack[outdepth]];

        /* propagate value up to buf[outdepth] */
        p_propagate_up_<NFACTORS>(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
            fp, fids, vals, mvals, nmodes);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
        propagateUpTimes[tid*8] += __rdtsc() - temp_timer;
        temp_timer = __rdtsc();
#endif

        val_t * const restrict outbuf = ovals + (noderow * NFACTORS);
        if(16 == NFACTORS) {
          p_add_hada_lock_<NFACTORS, SYNC_TYPE>(outbuf, buf[outdepth], buf[outdepth-1], noderow);

#pragma unroll(NFACTORS/SPLATT_VLEN)
          for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
            buf[outdepth][f] = _MM_SETZERO();
          }
        }
        else {
          splatt_set_lock<SYNC_TYPE>(noderow);
          p_add_hada_clear_<NFACTORS>(outbuf, buf[outdepth], buf[outdepth-1]);
          splatt_unset_lock<SYNC_TYPE>(noderow);

#ifdef SPLATT_COUNT_REDUCTION
          ++mttkrp_reduction;
#endif
        }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
        hadaClearTimes[tid*8] += __rdtsc() - temp_timer;
#endif

        /* backtrack to next unfinished node */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
#ifdef SPLATT_MEASURE_LOAD_BALANCE
      per_thread_times[tid*8] += omp_get_wtime();
#endif
    } /* end foreach outer slice */

/*#ifdef SPLATT_MEASURE_LOAD_BALANCE
#pragma omp barrier
#pragma omp master
    {
      if (mode == 3) {
        printf("assign_hada propagate_up hada_clear acquire\n");
        double freq = SpMP::get_cpu_freq();
        for (int i = 0; i < omp_get_max_threads(); ++i) {
          printf("[%d] %f %f %f %f\n", i, assignHadaTimes[tid*8]/freq, propagateUpTimes[tid*8]/freq, hadaClearTimes[tid*8]/freq, acquireTimes[tid*8]/freq);
        }
      }
    }
#pragma omp barrier
#endif*/
  }
  else {
#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] -= omp_get_wtime();
#endif

    /* fids[0] != NULL means we're using tiling, where we parallelize over tiles */
    for(idx_t s=0; s < nslices; ++s) {
      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

      /* push outer slice and fill stack */
      idxstack[0] = s;
      for(idx_t m=1; m <= outdepth; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first buf */
      val_t const * const restrict rootrow = mvals[0] + (fid*NFACTORS);
#pragma unroll(NFACTORS/SPLATT_VLEN)
      for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
        buf[0][f] = _MM_LOAD(rootrow + f*SPLATT_VLEN);
      }

      /* process entire subtree */
      idx_t depth = 0;
      while(idxstack[1] < fp[0][s+1]) {
        /* propagate values down to outdepth-1 */
        for(; depth < outdepth; ++depth) {
          val_t const * const restrict drow
              = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * NFACTORS);
          p_assign_hada_<NFACTORS>(buf[depth+1], buf[depth], drow);
        }

        /* write to output and clear buf[outdepth] for next subtree */
        fidx_t const noderow = fids[outdepth][idxstack[outdepth]];

        /* propagate value up to buf[outdepth] */
        p_propagate_up_<NFACTORS>(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
            fp, fids, vals, mvals, nmodes);

        val_t * const restrict outbuf = ovals + (noderow * NFACTORS);
        if(16 == NFACTORS) {
          p_add_hada_lock_<NFACTORS, SYNC_TYPE>(outbuf, buf[outdepth], buf[outdepth-1], noderow);

#pragma unroll(NFACTORS/SPLATT_VLEN)
          for(idx_t f=0; f < NFACTORS/SPLATT_VLEN; ++f) {
            buf[outdepth][f] = _MM_SETZERO();
          }
        }
        else {
          splatt_set_lock<SYNC_TYPE>(noderow);
          p_add_hada_clear_<NFACTORS>(outbuf, buf[outdepth], buf[outdepth-1]);
          splatt_unset_lock<SYNC_TYPE>(noderow);

#ifdef SPLATT_COUNT_REDUCTION
          ++mttkrp_reduction;
#endif
        }

        /* backtrack to next unfinished node */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
    } /* end foreach outer slice */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[tid*8] += omp_get_wtime();
#endif
  }
}


#ifdef BW_MEASUREMENT
static ctrs ctr1, ctr2;
bool bw_measurement_setup = false;
#endif


/* determine which function to call */
template<splatt_sync_type SYNC_TYPE>
static void p_root_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
#ifdef BW_MEASUREMENT
  double cpu_freq = SpMP::get_cpu_freq();

  if(!bw_measurement_setup) {
    setup();

    bw_measurement_setup = true;
  }
  readctrs(&ctr1);
  
  unsigned long long begin_cycle = __rdtsc();
#endif

  idx_t const nmodes = tensor->nmodes;

  sp_timer_t time;
  timer_fstart(&time);

  matrix_t * const M = mats[MAX_NMODES];
  if (nmodes != 3 || tensor->which_tile != SPLATT_NOTILE)
    par_memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  idx_t const nfactors = mats[0]->J;

#ifdef SPLATT_COUNT_FLOP
  mttkrp_flops = 0; 
#endif

  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[omp_get_thread_num()*8] = 0;
    double tbegin = omp_get_wtime();
#endif

    /* tile id */
    idx_t tid = 0;
    /* this mode may not be tiled due to minimum tiling depth */
    if(SPLATT_NOTILE == tensor->which_tile || opts[SPLATT_OPTION_TILEDEPTH] > 0) {
      double tbegin = omp_get_wtime();

      for(idx_t t=0; t < tensor->ntiles; ++t) {
        if(16 == nfactors) {
          p_csf_mttkrp_root_<16, SYNC_TYPE>(tensor, t, mats, thds);
        }
        else if(32 == nfactors) {
          p_csf_mttkrp_root_<32, SYNC_TYPE>(tensor, t, mats, thds);
        }
        else if(64 == nfactors) {
          p_csf_mttkrp_root_<64, SYNC_TYPE>(tensor, t, mats, thds);
        }
        else if(128 == nfactors) {
          p_csf_mttkrp_root_<128, SYNC_TYPE>(tensor, t, mats, thds);
        }
        else if(256 == nfactors) {
          p_csf_mttkrp_root_<256, SYNC_TYPE>(tensor, t, mats, thds);
        }
        else {
          p_csf_mttkrp_root<SYNC_TYPE>(tensor, t, mats, thds);
        }
        //#pragma omp barrier
      }
    }
    else {
      assert(SPLATT_DENSETILE == tensor->which_tile);
      /* distribute tiles to threads */
      #pragma omp for schedule(dynamic, 1) nowait
      for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
        tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
            mode, t);
        while(tid != TILE_END) {
          p_csf_mttkrp_root_tiled(tensor, tid, mats, thds);
          tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
        }
      }
    }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
#pragma omp barrier
#pragma omp master
    {
      double tend = omp_get_wtime();
      double barrier_time_sum = 0;
      for (int i = 0; i < omp_get_num_threads(); ++i) {
        barrier_time_sum += (tend - tbegin) - per_thread_times[i*8];
      }
      printf("%f load imbalance = %f\n", tend - tbegin, barrier_time_sum/(tend - tbegin)/omp_get_num_threads());
    }
#endif /* SPLATT_MEASURE_LOAD_BALANCE */

    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */

#ifdef BW_MEASUREMENT
  double dt = (__rdtsc() - begin_cycle)/cpu_freq;
  readctrs(&ctr2);

  double total_edc_rd = 0, total_edc_wr = 0;
  for (int j = 0; j < NEDC; ++j) {
    total_edc_rd += ctr2.edcrd[j] - ctr1.edcrd[j];
    total_edc_wr += ctr2.edcwr[j] - ctr1.edcwr[j];
  }
  double mcdram_rdbw = 64*total_edc_rd/dt;
  double mcdram_wrbw = 64*total_edc_wr/dt;

  double total_mc_rd = 0, total_mc_wr = 0;
  for (int j = 0; j < NMC; ++j) {
    total_mc_rd += ctr2.mcrd[j] - ctr1.mcrd[j];
    total_mc_wr += ctr2.mcwr[j] - ctr1.mcwr[j];
  }
  double ddr_rdbw = 64*total_mc_rd/dt;
  double ddr_wrbw = 64*total_mc_wr/dt;

  printf("mcdram_rdbw = %g gbps, mcdram_wrbw = %g gbps\n", mcdram_rdbw/1e9, mcdram_wrbw/1e9);
  printf("ddr_rdbw = %g gbps, ddr_wrbw = %g gbps\n", ddr_rdbw/1e9, ddr_wrbw/1e9);
#endif

  timer_stop(&time);
  if (opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_LOW) {
    size_t mbytes = 0;
    assert(mode == tensor->dim_perm[0]);
    for (int i = 0; i < tensor->nmodes; ++i) {
      if (i != mode) mbytes += tensor->dims[i] * M->J * sizeof(val_t);
    }
    size_t fbytes = tensor->storage;
    double gbps = (mbytes + fbytes)/time.seconds/1e9;
    double gflops = tensor->mttkrp_flops[mode]/time.seconds/1e9;

#ifdef SPLATT_COUNT_FLOP
    printf("       csf_mttkrp_root (%0.3f s, %.3f GBps, %.3f GFs)\n",
        time.seconds, gbps, (double)mttkrp_flops/1e9);
#else
    printf("       csf_mttkrp_root (%0.3f s, %.3f GBps)\n",
        time.seconds, gbps);
#endif
  }
}


template<splatt_sync_type SYNC_TYPE>
static void p_leaf_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
#ifdef BW_MEASUREMENT
  double cpu_freq = SpMP::get_cpu_freq();

  if(!bw_measurement_setup) {
    setup();

    bw_measurement_setup = true;
  }
  readctrs(&ctr1);
  
  unsigned long long begin_cycle = __rdtsc();
#endif

  idx_t const nmodes = tensor->nmodes;
  idx_t const depth = nmodes - 1;

  sp_timer_t time;
  timer_fstart(&time);

  matrix_t * const M = mats[MAX_NMODES];
  bool use_privatization =
    SPLATT_NOTILE == tensor->which_tile &&
    mttkrp_use_privatization(tensor->nnz, mats[mode]->I, opts);
    // if reduction overhead is sufficiently small, use privatization
  if(!use_privatization) par_memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  idx_t const nfactors = mats[0]->J;

#ifdef SPLATT_COUNT_FLOP
  mttkrp_flops = 0; 
#endif

  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    timer_start(&thds[omp_get_thread_num()].ttime);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[omp_get_thread_num()*8] = 0;
    double tbegin = omp_get_wtime();
#endif

    /* tile id */
    idx_t tid = 0;
    /* ignore opts[SPLATT_OPTION_TILEDEPTH] > depth case because that means we're
     * not tiling any mode */
    assert(opts[SPLATT_OPTION_TILEDEPTH] <= depth);
    if(SPLATT_NOTILE == tensor->which_tile) {
      if(use_privatization) {
        double reduction_time = -omp_get_wtime();
        if(0 == omp_get_thread_num()) {
          memset(M->vals, 0, nfactors*M->I*sizeof(val_t));
        }
        else {
          memset(thds[omp_get_thread_num()].scratch[1], 0, nfactors*M->I*sizeof(val_t));
        }
        reduction_time += omp_get_wtime();
        
        if(16 == nfactors) {
          p_csf_mttkrp_leaf_tiled_<16, true>(tensor, 0, mats, thds, opts);
        }
        else if(32 == nfactors) {
          p_csf_mttkrp_leaf_tiled_<32, true>(tensor, 0, mats, thds, opts);
        }
        else if(64 == nfactors) {
          p_csf_mttkrp_leaf_tiled_<64, true>(tensor, 0, mats, thds, opts);
        }
        else if(128 == nfactors) {
          p_csf_mttkrp_leaf_tiled_<128, true>(tensor, 0, mats, thds, opts);
        }
        else if(256 == nfactors) {
          p_csf_mttkrp_leaf_tiled_<256, true>(tensor, 0, mats, thds, opts);
        }
        else {
          p_csf_mttkrp_leaf_tiled<true>(tensor, 0, mats, thds, opts);
        }

#pragma omp barrier
        reduction_time -= omp_get_wtime();
#pragma omp for
        for(idx_t i=0; i < nfactors*M->I; ++i) {
          for(int t=1; t < nthreads; ++t) {
            M->vals[i] += ((val_t *)thds[t].scratch[1])[i];
          }
        }
        reduction_time += omp_get_wtime();
        if(0 == omp_get_thread_num()) {
          printf("reduction takes %f\n", reduction_time);
        }
      }
      else {
        if(16 == nfactors) {
          p_csf_mttkrp_leaf_<16, SYNC_TYPE>(tensor, 0, mats, thds);
        }
        else if(32 == nfactors) {
          p_csf_mttkrp_leaf_<32, SYNC_TYPE>(tensor, 0, mats, thds);
        }
        else if(64 == nfactors) {
          p_csf_mttkrp_leaf_<64, SYNC_TYPE>(tensor, 0, mats, thds);
        }
        else if(128 == nfactors) {
          p_csf_mttkrp_leaf_<128, SYNC_TYPE>(tensor, 0, mats, thds);
        }
        else if(256 == nfactors) {
          p_csf_mttkrp_leaf_<256, SYNC_TYPE>(tensor, 0, mats, thds);
        }
        else {
          p_csf_mttkrp_leaf<SYNC_TYPE>(tensor, 0, mats, thds);
        }
      }
    } /* SPLATT_NOTILE */
    else {
      #pragma omp for schedule(dynamic, 1) nowait
      for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
        tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
            mode, t);
        while(tid != TILE_END) {
          if(16 == nfactors) {
            p_csf_mttkrp_leaf_tiled_<16>(tensor, tid, mats, thds, opts);
          }
          else if(32 == nfactors) {
            p_csf_mttkrp_leaf_tiled_<32>(tensor, tid, mats, thds, opts);
          }
          else if(64 == nfactors) {
            p_csf_mttkrp_leaf_tiled_<64>(tensor, tid, mats, thds, opts);
          }
          else if(128 == nfactors) {
            p_csf_mttkrp_leaf_tiled_<128>(tensor, tid, mats, thds, opts);
          }
          else if(256 == nfactors) {
            p_csf_mttkrp_leaf_tiled_<256>(tensor, tid, mats, thds, opts);
          }
          else {
            p_csf_mttkrp_leaf_tiled(tensor, tid, mats, thds, opts);
          }
          tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
        }
      }
    } /* SPLATT_DENSETILE */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
#pragma omp barrier
#pragma omp master
    {
      double tend = omp_get_wtime();
      double barrier_time_sum = 0;
      for (int i = 0; i < nthreads; ++i) {
        barrier_time_sum += (tend - tbegin) - per_thread_times[i*8];
      }
      printf("%f load imbalance = %f\n", tend - tbegin, barrier_time_sum/(tend - tbegin)/nthreads);
    }
#endif /* SPLATT_MEASURE_LOAD_BALANCE */

    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */

#ifdef BW_MEASUREMENT
  double dt = (__rdtsc() - begin_cycle)/cpu_freq;
  readctrs(&ctr2);

  double total_edc_rd = 0, total_edc_wr = 0;
  for (int j = 0; j < NEDC; ++j) {
    total_edc_rd += ctr2.edcrd[j] - ctr1.edcrd[j];
    total_edc_wr += ctr2.edcwr[j] - ctr1.edcwr[j];
  }
  double mcdram_rdbw = 64*total_edc_rd/dt;
  double mcdram_wrbw = 64*total_edc_wr/dt;

  double total_mc_rd = 0, total_mc_wr = 0;
  for (int j = 0; j < NMC; ++j) {
    total_mc_rd += ctr2.mcrd[j] - ctr1.mcrd[j];
    total_mc_wr += ctr2.mcwr[j] - ctr1.mcwr[j];
  }
  double ddr_rdbw = 64*total_mc_rd/dt;
  double ddr_wrbw = 64*total_mc_wr/dt;

  printf("mcdram_rdbw = %g gbps, mcdram_wrbw = %g gbps\n", mcdram_rdbw/1e9, mcdram_wrbw/1e9);
  printf("ddr_rdbw = %g gbps, ddr_wrbw = %g gbps\n", ddr_rdbw/1e9, ddr_wrbw/1e9);
#endif

  timer_stop(&time);
  if (opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_LOW) {
    size_t mbytes = 0;
    assert(mode == tensor->dim_perm[tensor->nmodes-1]);
    for (int i = 0; i < tensor->nmodes; ++i) {
      if (i != mode) mbytes += tensor->dims[i] * M->J * sizeof(val_t);
    }
    size_t fbytes = tensor->storage;
    double gbps = (mbytes + fbytes)/time.seconds/1e9;
    double gflops = tensor->mttkrp_flops[mode]/time.seconds/1e9;

#ifdef SPLATT_COUNT_FLOP
    printf("       csf_mttkrp_leaf (%0.3f s, %.3f GBps, %.3f GFs)\n",
        time.seconds, gbps, (double)mttkrp_flops/1e9);
#else
    printf("       csf_mttkrp_leaf (%0.3f s, %.3f GBps)\n",
        time.seconds, gbps);
#endif
  }
}


template<splatt_sync_type SYNC_TYPE>
static void p_intl_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
#ifdef BW_MEASUREMENT
  double cpu_freq = SpMP::get_cpu_freq();

  if(!bw_measurement_setup) {
    setup();

    bw_measurement_setup = true;
  }
  readctrs(&ctr1);
  
  unsigned long long begin_cycle = __rdtsc();
#endif

  idx_t const nmodes = tensor->nmodes;
  idx_t const depth = csf_mode_depth(mode, tensor->dim_perm, nmodes);

  sp_timer_t time;
  timer_fstart(&time);

  matrix_t * const M = mats[MAX_NMODES];
  assert(M->I == mats[mode]->I);
  bool use_privatization =
    (SPLATT_NOTILE == tensor->which_tile || opts[SPLATT_OPTION_TILEDEPTH] > depth) &&
    mttkrp_use_privatization(tensor->nnz, mats[mode]->I, opts);
    // if reduction overhead is sufficiently small, use privatization
  if(!use_privatization) par_memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  idx_t const nfactors = mats[0]->J;

#ifdef SPLATT_COUNT_FLOP
  mttkrp_flops = 0; 
#endif
#ifdef SPLATT_COUNT_REDUCTION
  mttkrp_reduction = 0;
#endif

  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    timer_start(&thds[omp_get_thread_num()].ttime);

#ifdef SPLATT_MEASURE_LOAD_BALANCE
    per_thread_times[omp_get_thread_num()*8] = 0;
    double tbegin = omp_get_wtime();
#endif

    /* tile id */
    idx_t tid = 0;
    /* this mode may not be tiled due to minimum tiling depth */
    if(SPLATT_NOTILE == tensor->which_tile || opts[SPLATT_OPTION_TILEDEPTH] > depth) {
      if(use_privatization) {
        double reduction_time = -omp_get_wtime();
        if(0 == omp_get_thread_num()) {
          memset(M->vals, 0, nfactors*M->I*sizeof(val_t));
        }
        else {
          memset(thds[omp_get_thread_num()].scratch[1], 0, nfactors*M->I*sizeof(val_t));
        }
        reduction_time += omp_get_wtime();

        if(SPLATT_NOTILE == tensor->which_tile) {
          if(16 == nfactors) {
            p_csf_mttkrp_internal_tiled_<16, true>(tensor, 0, mats, mode, thds, opts);
          }
          else if(32 == nfactors) {
            p_csf_mttkrp_internal_tiled_<32, true>(tensor, 0, mats, mode, thds, opts);
          }
          else if(64 == nfactors) {
            p_csf_mttkrp_internal_tiled_<64, true>(tensor, 0, mats, mode, thds, opts);
          }
          else if(128 == nfactors) {
            p_csf_mttkrp_internal_tiled_<128, true>(tensor, 0, mats, mode, thds, opts);
          }
          else if(256 == nfactors) {
            p_csf_mttkrp_internal_tiled_<256, true>(tensor, 0, mats, mode, thds, opts);
          }
          else {
            p_csf_mttkrp_internal_tiled<true>(tensor, 0, mats, mode, thds, opts);
          }
        }
        else {
          #pragma omp for schedule(dynamic, 1) nowait
          for(idx_t t=0; t < tensor->ntiles; ++t) {
            if(16 == nfactors) {
              p_csf_mttkrp_internal_tiled_<16>(tensor, t, mats, mode, thds, opts);
            }
            else if(32 == nfactors) {
              p_csf_mttkrp_internal_tiled_<32>(tensor, t, mats, mode, thds, opts);
            }
            else if(64 == nfactors) {
              p_csf_mttkrp_internal_tiled_<64>(tensor, t, mats, mode, thds, opts);
            }
            else if(128 == nfactors) {
              p_csf_mttkrp_internal_tiled_<128>(tensor, t, mats, mode, thds, opts);
            }
            else if(256 == nfactors) {
              p_csf_mttkrp_internal_tiled_<256>(tensor, t, mats, mode, thds, opts);
            }
            else {
              p_csf_mttkrp_internal_tiled(tensor, t, mats, mode, thds, opts);
            }
          }
        }

#pragma omp barrier

        reduction_time -= omp_get_wtime();
#pragma omp for
        for(idx_t i=0; i < nfactors*M->I; ++i) {
          for(int t=1; t < nthreads; ++t) {
            M->vals[i] += ((val_t *)thds[t].scratch[1])[i];
          }
        }
        reduction_time += omp_get_wtime();
        if(0 == omp_get_thread_num()) {
          printf("reduction takes %f\n", reduction_time);
        }
      } /* privatization */
      else if(SPLATT_NOTILE == tensor->which_tile) {
        if(16 == nfactors) {
          p_csf_mttkrp_internal_<16, SYNC_TYPE>(tensor, 0, mats, mode, thds);
        }
        else if(32 == nfactors) {
          p_csf_mttkrp_internal_<32, SYNC_TYPE>(tensor, 0, mats, mode, thds);
        }
        else if(64 == nfactors) {
          p_csf_mttkrp_internal_<64, SYNC_TYPE>(tensor, 0, mats, mode, thds);
        }
        else if(128 == nfactors) {
          p_csf_mttkrp_internal_<128, SYNC_TYPE>(tensor, 0, mats, mode, thds);
        }
        else if(256 == nfactors) {
          p_csf_mttkrp_internal_<256, SYNC_TYPE>(tensor, 0, mats, mode, thds);
        }
        else {
          p_csf_mttkrp_internal<SYNC_TYPE>(tensor, 0, mats, mode, thds);
        }
      }
      else {
        #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          if(16 == nfactors) {
            p_csf_mttkrp_internal_<16, SYNC_TYPE>(tensor, t, mats, mode, thds);
          }
          else if(32 == nfactors) {
            p_csf_mttkrp_internal_<32, SYNC_TYPE>(tensor, t, mats, mode, thds);
          }
          else if(64 == nfactors) {
            p_csf_mttkrp_internal_<64, SYNC_TYPE>(tensor, t, mats, mode, thds);
          }
          else if(128 == nfactors) {
            p_csf_mttkrp_internal_<128, SYNC_TYPE>(tensor, t, mats, mode, thds);
          }
          else if(256 == nfactors) {
            p_csf_mttkrp_internal_<256, SYNC_TYPE>(tensor, t, mats, mode, thds);
          }
          else {
            p_csf_mttkrp_internal<SYNC_TYPE>(tensor, t, mats, mode, thds);
          }
        }
      }
    } else {
      assert(SPLATT_DENSETILE == tensor->which_tile);
      #pragma omp for schedule(dynamic, 1) nowait
      for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
        tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
            mode, t);
        while(tid != TILE_END) {
          if(16 == nfactors) {
            p_csf_mttkrp_internal_tiled_<16>(tensor, tid, mats, mode, thds, opts);
          }
          else if(32 == nfactors) {
            p_csf_mttkrp_internal_tiled_<32>(tensor, tid, mats, mode, thds, opts);
          }
          else if(64 == nfactors) {
            p_csf_mttkrp_internal_tiled_<64>(tensor, tid, mats, mode, thds, opts);
          }
          else if(128 == nfactors) {
            p_csf_mttkrp_internal_tiled_<128>(tensor, tid, mats, mode, thds, opts);
          }
          else if(256 == nfactors) {
            p_csf_mttkrp_internal_tiled_<256>(tensor, tid, mats, mode, thds, opts);
          }
          else {
            p_csf_mttkrp_internal_tiled(tensor, tid, mats, mode, thds, opts);
          }
          tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
        }
      }
    } /* SPLATT_DENSETILE */

#ifdef SPLATT_MEASURE_LOAD_BALANCE
#pragma omp barrier
#pragma omp master
    {
      double tend = omp_get_wtime();
      double barrier_time_sum = 0;
      for (int i = 0; i < nthreads; ++i) {
        barrier_time_sum += (tend - tbegin) - per_thread_times[i*8];
      }
      printf("%f load imbalance = %f\n", tend - tbegin, barrier_time_sum/(tend - tbegin)/nthreads);
    }
#endif /* SPLATT_MEASURE_LOAD_BALANCE */

    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */

#ifdef BW_MEASUREMENT
  double dt = (__rdtsc() - begin_cycle)/cpu_freq;
  readctrs(&ctr2);

  double total_edc_rd = 0, total_edc_wr = 0;
  for (int j = 0; j < NEDC; ++j) {
    total_edc_rd += ctr2.edcrd[j] - ctr1.edcrd[j];
    total_edc_wr += ctr2.edcwr[j] - ctr1.edcwr[j];
  }
  double mcdram_rdbw = 64*total_edc_rd/dt;
  double mcdram_wrbw = 64*total_edc_wr/dt;

  double total_mc_rd = 0, total_mc_wr = 0;
  for (int j = 0; j < NMC; ++j) {
    total_mc_rd += ctr2.mcrd[j] - ctr1.mcrd[j];
    total_mc_wr += ctr2.mcwr[j] - ctr1.mcwr[j];
  }
  double ddr_rdbw = 64*total_mc_rd/dt;
  double ddr_wrbw = 64*total_mc_wr/dt;

  printf("mcdram_rdbw = %g gbps, mcdram_wrbw = %g gbps\n", mcdram_rdbw/1e9, mcdram_wrbw/1e9);
  printf("ddr_rdbw = %g gbps, ddr_wrbw = %g gbps\n", ddr_rdbw/1e9, ddr_wrbw/1e9);
#endif

  timer_stop(&time);
  if (opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_LOW) {
    size_t mbytes = 0;
    assert(mode != tensor->dim_perm[0] && mode != tensor->dim_perm[tensor->nmodes-1]);
    for (int i = 0; i < tensor->nmodes; ++i) {
      if (i != mode) mbytes += tensor->dims[i] * M->J * sizeof(val_t);
    }
    size_t fbytes = tensor->storage;
    double gbps = (mbytes + fbytes)/time.seconds/1e9;
    double gflops = tensor->mttkrp_flops[mode]/time.seconds/1e9;

#ifdef SPLATT_COUNT_FLOP
    printf("       csf_mttkrp_internal (%0.3f s, %.3f GBps, %.3f GFs, %ld reductions)\n",
        time.seconds, gbps, (double)mttkrp_flops/1e9, mttkrp_reduction);
#else
    printf("       csf_mttkrp_internal (%0.3f s, %.3f GBps)\n",
        time.seconds, gbps);
#endif
  }
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

template<splatt_sync_type SYNC_TYPE>
void p_mttkrp_csf(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  double const * const opts)
{
  p_init_locks();

  /* clear output matrix */
  matrix_t * const M = mats[MAX_NMODES];
  M->I = tensors[0].dims[mode];

  omp_set_num_threads(opts[SPLATT_OPTION_NTHREADS]);

  idx_t nmodes = tensors[0].nmodes;
  /* find out which level in the tree this is */
  idx_t outdepth = MAX_NMODES;

  /* choose which MTTKRP function to use */
  splatt_csf_type which = (splatt_csf_type)opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {
  case SPLATT_CSF_ONEMODE:
    outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
    if(outdepth == 0) {
      p_root_decide<SYNC_TYPE>(tensors+0, mats, mode, thds, opts);
    } else if(outdepth == nmodes - 1) {
      p_leaf_decide<SYNC_TYPE>(tensors+0, mats, mode, thds, opts);
    } else {
      p_intl_decide<SYNC_TYPE>(tensors+0, mats, mode, thds, opts);
    }
    break;

  case SPLATT_CSF_TWOMODE:
    /* longest mode handled via second tensor's root */
    if(mode == tensors[0].dim_perm[nmodes-1]) {
      p_root_decide<SYNC_TYPE>(tensors+1, mats, mode, thds, opts);
    /* root and internal modes are handled via first tensor */
    } else {
      outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
      if(outdepth == 0) {
        p_root_decide<SYNC_TYPE>(tensors+0, mats, mode, thds, opts);
      } else {
        p_intl_decide<SYNC_TYPE>(tensors+0, mats, mode, thds, opts);
      }
    }
    break;

  case SPLATT_CSF_ALLMODE:
    p_root_decide<SYNC_TYPE>(tensors+mode, mats, mode, thds, opts);
    break;
  }
}


void mttkrp_csf(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  double const * const opts)
{
  splatt_sync_type sync_type = (splatt_sync_type)opts[SPLATT_OPTION_SYNCHRONIZATION];
  switch(sync_type) {
  case SPLATT_SYNC_OMP_LOCK:
    p_mttkrp_csf<SPLATT_SYNC_OMP_LOCK>(tensors, mats, mode, thds, opts);
    break; 
  case SPLATT_SYNC_TTAS:
    p_mttkrp_csf<SPLATT_SYNC_TTAS>(tensors, mats, mode, thds, opts);
    break; 
  case SPLATT_SYNC_RTM:
    p_mttkrp_csf<SPLATT_SYNC_RTM>(tensors, mats, mode, thds, opts);
    break; 
  case SPLATT_SYNC_CAS:
    p_mttkrp_csf<SPLATT_SYNC_CAS>(tensors, mats, mode, thds, opts);
    break; 
  case SPLATT_SYNC_VCAS:
    p_mttkrp_csf<SPLATT_SYNC_VCAS>(tensors, mats, mode, thds, opts);
    break; 
  case SPLATT_SYNC_NOSYNC:
    p_mttkrp_csf<SPLATT_SYNC_NOSYNC>(tensors, mats, mode, thds, opts);
    break; 
  default:
    assert(false);
    break;
  }
}






/******************************************************************************
 * DEPRECATED FUNCTIONS
 *****************************************************************************/








/******************************************************************************
 * SPLATT MTTKRP
 *****************************************************************************/

void mttkrp_splatt(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  if(ft->tiled == SPLATT_SYNCTILE) {
    mttkrp_splatt_sync_tiled(ft, mats, mode, thds, nthreads);
    return;
  }
  if(ft->tiled == SPLATT_COOPTILE) {
    mttkrp_splatt_coop_tiled(ft, mats, mode, thds, nthreads);
    return;
  }

  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];
  idx_t const nslices = ft->dims[mode];
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  par_memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict sptr = ft->sptr;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  storage_val_t const * const restrict vals = ft->vals;

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    timer_start(&thds[tid].ttime);

    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      val_t * const restrict mv = mvals + (s * rank);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        storage_val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          storage_val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t const * const restrict av = avals  + (fids[f] * rank);
        for(idx_t r=0; r < rank; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }
    }

    timer_stop(&thds[tid].ttime);
  } /* end parallel region */
}


void mttkrp_splatt_sync_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];

  idx_t const nslabs = ft->nslabs;
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  par_memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict slabptr = ft->slabptr;
  idx_t const * const restrict sids = ft->sids;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  storage_val_t const * const restrict vals = ft->vals;

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    timer_start(&thds[tid].ttime);

    #pragma omp for schedule(dynamic, 1) nowait
    for(idx_t s=0; s < nslabs; ++s) {
      /* foreach fiber in slice */
      for(idx_t f=slabptr[s]; f < slabptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t       * const restrict mv = mvals + (sids[f] * rank);
        val_t const * const restrict av = avals + (fids[f] * rank);
        for(idx_t r=0; r < rank; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }
    }

    timer_stop(&thds[tid].ttime);
  } /* end parallel region */
}


void mttkrp_splatt_coop_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];

  idx_t const nslabs = ft->nslabs;
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  par_memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict slabptr = ft->slabptr;
  idx_t const * const restrict sptr = ft->sptr;
  idx_t const * const restrict sids = ft->sids;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  storage_val_t const * const restrict vals = ft->vals;

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    val_t * const localm = (val_t *) thds[tid].scratch[1];
    timer_start(&thds[tid].ttime);

    /* foreach slab */
    for(idx_t s=0; s < nslabs; ++s) {
      /* foreach fiber in slab */
      #pragma omp for schedule(dynamic, 8)
      for(idx_t sl=slabptr[s]; sl < slabptr[s+1]; ++sl) {
        idx_t const slice = sids[sl];
        for(idx_t f=sptr[sl]; f < sptr[sl+1]; ++f) {
          /* first entry of the fiber is used to initialize accumF */
          idx_t const jjfirst  = fptr[f];
          val_t const vfirst   = vals[jjfirst];
          val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] = vfirst * bv[r];
          }

          /* foreach nnz in fiber */
          for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
            val_t const v = vals[jj];
            val_t const * const restrict bv = bvals + (inds[jj] * rank);
            for(idx_t r=0; r < rank; ++r) {
              accumF[r] += v * bv[r];
            }
          }

          /* scale inner products by row of A and update thread-local M */
          val_t       * const restrict mv = localm + ((slice % TILE_SIZES[0]) * rank);
          val_t const * const restrict av = avals + (fids[f] * rank);
          for(idx_t r=0; r < rank; ++r) {
            mv[r] += accumF[r] * av[r];
          }
        }
      }

      idx_t const start = s * TILE_SIZES[0];
      idx_t const stop  = SS_MIN((s+1) * TILE_SIZES[0], ft->dims[mode]);

      #pragma omp for schedule(static)
      for(idx_t i=start; i < stop; ++i) {
        /* map i back to global slice id */
        idx_t const localrow = i % TILE_SIZES[0];
        for(idx_t t=0; t < nthreads; ++t) {
          val_t * const threadm = (val_t *) thds[t].scratch[1];
          for(idx_t r=0; r < rank; ++r) {
            mvals[r + (i*rank)] += threadm[r + (localrow*rank)];
            threadm[r + (localrow*rank)] = 0.;
          }
        }
      }

    } /* end foreach slab */
    timer_stop(&thds[tid].ttime);
  } /* end omp parallel */
}



/******************************************************************************
 * GIGA MTTKRP
 *****************************************************************************/
void mttkrp_giga(
  spmatrix_t const * const spmat,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  idx_t const I = spmat->I;
  idx_t const rank = M->J;

  idx_t const * const restrict rowptr = spmat->rowptr;
  idx_t const * const restrict colind = spmat->colind;
  val_t const * const restrict vals   = spmat->vals;

  #pragma omp parallel
  {
    for(idx_t r=0; r < rank; ++r) {
      val_t       * const restrict mv =  M->vals + (r * I);
      val_t const * const restrict av =  A->vals + (r * A->I);
      val_t const * const restrict bv =  B->vals + (r * B->I);

      /* Joined Hadamard products of X, C, and B */
      #pragma omp for schedule(dynamic, 16)
      for(idx_t i=0; i < I; ++i) {
        for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
          idx_t const a = colind[y] / B->I;
          idx_t const b = colind[y] % B->I;
          scratch[y] = vals[y] * av[a] * bv[b];
        }
      }

      /* now accumulate rows into column of M1 */
      #pragma omp for schedule(dynamic, 16)
      for(idx_t i=0; i < I; ++i) {
        val_t sum = 0;
        for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
          sum += scratch[y];
        }
        mv[i] = sum;
      }
    }
  }
}


/******************************************************************************
 * TTBOX MTTKRP
 *****************************************************************************/
void mttkrp_ttbox(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  idx_t const I = tt->dims[mode];
  idx_t const rank = M->J;

  par_memset(M->vals, 0, I * rank * sizeof(val_t));

  idx_t const nnz = tt->nnz;
  fidx_t const * const restrict indM = tt->ind[mode];
  fidx_t const * const restrict indA =
    mode == 0 ? tt->ind[1] : tt->ind[0];
  fidx_t const * const restrict indB =
    mode == 2 ? tt->ind[1] : tt->ind[2];

  storage_val_t const * const restrict vals = tt->vals;

  for(idx_t r=0; r < rank; ++r) {
    val_t       * const restrict mv =  M->vals + (r * I);
    val_t const * const restrict av =  A->vals + (r * A->I);
    val_t const * const restrict bv =  B->vals + (r * B->I);

    /* stretch out columns of A and B */
    #pragma omp parallel for
    for(idx_t x=0; x < nnz; ++x) {
      scratch[x] = vals[x] * av[indA[x]] * bv[indB[x]];
    }

    /* now accumulate into m1 */
    for(idx_t x=0; x < nnz; ++x) {
      mv[indM[x]] += scratch[x];
    }
  }
}

void mttkrp_stream(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode)
{
  matrix_t * const M = mats[MAX_NMODES];
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = M->J;

  val_t * const outmat = M->vals;
  par_memset(outmat, 0, I * nfactors * sizeof(val_t));

  idx_t const nmodes = tt->nmodes;

  val_t * accum = (val_t *) splatt_malloc(nfactors * sizeof(val_t));

  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }

  storage_val_t const * const restrict vals = tt->vals;

  /* stream through nnz */
  for(idx_t n=0; n < tt->nnz; ++n) {
    /* initialize with value */
    for(idx_t f=0; f < nfactors; ++f) {
      accum[f] = vals[n];
    }

    for(idx_t m=0; m < nmodes; ++m) {
      if(m == mode) {
        continue;
      }
      val_t const * const restrict inrow = mvals[m] + (tt->ind[m][n] * nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        accum[f] *= inrow[f];
      }
    }

    /* write to output */
    val_t * const restrict outrow = outmat + (tt->ind[mode][n] * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      outrow[f] += accum[f];
    }
  }

  free(accum);
}



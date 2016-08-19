
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"
#include "thd_info.h"
#include "tile.h"
#include "util.h"
#include <limits.h>
#include <omp.h>
#include <algorithm>
#ifdef __INTEL_COMPILER
#include <immintrin.h>
#define SPLATT_INTRINSIC // use intrinsic
#endif
#include <unistd.h>
//#define SPLATT_SIM_CACHE
#ifdef SPLATT_SIM_CACHE
#include "gathertool.h"
#endif

#define SPLATT_NFACTOR_COMPILE_CONSTANT
#ifndef __AVX512F__ // RTM and ATOMIC_CACHE not yet implemented for AVX512
//#define SPLATT_RTM // use restricted transactional memory
//#define SPLATT_ATOMIC_CACHE
#endif
//#define SPLATT_CAS
//#define SPLATT_EMULATE_VECTOR_CAS

#define NLOCKS 1024
static int locks_initialized = 0;

#ifdef SPLATT_ATOMIC_CACHE
#define NBUCKETS 65536
#endif

//#define SPLATT_NO_LOCK
//#define SPLATT_MY_TTAS_LOCK
#define SPLATT_LOCK_PAD (8)
#ifdef SPLATT_MY_TTAS_LOCK
static volatile long locks[NLOCKS*SPLATT_LOCK_PAD];
#elif !defined(SPLATT_NO_LOCK)
static omp_lock_t locks[NLOCKS*SPLATT_LOCK_PAD];
#endif

//#define SPLATT_COUNT_FLOP
#ifdef SPLATT_COUNT_FLOP
static long mttkrp_flops = 0;
#endif

static void p_init_locks()
{
  if (!locks_initialized) {
    for(int i=0; i < NLOCKS; ++i) {
#ifdef SPLATT_MY_TTAS_LOCK
      locks[i*SPLATT_LOCK_PAD] = 0;
#elif !defined(SPLATT_NO_LOCK)
      omp_init_lock(locks+i*SPLATT_LOCK_PAD);
#endif
    }
    locks_initialized = 1;
  }
}

static inline void splatt_set_lock(int id)
{
  int i = (id%NLOCKS)*SPLATT_LOCK_PAD;
#ifdef SPLATT_MY_TTAS_LOCK
  //while (0 != locks[i] || 0 != __sync_lock_test_and_set(locks + i, 0xff)); // TTAS
  while (0 != __sync_lock_test_and_set(locks + i, 1)) { _mm_pause(); }; // TAS with backoff
#elif !defined(SPLATT_NO_LOCK)
  omp_set_lock(locks + i);
#endif
}

static inline void splatt_unset_lock(int id)
{
  int i = (id%NLOCKS)*SPLATT_LOCK_PAD;
#ifdef SPLATT_MY_TTAS_LOCK
  locks[i] = 0;
  asm volatile("" ::: "memory");
  //__sync_lock_release(locks + i); // icc generates mfence instruction for this
#elif !defined(SPLATT_NO_LOCK)
  omp_unset_lock(locks + i);
#endif
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
}


template<int NFACTORS>
static inline void p_add_hada_clear_(
  val_t * const restrict out,
  val_t * const restrict a,
  val_t const * const restrict b)
{
  __assume_aligned(out, 64);
  __assume_aligned(a, 64);
  __assume_aligned(b, 64);
  for(idx_t f=0; f < NFACTORS; ++f) {
    out[f] += a[f] * b[f];
    a[f] = 0;
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
}


template<int NFACTORS>
static inline void p_assign_hada_(
  val_t * const restrict out,
  val_t const * const restrict a,
  val_t const * const restrict b)
{
  __assume_aligned(out, 64);
  __assume_aligned(a, 64);
  __assume_aligned(b, 64);
  for(idx_t f=0; f < NFACTORS; ++f) {
    out[f] = a[f] * b[f];
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += NFACTORS;
#endif
}


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
#ifdef SPLATT_RTM
    if(_XBEGIN_STARTED == _xbegin()) {
      for(idx_t f=0; f < nfactors; ++f) {
        leafrow[f] += v * accumbuf[f];
      }
    }
    else
#endif
    {
#ifdef SPLATT_CAS
      for(idx_t f=0; f < nfactors; ++f) {
        double old_leafrow, new_leafrow;
        do {
          old_leafrow = leafrow[f];
          new_leafrow = old_leafrow + v * accumbuf[f];
        } while (!__sync_bool_compare_and_swap((long long *)(leafrow + f), *((long long *)(&old_leafrow)), *((long long *)(&new_leafrow))));
      }
#else
      splatt_set_lock(inds[jj]);
      for(idx_t f=0; f < nfactors; ++f) {
        leafrow[f] += v * accumbuf[f];
      }
      splatt_unset_lock(inds[jj]);
#endif
    }
  }
}


template<int NFACTORS>
static inline void p_csf_process_fiber_lock_(
  val_t * const leafmat,
  val_t const * const restrict accumbuf,
  idx_t const start,
  idx_t const end,
  fidx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  __assume_aligned(accumbuf, 64);
  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * NFACTORS);
    __assume_aligned(leafrow, 64);
    val_t const v = vals[jj];
#ifdef SPLATT_RTM
    if(_XBEGIN_STARTED == _xbegin()) {
      for(idx_t f=0; f < NFACTORS; ++f) {
        leafrow[f] += v * accumbuf[f];
      }
    }
    else
#endif
    {
#ifdef SPLATT_CAS
      for(idx_t f=0; f < NFACTORS; ++f) {
        double old_leafrow, new_leafrow;
        do {
          old_leafrow = leafrow[f];
          new_leafrow = old_leafrow + v * accumbuf[f];
        } while (!__sync_bool_compare_and_swap((long long *)(leafrow + f), *((long long *)(&old_leafrow)), *((long long *)(&new_leafrow))));
      }
#else
      splatt_set_lock(inds[jj]);
      for(idx_t f=0; f < NFACTORS; ++f) {
        leafrow[f] += v * accumbuf[f];
      }
      splatt_unset_lock(inds[jj]);
#endif
    }
  }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
  mttkrp_flops += 2*NFACTORS*(end - start);
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
}


template<int NFACTORS>
static inline void p_csf_process_fiber_nolock_(
  val_t * const leafmat,
  val_t const * const restrict accumbuf,
  idx_t const start,
  idx_t const end,
  fidx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  __assume_aligned(accumbuf, 64);
  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * NFACTORS);
    __assume_aligned(leafrow, 64);
    val_t const v = vals[jj];
    for(idx_t f=0; f < NFACTORS; ++f) {
      leafrow[f] += v * accumbuf[f];
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
}


template<int NFACTORS>
static inline void p_csf_process_fiber_(
  val_t * const restrict accumbuf,
  val_t const * const leafmat,
  idx_t const start,
  idx_t const end,
  fidx_t const * const inds,
  val_t const * const vals)
{
  /* foreach nnz in fiber */
  for(idx_t j=start; j < end; ++j) {
    val_t const v = vals[j] ;
    val_t const * const restrict row = leafmat + (NFACTORS * inds[j]);
    __assume_aligned(row, 64);
    __assume_aligned(accumbuf, 64);
    for(idx_t f=0; f < NFACTORS; ++f) {
      accumbuf[f] += v * row[f];
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
  val_t * const out,
  val_t * const * const buf,
  idx_t * const restrict idxstack,
  idx_t const init_depth,
  idx_t const init_idx,
  idx_t const * const * const fp,
  fidx_t const * const * const fids,
  val_t const * const restrict vals,
  val_t ** mvals,
  idx_t const nmodes)
{
  /* push initial idx initialize idxstack */
  idxstack[init_depth] = init_idx;
  for(idx_t m=init_depth+1; m < nmodes; ++m) {
    idxstack[m] = fp[m-1][idxstack[m-1]];
  }

  assert(init_depth < nmodes-1);

  /* clear out accumulation buffer */
  __assume_aligned(out, 64);
  for(idx_t f=0; f < NFACTORS; ++f) {
    __assume_aligned(buf[init_depth+1], 64);
    buf[init_depth+1][f] = 0;
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
      for(idx_t f=0; f < NFACTORS; ++f) {
        __assume_aligned(buf[depth+1], 64);
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
  for(idx_t f=0; f < NFACTORS; ++f) {
    __assume_aligned(buf[init_depth+1], 64);
    out[f] = buf[init_depth+1][f];
  }
}


template<int NFACTORS, bool TILED = false>
static void p_csf_mttkrp_root3_kernel_(
  const val_t *vals,
  const idx_t *sptr, const idx_t *fptr,
  const fidx_t *fids, const fidx_t *inds,
  const val_t *avals, const val_t *bvals, val_t *mv,
  val_t *accumF, val_t *accumO,
  idx_t s)
{
#ifdef SPLATT_INTRINSIC
#ifdef __AVX512F__
  __m512d accumO_v1, accumO_v2;
  if (TILED) {
    accumO_v1 = _mm512_load_pd(mv);
    accumO_v2 = _mm512_load_pd(mv + 8);
  }
  else {
    accumO_v1 = _mm512_setzero_pd();
    accumO_v2 = _mm512_setzero_pd();
  }
#else
  __m256d accumO_v1, accumO_v2, accumO_v3, accumO_v4;
  if (TILED) {
    accumO_v1 = _mm256_load_pd(mv);
    accumO_v2 = _mm256_load_pd(mv + 4);
    accumO_v3 = _mm256_load_pd(mv + 8);
    accumO_v4 = _mm256_load_pd(mv + 12);
  }
  else {
    accumO_v1 = _mm256_setzero_pd();
    accumO_v2 = _mm256_setzero_pd();
    accumO_v3 = _mm256_setzero_pd();
    accumO_v4 = _mm256_setzero_pd();
  }
#endif
#else
  if (TILED) {
    for(idx_t r=0; r < NFACTORS; ++r) {
      accumO[r] = mv[r];
    }
  }
  else {
    for(idx_t r=0; r < NFACTORS; ++r) {
      accumO[r] = 0;
    }
  }
#endif

  /* foreach fiber in slice */
  for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
    /* first entry of the fiber is used to initialize accumF */
    idx_t const jjfirst  = fptr[f];
    val_t const vfirst   = vals[jjfirst];
    val_t const * const restrict bv = bvals + (inds[jjfirst] * NFACTORS);
#ifdef SPLATT_SIM_CACHE
    Load(vals + jjfirst);
    Load(inds + jjfirst);
#endif
#ifdef SPLATT_INTRINSIC
#ifdef __AVX512F__
    __m512d accumF_v1 = _mm512_mul_pd(_mm512_set1_pd(vfirst), _mm512_load_pd(bv));
    __m512d accumF_v2 = _mm512_mul_pd(_mm512_set1_pd(vfirst), _mm512_load_pd(bv + 8));

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
      val_t const v = vals[jj];
      val_t const * const restrict bv = bvals + (inds[jj] * NFACTORS);
      accumF_v1 = _mm512_fmadd_pd(_mm512_set1_pd(v), _mm512_load_pd(bv), accumF_v1);
      accumF_v2 = _mm512_fmadd_pd(_mm512_set1_pd(v), _mm512_load_pd(bv + 8), accumF_v2);
    }

    /* scale inner products by row of A and update to M */
    val_t const * const restrict av = avals  + (fids[f] * NFACTORS);
    accumO_v1 = _mm512_fmadd_pd(accumF_v1, _mm512_load_pd(av), accumO_v1);
    accumO_v2 = _mm512_fmadd_pd(accumF_v2, _mm512_load_pd(av + 8), accumO_v2);
#else
    __m256d accumF_v1 = _mm256_mul_pd(_mm256_set1_pd(vfirst), _mm256_load_pd(bv));
    __m256d accumF_v2 = _mm256_mul_pd(_mm256_set1_pd(vfirst), _mm256_load_pd(bv + 4));
    __m256d accumF_v3 = _mm256_mul_pd(_mm256_set1_pd(vfirst), _mm256_load_pd(bv + 8));
    __m256d accumF_v4 = _mm256_mul_pd(_mm256_set1_pd(vfirst), _mm256_load_pd(bv + 12));

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
      val_t const v = vals[jj];
      val_t const * const restrict bv = bvals + (inds[jj] * NFACTORS);
      accumF_v1 = _mm256_fmadd_pd(_mm256_set1_pd(v), _mm256_load_pd(bv), accumF_v1);
      accumF_v2 = _mm256_fmadd_pd(_mm256_set1_pd(v), _mm256_load_pd(bv + 4), accumF_v2);
      accumF_v3 = _mm256_fmadd_pd(_mm256_set1_pd(v), _mm256_load_pd(bv + 8), accumF_v3);
      accumF_v4 = _mm256_fmadd_pd(_mm256_set1_pd(v), _mm256_load_pd(bv + 12), accumF_v4);
    }

    /* scale inner products by row of A and update to M */
    val_t const * const restrict av = avals  + (fids[f] * NFACTORS);
    accumO_v1 = _mm256_fmadd_pd(accumF_v1, _mm256_load_pd(av), accumO_v1);
    accumO_v2 = _mm256_fmadd_pd(accumF_v2, _mm256_load_pd(av + 4), accumO_v2);
    accumO_v3 = _mm256_fmadd_pd(accumF_v3, _mm256_load_pd(av + 8), accumO_v3);
    accumO_v4 = _mm256_fmadd_pd(accumF_v4, _mm256_load_pd(av + 12), accumO_v4);
#endif
#else // SPLATT_INTRINSIC
    for(idx_t r=0; r < NFACTORS; ++r) {
      accumF[r] = vfirst * bv[r];
#ifdef SPLATT_SIM_CACHE
      Load(bv + r);
#endif
    }

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
      val_t const v = vals[jj];
      val_t const * const restrict bv = bvals + (inds[jj] * NFACTORS);
#ifdef SPLATT_SIM_CACHE
      Load(vals + jj);
      Load(inds + jj);
#endif
      for(idx_t r=0; r < NFACTORS; ++r) {
        accumF[r] += v * bv[r];
#ifdef SPLATT_SIM_CACHE
        Load(bv + r);
#endif
      }
    }

    /* scale inner products by row of A and update to M */
    val_t const * const restrict av = avals  + (fids[f] * NFACTORS);
    for(idx_t r=0; r < NFACTORS; ++r) {
      accumO[r] += accumF[r] * av[r];
#ifdef SPLATT_SIM_CACHE
      Load(av + r);
#endif
    }
#endif // SPLATT_INTRINSIC
  }

#ifdef SPLATT_INTRINSIC
#ifdef __AVX512F__
  _mm512_stream_pd(mv, accumO_v1);
  _mm512_stream_pd(mv + 8, accumO_v2);
#else
  _mm256_stream_pd(mv, accumO_v1);
  _mm256_stream_pd(mv + 4, accumO_v2);
  _mm256_stream_pd(mv + 8, accumO_v3);
  _mm256_stream_pd(mv + 12, accumO_v4);
#endif
#else
#pragma vector nontemporal(mv)
  for(idx_t r=0; r < NFACTORS; ++r) {
    mv[r] = accumO[r];
#ifdef SPLATT_SIM_CACHE
    Load(mv + r);
#endif
  }
#endif
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
  val_t * const restrict accumO = (val_t *) thds[tid].scratch[2];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

#ifdef SPLATT_NFACTOR_COMPILE_CONSTANT
  if (nfactors == 16) {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      val_t * const restrict mv = ovals + (fid * nfactors);

      p_csf_mttkrp_root3_kernel_<16, true>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, accumF, accumO, s);
    }
  }
  else
#endif
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
double barrierTimes[1024];
#endif

template<int NFACTORS>
static void p_csf_mttkrp_root3_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
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

  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();

  val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
  val_t * const restrict accumO = (val_t *) thds[tid].scratch[2];

#ifdef SPLATT_SIM_CACHE
  static int simulation_cnt = 0;
#pragma omp barrier
#pragma omp master
  if (simulation_cnt < 3) {
    simulateCache = true;
    for (int i = 0; i < nthreads; ++i) {
      tidsToSimulate.push_back(i);
    }
    RegisterRegion((void *)avals, (void *)(avals + ct->dims[ct->dim_perm[1]]*NFACTORS), "avals");
    RegisterRegion((void *)bvals, (void *)(bvals + ct->dims[ct->dim_perm[2]]*NFACTORS), "bvals");
    RegisterRegion((void *)vals, (void *)(vals + ct->nnz), "vals");
    RegisterRegion((void *)inds, (void *)(inds + ct->nnz), "inds");
    Reset();
  }
#pragma omp barrier
#endif

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

#ifdef SPLATT_MEASURE_LOAD_BALANCE
#pragma omp barrier
  double tBegin = omp_get_wtime();
#endif

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
      val_t *mv = ovals + (fid * NFACTORS);
      p_csf_mttkrp_root3_kernel_<NFACTORS>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, accumF, accumO, s);
    }
  }
  else {
    idx_t const nrows = mats[MAX_NMODES]->I;
    idx_t rows_per_thread = (nrows + nthreads - 1)/nthreads;
    idx_t row_begin = SS_MIN(rows_per_thread*tid, nrows);
    idx_t row_end = SS_MIN(row_begin + rows_per_thread, nrows);

    idx_t sbegin = std::lower_bound(sids, sids + nslices, row_begin) - sids;

    for(idx_t s=sbegin; s < nslices && sids[s] < row_end; ++s) {
      idx_t fid = sids[s];
      val_t *mv = ovals + (fid * NFACTORS);
      p_csf_mttkrp_root3_kernel_<NFACTORS, true>(
        vals, sptr, fptr, fids, inds, avals, bvals, mv, accumF, accumO, s);
    }
  }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
  double t = omp_get_wtime();
#pragma omp barrier
  barrierTimes[tid] = omp_get_wtime() - t;

#pragma omp barrier
#pragma omp master
  {
    double tEnd = omp_get_wtime();
    double barrierTimeSum = 0;
    for (int i = 0; i < nthreads; ++i) {
      barrierTimeSum += barrierTimes[i];
    }
    printf("%f load imbalance = %f\n", tEnd - tBegin, barrierTimeSum/(tEnd - tBegin)/nthreads);
  }
#endif // SPLATT_MEASURE_LOAD_BALANCE
#ifdef SPLATT_SIM_CACHE
#pragma omp barrier
#pragma omp master
  if (simulation_cnt < 3) {
    Analyze();
    simulateCache = false;
    ++simulation_cnt;
    ClearRegions();
  }
#pragma omp barrier
#endif
}

static void p_csf_mttkrp_root3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  idx_t const nfactors = mats[MAX_NMODES]->J;

#ifdef SPLATT_NFACTOR_COMPILE_CONSTANT
  if (nfactors == 16) {
    p_csf_mttkrp_root3_<16>(ct, tile_id, mats, thds);
  }
  else
#endif
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

#ifdef SPLATT_ATOMIC_CACHE
int hit_cnts[256];
int miss_cnts[256];
int global_atomic_cache_keys[32*NBUCKETS];
__declspec(aligned(64)) val_t global_atomic_cache_values[32*NBUCKETS*16];
#endif

template<int NFACTORS, bool TILED = false>
static void p_csf_mttkrp_internal3_kernel_(
  const val_t *vals,
  const idx_t *sptr, const idx_t *fptr,
  const fidx_t *fids, const fidx_t *inds,
  const val_t *rv, const val_t *bvals, val_t *ovals,
  val_t *accumF,
  idx_t s)
{
  /* foreach fiber in slice */
  for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
    /* first entry of the fiber is used to initialize accumF */
    idx_t const jjfirst  = fptr[f];
    val_t const vfirst   = vals[jjfirst];
    val_t const * const restrict bv = bvals + (inds[jjfirst] * NFACTORS);
#ifdef SPLATT_INTRINSIC
#ifdef __AVX512F__
    __m512d accumF_v1 = _mm512_mul_pd(_mm512_set1_pd(vfirst), _mm512_load_pd(bv));
    __m512d accumF_v2 = _mm512_mul_pd(_mm512_set1_pd(vfirst), _mm512_load_pd(bv + 8));
#else
    __m256d accumF_v1 = _mm256_mul_pd(_mm256_set1_pd(vfirst), _mm256_load_pd(bv));
    __m256d accumF_v2 = _mm256_mul_pd(_mm256_set1_pd(vfirst), _mm256_load_pd(bv + 4));
    __m256d accumF_v3 = _mm256_mul_pd(_mm256_set1_pd(vfirst), _mm256_load_pd(bv + 8));
    __m256d accumF_v4 = _mm256_mul_pd(_mm256_set1_pd(vfirst), _mm256_load_pd(bv + 12));
#endif

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
      val_t const v = vals[jj];
      val_t const * const restrict bv = bvals + (inds[jj] * NFACTORS);
#ifdef __AVX512F__
      accumF_v1 = _mm512_fmadd_pd(_mm512_set1_pd(v), _mm512_load_pd(bv), accumF_v1);
      accumF_v2 = _mm512_fmadd_pd(_mm512_set1_pd(v), _mm512_load_pd(bv + 8), accumF_v2);
#else
      accumF_v1 = _mm256_fmadd_pd(_mm256_set1_pd(v), _mm256_load_pd(bv), accumF_v1);
      accumF_v2 = _mm256_fmadd_pd(_mm256_set1_pd(v), _mm256_load_pd(bv + 4), accumF_v2);
      accumF_v3 = _mm256_fmadd_pd(_mm256_set1_pd(v), _mm256_load_pd(bv + 8), accumF_v3);
      accumF_v4 = _mm256_fmadd_pd(_mm256_set1_pd(v), _mm256_load_pd(bv + 12), accumF_v4);
#endif
    }

    /* scale inner products by row of A and update to M */
    fidx_t oid = fids[f];
    val_t * const restrict ov = ovals  + (oid * NFACTORS);
    if (TILED) {
#ifdef __AVX512F__
        _mm512_store_pd(ov, _mm512_fmadd_pd(accumF_v1, _mm512_load_pd(rv), _mm512_load_pd(ov)));
        _mm512_store_pd(ov + 8, _mm512_fmadd_pd(accumF_v2, _mm512_load_pd(rv + 8), _mm512_load_pd(ov + 8)));
#else
        _mm256_store_pd(ov, _mm256_fmadd_pd(accumF_v1, _mm256_load_pd(rv), _mm256_load_pd(ov)));
        _mm256_store_pd(ov + 4, _mm256_fmadd_pd(accumF_v2, _mm256_load_pd(rv + 4), _mm256_load_pd(ov + 4)));
        _mm256_store_pd(ov + 8, _mm256_fmadd_pd(accumF_v3, _mm256_load_pd(rv + 8), _mm256_load_pd(ov + 8)));
        _mm256_store_pd(ov + 12, _mm256_fmadd_pd(accumF_v4, _mm256_load_pd(rv + 12), _mm256_load_pd(ov + 12)));
#endif
    } // TILED
    else {
#ifdef SPLATT_RTM
      if (_xbegin() == _XBEGIN_STARTED) {
        _mm256_store_pd(ov, _mm256_fmadd_pd(accumF_v1, _mm256_load_pd(rv), _mm256_load_pd(ov)));
        _mm256_store_pd(ov + 4, _mm256_fmadd_pd(accumF_v2, _mm256_load_pd(rv + 4), _mm256_load_pd(ov + 4)));
        _mm256_store_pd(ov + 8, _mm256_fmadd_pd(accumF_v3, _mm256_load_pd(rv + 8), _mm256_load_pd(ov + 8)));
        _mm256_store_pd(ov + 12, _mm256_fmadd_pd(accumF_v4, _mm256_load_pd(rv + 12), _mm256_load_pd(ov + 12)));
        _xend();
      }
      else
#endif
      {
#ifdef SPLATT_ATOMIC_CACHE
        int bucket_id = oid%NBUCKETS;
        val_t *bucket_v = atomic_cache_values + bucket_id*NFACTORS;
        int key = atomic_cache_keys[bucket_id];

        if (key == oid) {
          _mm256_store_pd(bucket_v, _mm256_fmadd_pd(accumF_v1, _mm256_load_pd(rv), _mm256_load_pd(bucket_v)));
          _mm256_store_pd(bucket_v + 4, _mm256_fmadd_pd(accumF_v2, _mm256_load_pd(rv + 4), _mm256_load_pd(bucket_v + 4)));
          _mm256_store_pd(bucket_v + 8, _mm256_fmadd_pd(accumF_v3, _mm256_load_pd(rv + 8), _mm256_load_pd(bucket_v + 8)));
          _mm256_store_pd(bucket_v + 12, _mm256_fmadd_pd(accumF_v4, _mm256_load_pd(rv + 12), _mm256_load_pd(bucket_v + 12)));

          //++hit_cnt;
        }
        else {
          if (key != -1) {
            val_t *ov2 = ovals + (key * NFACTORS);
            splatt_set_lock(key);
            _mm256_store_pd(ov2, _mm256_add_pd(_mm256_load_pd(ov2), _mm256_load_pd(bucket_v)));
            _mm256_store_pd(ov2 + 4, _mm256_add_pd(_mm256_load_pd(ov2 + 4), _mm256_load_pd(bucket_v + 4)));
            _mm256_store_pd(ov2 + 8, _mm256_add_pd(_mm256_load_pd(ov2 + 8), _mm256_load_pd(bucket_v + 8)));
            _mm256_store_pd(ov2 + 12, _mm256_add_pd(_mm256_load_pd(ov2 + 12), _mm256_load_pd(bucket_v + 12)));
            splatt_unset_lock(key);
          }

          _mm256_store_pd(bucket_v, _mm256_mul_pd(accumF_v1, _mm256_load_pd(rv)));
          _mm256_store_pd(bucket_v + 4, _mm256_mul_pd(accumF_v2, _mm256_load_pd(rv + 4)));
          _mm256_store_pd(bucket_v + 8, _mm256_mul_pd(accumF_v3, _mm256_load_pd(rv + 8)));
          _mm256_store_pd(bucket_v + 12, _mm256_mul_pd(accumF_v4, _mm256_load_pd(rv + 12)));
          atomic_cache_keys[bucket_id] = oid;

          //++miss_cnt;
        }
#elif defined(SPLATT_CAS)
        __m128d old_ov, new_ov, acc;

        do {
          old_ov = _mm_load_pd(ov);
#ifdef __AVX512F__
          acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v1), 0));
#else
          acc = _mm256_extractf128_pd(accumF_v1, 0);
#endif
          new_ov = _mm_fmadd_pd(_mm_load_pd(rv), acc, old_ov);
        } while (!__sync_bool_compare_and_swap((__int128 *)ov, *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#ifndef SPLATT_EMULATE_VECTOR_CAS
        do {
          old_ov = _mm_load_pd(ov + 2);
#ifdef __AVX512F__
          acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v1), 1));
#else
          acc = _mm256_extractf128_pd(accumF_v1, 1);
#endif
          new_ov = _mm_fmadd_pd(_mm_load_pd(rv + 2), acc, old_ov);
        } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 2), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

        do {
          old_ov = _mm_load_pd(ov + 4);
#ifdef __AVX512F__
          acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v1), 2));
#else
          acc = _mm256_extractf128_pd(accumF_v2, 0);
#endif
          new_ov = _mm_fmadd_pd(_mm_load_pd(rv + 4), acc, old_ov);
        } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 4), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
        do {
          old_ov = _mm_load_pd(ov + 6);
#ifdef __AVX512F__
          acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v1), 3));
#else
          acc = _mm256_extractf128_pd(accumF_v2, 1);
#endif
          new_ov = _mm_fmadd_pd(_mm_load_pd(rv + 6), acc, old_ov);
        } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 6), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#endif

        do {
          old_ov = _mm_load_pd(ov + 8);
#ifdef __AVX512F__
          acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v2), 0));
#else
          acc = _mm256_extractf128_pd(accumF_v3, 0);
#endif
          new_ov = _mm_fmadd_pd(_mm_load_pd(rv + 8), acc, old_ov);
        } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 8), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#ifndef SPLATT_EMULATE_VECTOR_CAS
        do {
          old_ov = _mm_load_pd(ov + 10);
#ifdef __AVX512F__
          acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v2), 1));
#else
          acc = _mm256_extractf128_pd(accumF_v3, 1);
#endif
          new_ov = _mm_fmadd_pd(_mm_load_pd(rv + 10), acc, old_ov);
        } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 10), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

        do {
          old_ov = _mm_load_pd(ov + 12);
#ifdef __AVX512F__
          acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v2), 2));
#else
          acc = _mm256_extractf128_pd(accumF_v4, 0);
#endif
          new_ov = _mm_fmadd_pd(_mm_load_pd(rv + 12), acc, old_ov);
        } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 12), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
        do {
          old_ov = _mm_load_pd(ov + 14);
#ifdef __AVX512F__
          acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v2), 3));
#else
          acc = _mm256_extractf128_pd(accumF_v4, 1);
#endif
          new_ov = _mm_fmadd_pd(_mm_load_pd(rv + 14), acc, old_ov);
        } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 14), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#endif
#else // !SPLATT_CAS
        splatt_set_lock(oid);
#ifdef __AVX512F__
        _mm512_store_pd(ov, _mm512_fmadd_pd(accumF_v1, _mm512_load_pd(rv), _mm512_load_pd(ov)));
        _mm512_store_pd(ov + 8, _mm512_fmadd_pd(accumF_v2, _mm512_load_pd(rv + 8), _mm512_load_pd(ov + 8)));
#else
        _mm256_store_pd(ov, _mm256_fmadd_pd(accumF_v1, _mm256_load_pd(rv), _mm256_load_pd(ov)));
        _mm256_store_pd(ov + 4, _mm256_fmadd_pd(accumF_v2, _mm256_load_pd(rv + 4), _mm256_load_pd(ov + 4)));
        _mm256_store_pd(ov + 8, _mm256_fmadd_pd(accumF_v3, _mm256_load_pd(rv + 8), _mm256_load_pd(ov + 8)));
        _mm256_store_pd(ov + 12, _mm256_fmadd_pd(accumF_v4, _mm256_load_pd(rv + 12), _mm256_load_pd(ov + 12)));
#endif
        splatt_unset_lock(oid);
#endif // !SPLATT_CAS
      }
    } // !TILED
#else // SPLATT_INTRINSIC
    for(idx_t r=0; r < NFACTORS; ++r) {
      accumF[r] = vfirst * bv[r];
    }

    /* foreach nnz in fiber */
    for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
      val_t const v = vals[jj];
      val_t const * const restrict bv = bvals + (inds[jj] * NFACTORS);
      for(idx_t r=0; r < NFACTORS; ++r) {
        accumF[r] += v * bv[r];
      }
    }

    /* write to fiber row */
    val_t * const restrict ov = ovals  + (fids[f] * NFACTORS);
    if (TILED) {
      for(idx_t r=0; r < NFACTORS; ++r) {
        ov[r] += rv[r] * accumF[r];
      }
    } // TILED
    else {
#ifdef SPLATT_RTM
      if (_xbegin() == _XBEGIN_STARTED) {
        for(idx_t r=0; r < NFACTORS; ++r) {
          ov[r] += rv[r] * accumF[r];
        }
        _xend();
      }
      else
#endif
      {
#ifdef SPLATT_CAS
        for(idx_t r=0; r < NFACTORS; ++r) {
          double old_ov, new_ov;
          do {
            old_ov = ov[r];
            new_ov = old_ov + rv[r]*accumF[r];
          } while (!__sync_bool_compare_and_swap((long long *)(ov + r), *((long long *)(&old_ov)), *((long long *)(&new_ov))));
        }
#else
        splatt_set_lock(fids[f]);
        for(idx_t r=0; r < NFACTORS; ++r) {
          ov[r] += rv[r] * accumF[r];
        }
        splatt_unset_lock(fids[f]);
#endif
      }
    } // !TILED
#endif // SPLATT_INTRINSIC
  }
}

template<int NFACTORS>
static void p_csf_mttkrp_internal3_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
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

  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();
  
  val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

#ifdef SPLATT_ATOMIC_CACHE
  int *atomic_cache_keys = global_atomic_cache_keys + tid*NBUCKETS;
  val_t *atomic_cache_values = global_atomic_cache_values + tid*NBUCKETS*NFACTORS;
  for (int i = 0; i < NBUCKETS; ++i) {
    atomic_cache_keys[i] = -1;
  }
  int hit_cnt = 0, miss_cnt = 0;
#endif

#ifdef SPLATT_MEASURE_LOAD_BALANCE
#pragma omp barrier
  double tBegin = omp_get_wtime();
#endif

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

      p_csf_mttkrp_internal3_kernel_<NFACTORS>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, accumF, s);
    }
  }
  else {
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      idx_t fid = sids[s];

      /* root row */
      val_t const * const restrict rv = avals + (fid * NFACTORS);

      p_csf_mttkrp_internal3_kernel_<NFACTORS>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, accumF, s);
    }
  }

#ifdef SPLATT_ATOMIC_CACHE
  for (int bucket_id = 0; bucket_id < NBUCKETS; ++bucket_id) {
    int key = atomic_cache_keys[bucket_id];
    if (key != -1) {
      val_t * const restrict ov = ovals  + (key * NFACTORS);
      val_t *bucket_v = atomic_cache_values + bucket_id*NFACTORS;

      splatt_set_lock(key);
      _mm256_store_pd(ov, _mm256_add_pd(_mm256_load_pd(ov), _mm256_load_pd(bucket_v)));
      _mm256_store_pd(ov + 4, _mm256_add_pd(_mm256_load_pd(ov + 4), _mm256_load_pd(bucket_v + 4)));
      _mm256_store_pd(ov + 8, _mm256_add_pd(_mm256_load_pd(ov + 8), _mm256_load_pd(bucket_v + 8)));
      _mm256_store_pd(ov + 12, _mm256_add_pd(_mm256_load_pd(ov + 12), _mm256_load_pd(bucket_v + 12)));
      splatt_unset_lock(key);
    }
  }
#endif

#ifdef SPLATT_MEASURE_LOAD_BALANCE
  double t = omp_get_wtime();
#pragma omp barrier
  barrierTimes[tid] = omp_get_wtime() - t;

#pragma omp barrier
#pragma omp master
  {
    double tEnd = omp_get_wtime();
    double barrierTimeSum = 0;
    for (int i = 0; i < nthreads; ++i) {
      barrierTimeSum += barrierTimes[i];
    }
    printf("%f load imbalance = %f\n", tEnd - tBegin, barrierTimeSum/(tEnd - tBegin)/nthreads);
  }
#endif // SPLATT_MEASURE_LOAD_BALANCE

#ifdef SPLATT_ATOMIC_CACHE
  hit_cnts[tid] = hit_cnt;
  miss_cnts[tid] = miss_cnt;
#pragma omp barrier
#pragma omp master
  {
    hit_cnt = 0;
    miss_cnt = 0;
    for (int i = 0; i < nthreads; ++i) {
      hit_cnt += hit_cnts[i];
      miss_cnt += miss_cnts[i];
    }
    printf("hit = %d miss = %d hit-rate = %f\n", hit_cnt, miss_cnt, (double)hit_cnt/(hit_cnt + miss_cnt));
  }
#endif
}

static void p_csf_mttkrp_internal3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  idx_t const nfactors = mats[MAX_NMODES]->J;

#ifdef SPLATT_NFACTOR_COMPILE_CONSTANT
  if (nfactors == 16) {
    p_csf_mttkrp_internal3_<16>(ct, tile_id, mats, thds);
  }
  else
#endif
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
        splatt_set_lock(fids[f]);
        for(idx_t r=0; r < nfactors; ++r) {
          ov[r] += rv[r] * accumF[r];
        }
        splatt_unset_lock(fids[f]);
      }
    }
  }
}

template<int NFACTORS, bool TILED = false>
static void p_csf_mttkrp_leaf3_kernel_(
  const val_t *vals,
  const idx_t *sptr, const idx_t *fptr,
  const fidx_t *fids, const fidx_t *inds,
  const val_t *rv, const val_t *bvals, val_t *ovals,
  val_t *accumF,
  idx_t s)
{
  /* foreach fiber in slice */
  for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
    /* fill fiber with hada */
    val_t const * const restrict av = bvals  + (fids[f] * NFACTORS);
#ifdef SPLATT_INTRINSIC
#ifdef __AVX512F__
    __m512d accumF_v1 = _mm512_mul_pd(_mm512_load_pd(rv), _mm512_load_pd(av));
    __m512d accumF_v2 = _mm512_mul_pd(_mm512_load_pd(rv + 8), _mm512_load_pd(av + 8));
#else
    __m256d accumF_v1 = _mm256_mul_pd(_mm256_load_pd(rv), _mm256_load_pd(av));
    __m256d accumF_v2 = _mm256_mul_pd(_mm256_load_pd(rv + 4), _mm256_load_pd(av + 4));
    __m256d accumF_v3 = _mm256_mul_pd(_mm256_load_pd(rv + 8), _mm256_load_pd(av + 8));
    __m256d accumF_v4 = _mm256_mul_pd(_mm256_load_pd(rv + 12), _mm256_load_pd(av + 12));
#endif

    /* foreach nnz in fiber, scale with hada and write to ovals */
    for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
      val_t const v = vals[jj];
      val_t * const restrict ov = ovals + (inds[jj] * NFACTORS);
      if (TILED) {
#ifdef __AVX512F__
        _mm512_store_pd(ov, _mm512_fmadd_pd(_mm512_set1_pd(v), accumF_v1, _mm512_load_pd(ov)));
        _mm512_store_pd(ov + 8, _mm512_fmadd_pd(_mm512_set1_pd(v), accumF_v2, _mm512_load_pd(ov + 8)));
#else
        _mm256_store_pd(ov, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v1, _mm256_load_pd(ov)));
        _mm256_store_pd(ov + 4, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v2, _mm256_load_pd(ov + 4)));
        _mm256_store_pd(ov + 8, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v3, _mm256_load_pd(ov + 8)));
        _mm256_store_pd(ov + 12, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v4, _mm256_load_pd(ov + 12)));
#endif
      }
      else {
#ifdef SPLATT_RTM
        if (_xbegin() == _XBEGIN_STARTED) {
          _mm256_store_pd(ov, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v1, _mm256_load_pd(ov)));
          _mm256_store_pd(ov + 4, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v2, _mm256_load_pd(ov + 4)));
          _mm256_store_pd(ov + 8, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v3, _mm256_load_pd(ov + 8)));
          _mm256_store_pd(ov + 12, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v4, _mm256_load_pd(ov + 12)));
          _xend();
        }
        else
#endif
        {
#ifdef SPLATT_CAS
          __m128d old_ov, new_ov, acc;

          do {
            old_ov = _mm_load_pd(ov);
#ifdef __AVX512F__
            acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v1), 0));
#else
            acc = _mm256_extractf128_pd(accumF_v1, 0);
#endif
            new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
          } while (!__sync_bool_compare_and_swap((__int128 *)ov, *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#ifndef SPLATT_EMULATE_VECTOR_CAS
          do {
            old_ov = _mm_load_pd(ov + 2);
#ifdef __AVX512F__
            acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v1), 1));
#else
            acc = _mm256_extractf128_pd(accumF_v1, 1);
#endif
            new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
          } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 2), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

          do {
            old_ov = _mm_load_pd(ov + 4);
#ifdef __AVX512F__
            acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v1), 2));
#else
            acc = _mm256_extractf128_pd(accumF_v2, 0);
#endif
            new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
          } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 4), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
          do {
            old_ov = _mm_load_pd(ov + 6);
#ifdef __AVX512F__
            acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v1), 3));
#else
            acc = _mm256_extractf128_pd(accumF_v2, 1);
#endif
            new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
          } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 6), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#endif

          do {
            old_ov = _mm_load_pd(ov + 8);
#ifdef __AVX512F__
            acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v2), 0));
#else
            acc = _mm256_extractf128_pd(accumF_v3, 0);
#endif
            new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
          } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 8), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#ifndef SPLATT_EMULATE_VECTOR_CAS
          do {
            old_ov = _mm_load_pd(ov + 10);
#ifdef __AVX512F__
            acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v2), 1));
#else
            acc = _mm256_extractf128_pd(accumF_v3, 1);
#endif
            new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
          } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 10), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));

          do {
            old_ov = _mm_load_pd(ov + 12);
#ifdef __AVX512F__
            acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v2), 2));
#else
            acc = _mm256_extractf128_pd(accumF_v4, 0);
#endif
            new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
          } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 12), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
          do {
            old_ov = _mm_load_pd(ov + 14);
#ifdef __AVX512F__
            acc = _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(accumF_v2), 3));
#else
            acc = _mm256_extractf128_pd(accumF_v4, 1);
#endif
            new_ov = _mm_fmadd_pd(_mm_set1_pd(v), acc, old_ov);
          } while (!__sync_bool_compare_and_swap((__int128 *)(ov + 14), *((__int128 *)&old_ov), *((__int128 *)&new_ov)));
#endif
#else
          splatt_set_lock(inds[jj]);
#ifdef __AVX512F__
          _mm512_store_pd(ov, _mm512_fmadd_pd(_mm512_set1_pd(v), accumF_v1, _mm512_load_pd(ov)));
          _mm512_store_pd(ov + 8, _mm512_fmadd_pd(_mm512_set1_pd(v), accumF_v2, _mm512_load_pd(ov + 8)));
#else
          _mm256_store_pd(ov, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v1, _mm256_load_pd(ov)));
          _mm256_store_pd(ov + 4, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v2, _mm256_load_pd(ov + 4)));
          _mm256_store_pd(ov + 8, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v3, _mm256_load_pd(ov + 8)));
          _mm256_store_pd(ov + 12, _mm256_fmadd_pd(_mm256_set1_pd(v), accumF_v4, _mm256_load_pd(ov + 12)));
#endif
          splatt_unset_lock(inds[jj]);
#endif
        }
      } // !TILED
    }

#else // SPLATT_INTRINSIC
    for(idx_t r=0; r < NFACTORS; ++r) {
      accumF[r] = rv[r] * av[r];
    }

    /* foreach nnz in fiber, scale with hada and write to ovals */
    for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
      val_t const v = vals[jj];
      val_t * const restrict ov = ovals + (inds[jj] * NFACTORS);
      if (TILED) {
        for(idx_t r=0; r < NFACTORS; ++r) {
          ov[r] += v * accumF[r];
        }
      }
      else {
#ifdef SPLATT_RTM
        if (_xbegin() == _XBEGIN_STARTED) {
          for(idx_t r=0; r < NFACTORS; ++r) {
            ov[r] += v * accumF[r];
          }
          _xend();
        }
        else
#endif
        {
#ifdef SPLATT_CAS
          for(idx_t r=0; r < NFACTORS; ++r) {
            double old_ov, new_ov;
            do {
              old_ov = ov[r];
              new_ov = old_ov + v*accumF[r];
            } while (!__sync_bool_compare_and_swap((long long *)(ov + r), *((long long *)(&old_ov)), *((long long *)(&new_ov))));
          }
#else
          splatt_set_lock(inds[jj]);
          for(idx_t r=0; r < NFACTORS; ++r) {
            ov[r] += v * accumF[r];
          }
          splatt_unset_lock(inds[jj]);
#endif
        }
      } // !TILED
    }
#endif // SPLATT_INTRINSIC
  }
}

template<int NFACTORS>
static void p_csf_mttkrp_leaf3_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
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

  val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

#ifdef SPLATT_MEASURE_LOAD_BALANCE
#pragma omp barrier
  double tBegin = omp_get_wtime();
#endif

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
      idx_t const fid = s;

      /* root row */
      val_t const * const restrict rv = avals + (fid * NFACTORS);

      p_csf_mttkrp_leaf3_kernel_<NFACTORS>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, accumF, s);
    }
  }
  else {
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = sids[s];

      /* root row */
      val_t const * const restrict rv = avals + (fid * NFACTORS);

      p_csf_mttkrp_leaf3_kernel_<NFACTORS>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, accumF, s);
    }
  }

#ifdef SPLATT_MEASURE_LOAD_BALANCE
  double t = omp_get_wtime();
#pragma omp barrier
  barrierTimes[tid] = omp_get_wtime() - t;

#pragma omp barrier
#pragma omp master
  {
    double tEnd = omp_get_wtime();
    double barrierTimeSum = 0;
    for (int i = 0; i < nthreads; ++i) {
      barrierTimeSum += barrierTimes[i];
    }
    printf("%f load imbalance = %f\n", tEnd - tBegin, barrierTimeSum/(tEnd - tBegin)/nthreads);
  }
#endif // SPLATT_MEASURE_LOAD_BALANCE
}


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

#ifdef SPLATT_NFACTOR_COMPILE_CONSTANT
  if (nfactors == 16) {
    p_csf_mttkrp_leaf3_<16>(ct, tile_id, mats, thds);
  }
  else
#endif
  {
    val_t * const restrict accumF
        = (val_t *) thds[omp_get_thread_num()].scratch[0];

    idx_t const nslices = ct->pt[tile_id].nfibs[0];
    #pragma omp for schedule(dynamic, 16) nowait
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

        /* foreach nnz in fiber, scale with hada and write to ovals */
        for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t * const restrict ov = ovals + (inds[jj] * nfactors);
          splatt_set_lock(inds[jj]);
          for(idx_t r=0; r < nfactors; ++r) {
            ov[r] += v * accumF[r];
          }
          splatt_unset_lock(inds[jj]);
        }
      }
    }
  }
}


static void p_csf_mttkrp_root_tiled(
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
}



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
    p_csf_mttkrp_root3(ct, tile_id, mats, thds);
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
      fidx_t const fid = s;

      assert(fid < mats[MAX_NMODES]->I);

      if(16 == nfactors) {
        p_propagate_up_<16>(buf[0], buf, idxstack, 0, s, fp, fids,
            vals, mvals, nmodes);
      }
      else {
        p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids,
            vals, mvals, nmodes, nfactors);
      }

      val_t * const restrict orow = ovals + (fid * nfactors);
      val_t const * const restrict obuf = buf[0];
      for(idx_t f=0; f < nfactors; ++f) {
        orow[f] += obuf[f];
      }
    } /* end foreach outer slice */
  }
  else {
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

      if(16 == nfactors) {
        p_propagate_up_<16>(buf[0], buf, idxstack, 0, s, fp, fids,
            vals, mvals, nmodes);
      }
      else {
        p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids,
            vals, mvals, nmodes, nfactors);
      }

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
  }
}


static void p_csf_mttkrp_leaf_tiled3(
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

  val_t * const restrict accumF
      = (val_t *) thds[omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

#ifdef SPLATT_NFACTOR_COMPILE_CONSTANT
  if (nfactors == 16) {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      /* root row */
      val_t const * const restrict rv = avals + (fid * nfactors);

      p_csf_mttkrp_leaf3_kernel_<16, true>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, accumF, s);
    }
  }
  else
#endif
  {
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

        /* foreach nnz in fiber, scale with hada and write to ovals */
        for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t * const restrict ov = ovals + (inds[jj] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            ov[r] += v * accumF[r];
          }
        }
      }
    }
  }
}


static void p_csf_mttkrp_leaf_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf_tiled3(ct, tile_id, mats, thds);
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

  /* foreach outer slice */
  idx_t const nouter = ct->pt[tile_id].nfibs[0];
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
      p_csf_process_fiber_nolock(mats[MAX_NMODES]->vals, buf[depth],
          nfactors, start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */
}


template<int NFACTORS>
static void p_csf_mttkrp_leaf_tiled_(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf_tiled3(ct, tile_id, mats, thds);
    return;
  }

  /* extract tensor structures */
  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (NFACTORS * m);
  }

  /* foreach outer slice */
  idx_t const nouter = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nouter; ++s) {
    fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    idxstack[0] = s;

    /* clear out stale data */
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const rootrow = mvals[0] + (fid*NFACTORS);
    val_t * const rootbuf = buf[0];
    __assume_aligned(rootrow, 64);
    __assume_aligned(rootbuf, 64);
    for(idx_t f=0; f < NFACTORS; ++f) {
      rootbuf[f] = rootrow[f];
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
      p_csf_process_fiber_nolock_<NFACTORS>(mats[MAX_NMODES]->vals, buf[depth],
          start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */
}


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
    p_csf_mttkrp_leaf3(ct, tile_id, mats, thds);
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
      p_csf_process_fiber_lock(mats[MAX_NMODES]->vals, buf[depth],
          nfactors, start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */
}


template<int NFACTORS>
static void p_csf_mttkrp_leaf_(
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
    p_csf_mttkrp_leaf3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (NFACTORS * m);
  }

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    idxstack[0] = s;

    /* clear out stale data */
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const restrict rootrow = mvals[0] + (fid*NFACTORS);
    val_t * const rootbuf = buf[0];
    __assume_aligned(rootrow, 64);
    __assume_aligned(rootbuf, 64);
    for(idx_t f=0; f < NFACTORS; ++f) {
      rootbuf[f] = rootrow[f];
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
      p_csf_process_fiber_lock_<NFACTORS>(mats[MAX_NMODES]->vals, buf[depth],
          start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */
}


static void p_csf_mttkrp_internal_tiled3(
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
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];

#ifdef SPLATT_NFACTOR_COMPILE_CONSTANT
  if (nfactors == 16) {
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];

      /* root row */
      val_t const * const restrict rv = avals + (fid * nfactors);

      p_csf_mttkrp_internal3_kernel_<16, true>(
        vals, sptr, fptr, fids, inds, rv, bvals, ovals, accumF, s);
    }
  }
  else
#endif
  {
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
        for(idx_t r=0; r < nfactors; ++r) {
          ov[r] += rv[r] * accumF[r];
        }
      }
    }
  }
}


static void p_csf_mttkrp_internal_tiled(
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
    p_csf_mttkrp_internal_tiled3(ct, tile_id, mats, thds);
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
}


template<int NFACTORS>
static void p_csf_mttkrp_internal_tiled_(
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
    p_csf_mttkrp_internal_tiled3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (NFACTORS * m);
    memset(buf[m], 0, NFACTORS * sizeof(val_t));
  }
  val_t * const ovals = mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nslices; ++s) {
    fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    /* push outer slice and fill stack */
    idxstack[0] = s;
    for(idx_t m=1; m <= outdepth; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* fill first buf */
    val_t const * const restrict rootrow = mvals[0] + (fid*NFACTORS);
    __assume_aligned(buf[0], 64);
    for(idx_t f=0; f < NFACTORS; ++f) {
      buf[0][f] = rootrow[f];
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
}


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
    p_csf_mttkrp_internal3(ct, tile_id, mats, thds);
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
      splatt_set_lock(noderow);
      p_add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);
      splatt_unset_lock(noderow);

      /* backtrack to next unfinished node */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end foreach outer slice */
}


template<int NFACTORS>
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
    p_csf_mttkrp_internal3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  fidx_t const * const * const restrict fids
      = (fidx_t const * const *) ct->pt[tile_id].fids;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (NFACTORS * m);
    memset(buf[m], 0, NFACTORS * sizeof(val_t));
  }
  val_t * const ovals = mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  if(NULL == fids[0]) {
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      fidx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

      /* push outer slice and fill stack */
      idxstack[0] = s;
      for(idx_t m=1; m <= outdepth; ++m) {
        idxstack[m] = fp[m-1][idxstack[m-1]];
      }

      /* fill first buf */
      val_t const * const restrict rootrow = mvals[0] + (fid*NFACTORS);
      __assume_aligned(buf[0], 64);
      __assume_aligned(rootrow, 64);
      for(idx_t f=0; f < NFACTORS; ++f) {
        buf[0][f] = rootrow[f];
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
          val_t * const restrict out = outbuf;
          val_t * const restrict a = buf[outdepth];
          val_t const * const restrict b = buf[outdepth-1];

          __assume_aligned(out, 64);
          __assume_aligned(a, 64);
          __assume_aligned(b, 64);

#ifdef SPLATT_RTM
          if(_XBEGIN_STARTED == _xbegin()) {
            for(idx_t f=0; f < NFACTORS; ++f) {
              out[f] += a[f]*b[f];
            }
          }
          else
#endif
          {
#ifdef SPLATT_CAS
            __m128d old_out, new_out, acc;

            do {
              old_out = _mm_load_pd(out);
              new_out = _mm_fmadd_pd(_mm_load_pd(a), _mm_load_pd(b), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)out, *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 2);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 2), _mm_load_pd(b + 2), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 2), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 4);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 4), _mm_load_pd(b + 4), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 4), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 6);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 6), _mm_load_pd(b + 6), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 6), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 8);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 8), _mm_load_pd(b + 8), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 8), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 10);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 10), _mm_load_pd(b + 10), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 10), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 12);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 12), _mm_load_pd(b + 12), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 12), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 14);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 14), _mm_load_pd(b + 14), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 14), *((__int128 *)&old_out), *((__int128 *)&new_out)));

#else /* !SPLATT_CAS */
            splatt_set_lock(noderow);
            for(idx_t f=0; f < NFACTORS; ++f) {
              out[f] += a[f]*b[f];
            }
            splatt_unset_lock(noderow);
#endif /* !SPLATT_CAS */
          }

          for(idx_t f=0; f < NFACTORS; ++f) {
            a[f] = 0;
          }
#ifdef SPLATT_COUNT_FLOP
#pragma omp atomic
          mttkrp_flops += 2*NFACTORS;
#endif
        }
        else {
          splatt_set_lock(noderow);
          p_add_hada_clear_<NFACTORS>(outbuf, buf[outdepth], buf[outdepth-1]);
          splatt_unset_lock(noderow);
        }

        /* backtrack to next unfinished node */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
    } /* end foreach outer slice */
  }
  else {
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
      __assume_aligned(buf[0], 64);
      __assume_aligned(rootrow, 64);
      for(idx_t f=0; f < NFACTORS; ++f) {
        buf[0][f] = rootrow[f];
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
          val_t * const restrict out = outbuf;
          val_t * const restrict a = buf[outdepth];
          val_t const * const restrict b = buf[outdepth-1];

          __assume_aligned(out, 64);
          __assume_aligned(a, 64);
          __assume_aligned(b, 64);

#ifdef SPLATT_RTM
          if(_XBEGIN_STARTED == _xbegin()) {
            for(idx_t f=0; f < NFACTORS; ++f) {
              out[f] += a[f]*b[f];
            }
          }
          else
#endif
          {
#ifdef SPLATT_CAS
            __m128d old_out, new_out, acc;

            do {
              old_out = _mm_load_pd(out);
              new_out = _mm_fmadd_pd(_mm_load_pd(a), _mm_load_pd(b), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)out, *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 2);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 2), _mm_load_pd(b + 2), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 2), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 4);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 4), _mm_load_pd(b + 4), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 4), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 6);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 6), _mm_load_pd(b + 6), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 6), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 8);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 8), _mm_load_pd(b + 8), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 8), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 10);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 10), _mm_load_pd(b + 10), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 10), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 12);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 12), _mm_load_pd(b + 12), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 12), *((__int128 *)&old_out), *((__int128 *)&new_out)));

            do {
              old_out = _mm_load_pd(out + 14);
              new_out = _mm_fmadd_pd(_mm_load_pd(a + 14), _mm_load_pd(b + 14), old_out);
            } while (!__sync_bool_compare_and_swap((__int128 *)(out + 14), *((__int128 *)&old_out), *((__int128 *)&new_out)));

#else /* !SPLATT_CAS */
            splatt_set_lock(noderow);
            for(idx_t f=0; f < NFACTORS; ++f) {
              out[f] += a[f]*b[f];
            }
            splatt_unset_lock(noderow);
#endif /* !SPLATT_CAS */
          }

          for(idx_t f=0; f < NFACTORS; ++f) {
            a[f] = 0;
          }
        }
        else {
          splatt_set_lock(noderow);
          p_add_hada_clear_<NFACTORS>(outbuf, buf[outdepth], buf[outdepth-1]);
          splatt_unset_lock(noderow);
        }

        /* backtrack to next unfinished node */
        do {
          ++idxstack[depth];
          --depth;
        } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      } /* end DFS */
    } /* end foreach outer slice */
  }
}


/* determine which function to call */
static void p_root_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;

  sp_timer_t time;
  timer_fstart(&time);

  matrix_t * const M = mats[MAX_NMODES];
#ifdef SPLATT_NFACTOR_COMPILE_CONSTANT
  if (nmodes != 3 || tensor->which_tile != SPLATT_NOTILE || M->J != 16)
#endif
    par_memset(M->vals, 0, M->I * M->J * sizeof(val_t));

#ifdef SPLATT_COUNT_FLOP
  mttkrp_flops = 0; 
#endif

  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);
    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      p_csf_mttkrp_root(tensor, 0, mats, thds);
      break;
    case SPLATT_DENSETILE:
      /* this mode may not be tiled due to minimum tiling depth */
      if(opts[SPLATT_OPTION_TILEDEPTH] > 0) {
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          p_csf_mttkrp_root(tensor, t, mats, thds);
          //#pragma omp barrier
        }
      } else {
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
      break;

    /* XXX */
    case SPLATT_SYNCTILE:
      assert(false);
      break;
    case SPLATT_COOPTILE:
      assert(false);
      break;
    }
    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */

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
    printf("       csf_mttkrp_root (%0.3fs, %.3f GBps, %.3f GFs)\n",
        time.seconds, gbps, (double)mttkrp_flops/1e9);
#else
    printf("       csf_mttkrp_root (%0.3fs, %.3f GBps)\n",
        time.seconds, gbps);
#endif
  }
}


static void p_leaf_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  idx_t const depth = nmodes - 1;

  sp_timer_t time;
  timer_fstart(&time);

  matrix_t * const M = mats[MAX_NMODES];
  par_memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  idx_t const nfactors = mats[0]->J;

#ifdef SPLATT_COUNT_FLOP
  mttkrp_flops = 0; 
#endif

  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);

    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      if(16 == nfactors) {
        p_csf_mttkrp_leaf_<16>(tensor, 0, mats, thds);
      }
      else {
        p_csf_mttkrp_leaf(tensor, 0, mats, thds);
      }
      break;
    case SPLATT_DENSETILE:
      /* this mode may not be tiled due to minimum tiling depth */
      if(opts[SPLATT_OPTION_TILEDEPTH] > depth) {
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          if(16 == nfactors) {
            p_csf_mttkrp_leaf_<16>(tensor, 0, mats, thds);
          }
          else {
            p_csf_mttkrp_leaf(tensor, 0, mats, thds);
          }
        }
      } else {
        #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
          tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
              mode, t);
          while(tid != TILE_END) {
            if(16 == nfactors) {
              p_csf_mttkrp_leaf_tiled_<16>(tensor, tid, mats, thds);
            }
            else {
              p_csf_mttkrp_leaf_tiled(tensor, tid, mats, thds);
            }
            tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
          }
        }
      }
      break;

    /* XXX */
    case SPLATT_SYNCTILE:
      break;
    case SPLATT_COOPTILE:
      break;
    }
    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */

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
    printf("       csf_mttkrp_leaf (%0.3fs, %.3f GBps, %.3f GFs)\n",
        time.seconds, gbps, (double)mttkrp_flops/1e9);
#else
    printf("       csf_mttkrp_leaf (%0.3fs, %.3f GBps)\n",
        time.seconds, gbps);
#endif
  }
}


static void p_intl_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  idx_t const depth = csf_mode_depth(mode, tensor->dim_perm, nmodes);

  sp_timer_t time;
  timer_fstart(&time);

  matrix_t * const M = mats[MAX_NMODES];
  par_memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  idx_t const nfactors = mats[0]->J;

#ifdef SPLATT_COUNT_FLOP
  mttkrp_flops = 0; 
#endif

  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);
    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      if(16 == nfactors) {
        p_csf_mttkrp_internal_<16>(tensor, 0, mats, mode, thds);
      }
      else {
        p_csf_mttkrp_internal(tensor, 0, mats, mode, thds);
      }
      break;
    case SPLATT_DENSETILE:
      /* this mode may not be tiled due to minimum tiling depth */
      if(opts[SPLATT_OPTION_TILEDEPTH] > depth) {
        #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          if(16 == nfactors) {
            p_csf_mttkrp_internal_<16>(tensor, t, mats, mode, thds);
          }
          else {
            p_csf_mttkrp_internal(tensor, t, mats, mode, thds);
          }
        }
      } else {
        #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
          tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
              mode, t);
          while(tid != TILE_END) {
            if(16 == nfactors) {
              p_csf_mttkrp_internal_tiled_<16>(tensor, tid, mats, mode, thds);
            }
            else {
              p_csf_mttkrp_internal_tiled(tensor, tid, mats, mode, thds);
            }
            tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
          }
        }
      }
      break;

    /* XXX */
    case SPLATT_SYNCTILE:
      assert(false);
      break;
    case SPLATT_COOPTILE:
      assert(false);
      break;
    }

    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */

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
    printf("       csf_mttkrp_internal (%0.3fs, %.3f GBps, %.3f GFs)\n",
        time.seconds, gbps, (double)mttkrp_flops/1e9);
#else
    printf("       csf_mttkrp_internal (%0.3fs, %.3f GBps)\n",
        time.seconds, gbps);
#endif
  }
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mttkrp_csf(
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
      p_root_decide(tensors+0, mats, mode, thds, opts);
    } else if(outdepth == nmodes - 1) {
      p_leaf_decide(tensors+0, mats, mode, thds, opts);
    } else {
      p_intl_decide(tensors+0, mats, mode, thds, opts);
    }
    break;

  case SPLATT_CSF_TWOMODE:
    /* longest mode handled via second tensor's root */
    if(mode == tensors[0].dim_perm[nmodes-1]) {
      p_root_decide(tensors+1, mats, mode, thds, opts);
    /* root and internal modes are handled via first tensor */
    } else {
      outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
      if(outdepth == 0) {
        p_root_decide(tensors+0, mats, mode, thds, opts);
      } else {
        p_intl_decide(tensors+0, mats, mode, thds, opts);
      }
    }
    break;

  case SPLATT_CSF_ALLMODE:
    p_root_decide(tensors+mode, mats, mode, thds, opts);
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
  val_t const * const restrict vals = ft->vals;

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
  val_t const * const restrict vals = ft->vals;

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
  val_t const * const restrict vals = ft->vals;

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

  val_t const * const restrict vals = tt->vals;

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

  val_t const * const restrict vals = tt->vals;

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





/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <omp.h>
#include "base.h"
#include "util.h"

#include "mkl.h"

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
val_t rand_val(void)
{
  /* TODO: modify this to work based on the size of idx_t */
  val_t v =  3.0 * ((val_t) rand() / (val_t) RAND_MAX);
  if(rand() % 2 == 0) {
    v *= -1;
  }
  return v;
}


idx_t rand_idx(void)
{
  /* TODO: modify this to work based on the size of idx_t */
  return (idx_t) (rand() << 16) | rand();
}


static int vsl_stream_initialized[4096];
static VSLStreamStatePtr vsl_streams[4096];

void fill_rand(
  val_t * const restrict vals,
  idx_t const nelems)
{
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    idx_t i_per_thread = (nelems + nthreads - 1)/nthreads;
    idx_t i_begin = SS_MIN(i_per_thread*tid, nelems);
    idx_t i_end = SS_MIN(i_begin + i_per_thread, nelems);

    if (!vsl_stream_initialized[tid]) {
      vslNewStream(&vsl_streams[tid], VSL_BRNG_SFMT19937, 0);
      vsl_stream_initialized[tid] = 1;
    }

    vslSkipAheadStream(vsl_streams[tid], i_begin);

    vdRngUniform(
      VSL_RNG_METHOD_UNIFORM_STD, vsl_streams[tid],
      i_end - i_begin, vals + i_begin, -3, 3);

    vslSkipAheadStream(vsl_streams[tid], nelems - i_end);
  }
}


char * bytes_str(
  size_t const bytes)
{
  double size = (double)bytes;
  int suff = 0;
  const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
  while(size > 1024 && suff < 5) {
    size /= 1024.;
    ++suff;
  }
  char * ret = NULL;
  if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    fprintf(stderr, "SPLATT: asprintf failed with%"SPLATT_PF_IDX" bytes.\n", bytes);
    ret = NULL;
  }
  return ret;
}



idx_t argmax_elem(
  idx_t const * const arr,
  idx_t const N)
{
  idx_t mkr = 0;
  for(idx_t i=1; i < N; ++i) {
    if(arr[i] > arr[mkr]) {
      mkr = i;
    }
  }
  return mkr;
}


idx_t argmin_elem(
  idx_t const * const arr,
  idx_t const N)
{
  idx_t mkr = 0;
  for(idx_t i=1; i < N; ++i) {
    if(arr[i] < arr[mkr]) {
      mkr = i;
    }
  }
  return mkr;
}

void par_memcpy(void *dst, const void *src, size_t n)
{
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    size_t n_per_thread = (n + nthreads - 1)/nthreads;
    size_t n_begin = SS_MIN(n_per_thread*tid, n);
    size_t n_end = SS_MIN(n_begin + n_per_thread, n);

    memcpy((char *)dst + n_begin, (char *)src + n_begin, n_end - n_begin);
  }
}

void par_memset(void *ptr, int c, size_t n)
{
#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    size_t n_per_thread = (n + nthreads - 1)/nthreads;
    size_t n_begin = SS_MIN(n_per_thread*tid, n);
    size_t n_end = SS_MIN(n_begin + n_per_thread, n);

    memset((char *)ptr + n_begin, c, n_end - n_begin);
  }
}

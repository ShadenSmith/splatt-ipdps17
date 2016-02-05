

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "util.h"
#include "timer.h"
#include <mkl.h>

#include <math.h>
#include <omp.h>


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static void p_mat_2norm(
  matrix_t * const A,
  val_t * const restrict lambda,
  rank_info * const rinfo,
  thd_info * const thds)
{
  idx_t const I = A->I;
  idx_t const J = A->J;
  val_t * const restrict vals = A->vals;

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const mylambda = (val_t *) thds[tid].scratch[0];
    for(idx_t j=0; j < J; ++j) {
      mylambda[j] = 0;
    }

    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        mylambda[j] += vals[j + (i*J)] * vals[j + (i*J)];
      }
    }

    /* do reduction on partial sums */
    thd_reduce(thds, 0, J, REDUCE_SUM);

    #pragma omp master
    {
#ifdef SPLATT_USE_MPI
      /* now do an MPI reduction to get the global lambda */
      timer_start(&timers[TIMER_MPI_NORM]);
      timer_start(&timers[TIMER_MPI_IDLE]);
      MPI_Barrier(rinfo->comm_3d);
      timer_stop(&timers[TIMER_MPI_IDLE]);

      timer_start(&timers[TIMER_MPI_COMM]);
      MPI_Allreduce(mylambda, lambda, J, SPLATT_MPI_VAL, MPI_SUM, rinfo->comm_3d);
      timer_stop(&timers[TIMER_MPI_COMM]);
      timer_stop(&timers[TIMER_MPI_NORM]);
#else
      memcpy(lambda, mylambda, J * sizeof(val_t));
#endif
    }

    #pragma omp barrier

    #pragma omp for schedule(static)
    for(idx_t j=0; j < J; ++j) {
      lambda[j] = sqrt(lambda[j]);
    }

    /* do the normalization */
    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        vals[j+(i*J)] /= lambda[j];
      }
    }
  } /* end omp for */
}


static void p_mat_maxnorm(
  matrix_t * const A,
  val_t * const restrict lambda,
  rank_info * const rinfo,
  thd_info * const thds)
{
  idx_t const I = A->I;
  idx_t const J = A->J;
  val_t * const restrict vals = A->vals;

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const mylambda = (val_t *) thds[tid].scratch[0];
    for(idx_t j=0; j < J; ++j) {
      mylambda[j] = 0;
    }

    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        mylambda[j] = SS_MAX(mylambda[j], vals[j+(i*J)]);
      }
    }

#pragma omp barrier

    /* do reduction on partial maxes */
    //thd_reduce(thds, 0, J, REDUCE_MAX);

    #pragma omp master
    {
      for(idx_t i=1; i < omp_get_num_threads(); ++i) {
        for(idx_t j=0; j < J; ++j) {
          mylambda[j] = SS_MAX(mylambda[j], ((val_t *)thds[i].scratch[0])[j]);
        }
      }
#ifdef SPLATT_USE_MPI
      /* now do an MPI reduction to get the global lambda */
      timer_start(&timers[TIMER_MPI_NORM]);
      timer_start(&timers[TIMER_MPI_IDLE]);
      MPI_Barrier(rinfo->comm_3d);
      timer_stop(&timers[TIMER_MPI_IDLE]);

      timer_start(&timers[TIMER_MPI_COMM]);
      MPI_Allreduce(mylambda, lambda, J, SPLATT_MPI_VAL, MPI_MAX, rinfo->comm_3d);
      timer_stop(&timers[TIMER_MPI_COMM]);
      timer_stop(&timers[TIMER_MPI_NORM]);

      for(idx_t j=0; j < J; ++j) {
        lambda[j] = SS_MAX(lambda[j], 1.);
      }
#else
      for(idx_t j=0; j < J; ++j) {
        lambda[j] = SS_MAX(mylambda[j], 1.);
      }
#endif
    }

    #pragma omp barrier

    /* do the normalization */
    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        vals[j+(i*J)] /= lambda[j];
      }
    }
  } /* end omp parallel */
}


/**
* @brief Solve the system LX = B.
*
* @param L The lower triangular matrix of coefficients.
* @param B The right-hand side which is overwritten with X.
*/
static void p_mat_forwardsolve(
  matrix_t const * const L,
  matrix_t * const B)
{
  /* check dimensions */
  idx_t const N = L->I;

  val_t const * const restrict lv = L->vals;
  val_t * const restrict bv = B->vals;

  /* first row of X is easy */
  for(idx_t j=0; j < N; ++j) {
    bv[j] /= lv[0];
  }

  /* now do forward substitution */
  for(idx_t i=1; i < N; ++i) {
    /* X(i,f) = B(i,f) - \sum_{j=0}^{i-1} L(i,j)X(i,j) */
    for(idx_t j=0; j < i; ++j) {
      for(idx_t f=0; f < N; ++f) {
        bv[f+(i*N)] -= lv[j+(i*N)] * bv[f+(j*N)];
      }
    }
    for(idx_t f=0; f < N; ++f) {
      bv[f+(i*N)] /= lv[i+(i*N)];
    }
  }
}

/**
* @brief Solve the system UX = B.
*
* @param U The upper triangular matrix of coefficients.
* @param B The right-hand side which is overwritten with X.
*/
static void p_mat_backwardsolve(
  matrix_t const * const U,
  matrix_t * const B)
{
  /* check dimensions */
  idx_t const N = U->I;

  val_t const * const restrict rv = U->vals;
  val_t * const restrict bv = B->vals;

  /* last row of X is easy */
  for(idx_t f=0; f < N; ++f) {
    idx_t const i = N-1;
    bv[f+(i*N)] /= rv[i+(i*N)];
  }

  /* now do backward substitution */
  for(idx_t row=2; row <= N; ++row) {
    /* operate with (N - row) to make unsigned comparisons easy */
    idx_t const i = N - row;

    /* X(i,f) = B(i,f) - \sum_{j=0}^{i-1} R(i,j)X(i,j) */
    for(idx_t j=i+1; j < N; ++j) {
      for(idx_t f=0; f < N; ++f) {
        bv[f+(i*N)] -= rv[j+(i*N)] * bv[f+(j*N)];
      }
    }
    for(idx_t f=0; f < N; ++f) {
      bv[f+(i*N)] /= rv[i+(i*N)];
    }
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mat_syminv(
  matrix_t * const A)
{
  /* check dimensions */
  assert(A->I == A->J);

  idx_t const N = A->I;

  matrix_t * L = mat_alloc(N, N);

  /* do a Cholesky factorization on A */
  mat_cholesky(A, L);

  /* setup identity matrix */
  memset(A->vals, 0, N*N*sizeof(val_t));
  for(idx_t n=0; n < N; ++n) {
    A->vals[n+(n*N)] = 1.;
  }

  /* Solve L*Y = I */
  p_mat_forwardsolve(L, A);

  /* transpose L */
  for(idx_t i=0; i < N; ++i) {
    for(idx_t j=i+1; j < N; ++j) {
      L->vals[j+(i*N)] = L->vals[i+(j*N)];
      L->vals[i+(j*N)] = 0.;
    }
  }

  /* Solve U*A = Y */
  p_mat_backwardsolve(L, A);

  mat_free(L);
}


void mat_cholesky(
  matrix_t const * const A,
  matrix_t * const L)
{
  /* check dimensions */
  assert(A->I == A->J);
  assert(A->I == L->J);
  assert(L->I == L->J);

  idx_t const N = A->I;
  val_t const * const restrict av = A->vals;
  val_t * const restrict lv = L->vals;

  memset(lv, 0, N*N*sizeof(val_t));
  for (idx_t i = 0; i < N; ++i) {
    for (idx_t j = 0; j <= i; ++j) {
      val_t inner = 0;
      for (idx_t k = 0; k < j; ++k) {
        inner += lv[k+(i*N)] * lv[k+(j*N)];
      }

      if(i == j) {
        lv[j+(i*N)] = sqrt(av[i+(i*N)] - inner);
      } else {
        lv[j+(i*N)] = 1.0 / lv[j+(j*N)] * (av[j+(i*N)] - inner);
      }
    }
  }
}


void mat_aTa_hada(
  matrix_t ** mats,
  idx_t const start,
  idx_t const nmults,
  idx_t const nmats,
  matrix_t * const buf,
  matrix_t * const ret)
{
  idx_t const F = mats[0]->J;

  /* check matrix dimensions */
  assert(ret->I == ret->J);
  assert(ret->I == F);
  assert(buf->I == F);
  assert(buf->J == F);
  assert(ret->vals != NULL);
  assert(mats[0]->rowmajor);
  assert(ret->rowmajor);

  val_t       * const restrict rv   = ret->vals;
  val_t       * const restrict bufv = buf->vals;
  for(idx_t i=0; i < F; ++i) {
    for(idx_t j=i; j < F; ++j) {
      rv[j+(i*F)] = 1.;
    }
  }

  for(idx_t mode=0; mode < nmults; ++mode) {
    idx_t const m = (start+mode) % nmats;
    idx_t const I  = mats[m]->I;
    val_t const * const Av = mats[m]->vals;
    memset(bufv, 0, F * F * sizeof(val_t));

    /* compute upper triangular matrix */
    for(idx_t i=0; i < I; ++i) {
      for(idx_t mi=0; mi < F; ++mi) {
        for(idx_t mj=mi; mj < F; ++mj) {
          bufv[mj + (mi*F)] += Av[mi + (i*F)] * Av[mj + (i*F)];
        }
      }
    }

    /* hadamard product */
    for(idx_t mi=0; mi < F; ++mi) {
      for(idx_t mj=mi; mj < F; ++mj) {
        rv[mj + (mi*F)] *= bufv[mj + (mi*F)];
      }
    }
  }

  /* copy to lower triangular matrix */
  for(idx_t i=1; i < F; ++i) {
    for(idx_t j=0; j < i; ++j) {
      rv[j + (i*F)] = rv[i + (j*F)];
    }
  }
}


void mat_aTa(
  matrix_t const * const A,
  matrix_t * const ret,
  rank_info * const rinfo,
  thd_info * const thds,
  idx_t const nthreads)
{
  timer_start(&timers[TIMER_ATA]);
#ifdef SPLATT_USE_MKL
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A->J, A->J, A->I, 1, A->vals, A->J, A->vals, A->J, 1, ret->vals, ret->J);
  // can use dsyrk?
#else
  /* check matrix dimensions */
  assert(ret->I == ret->J);
  assert(ret->I == A->J);
  assert(ret->vals != NULL);
  assert(A->rowmajor);
  assert(ret->rowmajor);

  idx_t const I = A->I;
  idx_t const F = A->J;
  val_t const * const restrict Av = A->vals;

  omp_set_num_threads(nthreads);

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const accum = (val_t *) thds[tid].scratch[0];

    /* compute upper triangular portion */
    memset(accum, 0, F * F * sizeof(val_t));

    /* compute each thread's partial matrix product */
    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t mi=0; mi < F; ++mi) {
        for(idx_t mj=mi; mj < F; ++mj) {
          accum[mj + (mi*F)] += Av[mi + (i*F)] * Av[mj + (i*F)];
        }
      }
    }

    /* parallel reduction on accum */
    thd_reduce(thds, 0, F * F, REDUCE_SUM);

    /* copy to lower triangular matrix */
    #pragma omp master
    for(idx_t i=1; i < F; ++i) {
      for(idx_t j=0; j < i; ++j) {
        accum[j + (i*F)] = accum[i + (j*F)];
      }
    }
  }

#ifdef SPLATT_USE_MPI
  timer_start(&timers[TIMER_MPI_ATA]);
  timer_start(&timers[TIMER_MPI_IDLE]);
  MPI_Barrier(rinfo->comm_3d);
  timer_stop(&timers[TIMER_MPI_IDLE]);

  timer_start(&timers[TIMER_MPI_COMM]);
  MPI_Allreduce(thds[0].scratch[0], ret->vals, F * F, SPLATT_MPI_VAL, MPI_SUM,
      rinfo->comm_3d);
  timer_stop(&timers[TIMER_MPI_COMM]);
  timer_stop(&timers[TIMER_MPI_ATA]);
#else
  memcpy(ret->vals, (val_t *) thds[0].scratch[0], F * F * sizeof(val_t));
#endif
#endif

  timer_stop(&timers[TIMER_ATA]);
}

void mat_matmul(
  matrix_t const * const A,
  matrix_t const * const B,
  matrix_t  * const C)
{
  timer_start(&timers[TIMER_MATMUL]);

#define SPLATT_USE_MKL
#ifdef SPLATT_USE_MKL
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->I, B->J, A->J, 1, A->vals, A->J, B->vals, B->J, 0, C->vals, C->J);
#else

  /* check dimensions */
  assert(A->J == B->I);
  assert(C->I * C->J >= A->I * B->J);
  C->I = A->I;
  C->J = B->J;

  val_t const * const restrict av = A->vals;
  val_t const * const restrict bv = B->vals;
  val_t       * const restrict cv = C->vals;

  idx_t const M  = A->I;
  idx_t const N  = B->J;
  idx_t const Na = A->J;

  /* tiled matrix multiplication */
  idx_t const TILE = 16;
  #pragma omp parallel for schedule(static)
  for(idx_t i=0; i < M; ++i) {
    for(idx_t jt=0; jt < N; jt += TILE) {
      for(idx_t kt=0; kt < Na; kt += TILE) {
        idx_t const JSTOP = SS_MIN(jt+TILE, N);
        for(idx_t j=jt; j < JSTOP; ++j) {
          val_t accum = 0;
          idx_t const KSTOP = SS_MIN(kt+TILE, Na);
          for(idx_t k=kt; k < KSTOP; ++k) {
            accum += av[k + (i*Na)] * bv[j + (k*N)];
          }
          cv[j + (i*N)] = accum;
        }
      }
    }
  }
#endif

  timer_stop(&timers[TIMER_MATMUL]);
}

void mat_normalize(
  matrix_t * const A,
  val_t * const restrict lambda,
  splatt_mat_norm const which,
  rank_info * const rinfo,
  thd_info * const thds,
  idx_t const nthreads)
{
  timer_start(&timers[TIMER_MATNORM]);

  omp_set_num_threads(nthreads);

  switch(which) {
  case MAT_NORM_2:
    p_mat_2norm(A, lambda, rinfo, thds);
    break;
  case MAT_NORM_MAX:
    p_mat_maxnorm(A, lambda, rinfo, thds);
    break;
  default:
    fprintf(stderr, "SPLATT: mat_normalize supports 2 and MAX only.\n");
    abort();
  }
  timer_stop(&timers[TIMER_MATNORM]);
}


void calc_gram_inv(
  idx_t const mode,
  idx_t const nmodes,
  matrix_t ** aTa)
{
  timer_start(&timers[TIMER_INV]);

  idx_t const rank = aTa[0]->J;
  val_t * const restrict av = aTa[MAX_NMODES]->vals;

  /* ata[MAX_NMODES] = hada(aTa[0], aTa[1], ...) */
  for(idx_t x=0; x < rank*rank; ++x) {
    av[x] = 1.;
  }
  for(idx_t m=1; m < nmodes; ++m) {
    idx_t const madjust = (mode + m) % nmodes;
    val_t const * const vals = aTa[madjust]->vals;
    for(idx_t x=0; x < rank*rank; ++x) {
      av[x] *= vals[x];
    }
  }

  /* M2 = M2^-1 */
  mat_syminv(aTa[MAX_NMODES]);
  timer_stop(&timers[TIMER_INV]);
}


void mat_transpose(
  matrix_t const * const A,
  matrix_t * B)
{
  idx_t const I = A->I;
  idx_t const J = A->J;

  assert(B->vals != NULL);
  assert(B->I * B->J >= I*J);

  B->I = J;
  B->J = I;

  #pragma omp parallel
  {
    val_t const * const av = A->vals;
    val_t       * const bv = B->vals;

    for(idx_t j=0; j < J; ++j) {
      #pragma omp for nowait
      for(idx_t i=0; i < I; ++i) {
        bv[i + (j*I)] = av[j + (i*J)];
      }
    }
  } /* end omp parallel */

}


matrix_t * mat_alloc(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = (matrix_t *) splatt_malloc(sizeof(matrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->vals = (val_t *) splatt_malloc(nrows * ncols * sizeof(val_t));
  mat->rowmajor = 1;
  return mat;
}

matrix_t * mat_rand(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = mat_alloc(nrows, ncols);
  val_t * const vals = mat->vals;

  fill_rand(vals, nrows * ncols);

  return mat;
}

void mat_free(
  matrix_t * mat)
{
  free(mat->vals);
  free(mat);
}

matrix_t * mat_mkrow(
  matrix_t const * const mat)
{
  assert(mat->rowmajor == 0);

  idx_t const I = mat->I;
  idx_t const J = mat->J;

  matrix_t * row = mat_alloc(I, J);
  val_t       * const restrict rowv = row->vals;
  val_t const * const restrict colv = mat->vals;

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      rowv[j + (i*J)] = colv[i + (j*I)];
    }
  }

  return row;
}

matrix_t * mat_mkcol(
  matrix_t const * const mat)
{
  assert(mat->rowmajor == 1);
  idx_t const I = mat->I;
  idx_t const J = mat->J;

  matrix_t * col = mat_alloc(I, J);
  val_t       * const restrict colv = col->vals;
  val_t const * const restrict rowv = mat->vals;

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      colv[i + (j*I)] = rowv[j + (i*J)];
    }
  }

  col->rowmajor = 0;

  return col;
}


spmatrix_t * spmat_alloc(
  idx_t const nrows,
  idx_t const ncols,
  idx_t const nnz)
{
  spmatrix_t * mat = (spmatrix_t*) splatt_malloc(sizeof(spmatrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->nnz = nnz;
  mat->rowptr = (idx_t*) splatt_malloc((nrows+1) * sizeof(idx_t));
  mat->colind = (idx_t*) splatt_malloc(nnz * sizeof(idx_t));
  mat->vals   = (val_t*) splatt_malloc(nnz * sizeof(val_t));
  return mat;
}

void spmat_free(
  spmatrix_t * mat)
{
  free(mat->rowptr);
  free(mat->colind);
  free(mat->vals);
  free(mat);
}


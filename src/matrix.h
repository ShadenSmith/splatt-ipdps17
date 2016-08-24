#ifndef SPLATT_MATRIX_H
#define SPLATT_MATRIX_H

#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  idx_t I;
  idx_t J;
  val_t *vals;
  int rowmajor;
} matrix_t;

typedef struct
{
  idx_t I;
  idx_t J;
  idx_t nnz;
  idx_t * rowptr;
  idx_t * colind;
  val_t * vals;
} spmatrix_t;

typedef enum
{
  MAT_NORM_2,
  MAT_NORM_MAX
} splatt_mat_norm;


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "splatt_mpi.h"
#include "thd_info.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

#define mat_cholesky splatt_mat_cholesky
/**
* @brief Compute the Cholesky factorization of A.
*
* @param A The SPD matrix A.
* @param L The lower-triangular result.
*/
void mat_cholesky(
  matrix_t const * const A,
  matrix_t * const L);


#define mat_matmul splatt_mat_matmul
/**
* @brief Dense matrix-matrix multiplication, C = AB.
*
* @param A The left multiplication parameter.
* @param B The right multiplication parameter.
* @param C The result matrix. NOTE: C is not zeroed before multiplication!
*/
void mat_matmul(
  matrix_t const * const A,
  matrix_t const * const B,
  matrix_t  * const C);


#define mat_syminv splatt_mat_syminv
/**
* @brief Compute the 'inverse' of symmetric matrix A.
*
* @param A Symmetric FxF matrix.
* @param Abuf FxF buffer space.
*/
void mat_syminv(
  matrix_t * const A);


#define mat_solve_normals splatt_mat_solve_normals
void mat_solve_normals(
  idx_t const mode,
  idx_t const nmodes,
	matrix_t * * ata,
  matrix_t * rhs,
  val_t const reg);


#define mat_aTa_hada splatt_mat_aTa_hada
/**
* @brief Compute (A^T A * B^T B * C^T C ...) where * is the Hadamard product.
*
* @param mats An array of matrices.
* @param start The first matrix to include.
* @param end The last matrix to include. This can be before start because we
*            operate modulo nmats.
* @param nmats The number of matrices.
* @param ret The FxF output matrix.
*/
void mat_aTa_hada(
  matrix_t ** mats,
  idx_t const start,
  idx_t const end,
  idx_t const nmats,
  matrix_t * const buf,
  matrix_t * const ret);


#define mat_aTa splatt_mat_aTa
/**
* @brief Compute A^T * A with a nice row-major pattern.
*
* @param A The input matrix.
* @param ret The output matrix, A^T * A.
* @param thds Data structure for thread scratch space.
*/
void mat_aTa(
  matrix_t const * const A,
  matrix_t * const ret,
  rank_info * const rinfo,
  thd_info * const thds,
  idx_t const nthreads);

#define calc_gram_inv splatt_calc_gram_inv
/**
* @brief Calculate (BtB * CtC * ...)^-1, where * is the Hadamard product. This
*        is the Gram Matrix of the CPD.
*
* @param mode Which mode we are operating on (it is not used in the product).
* @param nmodes The number of modes in the tensor.
* @param aTa An array of matrices (length MAX_NMODES)containing BtB, CtC, etc.
*            [OUT] The result is stored in ata[MAX_NMODES].
*/
void calc_gram_inv(
  idx_t const mode,
  idx_t const nmodes,
  matrix_t ** aTa);

#define mat_normalize splatt_mat_normalize
/**
* @brief Normalize the columns of A and return the norms in lambda.
*        Supported norms are:
*          1. 2-norm
*          2. max-norm
*
* @param A The matrix to normalize.
* @param lambda The vector of column norms.
* @param which Which norm to use.
*/
void mat_normalize(
  matrix_t * const A,
  val_t * const restrict lambda,
  splatt_mat_norm const which,
  rank_info * const rinfo,
  thd_info * const thds,
  idx_t const nthreads);


#define mat_rand splatt_mat_rand
/**
* @brief Return a randomly initialized matrix (from util's rand_val()).
*
* @param nrows The number of rows in the matrix.
* @param ncols The number of columns in the matrix.
*
* @return The random matrix.
*/
matrix_t * mat_rand(
  idx_t const nrows,
  idx_t const ncols);


#define mat_alloc splatt_mat_alloc
/**
* @brief Allocate a dense matrix. The values will not be initialized. This
*        matrix must be freed with mat_free().
*
* @param nrows The number of rows in the matrix.
* @param ncols The number of columns in the matrix.
*
* @return The allocated matrix.
*/
matrix_t * mat_alloc(
  idx_t const nrows,
  idx_t const ncols);


#define mat_transpose splatt_mat_transpose
/**
* @brief Fill matrix B with transpose(A). Assumes B is already allocated.
*
* @param A The matrix to transpose.
* @param[out] B The transposed matrix.
*/
void mat_transpose(
  matrix_t const * const A,
  matrix_t * B);


#define mat_free splatt_mat_free
/**
* @brief Free a matrix allocated with mat_alloc(). This also frees the matrix
*        pointer!
*
* @param mat The matrix to be freed.
*/
void mat_free(
  matrix_t * mat);


#define spmat_alloc splatt_spmat_alloc
/**
* @brief Allocate a sparse matrix in CSR format. The values will not be
*        initialized. This matrix must be freed with spmat_free().
*
* @param nrows The number of rows in the sparse matrix.
* @param ncols The number of columns in the sparse matrix.
* @param nnz The number of nonzero values in the sparse matrix.
*
* @return The allocated CSR matrix.
*/
spmatrix_t * spmat_alloc(
  idx_t const nrows,
  idx_t const ncols,
  idx_t const nnz);


#define spmat_free splatt_spmat_free
/**
* @brief Free a sparse matrix allocated with spmat_alloc(). This also frees the
*        matrix pointer!
*
* @param mat The sparse matrix to be freed.
*/
void spmat_free(
  spmatrix_t * mat);


#define mat_mkrow splatt_mat_mkrow
/**
* @brief Copies a column-major matrix and returns a row-major version.
*
* @param mat The column-major matrix to copy.
*
* @return A row-major copy of mat.
*/
matrix_t * mat_mkrow(
  matrix_t const * const mat);


#define mat_mkcol splatt_mat_mkcol
/**
* @brief Copies a row-major matrix and returns a column-major version.
*
* @param mat The row-major matrix to copy.
*
* @return A column-major copy of mat.
*/
matrix_t * mat_mkcol(
  matrix_t const * const mat);

#ifdef __cplusplus
}
#endif

#endif

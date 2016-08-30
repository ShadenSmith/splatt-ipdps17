
/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include <omp.h>

#include "csf.h"
#include "sort.h"
#include "tile.h"
#include "util.h"
#include "mttkrp.h"

#include "io.h"
#include <omp.h>

#ifdef __AVX512F__
//#define HBW_ALLOC
  /* define this and run with "numactl -m 1" and MEMKIND_HBW_NODES=0
   * to allocate tensor data to DDR */
#endif
#ifdef HBW_ALLOC
#include <hbwmalloc.h>
#endif

/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

int splatt_csf_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_csf ** tensors,
    double const * const options)
{
  sptensor_t * tt = tt_read(fname);
  if(tt == NULL) {
    return SPLATT_ERROR_BADINPUT;
  }

  tt_remove_empty(tt);

  *tensors = csf_alloc(tt, options);
  *nmodes = tt->nmodes;

  tt_free(tt);

  return SPLATT_SUCCESS;
}

int splatt_csf_convert(
    splatt_idx_t const nmodes,
    splatt_idx_t const nnz,
    splatt_fidx_t ** const inds,
    splatt_storage_val_t * const vals,
    splatt_csf ** tensors,
    double const * const options)
{
  sptensor_t tt;
  tt_fill(&tt, nnz, nmodes, inds, vals);
  tt_remove_empty(&tt);

  *tensors = csf_alloc(&tt, options);

  return SPLATT_SUCCESS;
}


void splatt_free_csf(
    splatt_csf * tensors,
    double const * const options)
{
  csf_free(tensors, options);
}




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Find a permutation of modes that results in non-increasing mode size.
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_small(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t * const perm_dims)
{
  idx_t sorted[MAX_NMODES];
  idx_t matched[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    sorted[m] = dims[m];
    matched[m] = 0;
  }
  quicksort(sorted, nmodes);

  /* silly n^2 comparison to grab modes from sorted dimensions.
   * TODO: make a key/val sort...*/
  for(idx_t mfind=0; mfind < nmodes; ++mfind) {
    for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
      if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
        perm_dims[mfind] = mcheck;
        matched[mcheck] = 1;
        break;
      }
    }
  }
}


/**
 * Same as p_order_dims_small but don't put dims that can be privated
 * as the root
 */
static void p_order_dims_small_no_privatization(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t * const perm_dims,
  idx_t nnz,
  double const * const opts)
{
  p_order_dims_small(dims, nmodes, perm_dims);

  /* find where custom_mode was placed and adjust from there */
  for(idx_t m=0; m < nmodes; ++m) {
    idx_t perm_dim = perm_dims[m];
    if(dims[perm_dims[m]] >= 32*1024) { // don't make it root mode if it's too short to avoid load imbalance
      memmove(perm_dims + 1, perm_dims, (m) * sizeof(m));
      perm_dims[0] = perm_dim;
      break;
    }
  }
}


/**
* @brief Find a permutation of modes such that the first mode is 'custom-mode'
*        and the remaining are naturally ordered (0, 1, ...).
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param custom_mode The mode to place first.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_inorder(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const custom_mode,
  idx_t * const perm_dims)
{
  /* initialize to natural ordering */
  for(idx_t m=0; m < nmodes; ++m) {
    perm_dims[m] = m;
  }

  /* find where custom_mode was placed and adjust from there */
  for(idx_t m=0; m < nmodes; ++m) {
    if(perm_dims[m] == custom_mode) {
      memmove(perm_dims + 1, perm_dims, (m) * sizeof(m));
      perm_dims[0] = custom_mode;
      break;
    }
  }
}

static void p_order_dims_round_robin(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const custom_mode,
  idx_t * const perm_dims)
{
  for(idx_t m=0; m < nmodes; ++m) {
    perm_dims[m] = (custom_mode + m)%nmodes;
  }
}

static void p_order_dims_all_permute(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const custom_mode,
  idx_t * const perm_dims)
{
  if (custom_mode == 0) {
    perm_dims[0] = 0;
    perm_dims[1] = 1;
    perm_dims[2] = 2;
  } else if (custom_mode == 1) {
    perm_dims[0] = 0;
    perm_dims[1] = 2;
    perm_dims[2] = 1;
  } else if (custom_mode == 2) {
    perm_dims[0] = 1;
    perm_dims[1] = 0;
    perm_dims[2] = 2;
  } else if (custom_mode == 3) {
    perm_dims[0] = 1;
    perm_dims[1] = 2;
    perm_dims[2] = 0;
  } else if (custom_mode == 4) {
    perm_dims[0] = 2;
    perm_dims[1] = 0;
    perm_dims[2] = 1;
  } else if (custom_mode == 5) {
    perm_dims[0] = 2;
    perm_dims[1] = 1;
    perm_dims[2] = 0;
  }
  else { assert(0); }
}

/**
* @brief Find a permutation of modes such that the first mode is 'custom-mode'
*        and the remaining are sorted in non-increasing order.
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param custom_mode The mode to place first.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_minusone(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const custom_mode,
  idx_t * const perm_dims)
{
  p_order_dims_small(dims, nmodes, perm_dims);

  /* find where custom_mode was placed and adjust from there */
  for(idx_t m=0; m < nmodes; ++m) {
    if(perm_dims[m] == custom_mode) {
      memmove(perm_dims + 1, perm_dims, (m) * sizeof(m));
      perm_dims[0] = custom_mode;
      break;
    }
  }
}


/**
* @brief Find a permutation of modes that results in non-decreasing mode size.
*
* @param dims The tensor dimensions.
* @param nmodes The number of modes.
* @param perm_dims The resulting permutation.
*/
static void p_order_dims_large(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t * const perm_dims)
{
  idx_t sorted[MAX_NMODES];
  idx_t matched[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    sorted[m] = dims[m];
    matched[m] = 0;
  }
  /* sort small -> large */
  quicksort(sorted, nmodes);

  /* reverse list */
  for(idx_t m=0; m < nmodes/2; ++m) {
    idx_t tmp = sorted[nmodes-m-1];
    sorted[nmodes-m-1] = sorted[m];
    sorted[m] = tmp;
  }

  /* silly n^2 comparison to grab modes from sorted dimensions.
   * TODO: make a key/val sort...*/
  for(idx_t mfind=0; mfind < nmodes; ++mfind) {
    for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
      if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
        perm_dims[mfind] = mcheck;
        matched[mcheck] = 1;
        break;
      }
    }
  }

}


void
splatt_csf_write_file(
  splatt_csf const * const ct,
  FILE * fout)
{
  fwrite(&ct->nnz, sizeof(ct->nnz), 1, fout);
  fwrite(&ct->nmodes, sizeof(ct->nmodes), 1, fout);
  fwrite(ct->dims, sizeof(*ct->dims), ct->nmodes, fout);
  fwrite(ct->dim_perm, sizeof(*ct->dim_perm), ct->nmodes, fout);
  fwrite(&ct->which_tile, sizeof(ct->which_tile), 1, fout);
  fwrite(&ct->ntiles, sizeof(ct->ntiles), 1, fout);
  fwrite(ct->tile_dims, sizeof(*ct->tile_dims), ct->nmodes, fout);

  for(idx_t t=0; t < ct->ntiles; ++t) {
    csf_sparsity const * const ft = ct->pt + t;

    fwrite(ft->nfibs, sizeof(*ft->nfibs), ct->nmodes, fout);

    for(idx_t m=0; m < ct->nmodes-1; ++m) {
      fwrite(ft->fptr[m], sizeof(*ft->fptr[m]), ft->nfibs[m] + 1, fout);
      if (m != 0) { // FIXME
        fwrite(ft->fids[m], sizeof(*ft->fids[m]), ft->nfibs[m], fout);
      }
    }

    fwrite(ft->fids[ct->nmodes - 1], sizeof(*ft->fids[ct->nmodes - 1]), ft->nfibs[ct->nmodes-1], fout);
    fwrite(ft->vals, sizeof(*ft->vals), ft->nfibs[ct->nmodes-1], fout);
  }
}

void
splatt_csf_write(
  splatt_csf const * const ct,
  char const * const ofname,
  int ncopies)
{
  FILE * fout = fopen(ofname,"w");
  if (fout == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", ofname);
    return;
  }

  timer_start(&timers[TIMER_IO]);
  fwrite(&ncopies, sizeof(ncopies), 1, fout);
  for (int i = 0; i < ncopies; ++i) {
    splatt_csf_write_file(ct + i, fout);
  }
  timer_stop(&timers[TIMER_IO]);

  fclose(fout);
}


void splatt_csf_read_file(
  splatt_csf *ct, FILE * fin)
{
  fread(&ct->nnz, sizeof(ct->nnz), 1, fin);
  fread(&ct->nmodes, sizeof(ct->nmodes), 1, fin);
  fread(ct->dims, sizeof(*ct->dims), ct->nmodes, fin);
  fread(ct->dim_perm, sizeof(*ct->dim_perm), ct->nmodes, fin);
  fread(&ct->which_tile, sizeof(ct->which_tile), 1, fin);
  fread(&ct->ntiles, sizeof(ct->ntiles), 1, fin);
  fread(&ct->tile_dims, sizeof(*ct->tile_dims), ct->nmodes, fin);

  ct->pt = splatt_malloc(sizeof(*(ct->pt))*ct->ntiles);

  for(idx_t t=0; t < ct->ntiles; ++t) {
    csf_sparsity * ft = ct->pt + t;

    fread(ft->nfibs, sizeof(*ft->nfibs), ct->nmodes, fin);

    for(idx_t m=0; m < ct->nmodes-1; ++m) {
      ft->fptr[m] = splatt_malloc((ft->nfibs[m]+1) * sizeof(**(ft->fptr)));
      fread(ft->fptr[m], sizeof(*ft->fptr[m]), ft->nfibs[m]+1, fin);
      if (m != 0) { // FIXME
        ft->fids[m] = splatt_malloc(ft->nfibs[m] * sizeof(**(ft->fids)));
        fread(ft->fids[m], sizeof(*ft->fids[m]), ft->nfibs[m], fin);
      }
      else {
        ft->fids[m] = NULL;
      }
    }

#ifdef HBW_ALLOC
    hbw_posix_memalign((void **)&ft->fids[ct->nmodes-1], 4096, ft->nfibs[ct->nmodes-1] * sizeof(**(ft->fids)));
    hbw_posix_memalign((void **)&ft->vals, 4096, ft->nfibs[ct->nmodes-1] * sizeof(*(ft->vals)));
#else
    ft->fids[ct->nmodes-1] = splatt_malloc(ft->nfibs[ct->nmodes-1] * sizeof(**(ft->fids)));
    ft->vals               = splatt_malloc(ft->nfibs[ct->nmodes-1] * sizeof(*(ft->vals)));
#endif

    fread(ft->fids[ct->nmodes-1], sizeof(*ft->fids[ct->nmodes-1]), ft->nfibs[ct->nmodes-1], fin);
    fread(ft->vals, sizeof(*ft->vals), ft->nfibs[ct->nmodes-1], fin);
  }
}

int splatt_csf_equals(splatt_csf *ct1, splatt_csf *ct2)
{
  if (ct1->nnz != ct2->nnz) return 0;
  if (ct1->nmodes != ct2->nmodes) return 0;
  if (memcmp(ct1->dims, ct2->dims, sizeof(*ct1->dims)*ct1->nmodes)) return 0;
  if (memcmp(ct1->dim_perm, ct2->dim_perm, sizeof(*ct1->dim_perm)*ct1->nmodes)) return 0;
  if (ct1->which_tile != ct2->which_tile) return 0;
  if (ct1->ntiles != ct2->ntiles) return 0;
  if (memcmp(ct1->tile_dims, ct2->tile_dims, sizeof(*ct1->tile_dims)*ct1->nmodes)) return 0;
  
  for(idx_t t=0; t < ct1->ntiles; ++t) {
    csf_sparsity const * const ft1 = ct1->pt + t;
    csf_sparsity const * const ft2 = ct2->pt + t;

    if (memcmp(ft1->nfibs, ft2->nfibs, sizeof(*ft1->nfibs)*ct1->nmodes)) return 0;
    for(idx_t m=0; m < ct1->nmodes-1; ++m) {
      if (memcmp(ft1->fptr[m], ft2->fptr[m], sizeof(*ft1->fptr[m])*(ft1->nfibs[m] + 1))) return 0;
      if (m != 0 && memcmp(ft1->fids[m], ft2->fids[m], sizeof(*ft1->fids[m])*ft1->nfibs[m])) return 0;
    }

    if (memcmp(ft1->fids[ct1->nmodes - 1], ft2->fids[ct2->nmodes - 1], sizeof(*ft1->fids[ct1->nmodes - 1])*ft1->nfibs[ct1->nmodes-1])) return 0;
    if (memcmp(ft1->vals, ft2->vals, sizeof(*ft1->vals)*ft1->nfibs[ct1->nmodes-1])) return 0;
  }

  return 1;
}

void
splatt_csf_read(
  splatt_csf *ct,
  char const * const ifname,
  int ncopies)
{
  FILE * fin = fopen(ifname,"r");
  if (fin == NULL) {
    fprintf(stderr, "SPLATT ERROR: failed to open '%s'\n.", ifname);
    return;
  }

  timer_start(&timers[TIMER_IO]);
  int file_ncopies = 2;
  fread(&file_ncopies, sizeof(file_ncopies), 1, fin);
  if (ncopies == -1) {
    splatt_csf_read_file(ct, fin);
    ncopies = ct->nmodes;
    for (int i = 1; i < ncopies; ++i) {
      splatt_csf_read_file(ct + i, fin);
    }
  }
  else {
    if (file_ncopies < ncopies) {
      fprintf(stderr, "SPLATT ERROR: %d copies are required but %s has only %d\n", ncopies, ifname, file_ncopies);
    }
    for (int i = 0; i < ncopies; ++i) {
      splatt_csf_read_file(ct + i, fin);
    }
  }
  timer_stop(&timers[TIMER_IO]);

  fclose(fin);
}



/**
* @brief Print a CSF tensor in human-readable format.
*
* @param ct The tensor to print.
*/
static void p_print_csf(
  splatt_csf const * const ct)
{
  printf("-----------\n");
  printf("nmodes: %"SPLATT_PF_IDX" nnz: %"SPLATT_PF_IDX" ntiles: "
         "%"SPLATT_PF_IDX"\n", ct->nmodes, ct->nnz, ct->ntiles);
  printf("dims: %"SPLATT_PF_IDX"", ct->dims[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("x%"SPLATT_PF_IDX"", ct->dims[m]);
  }
  printf(" (%"SPLATT_PF_IDX"", ct->dim_perm[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("->%"SPLATT_PF_IDX"", ct->dim_perm[m]);
  }
  printf(") ");
  printf("tile dims: %"SPLATT_PF_IDX"", ct->tile_dims[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("x%"SPLATT_PF_IDX"", ct->tile_dims[m]);
  }
  printf("\n");

  for(idx_t t=0; t < ct->ntiles; ++t) {
    csf_sparsity const * const ft = ct->pt + t;
    /* skip empty tiles */
    if(ft->vals == NULL) {
      continue;
    }

    /* write slices */
    printf("tile: %"SPLATT_PF_IDX" fptr:\n", t);
    printf("[%"SPLATT_PF_IDX"] ", ft->nfibs[0]);
    for(idx_t f=0; f < ft->nfibs[0]; ++f) {
      if(ft->fids[0] == NULL) {
        printf(" %"SPLATT_PF_IDX"", ft->fptr[0][f]);
      } else {
        printf(" (%"SPLATT_PF_IDX", %"SPLATT_PF_FIDX")", ft->fptr[0][f],
            ft->fids[0][f]);
      }
    }
    printf(" %"SPLATT_PF_IDX"\n", ft->fptr[0][ft->nfibs[0]]);

    /* inner nodes */
    for(idx_t m=1; m < ct->nmodes-1; ++m) {
      printf("[%"SPLATT_PF_IDX"] ", ft->nfibs[m]);
      for(idx_t f=0; f < ft->nfibs[m]; ++f) {
        printf(" (%"SPLATT_PF_IDX", %"SPLATT_PF_FIDX")", ft->fptr[m][f],
            ft->fids[m][f]);
      }
      printf(" %"SPLATT_PF_IDX"\n", ft->fptr[m][ft->nfibs[m]]);
    }

    /* vals/inds */
    printf("[%"SPLATT_PF_IDX"] ", ft->nfibs[ct->nmodes-1]);
    for(idx_t f=0; f < ft->nfibs[ct->nmodes-1]; ++f) {
      printf(" %3"SPLATT_PF_FIDX"", ft->fids[ct->nmodes-1][f]);
    }
    printf("\n");
    for(idx_t n=0; n < ft->nfibs[ct->nmodes-1]; ++n) {
      printf(" %0.1f", ft->vals[n]);
    }
    printf("\n");
  }

  printf("-----------\n\n");
}


static void p_set_nfibs_root(
  splatt_csf * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr)
{
  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  assert(nnzstart < nnzend);

  /* the mode after accounting for dim_perm */
  fidx_t const * const restrict ttind = tt->ind[ct->dim_perm[0]] + nnzstart;

  /* grab sparsity pattern */
  csf_sparsity * const pt = ct->pt + tile_id;

  /* count fibers */
  idx_t nfibs = 1;
  for(idx_t x=1; x < nnz; ++x) {
    assert(ttind[x-1] <= ttind[x]);
    if(ttind[x] != ttind[x-1]) {
      assert(nfibs == ttind[x]);
      ++nfibs;
    }
  }
  ct->pt[tile_id].nfibs[0] = nfibs;
  assert(nfibs <= ct->dims[ct->dim_perm[0]]);
}

/**
* @brief Construct the sparsity structure of the outer-mode of a CSF tensor.
*
* @param ct The CSF tensor to construct.
* @param tt The coordinate tensor to construct from. Assumed to be already
*            sorted.
* @param tile_id The ID of the tile to construct.
* @param nnztile_ptr A pointer into 'tt' that marks the start of each tile.
*/
static void p_mk_outerptr_hub(
  splatt_csf * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr)
{
  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  assert(nnzstart < nnzend);

  /* the mode after accounting for dim_perm */
  fidx_t const * const restrict ttind = tt->ind[ct->dim_perm[0]] + nnzstart;

  /* grab sparsity pattern */
  csf_sparsity * const pt = ct->pt + tile_id;

  idx_t *nfibs = malloc(sizeof(idx_t)*(omp_get_max_threads() + 1));
  nfibs[0] = 1;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    idx_t x_per_thread = (nnz + nthreads - 1)/nthreads;
    idx_t x_begin = SS_MAX(SS_MIN(x_per_thread*tid, nnz), 1);
    idx_t x_end = SS_MIN(x_per_thread*(tid + 1), nnz);

    idx_t nfibs_private = 0;

    for(idx_t x = x_begin; x < x_end; ++x) {
      assert(ttind[x-1] <= ttind[x]);
      if(ttind[x] != ttind[x-1]) {
        ++nfibs_private;
      }
    }
    nfibs[tid + 1] = nfibs_private;

#pragma omp barrier
#pragma omp master
    {
      /* prefix sum */
      for(int t = 0; t < nthreads; ++t) {
        nfibs[t + 1] += nfibs[t];
      }
      ct->pt[tile_id].nfibs[0] = nfibs[nthreads];
      assert(nfibs[nthreads] <= ct->dims[ct->dim_perm[0]]);

      pt->fptr[0] = splatt_malloc((nfibs[nthreads]+1) * sizeof(**(pt->fptr)));
      if(ct->ntiles > 1) {
        pt->fids[0] = splatt_malloc(nfibs[nthreads] * sizeof(**(pt->fids)));
      } else {
        pt->fids[0] = NULL;
      }
    }
#pragma omp barrier

    idx_t  * const restrict fp = pt->fptr[0];
    fidx_t  * const restrict fi = pt->fids[0];
#pragma omp master
    {
      fp[0] = 0;
      if(fi != NULL) {
        fi[0] = ttind[0];
      }
      fp[nfibs[nthreads]] = nnz;
    }

    idx_t nfound = nfibs[tid];
    if(fi != NULL) {
      for(idx_t n=x_begin; n < x_end; ++n) {
        /* check for end of outer index */
        if(ttind[n] != ttind[n-1]) {
          fi[nfound] = ttind[n];
          fp[nfound++] = n;
        }
      }
    }
    else {
      for(idx_t n=x_begin; n < x_end; ++n) {
        /* check for end of outer index */
        if(ttind[n] != ttind[n-1]) {
          assert(nfound == ttind[n]);
          fp[nfound++] = n;
        }
      }
    }
  } /* omp parallel */
}

/**
* @brief Construct the sparsity structure of the outer-mode of a CSF tensor.
*
* @param ct The CSF tensor to construct.
* @param tt The coordinate tensor to construct from. Assumed to be already
*            sorted.
* @param tile_id The ID of the tile to construct.
* @param nnztile_ptr A pointer into 'tt' that marks the start of each tile.
*/
static void p_mk_outerptr(
  splatt_csf * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr)
{
  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  assert(nnzstart < nnzend);

  /* the mode after accounting for dim_perm */
  fidx_t const * const restrict ttind = tt->ind[ct->dim_perm[0]] + nnzstart;

  /* grab sparsity pattern */
  csf_sparsity * const pt = ct->pt + tile_id;

  if(omp_in_parallel()) {
    /* nfibs already counted in p_set_nfibs_root */
    idx_t nfibs = ct->pt[tile_id].nfibs[0];

    idx_t  * const restrict fp = pt->fptr[0];
    fidx_t  * const restrict fi = pt->fids[0];
    fp[0] = 0;
    if(fi != NULL) {
      fi[0] = ttind[0];
    }

    idx_t nfound = 1;
    if(fi != NULL) {
      for(idx_t n=1; n < nnz; ++n) {
        /* check for end of outer index */
        if(ttind[n] != ttind[n-1]) {
          fi[nfound] = ttind[n];
          fp[nfound++] = n;
        }
      }
    }
    else {
      for(idx_t n=1; n < nnz; ++n) {
        /* check for end of outer index */
        if(ttind[n] != ttind[n-1]) {
          assert(nfound == ttind[n]);
          fp[nfound++] = n;
        }
      }
    }

    fp[nfibs] = nnz;
  } /* omp_in_parallel */
  else {
    idx_t *nfibs = malloc(sizeof(idx_t)*(omp_get_max_threads() + 1));
    nfibs[0] = 1;

#pragma omp parallel
    {
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();

      idx_t x_per_thread = (nnz + nthreads - 1)/nthreads;
      idx_t x_begin = SS_MAX(SS_MIN(x_per_thread*tid, nnz), 1);
      idx_t x_end = SS_MIN(x_per_thread*(tid + 1), nnz);

      idx_t nfibs_private = 0;

      for(idx_t x = x_begin; x < x_end; ++x) {
        assert(ct->nslice_hubs > 0 || ttind[x-1] <= ttind[x]);
        if(ttind[x] != ttind[x-1]) {
          ++nfibs_private;
        }
      }
      nfibs[tid + 1] = nfibs_private;

#pragma omp barrier
#pragma omp master
      {
        /* prefix sum */
        for(int t = 0; t < nthreads; ++t) {
          nfibs[t + 1] += nfibs[t];
        }
        ct->pt[tile_id].nfibs[0] = nfibs[nthreads];
        assert(nfibs[nthreads] <= ct->dims[ct->dim_perm[0]]);

        pt->fptr[0] = splatt_malloc((nfibs[nthreads]+1) * sizeof(**(pt->fptr)));
        if(ct->ntiles > 1 || ct->nslice_hubs > 0) {
          pt->fids[0] = splatt_malloc(nfibs[nthreads] * sizeof(**(pt->fids)));
        } else {
          pt->fids[0] = NULL;
        }
      }
#pragma omp barrier

      idx_t  * const restrict fp = pt->fptr[0];
      fidx_t  * const restrict fi = pt->fids[0];
#pragma omp master
      {
        fp[0] = 0;
        if(fi != NULL) {
          fi[0] = ttind[0];
        }
        fp[nfibs[nthreads]] = nnz;
      }

      idx_t nfound = nfibs[tid];
      if(fi != NULL) {
        for(idx_t n=x_begin; n < x_end; ++n) {
          /* check for end of outer index */
          if(ttind[n] != ttind[n-1]) {
            fi[nfound] = ttind[n];
            fp[nfound++] = n;
          }
        }
      }
      else {
        for(idx_t n=x_begin; n < x_end; ++n) {
          /* check for end of outer index */
          if(ttind[n] != ttind[n-1]) {
            assert(nfound == ttind[n]);
            fp[nfound++] = n;
          }
        }
      }
    } /* omp parallel */
  } /* !omp_in_parallel */
}


static void p_set_nfibs(
  splatt_csf * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr,
  idx_t const mode)
{
  assert(mode < ct->nmodes);

  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  /* outer mode is easy; just look at outer indices */
  if(mode == 0) {
    p_set_nfibs_root(ct, tt, tile_id, nnztile_ptr);
    return;
  }
  /* the mode after accounting for dim_perm */
  fidx_t const * const restrict ttind = tt->ind[ct->dim_perm[mode]] + nnzstart;

  csf_sparsity * const pt = ct->pt + tile_id;

  /* we will edit this to point to the new fiber idxs instead of nnz */
  idx_t * const restrict fprev = pt->fptr[mode-1];

  /* first count nfibers */
  double t = omp_get_wtime();
  idx_t nfibs = 0;
  /* foreach 'slice' in the previous dimension */
  for(idx_t s=0; s < pt->nfibs[mode-1]; ++s) {
    ++nfibs; /* one by default per 'slice' */
    /* count fibers in current hyperplane*/
    for(idx_t f=fprev[s]+1; f < fprev[s+1]; ++f) {
      if(ttind[f] != ttind[f-1]) {
        ++nfibs;
      }
    }
  }
  pt->nfibs[mode] = nfibs;
}


/**
* @brief Construct the sparsity structure of any mode but the last. The first
*        (root) mode is handled by p_mk_outerptr and the first is simply a copy
*        of the nonzeros.
*
* @param ct The CSF tensor to construct.
* @param tt The coordinate tensor to construct from. Assumed to be already
*            sorted.
* @param tile_id The ID of the tile to construct.
* @param nnztile_ptr A pointer into 'tt' that marks the start of each tile.
* @param mode Which mode we are constructing.
*/
static void p_mk_fptr(
  splatt_csf * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr,
  idx_t const mode)
{
  assert(mode < ct->nmodes);

  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  /* outer mode is easy; just look at outer indices */
  if(mode == 0) {
    p_mk_outerptr(ct, tt, tile_id, nnztile_ptr);
    return;
  }
  /* the mode after accounting for dim_perm */
  fidx_t const * const restrict ttind = tt->ind[ct->dim_perm[mode]] + nnzstart;

  csf_sparsity * const pt = ct->pt + tile_id;

  /* we will edit this to point to the new fiber idxs instead of nnz */
  idx_t * const restrict fprev = pt->fptr[mode-1];

  if(omp_in_parallel()) {
    /* nfibers already counted in p_set_nfibs */
    idx_t nfibs = pt->nfibs[mode];


    idx_t * const restrict fp = pt->fptr[mode];
    fidx_t * const restrict fi = pt->fids[mode];
    fp[0] = 0;

    /* now fill in fiber info */
    idx_t nfound = 0;
    for(idx_t s=0; s < pt->nfibs[mode-1]; ++s) {
      idx_t const start = fprev[s]+1;
      idx_t const end = fprev[s+1];

      /* mark start of subtree */
      fprev[s] = nfound;
      fi[nfound] = ttind[start-1];
      fp[nfound++] = start-1;

      /* mark fibers in current hyperplane */
      for(idx_t f=start; f < end; ++f) {
        if(ttind[f] != ttind[f-1]) {
          fi[nfound] = ttind[f];
          fp[nfound++] = f;
        }
      }
    }

    /* mark end of last hyperplane */
    fprev[pt->nfibs[mode-1]] = nfibs;
    fp[nfibs] = nnz;
  } /* omp_in_parallel */
  else {
    idx_t nfibs[omp_get_max_threads() + 1];
    nfibs[0] = 0;

#pragma omp parallel
    {
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();

      idx_t s_per_thread = (pt->nfibs[mode-1] + nthreads - 1)/nthreads;
      idx_t s_begin = SS_MIN(s_per_thread*tid, pt->nfibs[mode-1]);
      idx_t s_end = SS_MIN(s_begin + s_per_thread, pt->nfibs[mode-1]);

      /* first count nfibers */
      idx_t nfibs_private = 0;
      /* foreach 'slice' in the previous dimension */
      for(idx_t s=s_begin; s < s_end; ++s) {
        ++nfibs_private; /* one by default per 'slice' */
        /* count fibers in current hyperplane*/
        for(idx_t f=fprev[s]+1; f < fprev[s+1]; ++f) {
          if(ttind[f] != ttind[f-1]) {
            ++nfibs_private;
          }
        }
      }
      nfibs[tid + 1] = nfibs_private;
      idx_t fprev_end = fprev[s_end];

#pragma omp barrier
#pragma omp master
      {
        /* prefix sum */
        for(int t = 0; t < nthreads; ++t) {
          nfibs[t + 1] += nfibs[t];
        }
        pt->nfibs[mode] = nfibs[nthreads];

        pt->fptr[mode] = splatt_malloc((nfibs[nthreads]+1) * sizeof(**(pt->fptr)));
        pt->fids[mode] = splatt_malloc(nfibs[nthreads] * sizeof(**(pt->fids)));
      }
#pragma omp barrier

      idx_t * const restrict fp = pt->fptr[mode];
      fidx_t * const restrict fi = pt->fids[mode];

#pragma omp master
      {
        fp[0] = 0;
      }

      /* now fill in fiber info */
      idx_t nfound = nfibs[tid];
      for(idx_t s=s_begin; s < s_end; ++s) {
        idx_t const start = fprev[s]+1;
        idx_t const end = s == s_end - 1 ? fprev_end : fprev[s+1];

        /* mark start of subtree */
        fprev[s] = nfound;
        fi[nfound] = ttind[start-1];
        fp[nfound++] = start-1;

        /* mark fibers in current hyperplane */
        for(idx_t f=start; f < end; ++f) {
          if(ttind[f] != ttind[f-1]) {
            fi[nfound] = ttind[f];
            fp[nfound++] = f;
          }
        }
      }

      if(tid == nthreads - 1) {
        /* mark end of last hyperplane */
        fprev[pt->nfibs[mode-1]] = nfibs[nthreads];
        fp[nfibs[nthreads]] = nnz;
      }
    } /* omp parallel */
  } /* !omp_in_parallel */
}

/**
* @brief Allocate and fill a CSF tensor from a coordinate tensor without
*        tiling.
*
* @param ct The CSF tensor to fill out.
* @param tt The sparse tensor to start from.
*/
static void p_csf_alloc_untiled(
  splatt_csf * const ct,
  sptensor_t * const tt)
{
  idx_t const nmodes = tt->nmodes;
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);

  ct->ntiles = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ct->tile_dims[m] = 1;
  }
  ct->pt = splatt_malloc(sizeof(*(ct->pt)));

  csf_sparsity * const pt = ct->pt;

  /* check hub slices in root mode */
  idx_t last_fp = 0;
  ct->hub_slices = (splatt_fidx_t *)splatt_malloc(4 * omp_get_max_threads() * sizeof(int));
  ct->nslice_hubs = 0;
  idx_t nnz_hubs = 0;
#define FINE_GRAIN_PARTITION_OF_HUBS
#ifdef FINE_GRAIN_PARTITION_OF_HUBS
  for(idx_t i=1; i < ct->nnz; ++i) {
    if(tt->ind[ct->dim_perm[0]][i] != tt->ind[ct->dim_perm[0]][i-1]) {
      if(i - last_fp >= 0.5*ct->nnz/omp_get_max_threads()) {
        ct->hub_slices[ct->nslice_hubs++] = tt->ind[ct->dim_perm[0]][i-1];
        nnz_hubs += i - last_fp;
        assert(ct->nslice_hubs < 4 * omp_get_max_threads());
        printf("%d hub slice\n", ct->hub_slices[ct->nslice_hubs - 1]);
      }
      last_fp = i;
    }
  }
  if(ct->nnz - last_fp >= 0.5*ct->nnz/omp_get_max_threads()) {
    ct->hub_slices[ct->nslice_hubs++] = tt->ind[ct->dim_perm[0]][ct->nnz - 1];
    nnz_hubs += ct->nnz - last_fp;
    printf("%d hub slice\n", ct->hub_slices[ct->nslice_hubs - 1]);
  }
  printf("nslice_hubs = %d nnz_hubs = %ld\n", ct->nslice_hubs, nnz_hubs);
#endif

  if(ct->nslice_hubs > 0) {
    fidx_t **new_ind = splatt_malloc(nmodes*sizeof(fidx_t *));
    for(idx_t i=0; i < nmodes; ++i) {
#ifdef HBW_ALLOC
      hbw_posix_memalign((void **)&new_ind[i], 4096, tt->nnz*sizeof(fidx_t));
#else
      new_ind[i] = splatt_malloc(tt->nnz*sizeof(fidx_t));
#endif
    }
    storage_val_t *new_vals;
#ifdef HBW_ALLOC
    hbw_posix_memalign((void **)&new_vals, 4096, tt->nnz*sizeof(new_vals[0]));
#else
    new_vals = splatt_malloc(tt->nnz*sizeof(new_vals[0]));
#endif

    idx_t non_hub_idx = 0, hub_idx = tt->nnz - nnz_hubs;

    int hub_slice_idx = 0;
    for (idx_t i=0; i < ct->nnz; ++i) {
      while(ct->hub_slices[hub_slice_idx] < tt->ind[ct->dim_perm[0]][i] && hub_slice_idx < ct->nslice_hubs) {
        ++hub_slice_idx;
      }

      if(ct->hub_slices[hub_slice_idx] == tt->ind[ct->dim_perm[0]][i]) {
        if(hub_idx >= tt->nnz) {
          printf("i=%ld hub_slice_idx=%d\n", i, hub_slice_idx);
        }
        assert(hub_idx < tt->nnz);
        for(int m=0; m < nmodes; ++m) {
          new_ind[m][hub_idx] = tt->ind[m][i];
        }
        new_vals[hub_idx] = tt->vals[i];
        ++hub_idx;
      }
      else {
        assert(non_hub_idx < tt->nnz - nnz_hubs);
        for(int m=0; m < nmodes; ++m) {
          new_ind[m][non_hub_idx] = tt->ind[m][i];
        }
        new_vals[non_hub_idx] = tt->vals[i];
        ++non_hub_idx;
      }
    }
    assert(non_hub_idx == tt->nnz - nnz_hubs);
    assert(hub_idx == tt->nnz);

    for(int m=0; m < nmodes; ++m) {
#ifdef HBW_ALLOC
      hbw_free(tt->ind[m]);
#else
      splatt_free(tt->ind[m]);
#endif
      tt->ind[m] = new_ind[m];
    }
    splatt_free(new_ind);

#ifdef HBW_ALLOC
    hbw_free(tt->vals);
#else
    splatt_free(tt->vals);
#endif
    tt->vals = new_vals;
  }
  else {
    ct->hub_slices = NULL;
    ct->nslice_hubs = 0;
  }

  /* last row of fptr is just nonzero inds */
  pt->nfibs[nmodes-1] = ct->nnz;
#ifdef HBW_ALLOC
  hbw_posix_memalign((void **)&pt->fids[nmodes-1], 4096, ct->nnz * sizeof(**(pt->fids)));
  hbw_posix_memalign((void **)&pt->vals, 4096, ct->nnz * sizeof(*(pt->vals)));
#else
  pt->fids[nmodes-1] = splatt_malloc(ct->nnz * sizeof(**(pt->fids)));
  pt->vals           = splatt_malloc(ct->nnz * sizeof(*(pt->vals)));
#endif
#pragma omp parallel for
  for (idx_t i=0; i < ct->nnz; ++i) {
    pt->fids[nmodes-1][i] = tt->ind[ct->dim_perm[nmodes-1]][i];
  }
#pragma omp parallel for
  for(idx_t i=0; i < ct->nnz; ++i) {
    pt->vals[i] = tt->vals[i];
  }

  /* setup a basic tile ptr for one tile */
  idx_t nnz_ptr[2];
  nnz_ptr[0] = 0;
  nnz_ptr[1] = tt->nnz;

  /* create fptr entries for the rest of the modes, working down from roots.
   * Skip the bottom level (nnz) */
  for(idx_t m=0; m < tt->nmodes-1; ++m) {
    p_mk_fptr(ct, tt, 0, nnz_ptr, m);
  }
}


/**
* @brief Reorder the nonzeros in a sparse tensor using dense tiling and fill
*        a CSF tensor with the data.
*
* @param ct The CSF tensor to fill.
* @param tt The sparse tensor to start from.
* @param splatt_opts Options array for SPLATT - used for tile dimensions.
*/
static void p_csf_alloc_densetile(
  splatt_csf * const ct,
  sptensor_t * const tt,
  double const * const splatt_opts)
{
  idx_t const nmodes = tt->nmodes;

  idx_t ntiles = 1;
  for(idx_t m=0; m < ct->nmodes; ++m) {
    idx_t const depth = csf_mode_depth(m, ct->dim_perm, ct->nmodes);
    if(depth >= splatt_opts[SPLATT_OPTION_TILEDEPTH]) {
      ct->tile_dims[m] = (idx_t) splatt_opts[SPLATT_OPTION_NTHREADS];
    } else {
      ct->tile_dims[m] = 1;
    }
    ntiles *= ct->tile_dims[m];
  }

  /* perform tensor tiling */
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);
  idx_t * nnz_ptr = tt_densetile(tt, ct->tile_dims);

  ct->ntiles = ntiles;
  ct->pt = splatt_malloc(ntiles * sizeof(*(ct->pt)));
  ct->hub_slices = NULL;
  ct->nslice_hubs = 0;

  fidx_t *fids_buf = NULL;
  val_t *vals_buf = NULL;

#ifdef HBW_ALLOC
  int ret = hbw_posix_memalign((void **)&fids_buf, 4096, ct->nnz * sizeof(**(ct->pt[0].fids)));
  if(ret != 0) {
    fprintf(stderr, "SPLATT: hbw_posix_memalign() returned %d.\n", ret);
    assert(0);
  }
  ret = hbw_posix_memalign((void **)&vals_buf, 4096, ct->nnz * sizeof(*(ct->pt[0].vals)));
  if(ret != 0) {
    fprintf(stderr, "SPLATT: hbw_posix_memalign() returned %d.\n", ret);
    assert(0);
  }
#else
  fids_buf = splatt_malloc(ct->nnz * sizeof(**(ct->pt[0].fids)));
  vals_buf = splatt_malloc(ct->nnz * sizeof(*(ct->pt[0].vals)));
#endif

  for(idx_t m=0; m < nmodes-1; ++m) {
#pragma omp parallel for if (ntiles > 1) schedule(dynamic, 1)
    for(idx_t t=0; t < ntiles; ++t) {
      idx_t const startnnz = nnz_ptr[t];
      idx_t const endnnz   = nnz_ptr[t+1];
      assert(endnnz >= startnnz);
      idx_t const ptnnz = endnnz - startnnz;

      csf_sparsity * const pt = ct->pt + t;

      /* empty tile */
      if(ptnnz == 0) {
        if(0 == m) {
          for(idx_t i=0; i < ct->nmodes; ++i) {
            pt->fptr[i] = NULL;
            pt->fids[i] = NULL;
            pt->nfibs[i] = 0;
          }
          /* first fptr may be accessed anyway */
          if(!omp_in_parallel()) {
            pt->fptr[0] = (idx_t *) splatt_malloc(2 * sizeof(**(pt->fptr)));
            pt->fptr[0][0] = 0;
            pt->fptr[0][1] = 0;
          }
          pt->vals = NULL;
        }
      }
      else {
        /* last row of fptr is just nonzero inds */
        pt->nfibs[nmodes-1] = ptnnz;

        pt->fids[nmodes-1] = fids_buf + startnnz;
        if (omp_in_parallel()) {
          for (idx_t i = 0; i < ptnnz; ++i) {
            pt->fids[nmodes-1][i] = tt->ind[ct->dim_perm[nmodes-1]][startnnz + i];
          }
        }
        else {
#pragma omp parallel for
          for (idx_t i = 0; i < ptnnz; ++i) {
            pt->fids[nmodes-1][i] = tt->ind[ct->dim_perm[nmodes-1]][startnnz + i];
          }
        }

        pt->vals = vals_buf + startnnz;
        if (omp_in_parallel()) {
          for(idx_t j=0; j < ptnnz; ++j) {
            pt->vals[j] = tt->vals[startnnz + j];
          }
        }
        else {
#pragma omp parallel for
          for(idx_t j=0; j < ptnnz; ++j) {
            pt->vals[j] = tt->vals[startnnz + j];
          }
        }

        /* create fptr entries for the rest of the modes */
        if(omp_in_parallel()) {
          p_set_nfibs(ct, tt, t, nnz_ptr, m);
        }
        else {
          p_mk_fptr(ct, tt, t, nnz_ptr, m);
        }
      }
    } /* for each tile */

    if(ntiles > 1) {
      idx_t nfibs_acc = 0;
      idx_t nempty = 0;
      for(idx_t t=0; t < ntiles; ++t) {
        if(nnz_ptr[t+1] - nnz_ptr[t] == 0) {
          if(0 == m) ++nempty;
        }
        else nfibs_acc += ct->pt[t].nfibs[m];
      }

      ct->pt[0].fptr[m] = splatt_malloc((nfibs_acc + ntiles + nempty) * sizeof(**(ct->pt->fptr)));
      ct->pt[0].fids[m] = splatt_malloc(nfibs_acc * sizeof(**(ct->pt->fids)));

      nfibs_acc = 0;
      nempty = 0;
      for(idx_t t=0; t < ntiles; ++t) {
        ct->pt[t].fptr[m] = ct->pt[0].fptr[m] + nfibs_acc + t + nempty;
        ct->pt[t].fids[m] = ct->pt[0].fids[m] + nfibs_acc;
        if(nnz_ptr[t+1] - nnz_ptr[t] == 0) {
          if(0 == m) {
            ct->pt[t].fptr[0][0] = 0;
            ct->pt[t].fptr[0][1] = 0;
            ++nempty;
          }
        }
        else nfibs_acc += ct->pt[t].nfibs[m];
      }

#pragma omp parallel for schedule(dynamic, 1)
      for(idx_t t=0; t < ntiles; ++t) {
        if(nnz_ptr[t+1] > nnz_ptr[t]) {
          p_mk_fptr(ct, tt, t, nnz_ptr, m);
        }
      } /* for each tile */
    }
  } /* for each mode */

  free(nnz_ptr);
}


/**
* @brief Allocate and fill a CSF tensor.
*
* @param ct The CSF tensor to fill.
* @param tt The coordinate tensor to work from.
* @param mode_type The allocation scheme for the CSF tensor.
* @param mode Which mode we are converting for (if applicable).
* @param splatt_opts Used to determine tiling scheme.
*/
static void p_mk_csf(
  splatt_csf * const ct,
  sptensor_t * const tt,
  csf_mode_type mode_type,
  idx_t const mode,
  double const * const splatt_opts)
{
  ct->nnz = tt->nnz;
  ct->nmodes = tt->nmodes;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ct->dims[m] = tt->dims[m];
  }

  /* get the indices in order */
  csf_find_mode_order(tt->dims, tt->nmodes, mode_type, mode, ct->dim_perm, ct->nnz, splatt_opts);

  ct->which_tile = (splatt_tile_type)splatt_opts[SPLATT_OPTION_TILE];
  switch(ct->which_tile) {
  case SPLATT_NOTILE:
    p_csf_alloc_untiled(ct, tt);
    break;
  case SPLATT_DENSETILE:
    p_csf_alloc_densetile(ct, tt, splatt_opts);
    break;
  default:
    fprintf(stderr, "SPLATT: tiling '%d' unsupported for CSF tensors.\n",
        ct->which_tile);
    break;
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


void csf_free(
  splatt_csf * const csf,
  double const * const opts)
{
  idx_t ntensors = 0;
  splatt_csf_type which = (splatt_csf_type)opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {
  case SPLATT_CSF_ONEMODE:
    ntensors = 1;
    break;
  case SPLATT_CSF_TWOMODE:
    ntensors = 2;
    break;
  case SPLATT_CSF_ALLMODE:
    ntensors = csf[0].nmodes;
    break;
  }

  for(idx_t i=0; i < ntensors; ++i) {
    csf_free_mode(csf + i);
  }

  free(csf);
}


void csf_free_mode(
    splatt_csf * const csf)
{
  /* free each tile of sparsity pattern */
#ifdef HBW_ALLOC
  hbw_free(csf->pt[0].vals);
  hbw_free(csf->pt[0].fids[csf->nmodes-1]);
#else
  free(csf->pt[0].vals);
  free(csf->pt[0].fids[csf->nmodes-1]);
#endif
  for(idx_t m=0; m < csf->nmodes-1; ++m) {
    free(csf->pt[0].fptr[m]);
    free(csf->pt[0].fids[m]);
  }
  splatt_free(csf->hub_slices);
  free(csf->pt);
}



void csf_find_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  csf_mode_type which,
  idx_t const mode,
  idx_t * const perm_dims,
  idx_t nnz,
  double const * const opts)
{
  switch(which) {
  case CSF_SORTED_SMALLFIRST:
    if(nmodes >= 6) {
      // FIXME: temporaily using this for outpatient6
      p_order_dims_small(dims, nmodes, perm_dims);
    }
    else {
      p_order_dims_small_no_privatization(dims, nmodes, perm_dims, nnz, opts);
    }
    break;

  case CSF_SORTED_BIGFIRST:
    p_order_dims_large(dims, nmodes, perm_dims);
    break;

  case CSF_INORDER_MINUSONE:
    p_order_dims_inorder(dims, nmodes, mode, perm_dims);
    break;

  case CSF_SORTED_MINUSONE:
    p_order_dims_minusone(dims, nmodes, mode, perm_dims);
    break;

  case CSF_ROUND_ROBIN:
    p_order_dims_round_robin(dims, nmodes, mode, perm_dims);
    break;

  case CSF_ALLPERMUTE:
    p_order_dims_all_permute(dims, nmodes, mode, perm_dims);
    break;

  default:
    fprintf(stderr, "SPLATT: csf_mode_type '%d' not recognized.\n", which);
    break;
  }
}


size_t csf_storage(
  splatt_csf const * const tensors,
  double const * const opts)
{
  idx_t ntensors = 0;
  splatt_csf_type which_alloc = (splatt_csf_type)opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which_alloc) {
  case SPLATT_CSF_ONEMODE:
    ntensors = 1;
    break;
  case SPLATT_CSF_TWOMODE:
    ntensors = 2;
    break;
  case SPLATT_CSF_ALLMODE:
    ntensors = tensors[0].nmodes;
    break;
  }

  size_t total_bytes = 0;
  for(idx_t m=0; m < ntensors; ++m) {
    size_t bytes = 0;

    splatt_csf * const ct = (splatt_csf *)(tensors + m);
    bytes += ct->nnz * sizeof(*(ct->pt->vals)); /* vals */
    bytes += ct->nnz * sizeof(**(ct->pt->fids)); /* fids[nmodes] */
    bytes += ct->ntiles * sizeof(*(ct->pt)); /* pt */

    for(idx_t t=0; t < ct->ntiles; ++t) {
      csf_sparsity const * const pt = ct->pt + t;

      for(idx_t m=0; m < ct->nmodes-1; ++m) {
        bytes += (pt->nfibs[m]+1) * sizeof(**(pt->fptr)); /* fptr */
        if(pt->fids[m] != NULL) {
          bytes += pt->nfibs[m] * sizeof(**(pt->fids)); /* fids */
        }
      }
    }

    ct->storage = bytes;
    total_bytes += bytes;
  }

  return total_bytes;
}


splatt_csf * csf_alloc(
  sptensor_t * const tt,
  double const * const opts)
{
  splatt_csf * ret = NULL;

  double * tmp_opts = NULL;
  idx_t last_mode = 0;

  int tmp = 0;

  switch((splatt_csf_type) opts[SPLATT_OPTION_CSF_ALLOC]) {
  case SPLATT_CSF_ONEMODE:
    ret = splatt_malloc(sizeof(*ret));
    p_mk_csf(ret, tt, CSF_SORTED_SMALLFIRST, 0, opts);
    break;

  case SPLATT_CSF_TWOMODE:
    ret = splatt_malloc(2 * sizeof(*ret));
    /* regular CSF allocation */
    p_mk_csf(ret + 0, tt, CSF_SORTED_SMALLFIRST, 0, opts);

    /* make a copy of opts and don't tile the last mode
     * TODO make this configurable? */
    tmp_opts = splatt_default_opts();
    memcpy(tmp_opts, opts, SPLATT_OPTION_NOPTIONS * sizeof(*opts));
    tmp_opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

    /* allocate with no tiling for the last mode */
    last_mode = ret[0].dim_perm[tt->nmodes-1];
    p_mk_csf(ret + 1, tt, CSF_SORTED_MINUSONE, last_mode, tmp_opts);

    free(tmp_opts);
    break;

  case SPLATT_CSF_ALLMODE:
    ret = splatt_malloc(tt->nmodes * sizeof(*ret));
    for(idx_t m=0; m < tt->nmodes; ++m) {
      p_mk_csf(ret + m, tt, CSF_SORTED_MINUSONE, m, opts);
    }
    break;

  case SPLATT_CSF_ALLMODE_ROUND_ROBIN:
    ret = splatt_malloc(tt->nmodes * sizeof(*ret));
    for(idx_t m=0; m < tt->nmodes; ++m) {
      p_mk_csf(ret + m, tt, CSF_ROUND_ROBIN, m, opts);
    }
    break;

  case SPLATT_CSF_ALLPERMUTE:
    assert(tt->nmodes == 3);
    ret = splatt_malloc(6 * sizeof(*ret));
    for(idx_t m=0; m < 6; ++m) {
      p_mk_csf(ret + m, tt, CSF_ALLPERMUTE, m, opts);
    }
    break;
  }

  return ret;
}


void csf_alloc_mode(
  sptensor_t * const tt,
  csf_mode_type which_ordering,
  idx_t const mode_special,
  splatt_csf * const csf,
  double const * const opts)
{
  p_mk_csf(csf, tt, which_ordering, mode_special, opts);
}


val_t csf_frobsq(
    splatt_csf const * const tensor)
{
  idx_t const nmodes = tensor->nmodes;
  val_t norm = 0;
  #pragma omp parallel reduction(+:norm)
  {
    for(idx_t t=0; t < tensor->ntiles; ++t) {
      val_t const * const vals = tensor->pt[t].vals;
      if(vals == NULL) {
        continue;
      }

      idx_t const nnz = tensor->pt[t].nfibs[nmodes-1];

      #pragma omp for nowait
      for(idx_t n=0; n < nnz; ++n) {
        norm += vals[n] * vals[n];
      }
    }
  }

  return norm;
}


int csf_get_ncopies(double *opts, int nmodes)
{
  int ncopies = -1;
  splatt_csf_type which_csf = (splatt_csf_type)opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which_csf) {
  case SPLATT_CSF_ONEMODE:
    ncopies = 1;
    break;
  case SPLATT_CSF_TWOMODE:
    ncopies = 2;
    break;
  case SPLATT_CSF_ALLMODE:
    ncopies = nmodes;
    break;
  }

  return ncopies;
}

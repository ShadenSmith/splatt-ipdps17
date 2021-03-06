

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "matrix.h"
#include "sort.h"
#include "io.h"
#include "timer.h"

#include <math.h>

/******************************************************************************
 * PRIVATE FUNCTONS
 *****************************************************************************/
static inline int p_same_coord(
  sptensor_t const * const tt,
  idx_t const i,
  idx_t const j)
{
  idx_t const nmodes = tt->nmodes;
  if(nmodes == 3) {
    return (tt->ind[0][i] == tt->ind[0][j]) &&
           (tt->ind[1][i] == tt->ind[1][j]) &&
           (tt->ind[2][i] == tt->ind[2][j]);
  } else {
    for(idx_t m=0; m < nmodes; ++m) {
      if(tt->ind[m][i] != tt->ind[m][j]) {
        return 0;
      }
    }
    return 1;
  }
}




/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

val_t tt_normsq(sptensor_t const * const tt)
{
  val_t norm = 0.0;
  storage_val_t const * const restrict tv = tt->vals;
  for(idx_t n=0; n < tt->nnz; ++n) {
    norm += tv[n] * tv[n];
  }
  return norm;
}


double tt_density(
  sptensor_t const * const tt)
{
  double root = pow((double)tt->nnz, 1./(double)tt->nmodes);
  double density = 1.0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    density *= root / (double)tt->dims[m];
  }

  return density;
}


void tt_print_hist(sptensor_t const * const tt, int mode)
{
  int *hist = (int *)splatt_malloc(tt->dims[mode]*sizeof(*hist));
#pragma omp parallel for
  for(idx_t n=0; n < tt->dims[mode]; ++n) {
    hist[n] = 0;
  }
#pragma omp parallel for
  for(idx_t n=0; n < tt->nnz; ++n) {
#pragma omp atomic
    hist[tt->ind[mode][n]]++;
  }
  for(idx_t n=0; n < tt->dims[mode]; ++n) {
    if(hist[n] > 0) {
      printf("%ld %d\n", n, hist[n]);
    }
  }
  splatt_free(hist);
}


idx_t * tt_get_slices(
  sptensor_t const * const tt,
  idx_t const m,
  idx_t * nunique)
{
  /* get maximum number of unique slices */
  idx_t minidx = tt->dims[m];
  idx_t maxidx = 0;

  idx_t const nnz = tt->nnz;
  fidx_t const * const inds = tt->ind[m];

  /* find maximum number of uniques */
#pragma omp parallel for reduction(min:minidx), reduction(max:maxidx)
  for(idx_t n=0; n < nnz; ++n) {
    minidx = SS_MIN(minidx, inds[n]);
    maxidx = SS_MAX(maxidx, inds[n]);
  }
  /* +1 because maxidx is inclusive, not exclusive */
  idx_t const maxrange = 1 + maxidx - minidx;

  /* mark slices which are present and count uniques */
  idx_t * slice_mkrs = calloc(maxrange, sizeof(*slice_mkrs));
  idx_t found = 0;
#pragma omp parallel for reduction(+:found)
  for(idx_t n=0; n < nnz; ++n) {
    assert(inds[n] >= minidx);
    idx_t const idx = inds[n] - minidx;
    if(__sync_bool_compare_and_swap(slice_mkrs + idx, 0, 1)) {
      ++found;
    }
  }
  *nunique = found;

  /* now copy unique slices */
  idx_t * slices = (idx_t *) splatt_malloc(found * sizeof(*slices));
  idx_t ptr = 0;
  for(idx_t i=0; i < maxrange; ++i) {
    if(slice_mkrs[i] == 1) {
      slices[ptr++] = i + minidx;
    }
  }

  free(slice_mkrs);

  return slices;
}


idx_t tt_remove_dups(
  sptensor_t * const tt)
{
  tt_sort(tt, 0, NULL);

  idx_t const nmodes = tt->nmodes;

  idx_t newnnz = 0;
  for(idx_t nnz = 1; nnz < tt->nnz; ++nnz) {
    /* if the two nnz are the same, average them */
    if(p_same_coord(tt, newnnz, nnz)) {
      tt->vals[newnnz] += tt->vals[nnz];
    } else {
      /* new another nnz */
      ++newnnz;
      for(idx_t m=0; m < nmodes; ++m) {
        tt->ind[m][newnnz] = tt->ind[m][nnz];
      }
      tt->vals[newnnz] = tt->vals[nnz];
    }
  }
  ++newnnz;

  idx_t const diff = tt->nnz - newnnz;
  tt->nnz = newnnz;
  return diff;
}


idx_t tt_remove_empty(
  sptensor_t * const tt)
{
  idx_t dim_sizes[MAX_NMODES];

  idx_t nremoved = 0;

  /* Allocate indmap */
  idx_t const nmodes = tt->nmodes;
  idx_t const nnz = tt->nnz;

  idx_t maxdim = 0;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    maxdim = tt->dims[m] > maxdim ? tt->dims[m] : maxdim;
  }
  /* slice counts */
  idx_t * scounts = (idx_t *) splatt_malloc(maxdim * sizeof(*scounts));

  for(idx_t m=0; m < nmodes; ++m) {
    dim_sizes[m] = 0;
    memset(scounts, 0, maxdim * sizeof(*scounts));

    /* Fill in indmap */
    for(idx_t n=0; n < tt->nnz; ++n) {
      /* keep track of #unique slices */
      if(scounts[tt->ind[m][n]] == 0) {
        scounts[tt->ind[m][n]] = 1;
        ++dim_sizes[m];
      }
    }

    /* move on if no remapping is necessary */
    if(dim_sizes[m] == tt->dims[m]) {
      tt->indmap[m] = NULL;
      continue;
    }

    nremoved += tt->dims[m] - dim_sizes[m];

    /* Now scan to remove empty slices */
    idx_t ptr = 0;
    for(idx_t i=0; i < tt->dims[m]; ++i) {
      if(scounts[i] == 1) {
        scounts[i] = ptr++;
      }
    }

    tt->indmap[m] = splatt_malloc(dim_sizes[m] * sizeof(**tt->indmap));

    /* relabel all indices in mode m */
    tt->dims[m] = dim_sizes[m];
#pragma omp parallel for
    for(idx_t n=0; n < tt->nnz; ++n) {
      idx_t const global = tt->ind[m][n];
      idx_t const local = scounts[global];
      assert(local < dim_sizes[m]);
      tt->indmap[m][local] = global; /* store local -> global mapping */
      tt->ind[m][n] = local;
    }
  }

  splatt_free(scounts);
  return nremoved;
}



/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/
sptensor_t * tt_read(
  char const * const ifname)
{
  int l = strlen(ifname);
  sptensor_t * tt;
  if (l > 4 && !strcmp(ifname + l - 4, ".bin")) {
    tt = tt_read_binary_file(ifname);
  }
  else {
    tt = tt_read_file(ifname);
  }

  return tt;
}


sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes)
{
  sptensor_t * tt = splatt_malloc(sizeof(*tt));
  tt->tiled = SPLATT_NOTILE;

  tt->nnz = nnz;
#if SPLATT_SPTENSOR_HBW
  tt->vals = splatt_hbw_malloc(nnz * sizeof(*tt->vals));
#else
  tt->vals = splatt_malloc(nnz * sizeof(*tt->vals));
#endif

  tt->nmodes = nmodes;
  tt->type = (nmodes == 3) ? SPLATT_3MODE : SPLATT_NMODE;

  tt->dims = splatt_malloc(nmodes * sizeof(*tt->dims));
  tt->ind = splatt_malloc(nmodes * sizeof(*tt->ind));
#if SPLATT_SPTENSOR_HBW
  for(idx_t m=0; m < nmodes; ++m) {
    tt->ind[m] = splatt_hbw_malloc(nnz * sizeof(**tt->ind));
    tt->indmap[m] = NULL;
  }
#else
  for(idx_t m=0; m < nmodes; ++m) {
    tt->ind[m] = splatt_malloc(nnz * sizeof(**tt->ind));
    tt->indmap[m] = NULL;
  }
#endif

  return tt;
}


void tt_fill(
  sptensor_t * const tt,
  idx_t const nnz,
  idx_t const nmodes,
  fidx_t ** const inds,
  storage_val_t * const vals)
{
  tt->tiled = SPLATT_NOTILE;
  tt->nnz = nnz;
  tt->vals = vals;
  tt->ind = inds;

  tt->nmodes = nmodes;
  tt->type = (nmodes == 3) ? SPLATT_3MODE : SPLATT_NMODE;

  tt->dims = (idx_t*) splatt_malloc(nmodes * sizeof(*tt->dims));
  for(idx_t m=0; m < nmodes; ++m) {
    tt->indmap[m] = NULL;

    tt->dims[m] = inds[m][0];
    for(idx_t i=1; i < nnz; ++i) {
      tt->dims[m] = SS_MAX(tt->dims[m], inds[m][i]);
    }
  }
}



void tt_free(
  sptensor_t * tt)
{
#if SPLATT_SPTENSOR_HBW
  splatt_hbw_free(tt->vals);
#else
  splatt_free(tt->vals);
#endif

  for(idx_t m=0; m < tt->nmodes; ++m) {
#if SPLATT_SPTENSOR_HBW
    splatt_hbw_free(tt->ind[m]);
#else
    splatt_free(tt->ind[m]);
#endif
    splatt_free(tt->indmap[m]);
  }

  splatt_free(tt->dims);
  splatt_free(tt->ind);
  splatt_free(tt);
}


spmatrix_t * tt_unfold(
  sptensor_t * const tt,
  idx_t const mode)
{
  idx_t nrows = tt->dims[mode];
  idx_t ncols = 1;

  for(idx_t m=1; m < tt->nmodes; ++m) {
    ncols *= tt->dims[(mode + m) % tt->nmodes];
  }

  /* sort tt */
  tt_sort(tt, mode, NULL);

  /* allocate and fill matrix */
  spmatrix_t * mat = spmat_alloc(nrows, ncols, tt->nnz);
  idx_t * const rowptr = mat->rowptr;
  idx_t * const colind = mat->colind;
  val_t * const mvals  = mat->vals;

  /* make sure to skip ahead to the first non-empty slice */
  idx_t row = 0;
  for(idx_t n=0; n < tt->nnz; ++n) {
    /* increment row and account for possibly empty ones */
    while(row <= tt->ind[mode][n]) {
      rowptr[row++] = n;
    }
    mvals[n] = tt->vals[n];

    idx_t col = 0;
    idx_t mult = 1;
    for(idx_t m = 0; m < tt->nmodes; ++m) {
      idx_t const off = tt->nmodes - 1 - m;
      if(off == mode) {
        continue;
      }
      col += tt->ind[off][n] * mult;
      mult *= tt->dims[off];
    }

    colind[n] = col;
  }
  /* account for any empty rows at end, too */
  for(idx_t r=row; r <= nrows; ++r) {
    rowptr[r] = tt->nnz;
  }

  return mat;
}


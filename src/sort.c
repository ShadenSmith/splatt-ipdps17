

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <omp.h>

#include "sort.h"
#include "timer.h"
#include "util.h"

#ifdef __AVX512F__
//#define HBW_ALLOC
  /* define this and run with "numalloc -m 1" and MEMKIND_HBW_NODES=0
   * to allocate non-performance critical performance data to DDR */
#endif
#ifdef HBW_ALLOC
#include <hbwmalloc.h>
#endif

/******************************************************************************
 * DEFINES
 *****************************************************************************/
/* switch to insertion sort past this point */
#define MIN_QUICKSORT_SIZE 8



/******************************************************************************
 * STATIC FUNCTIONS
 *****************************************************************************/


/**
* @brief Compares ind*[i] and j[*] for three-mode tensors.
*
* @param ind0 The primary mode to compare. Defer tie-breaks to ind1.
* @param ind1 The secondary mode to compare. Defer tie-breaks to ind2.
* @param ind2 The final tie-breaking mode.
* @param i The index into ind*[]
* @param j[3] The indices we are comparing i against.
*
* @return Returns -1 if ind[i] < j, 1 if ind[i] > j, and 0 if they are equal.
*/
static inline int p_ttqcmp3(
  fidx_t const * const ind0,
  fidx_t const * const ind1,
  fidx_t const * const ind2,
  idx_t const i,
  idx_t const j[3])
{
  if(ind0[i] < j[0]) {
    return -1;
  } else if(j[0] < ind0[i]) {
    return 1;
  }
  if(ind1[i] < j[1]) {
    return -1;
  } else if(j[1] < ind1[i]) {
    return 1;
  }
  if(ind2[i] < j[2]) {
    return -1;
  } else if(j[2] < ind2[i]) {
    return 1;
  }

  return 0;
}


/**
* @brief Compares ind*[i] and ind*[j] for three-mode tensors.
*
* @param ind0 The primary mode to compare. Defer tie-breaks to ind1.
* @param ind1 The secondary mode to compare. Defer tie-breaks to ind2.
* @param ind2 The final tie-breaking mode.
* @param i The index into ind*.
* @param j The second index into ind*.
*
* @return Returns -1 if ind[i] < ind[j], 1 if ind[i] > ind[j], and 0 if they
*         are equal.
*/
static inline int p_ttcmp3(
  fidx_t const * const ind0,
  fidx_t const * const ind1,
  fidx_t const * const ind2,
  idx_t const i,
  idx_t const j)
{
  if(ind0[i] < ind0[j]) {
    return -1;
  } else if(ind0[j] < ind0[i]) {
    return 1;
  }
  if(ind1[i] < ind1[j]) {
    return -1;
  } else if(ind1[j] < ind1[i]) {
    return 1;
  }
  if(ind2[i] < ind2[j]) {
    return -1;
  } else if(ind2[j] < ind2[i]) {
    return 1;
  }
  return 0;
}

/**
* @brief Compares ind*[i] and j[*] for two-mode tensors.
*
* @param ind0 The primary mode to compare. Defer tie-breaks to ind1.
* @param ind1 The secondary mode to compare.
* @param i The index into ind*[]
* @param j[2] The indices we are comparing i against.
*
* @return Returns -1 if ind[i] < j, 1 if ind[i] > j, and 0 if they are equal.
*/
static inline int p_ttqcmp2(
  fidx_t const * const ind0,
  fidx_t const * const ind1,
  idx_t const i,
  idx_t const j[2])
{
  if(ind0[i] < j[0]) {
    return -1;
  } else if(j[0] < ind0[i]) {
    return 1;
  }
  if(ind1[i] < j[1]) {
    return -1;
  } else if(j[1] < ind1[i]) {
    return 1;
  }

  return 0;
}


/**
* @brief Compares ind*[i] and ind*[j] for two-mode tensors.
*
* @param ind0 The primary mode to compare. Defer tie-breaks to ind1.
* @param ind1 The secondary mode to compare.
* @param i The index into ind*.
* @param j The second index into ind*.
*
* @return Returns -1 if ind[i] < ind[j], 1 if ind[i] > ind[j], and 0 if they
*         are equal.
*/
static inline int p_ttcmp2(
  fidx_t const * const ind0,
  fidx_t const * const ind1,
  idx_t const i,
  idx_t const j)
{
  if(ind0[i] < ind0[j]) {
    return -1;
  } else if(ind0[j] < ind0[i]) {
    return 1;
  }
  if(ind1[i] < ind1[j]) {
    return -1;
  } else if(ind1[j] < ind1[i]) {
    return 1;
  }
  return 0;
}


/**
* @brief Compares ind*[i] and ind*[j] for n-mode tensors.
*
* @param tt The tensor we are sorting.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param i The index into ind*.
* @param j The second index into ind*.
*
* @return Returns -1 if ind[i] < ind[j], 1 if ind[i] > ind[j], and 0 if they
*         are equal.
*/
static inline int p_ttcmp(
  sptensor_t const * const tt,
  idx_t const * const cmplt,
  idx_t const i,
  idx_t const j)
{
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(tt->ind[cmplt[m]][i] < tt->ind[cmplt[m]][j]) {
      return -1;
    } else if(tt->ind[cmplt[m]][j] < tt->ind[cmplt[m]][i]) {
      return 1;
    }
  }
  return 0;
}


/**
* @brief Compares ind*[i] and ind*[j] for n-mode tensors.
*
* @param tt The tensor we are sorting.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param i The index into ind*.
* @param j The coordinate we are comparing against.
*
* @return Returns -1 if ind[i] < j, 1 if ind[i] > j, and 0 if they are equal.
*/
static inline int p_ttqcmp(
  sptensor_t const * const tt,
  idx_t const * const cmplt,
  idx_t const i,
  idx_t const j[MAX_NMODES])
{
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(tt->ind[cmplt[m]][i] < j[cmplt[m]]) {
      return -1;
    } else if(j[cmplt[m]] < tt->ind[cmplt[m]][i]) {
      return 1;
    }
  }
  return 0;
}


/**
* @brief Swap nonzeros i and j.
*
* @param tt The tensor to operate on.
* @param i The first nonzero to swap.
* @param j The second nonzero to swap with.
*/
static inline void p_ttswap(
  sptensor_t * const tt,
  idx_t const i,
  idx_t const j)
{
  storage_val_t vtmp = tt->vals[i];
  tt->vals[i] = tt->vals[j];
  tt->vals[j] = vtmp;

  idx_t itmp;
  for(idx_t m=0; m < tt->nmodes; ++m) {
    itmp = tt->ind[m][i];
    tt->ind[m][i] = tt->ind[m][j];
    tt->ind[m][j] = itmp;
  }
}


/**
* @brief Perform insertion sort on a 3-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_insertionsort3(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  fidx_t * const ind0 = tt->ind[cmplt[0]];
  fidx_t * const ind1 = tt->ind[cmplt[1]];
  fidx_t * const ind2 = tt->ind[cmplt[2]];
  storage_val_t * const vals = tt->vals;

  storage_val_t vbuf;
  idx_t ibuf;

  for(size_t i=start+1; i < end; ++i) {
    size_t j = i;
    while (j > start && p_ttcmp3(ind0, ind1, ind2, i, j-1) < 0) {
      --j;
    }

    vbuf = vals[i];

    /* shift all data */
    memmove(vals+j+1, vals+j, (i-j)*sizeof(vals[0]));
    vals[j] = vbuf;
    ibuf = ind0[i];
    memmove(ind0+j+1, ind0+j, (i-j)*sizeof(fidx_t));
    ind0[j] = ibuf;
    ibuf = ind1[i];
    memmove(ind1+j+1, ind1+j, (i-j)*sizeof(fidx_t));
    ind1[j] = ibuf;
    ibuf = ind2[i];
    memmove(ind2+j+1, ind2+j, (i-j)*sizeof(fidx_t));
    ind2[j] = ibuf;
  }
}

/**
* @brief Perform insertion sort on a 2-mode tensor between start and end,
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_insertionsort2(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  fidx_t * const ind0 = tt->ind[cmplt[0]];
  fidx_t * const ind1 = tt->ind[cmplt[1]];
  storage_val_t * const vals = tt->vals;

  storage_val_t vbuf;
  idx_t ibuf;

  for(size_t i=start+1; i < end; ++i) {
    size_t j = i;
    while (j > start && p_ttcmp2(ind0, ind1, i, j-1) < 0) {
      --j;
    }

    vbuf = vals[i];

    /* shift all data */
    memmove(vals+j+1, vals+j, (i-j)*sizeof(vals[0]));
    vals[j] = vbuf;
    ibuf = ind0[i];
    memmove(ind0+j+1, ind0+j, (i-j)*sizeof(fidx_t));
    ind0[j] = ibuf;
    ibuf = ind1[i];
    memmove(ind1+j+1, ind1+j, (i-j)*sizeof(fidx_t));
    ind1[j] = ibuf;
  }
}

/**
* @brief Perform insertion sort on an n-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_insertionsort(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  fidx_t * ind;
  storage_val_t * const vals = tt->vals;
  idx_t const nmodes = tt->nmodes;

  storage_val_t vbuf;
  idx_t ibuf;

  for(size_t i=start+1; i < end; ++i) {
    size_t j = i;
    while (j > start && p_ttcmp(tt, cmplt, i, j-1) < 0) {
      --j;
    }

    vbuf = vals[i];

    /* shift all data */
    memmove(vals+j+1, vals+j, (i-j)*sizeof(vals[0]));
    vals[j] = vbuf;
    for(idx_t m=0; m < nmodes; ++m) {
      ind = tt->ind[m];
      ibuf = ind[i];
      memmove(ind+j+1, ind+j, (i-j)*sizeof(fidx_t));
      ind[j] = ibuf;
    }
  }
}


/**
* @brief Perform quicksort on a 3-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_quicksort3(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  storage_val_t vmid;
  idx_t imid[3];

  fidx_t * const ind0 = tt->ind[cmplt[0]];
  fidx_t * const ind1 = tt->ind[cmplt[1]];
  fidx_t * const ind2 = tt->ind[cmplt[2]];
  storage_val_t * const vals = tt->vals;

  if((end-start) <= MIN_QUICKSORT_SIZE) {
    p_tt_insertionsort3(tt, cmplt, start, end);
  } else {
    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    imid[0] = ind0[k];
    imid[1] = ind1[k];
    imid[2] = ind2[k];
    ind0[k] = ind0[start];
    ind1[k] = ind1[start];
    ind2[k] = ind2[start];

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(p_ttqcmp3(ind0,ind1,ind2,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp3(ind0,ind1,ind2,j,imid) < 1) {
          storage_val_t vtmp = vals[i];
          vals[i] = vals[j];
          vals[j] = vtmp;
          idx_t itmp = ind0[i];
          ind0[i] = ind0[j];
          ind0[j] = itmp;
          itmp = ind1[i];
          ind1[i] = ind1[j];
          ind1[j] = itmp;
          itmp = ind2[i];
          ind2[i] = ind2[j];
          ind2[j] = itmp;
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(p_ttqcmp3(ind0,ind1,ind2,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp3(ind0,ind1,ind2,i,imid) == 1) {
      --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    ind0[start] = ind0[i];
    ind1[start] = ind1[i];
    ind2[start] = ind2[i];
    ind0[i] = imid[0];
    ind1[i] = imid[1];
    ind2[i] = imid[2];

    if(i > start + 1) {
      p_tt_quicksort3(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      p_tt_quicksort3(tt, cmplt, i, end);
    }
  }
}

/**
* @brief Perform quicksort on a 2-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_quicksort2(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  storage_val_t vmid;
  idx_t imid[2];

  fidx_t * const ind0 = tt->ind[cmplt[0]];
  fidx_t * const ind1 = tt->ind[cmplt[1]];
  storage_val_t * const vals = tt->vals;

  if((end-start) <= MIN_QUICKSORT_SIZE) {
    p_tt_insertionsort2(tt, cmplt, start, end);
  } else {
    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    imid[0] = ind0[k];
    imid[1] = ind1[k];
    ind0[k] = ind0[start];
    ind1[k] = ind1[start];

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(p_ttqcmp2(ind0,ind1,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp2(ind0,ind1,j,imid) < 1) {
          storage_val_t vtmp = vals[i];
          vals[i] = vals[j];
          vals[j] = vtmp;
          idx_t itmp = ind0[i];
          ind0[i] = ind0[j];
          ind0[j] = itmp;
          itmp = ind1[i];
          ind1[i] = ind1[j];
          ind1[j] = itmp;
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(p_ttqcmp2(ind0,ind1,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp2(ind0,ind1,i,imid) == 1) {
      --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    ind0[start] = ind0[i];
    ind1[start] = ind1[i];
    ind0[i] = imid[0];
    ind1[i] = imid[1];

    if(i > start + 1) {
      p_tt_quicksort2(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      p_tt_quicksort2(tt, cmplt, i, end);
    }
  }
}

#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)

static inline int cmp3(
  fidx_t *a1, fidx_t *a2, fidx_t *a3, int idx1, int idx2)
{
  if (a1[idx1] < a1[idx2]) {
    return -1;
  }
  else if (a1[idx1] > a1[idx2]) {
    return 1;
  }

  if (a2[idx1] < a2[idx2]) {
    return -1;
  }
  else if (a2[idx1] > a2[idx2]) {
    return 1;
  }

  if (a3[idx1] < a3[idx2]) {
    return -1;
  }
  else if (a3[idx1] > a3[idx2]) {
    return 1;
  }

  return 0;
}

static void merge(
  fidx_t *in_idx1, fidx_t *in_idx2, fidx_t* in_idx3, storage_val_t *in_val,
  int first1, int last1, int first2, int last2,
  fidx_t *out_idx1, fidx_t *out_idx2, fidx_t *out_idx3, storage_val_t *out_val)
{
  int out = 0;
  for ( ; first1 != last1; ++out) {
    if (first2 == last2) {
      for ( ; first1 != last1; ++first1, ++out) {
        out_idx1[out] = in_idx1[first1];
        out_idx2[out] = in_idx2[first1];
        out_idx3[out] = in_idx3[first1];
        out_val[out] = in_val[first1];
      }
      return;
    }
    /*int a_is_less = 0;
    if (a1[a_idx] < b1[b_idx]) {
      a_is_less = 1;
    }
    else if (a1[a_idx] == b1[b_idx]) {
      if (a2[a_idx] < b2[b_idx]) {
        a_is_less = 1;
      }
      else if (a2[a_idx] == b2[b_idx]) {
        if (a3[a_idx] < b3[b_idx]) {
          a_is_less = 1;
        }
      }
    }*/

    if (cmp3(in_idx1, in_idx2, in_idx3, first1, first2) < 0) {
      out_idx1[out] = in_idx1[first1];
      out_idx2[out] = in_idx2[first1];
      out_idx3[out] = in_idx3[first1];
      out_val[out] = in_val[first1];
      ++first1;
    }
    else {
      out_idx1[out] = in_idx1[first2];
      out_idx2[out] = in_idx2[first2];
      out_idx3[out] = in_idx3[first2];
      out_val[out] = in_val[first2];
      ++first2;
    }
  }
  for ( ; first2 != last2; ++first2, ++out) {
    out_idx1[out] = in_idx1[first2];
    out_idx2[out] = in_idx2[first2];
    out_idx3[out] = in_idx3[first2];
    out_val[out] = in_val[first2];
  }
}

static void kth_element_(
   int *out1, int *out2,
   fidx_t *in1, fidx_t *in2, fidx_t *in3,
   int begin1, int begin2,
   int left, int right,
   int n1, int n2,
   int k)
{
  while (1) {
    int i = (left + right)/2; // right < k -> i < k
    int j = k - i - 1;

    if ((j == -1 || cmp3(in1, in2, in3, begin1 + i, begin2 + j) >= 0) &&
      (j == n2 - 1 || cmp3(in1, in2, in3, begin1 + i, begin2 + j + 1) <= 0)) {
      *out1 = i; *out2 = j + 1;
      return;
    }
    else if (j >= 0 && cmp3(in1, in2, in3, begin1 + i, begin2 + j) <= 0 &&
      (i == n1 - 1 || cmp3(in1, in2, in3, begin1 + i + 1, begin2 + j) >= 0)) {
      *out1 = i + 1; *out2 = j;
      return;
    }
    else if (cmp3(in1, in2, in3, begin1 + i, begin2 + j) > 0 &&
        j != n2 - 1 &&
        cmp3(in1, in2, in3, begin1 + i, begin2 + j + 1) > 0) {
      // search in left half of a1
      right = i - 1;
    }
    else {
      // search in right half of a1
      left = i + 1;
    }
  }
}

/**
 * Partition the input so that
 * a[0:*out1) and b[0:*out2) contain the smallest k elements
 */
static void kth_element(
  int *out1, int *out2,
  fidx_t *in1, fidx_t *in2, fidx_t *in3,
  int begin1, int begin2,
  int n1, int n2, int k)
{
  // either of the inputs is empty
  if (n1 == 0) {
    *out1 = 0; *out2 = k;
    return;
  }
  if (n2 == 0) {
    *out1 = k; *out2 = 0;
    return;
  }
  if (k >= n1 + n2) {
    *out1 = n1; *out2 = n2;
    return;
  }

  // one is greater than the other
  if (k < n1 && cmp3(in1, in2, in3, begin1 + k, begin2) <= 0) {
    *out1 = k; *out2 = 0;
    return;
  }
  if (k - n1 >= 0 && cmp3(in1, in2, in3, begin1 + n1 - 1, begin2 + k - n1) <= 0) {
    *out1 = n1; *out2 = k - n1;
    return;
  }
  if (k < n2 && cmp3(in1, in2, in3, begin1, begin2 + k) >= 0) {
    *out1 = 0; *out2 = k;
    return;
  }
  if (k - n2 >= 0 && cmp3(in1, in2, in3, begin1 + k - n2, begin2 + n2 - 1) >= 0) {
    *out1 = k - n2; *out2 = n2;
    return;
  }
  // now k > 0

  // faster to do binary search on the shorter sequence
  if (n1 > n2) {
    SWAP(int, n1, n2);
    SWAP(int, begin1, begin2);
    SWAP(int *, out1, out2);
  }

  if (k < (n1 + n2)/2) {
    kth_element_(out1, out2, in1, in2, in3, begin1, begin2, 0, SS_MIN(n1 - 1, k), n1, n2, k);
  }
  else {
    // when k is big, faster to find (n1 + n2 - k)th biggest element
    int offset1 = SS_MAX(k - n2, 0), offset2 = SS_MAX(k - n1, 0);
    int new_k = k - offset1 - offset2;

    int new_n1 = SS_MIN(n1 - offset1, new_k + 1);
    int new_n2 = SS_MIN(n2 - offset2, new_k + 1);
    kth_element_(out1, out2, in1, in2, in3, begin1 + offset1, begin2 + offset2, 0, new_n1 - 1, new_n1, new_n2, new_k);

    *out1 += offset1;
    *out2 += offset2;
  }
}

/**
 * @param num_threads number of threads that participate in this merge
 * @param my_thread_num thread id (zeor-based) among the threads that participate in this merge
 */
static void parallel_merge(
  fidx_t *in1, fidx_t *in2, fidx_t *in3, storage_val_t *in_val,
  int in_begin1, int in_begin2,
  int n1, int n2,
  fidx_t *out1, fidx_t *out2, fidx_t *out3, storage_val_t *out_val,
  int num_threads, int my_thread_num)
{
   int n = n1 + n2;
   int n_per_thread = (n + num_threads - 1)/num_threads;
   int begin_rank = SS_MIN(n_per_thread*my_thread_num, n);
   int end_rank = SS_MIN(begin_rank + n_per_thread, n);

   int begin1, begin2, end1, end2;
   kth_element(&begin1, &begin2, in1, in2, in3, in_begin1, in_begin2, n1, n2, begin_rank);
   kth_element(&end1, &end2, in1, in2, in3, in_begin1, in_begin2, n1, n2, end_rank);

   begin1 += in_begin1;
   end1 += in_begin1;
   begin2 += in_begin2;
   end2 += in_begin2;

   while (begin1 > end1 && begin1 > in_begin1 && begin2 < in_begin2 + n2 && cmp3(in1, in2, in3, begin1 - 1, begin2) == 0) {
      begin1--; begin2++; 
   }
   while (begin2 > end2 && end1 > in_begin1 && end2 < in_begin2 + n2 && cmp3(in1, in2, in3, end1 - 1, end2) == 0) {
      end1--; end2++;
   }

   merge(
    in1, in2, in3, in_val, begin1, end1, begin2, end2,
    out1 + begin1 + begin2 - in_begin1 - in_begin2,
    out2 + begin1 + begin2 - in_begin1 - in_begin2,
    out3 + begin1 + begin2 - in_begin1 - in_begin2,
    out_val + begin1 + begin2 - in_begin1 - in_begin2);
}

void merge_sort(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t start, idx_t end)
{
   int len = end - start;
   if (len == 0) return;

   fidx_t *temp1 = (fidx_t *)splatt_malloc(len * sizeof(fidx_t));
   fidx_t *temp2 = (fidx_t *)splatt_malloc(len * sizeof(fidx_t));
   fidx_t *temp3 = (fidx_t *)splatt_malloc(len * sizeof(fidx_t));
   storage_val_t *temp_val = (storage_val_t *)splatt_malloc(len * sizeof(temp_val[0]));

   int thread_private_len[omp_get_max_threads()];
   int out_len = 0;

#pragma omp parallel
   {
      int num_threads = omp_get_num_threads();
      int my_thread_num = omp_get_thread_num();

      // thread-private sort
      int i_per_thread = (len + num_threads - 1)/num_threads;
      int i_begin = SS_MIN(i_per_thread*my_thread_num, len);
      int i_end = SS_MIN(i_begin + i_per_thread, len);

      double t = omp_get_wtime();

      p_tt_quicksort3(tt, cmplt, start + i_begin, start + i_end);

      if (0 == my_thread_num) printf("\tp_tt_quicksort3 takes %f\n", omp_get_wtime() - t);

      // merge sorted sequences
      int in_group_size;

      fidx_t *in_buf1 = tt->ind[cmplt[0]] + start;
      fidx_t *in_buf2 = tt->ind[cmplt[1]] + start;
      fidx_t *in_buf3 = tt->ind[cmplt[2]] + start;
      storage_val_t *in_val = tt->vals + start;

      fidx_t *out_buf1 = temp1;
      fidx_t *out_buf2 = temp2;
      fidx_t *out_buf3 = temp3;
      storage_val_t *out_val = temp_val;

      for (in_group_size = 1; in_group_size < num_threads; in_group_size *= 2)
      {
#pragma omp barrier

         // merge 2 in-groups into 1 out-group
         int out_group_size = in_group_size*2;
         int group_leader = my_thread_num/out_group_size*out_group_size;
         int group_sub_leader = SS_MIN(group_leader + in_group_size, num_threads - 1);
         int id_in_group = my_thread_num%out_group_size;
         int num_threads_in_group =
            SS_MIN(group_leader + out_group_size, num_threads) - group_leader;

         int in_group1_begin = SS_MIN(i_per_thread*group_leader, len);
         int in_group1_end = SS_MIN(in_group1_begin + i_per_thread*in_group_size, len);

         int in_group2_begin = SS_MIN(in_group1_begin + i_per_thread*in_group_size, len);
         int in_group2_end = SS_MIN(in_group2_begin + i_per_thread*in_group_size, len);

         parallel_merge(
            in_buf1, in_buf2, in_buf3, in_val,
            in_group1_begin, in_group2_begin,
            in_group1_end - in_group1_begin,
            in_group2_end - in_group2_begin,
            out_buf1 + in_group1_begin,
            out_buf2 + in_group1_begin,
            out_buf3 + in_group1_begin,
            out_val + in_group1_begin,
            num_threads_in_group,
            id_in_group);

         fidx_t *temp1 = in_buf1;
         fidx_t *temp2 = in_buf2;
         fidx_t *temp3 = in_buf3;
         storage_val_t *temp_val = in_val;

         in_buf1 = out_buf1;
         in_buf2 = out_buf2;
         in_buf3 = out_buf3;
         in_val = out_val;

         out_buf1 = temp1;
         out_buf2 = temp2;
         out_buf3 = temp3;
         out_val = temp_val;
      }

#pragma omp barrier

      if (0 == my_thread_num) {
        if (tt->ind[cmplt[0]] != in_buf1) {
          assert(start == 0);

          splatt_free(tt->ind[0]);
          splatt_free(tt->ind[1]);
          splatt_free(tt->ind[2]);
          splatt_free(tt->vals);

          tt->ind[cmplt[0]] = in_buf1;
          tt->ind[cmplt[1]] = in_buf2;
          tt->ind[cmplt[2]] = in_buf3;
          tt->vals = in_val;
        }
        else {
          splatt_free(temp1);
          splatt_free(temp2);
          splatt_free(temp3);
          splatt_free(temp_val);
        }
      }
   } /* omp parallel */
}

/**
 * idx = idx2*dim1 + idx1
 * -> ret = idx1*dim2 + idx2
 *        = (idx%dim1)*dim2 + idx/dim1
 */
static inline idx_t transpose_idx(idx_t idx, idx_t dim1, idx_t dim2)
{
  return idx%dim1*dim2 + idx/dim1;
}


/**
 * counting sort on the msb and comparison-based sort for
 * the remaining
 */
static void p_counting_sort_hybrid3(sptensor_t * const tt, idx_t *cmplt)
{
  idx_t m = cmplt[0];
  idx_t nslices = tt->dims[m];

  fidx_t **new_ind = splatt_malloc(tt->nmodes*sizeof(fidx_t *));
  for(idx_t i = 0; i < tt->nmodes; ++i) {
#ifdef HBW_ALLOC
    if(i != m) hbw_posix_memalign((void **)&new_ind[i], 4096, tt->nnz*sizeof(fidx_t));
#else
    if(i != m) new_ind[i] = splatt_malloc(tt->nnz*sizeof(fidx_t));
#endif
  }

  storage_val_t *new_vals;
  idx_t *histogram_array;
#ifdef HBW_ALLOC
  hbw_posix_memalign((void **)&new_vals, 4096, tt->nnz*sizeof(new_vals[0]));
  hbw_posix_memalign((void **)&histogram_array, 4096, (nslices*omp_get_max_threads() + 1)*sizeof(idx_t));
#else
  new_vals = splatt_malloc(tt->nnz*sizeof(new_vals[0]));
  histogram_array = splatt_malloc((nslices*omp_get_max_threads() + 1)*sizeof(idx_t));
#endif

#pragma omp parallel
  {
    double t = omp_get_wtime();

    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    idx_t *histogram = histogram_array + nslices*tid;
    memset(histogram, 0, nslices * sizeof(idx_t));

    idx_t j_per_thread = (tt->nnz + nthreads - 1)/nthreads;
    idx_t jbegin = SS_MIN(j_per_thread*tid, tt->nnz);
    idx_t jend = SS_MIN(jbegin + j_per_thread, tt->nnz);

    /* count */
    for (idx_t j = jbegin; j < jend; ++j) {
      idx_t idx = tt->ind[m][j];
      ++histogram[idx];
    }

#pragma omp barrier

    /* prefix sum */
    for (idx_t j = tid*nslices + 1; j < (tid + 1)*nslices; ++j) {
      idx_t transpose_j = transpose_idx(j, nthreads, nslices);
      idx_t transpose_j_minus_1 = transpose_idx(j - 1, nthreads, nslices);
      
      histogram_array[transpose_j] += histogram_array[transpose_j_minus_1];
    }

#pragma omp barrier
    if(0 == tid)
    {
      //printf("\tgather takes %f\n", omp_get_wtime() - t);
      t = omp_get_wtime();

      for (idx_t t = 1; t < nthreads; ++t) {
        idx_t j0 = nslices*t - 1, j1 = nslices*(t + 1) - 1;
        idx_t transpose_j0 = transpose_idx(j0, nthreads, nslices);
        idx_t transpose_j1 = transpose_idx(j1, nthreads, nslices);

        histogram_array[transpose_j1] += histogram_array[transpose_j0];
      }
    }
    else {
      t = omp_get_wtime();
    }
#pragma omp barrier

    if (tid > 0) {
      idx_t transpose_j0 = transpose_idx(nslices*tid - 1, nthreads, nslices);

      for (idx_t j = tid*nslices; j < (tid + 1)*nslices - 1; ++j) {
        idx_t transpose_j = transpose_idx(j, nthreads, nslices);

        histogram_array[transpose_j] += histogram_array[transpose_j0];
      }
    }

#pragma omp barrier
    t = omp_get_wtime();

    /* scatter */
    if(0 == m) {
      for(idx_t j = jend - 1; ; --j) {
        idx_t idx = tt->ind[m][j];
        --histogram[idx];

        idx_t offset = histogram[idx];

        new_vals[offset] = tt->vals[j];
        new_ind[1][offset] = tt->ind[1][j];
        new_ind[2][offset] = tt->ind[2][j];

        if (j == jbegin) break;
      }
    }
    else if (1 == m) {
      for(idx_t j = jend - 1; ; --j) {
        idx_t idx = tt->ind[m][j];
        --histogram[idx];

        idx_t offset = histogram[idx];

        new_vals[offset] = tt->vals[j];
        new_ind[0][offset] = tt->ind[0][j];
        new_ind[2][offset] = tt->ind[2][j];

        if (j == jbegin) break;
      }
    }
    else {
      for(idx_t j = jend - 1; ; --j) {
        idx_t idx = tt->ind[m][j];
        --histogram[idx];

        idx_t offset = histogram[idx];

        new_vals[offset] = tt->vals[j];
        new_ind[0][offset] = tt->ind[0][j];
        new_ind[1][offset] = tt->ind[1][j];

        if (j == jbegin) break;
      }
    }

    if (0 == tid) {
      //printf("\t[%d] scatter takes %f\n", tid, omp_get_wtime() - t);
    }
  } /* omp parallel */

  double t = omp_get_wtime();

  for(idx_t i = 0; i < tt->nmodes; ++i) {
    if(i != m) {
#ifdef HBW_ALLOC
      hbw_free(tt->ind[i]);
#else
      splatt_free(tt->ind[i]);
#endif
      tt->ind[i] = new_ind[i];
    }
  }
  splatt_free(new_ind);
#ifdef HBW_ALLOC
  hbw_free(tt->vals);
#else
  splatt_free(tt->vals);
#endif
  tt->vals = new_vals;

  histogram_array[nslices] = tt->nnz;
#pragma omp parallel for schedule(dynamic)
  for(idx_t i = 0; i < nslices; ++i) {
    p_tt_quicksort2(tt, cmplt + 1, histogram_array[i], histogram_array[i + 1]);
    for(idx_t j = histogram_array[i]; j < histogram_array[i + 1]; ++j) {
      tt->ind[m][j] = i;
    }
  }

#ifdef HBW_ALLOC
  hbw_free(histogram_array);
#else
  splatt_free(histogram_array);
#endif

  //printf("\tqsort takes %f\n", omp_get_wtime() - t);
}



/**
* @brief Perform quicksort on a n-mode tensor between start and end.
*
* @param tt The tensor to sort.
* @param cmplt Mode permutation used for defining tie-breaking order.
* @param start The first nonzero to sort.
* @param end The last nonzero to sort.
*/
static void p_tt_quicksort(
  sptensor_t * const tt,
  idx_t const * const cmplt,
  idx_t const start,
  idx_t const end)
{
  storage_val_t vmid;
  idx_t imid[MAX_NMODES];

  fidx_t * ind;
  storage_val_t * const vals = tt->vals;
  idx_t const nmodes = tt->nmodes;

  if((end-start) <= MIN_QUICKSORT_SIZE) {
    p_tt_insertionsort(tt, cmplt, start, end);
  } else {
    size_t i = start+1;
    size_t j = end-1;
    size_t k = start + ((end - start) / 2);

    /* grab pivot */
    vmid = vals[k];
    vals[k] = vals[start];
    for(idx_t m=0; m < nmodes; ++m) {
      ind = tt->ind[m];
      imid[m] = ind[k];
      ind[k] = ind[start];
    }

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(p_ttqcmp(tt,cmplt,i,imid) == 1) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(p_ttqcmp(tt,cmplt,j,imid) < 1) {
          p_ttswap(tt,i,j);
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(p_ttqcmp(tt,cmplt,j,imid) == 1) {
          --j;
        }
        ++i;
      }
    }

    /* if tt[i] > mid */
    if(p_ttqcmp(tt,cmplt,i,imid) == 1) {
      --i;
    }
    vals[start] = vals[i];
    vals[i] = vmid;
    for(idx_t m=0; m < nmodes; ++m) {
      ind = tt->ind[m];
      ind[start] = ind[i];
      ind[i] = imid[m];
    }

    if(i > start + 1) {
      p_tt_quicksort(tt, cmplt, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      p_tt_quicksort(tt, cmplt, i, end);
    }
  }
}

/**
 * counting sort on the msb and comparison-based sort for
 * the remaining
 */
static void p_counting_sort_hybrid(sptensor_t * const tt, idx_t *cmplt)
{
  idx_t m = cmplt[0];
  idx_t nslices = tt->dims[m];

  fidx_t **new_ind = splatt_malloc(tt->nmodes*sizeof(fidx_t *));
  for(idx_t i = 0; i < tt->nmodes; ++i) {
#ifdef HBW_ALLOC
    if(i != m) hbw_posix_memalign((void **)&new_ind[i], 4096, tt->nnz*sizeof(fidx_t));
#else
    if(i != m) new_ind[i] = splatt_malloc(tt->nnz*sizeof(fidx_t));
#endif
  }

  storage_val_t *new_vals;
  idx_t *histogram_array;
#ifdef HBW_ALLOC
  hbw_posix_memalign((void **)&new_vals, 4096, tt->nnz*sizeof(new_vals[0]));
  hbw_posix_memalign((void **)&histogram_array, 4096, (nslices*omp_get_max_threads() + 1)*sizeof(idx_t));
#else
  new_vals = splatt_malloc(tt->nnz*sizeof(new_vals[0]));
  histogram_array = splatt_malloc((nslices*omp_get_max_threads() + 1)*sizeof(idx_t));
#endif

#pragma omp parallel
  {
    double t = omp_get_wtime();

    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    idx_t *histogram = histogram_array + nslices*tid;
    memset(histogram, 0, nslices * sizeof(idx_t));

    idx_t j_per_thread = (tt->nnz + nthreads - 1)/nthreads;
    idx_t jbegin = SS_MIN(j_per_thread*tid, tt->nnz);
    idx_t jend = SS_MIN(jbegin + j_per_thread, tt->nnz);

    /* count */
    for (idx_t j = jbegin; j < jend; ++j) {
      idx_t idx = tt->ind[m][j];
      ++histogram[idx];
    }

#pragma omp barrier

    /* prefix sum */
    for (idx_t j = tid*nslices + 1; j < (tid + 1)*nslices; ++j) {
      idx_t transpose_j = transpose_idx(j, nthreads, nslices);
      idx_t transpose_j_minus_1 = transpose_idx(j - 1, nthreads, nslices);
      
      histogram_array[transpose_j] += histogram_array[transpose_j_minus_1];
    }

#pragma omp barrier
#pragma omp master
    {
      //printf("\tgather takes %f\n", omp_get_wtime() - t);
      t = omp_get_wtime();

      for (idx_t t = 1; t < nthreads; ++t) {
        idx_t j0 = nslices*t - 1, j1 = nslices*(t + 1) - 1;
        idx_t transpose_j0 = transpose_idx(j0, nthreads, nslices);
        idx_t transpose_j1 = transpose_idx(j1, nthreads, nslices);

        histogram_array[transpose_j1] += histogram_array[transpose_j0];
      }
    }
#pragma omp barrier

    if (tid > 0) {
      idx_t transpose_j0 = transpose_idx(nslices*tid - 1, nthreads, nslices);

      for (idx_t j = tid*nslices; j < (tid + 1)*nslices - 1; ++j) {
        idx_t transpose_j = transpose_idx(j, nthreads, nslices);

        histogram_array[transpose_j] += histogram_array[transpose_j0];
      }
    }

#pragma omp barrier

    /* scatter */
    for(idx_t j = jend - 1; ; --j) {
      idx_t idx = tt->ind[m][j];
      --histogram[idx];

      idx_t offset = histogram[idx];

      new_vals[offset] = tt->vals[j];
      for(int k = 0; k < tt->nmodes; ++k) {
        if(k != m) new_ind[k][offset] = tt->ind[k][j];
      }

      if (j == jbegin) break;
    }

    if (0 == tid) {
      //printf("\tscatter takes %f\n", omp_get_wtime() - t);
    }
  } /* omp parallel */

  double t = omp_get_wtime();

  for(idx_t i = 0; i < tt->nmodes; ++i) {
    if(i != m) {
#ifdef HBW_ALLOC
      hbw_free(tt->ind[i]);
#else
      splatt_free(tt->ind[i]);
#endif
      tt->ind[i] = new_ind[i];
    }
  }
  splatt_free(new_ind);

#ifdef HBW_ALLOC
  hbw_free(tt->vals);
#else
  splatt_free(tt->vals);
#endif
  tt->vals = new_vals;

  histogram_array[nslices] = tt->nnz;
#pragma omp parallel for schedule(dynamic)
  for(idx_t i = 0; i < nslices; ++i) {
    for(idx_t j = histogram_array[i]; j < histogram_array[i + 1]; ++j) {
      tt->ind[m][j] = i;
    }
    p_tt_quicksort(tt, cmplt + 1, histogram_array[i], histogram_array[i + 1]);
  }

#ifdef HBW_ALLOC
  hbw_free(histogram_array);
#else
  splatt_free(histogram_array);
#endif

  //printf("\tqsort takes %f\n", omp_get_wtime() - t);
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void tt_sort(
  sptensor_t * const tt,
  idx_t const mode,
  idx_t * dim_perm)
{
  tt_sort_range(tt, mode, dim_perm, 0, tt->nnz);
}


int tt_is_sorted(
  const sptensor_t * const tt, idx_t * cmplt,
  idx_t const start, idx_t const end)
{
  idx_t cnt_unsorted = 0;
#pragma omp parallel for reduction(+:cnt_unsorted)
  for(idx_t i=0; i < tt->nnz - 1; ++i) {
    if(p_ttcmp(tt, cmplt, i, i + 1) > 0) {
      assert(0);
      ++cnt_unsorted;
    }
  }
  return cnt_unsorted == 0;
}


void tt_sort_range(
  sptensor_t * const tt,
  idx_t const mode,
  idx_t * dim_perm,
  idx_t const start,
  idx_t const end)
{
  idx_t * cmplt;
  if(dim_perm == NULL) {
    cmplt = (idx_t*) splatt_malloc(tt->nmodes * sizeof(idx_t));
    cmplt[0] = mode;
    for(idx_t m=1; m < tt->nmodes; ++m) {
      cmplt[m] = (mode + m) % tt->nmodes;
    }
  } else {
    cmplt = dim_perm;
  }

  double t = omp_get_wtime();

  timer_start(&timers[TIMER_SORT]);
  switch(tt->type) {
  case SPLATT_NMODE:
    //p_tt_quicksort(tt, cmplt, start, end);
    p_counting_sort_hybrid(tt, cmplt);
    break;
  case SPLATT_3MODE:
    p_counting_sort_hybrid3(tt, cmplt);
    break;
  }

  if(dim_perm == NULL) {
    free(cmplt);
  }
  timer_stop(&timers[TIMER_SORT]);
}


void insertion_sort(
  idx_t * const a,
  idx_t const n)
{
  timer_start(&timers[TIMER_SORT]);
  for(size_t i=1; i < n; ++i) {
    idx_t b = a[i];
    size_t j = i;
    while (j > 0 &&  a[j-1] > b) {
      --j;
    }
    memmove(a+(j+1), a+j, sizeof(idx_t)*(i-j));
    a[j] = b;
  }
  timer_stop(&timers[TIMER_SORT]);
}


void quicksort(
  idx_t * const a,
  idx_t const n)
{
  timer_start(&timers[TIMER_SORT]);
  if(n < MIN_QUICKSORT_SIZE) {
    insertion_sort(a, n);
  } else {
    size_t i = 1;
    size_t j = n-1;
    size_t k = n >> 1;
    idx_t mid = a[k];
    a[k] = a[0];
    while(i < j) {
      if(a[i] > mid) { /* a[i] is on the wrong side */
        if(a[j] <= mid) { /* swap a[i] and a[j] */
          idx_t tmp = a[i];
          a[i] = a[j];
          a[j] = tmp;
          ++i;
        }
        --j;
      } else {
        if(a[j] > mid) { /* a[j] is on the right side */
          --j;
        }
        ++i;
      }
    }

    if(a[i] > mid) {
      --i;
    }
    a[0] = a[i];
    a[i] = mid;

    if(i > 1) {
      quicksort(a,i);
    }
    ++i; /* skip the pivot element */
    if(n-i > 1) {
      quicksort(a+i, n-i);
    }
  }
  timer_stop(&timers[TIMER_SORT]);
}


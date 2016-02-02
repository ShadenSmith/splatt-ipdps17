
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../splatt_mpi.h"
#include "../io.h"
#include "../timer.h"



/******************************************************************************
 * API FUNCTONS
 *****************************************************************************/
int splatt_mpi_csf_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_csf ** tensors,
    double const * const options,
    MPI_Comm comm)
{
  sptensor_t * tt = NULL;

  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);


  return SPLATT_SUCCESS;
}



int splatt_mpi_coord_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_idx_t * nnz,
    splatt_idx_t *** inds,
    splatt_val_t ** vals,
    double const * const options,
    MPI_Comm comm)
{
  sptensor_t * tt = mpi_simple_distribute(fname, comm);

  *nmodes = tt->nmodes;
  *nnz = tt->nnz;

  /* copy to output */
  *vals = tt->vals;
  *inds = splatt_malloc(tt->nmodes * sizeof(splatt_idx_t));
  for(idx_t m=0; m < tt->nmodes; ++m) {
    (*inds)[m] = tt->ind[m];
  }

  free(tt);

  return SPLATT_SUCCESS;
}


/******************************************************************************
 * PRIVATE FUNCTONS
 *****************************************************************************/

/**
* @brief Fill buf with the next 'nnz_to_read' tensor values.
*
* @param fin The file to read from.
* @param buf The sptensor buffer to fill.
* @param nnz_to_read The number of nonzeros to read.
*/
static void p_fill_tt_nnz(
  FILE * fin,
  sptensor_t * const buf,
  idx_t const nnz_to_read)
{
  idx_t const nmodes = buf->nmodes;

  char * ptr = NULL;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  idx_t nnzread = 0;
  while(nnzread < nnz_to_read && (read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        buf->ind[m][nnzread] = ind;
      }
      val_t const v = strtod(ptr, &ptr);
      buf->vals[nnzread++] = v;
    }
  }
}


static int * p_distribute_parts(
  sptensor_t * const ttbuf,
  char const * const pfname,
  rank_info * const rinfo)
{
  /* root may have more than target_nnz */
  idx_t const target_nnz = rinfo->global_nnz / rinfo->npes;
  int * parts = (int *) splatt_malloc(SS_MAX(ttbuf->nnz, target_nnz) * sizeof(int));

  if(rinfo->rank == 0) {
    int ret;
    FILE * fin = open_f(pfname, "r");

    /* send to all other ranks */
    for(int p=1; p < rinfo->npes; ++p) {
      /* read into buffer */
      for(idx_t n=0; n < target_nnz; ++n) {
        if((ret = fscanf(fin, "%d", &(parts[n]))) == 0) {
          fprintf(stderr, "SPLATT ERROR: not enough elements in '%s'\n",
              pfname);
          exit(1);
        }
      }
      MPI_Send(parts, target_nnz, MPI_INT, p, 0, rinfo->comm_3d);
    }

    /* now read my own part info */
    for(idx_t n=0; n < ttbuf->nnz; ++n) {
      if((ret = fscanf(fin, "%d", &(parts[n]))) == 0) {
        fprintf(stderr, "SPLATT ERROR: not enough elements in '%s'\n",
            pfname);
        exit(1);
      }
    }
    fclose(fin);
  } else {
    /* receive part info */
    MPI_Recv(parts, ttbuf->nnz, MPI_INT, 0, 0, rinfo->comm_3d,
        &(rinfo->status));
  }
  return parts;
}


static sptensor_t * p_rearrange_fine(
  sptensor_t * const ttbuf,
  char const * const pfname,
  idx_t * * ssizes,
  rank_info * const rinfo)
{
  /* first distribute partitioning information */
  int * parts = p_distribute_parts(ttbuf, pfname, rinfo);

  /* count how many to send to each process */
  int * nsend = (int *) calloc(rinfo->npes, sizeof(int));
  int * nrecv = (int *) calloc(rinfo->npes, sizeof(int));
  for(idx_t n=0; n < ttbuf->nnz; ++n) {
    nsend[parts[n]] += 1;
  }
  MPI_Alltoall(nsend, 1, MPI_INT, nrecv, 1, MPI_INT, rinfo->comm_3d);

  idx_t send_total = 0;
  idx_t recv_total = 0;
  for(int p=0; p < rinfo->npes; ++p) {
    send_total += nsend[p];
    recv_total += nrecv[p];
  }
  assert(send_total = ttbuf->nnz);

  /* how many nonzeros I'll own */
  idx_t nowned = recv_total;

  int * send_disp = (int *) splatt_malloc((rinfo->npes+1) * sizeof(int));
  int * recv_disp = (int *) splatt_malloc((rinfo->npes+1) * sizeof(int));

  /* recv_disp is const so we'll just fill it out once */
  recv_disp[0] = 0;
  for(int p=1; p <= rinfo->npes; ++p) {
    recv_disp[p] = recv_disp[p-1] + nrecv[p-1];
  }

  /* allocate my tensor and send buffer */
  sptensor_t * tt = tt_alloc(nowned, rinfo->nmodes);
  idx_t * isend_buf = (idx_t *) splatt_malloc(ttbuf->nnz * sizeof(idx_t));

  /* rearrange into sendbuf and send one mode at a time */
  for(idx_t m=0; m < ttbuf->nmodes; ++m) {
    /* prefix sum to make disps */
    send_disp[0] = send_disp[1] = 0;
    for(int p=2; p <= rinfo->npes; ++p) {
      send_disp[p] = send_disp[p-1] + nsend[p-2];
    }

    idx_t const * const ind = ttbuf->ind[m];
    for(idx_t n=0; n < ttbuf->nnz; ++n) {
      idx_t const index = send_disp[parts[n]+1]++;
      isend_buf[index] = ind[n];
    }

    /* exchange indices */
    MPI_Alltoallv(isend_buf, nsend, send_disp, SPLATT_MPI_IDX,
                  tt->ind[m], nrecv, recv_disp, SPLATT_MPI_IDX,
                  rinfo->comm_3d);
  }
  free(isend_buf);

  /* lastly, rearrange vals */
  val_t * vsend_buf = (val_t *) splatt_malloc(ttbuf->nnz * sizeof(val_t));
  send_disp[0] = send_disp[1] = 0;
  for(int p=2; p <= rinfo->npes; ++p) {
    send_disp[p] = send_disp[p-1] + nsend[p-2];
  }

  val_t const * const vals = ttbuf->vals;
  for(idx_t n=0; n < ttbuf->nnz; ++n) {
    idx_t const index = send_disp[parts[n]+1]++;
    vsend_buf[index] = vals[n];
  }
  /* exchange vals */
  MPI_Alltoallv(vsend_buf, nsend, send_disp, SPLATT_MPI_VAL,
                tt->vals,  nrecv, recv_disp, SPLATT_MPI_VAL,
                rinfo->comm_3d);
  free(vsend_buf);

  free(nsend);
  free(nrecv);
  free(send_disp);
  free(recv_disp);
  free(parts);

  return tt;
}


/**
* @brief Find the start/end slices in each mode for my partition of X.
*
* @param ssizes The nnz in each slice.
* @param nmodes The number of modes of X.
* @param nnz The number of nonzeros in total.
* @param rinfo MPI information.
*/
static void p_find_my_slices(
  idx_t ** const ssizes,
  idx_t const nmodes,
  idx_t const nnz,
  rank_info * const rinfo)
{
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  for(idx_t m=0; m < nmodes; ++m) {
    idx_t pnnz = nnz / rinfo->dims_3d[m]; /* nnz in a layer */
    /* current processor */
    int currp  = 0;
    idx_t lastn = 0;
    idx_t nnzcnt = 0;

    rinfo->layer_starts[m] = 0;
    rinfo->layer_ends[m] = dims[m];

#if 1
    for(idx_t s=0; s < dims[m]; ++s) {
      if(nnzcnt >= lastn + pnnz) {
        /* choose this slice or the previous, whichever is closer */
        if(s > 0) {
          idx_t const thisdist = nnzcnt - (lastn + pnnz);
          idx_t const prevdist = (lastn + pnnz) - (nnzcnt - ssizes[m][s-1]);
          if(prevdist < thisdist) {
            lastn = nnzcnt - ssizes[m][s-1];
          } else {
            lastn = nnzcnt;
          }
        } else {
          lastn = nnzcnt;
        }

        ++currp;

        /* adjust target nnz based on what is left */
        pnnz = (nnz - lastn) / SS_MAX(1, rinfo->dims_3d[m] - currp);

        if(currp == rinfo->coords_3d[m]) {
          rinfo->layer_starts[m] = s;
        } else if(currp == rinfo->coords_3d[m]+1
            && currp != rinfo->dims_3d[m]) {
          /* only set layer_end if we aren't at the end of the tensor */
          rinfo->layer_ends[m] = s;
          break;
        }
      }
      nnzcnt += ssizes[m][s];
    }
#else
    idx_t const ideal = dims[m] / rinfo->dims_3d[m];
    rinfo->layer_starts[m] = rinfo->coords_3d[m] * ideal;
    rinfo->layer_ends[m] = SS_MIN(dims[m], (rinfo->coords_3d[m]+1) * ideal);
#endif

    /* it is possible to have a very small dimension and too many ranks */
    if(rinfo->dims_3d[m] > 1 && rinfo->layer_starts[m] == 0
        && rinfo->layer_ends[m] == dims[m]) {
      fprintf(stderr, "SPLATT: rank: %d too many MPI ranks for mode %lu.\n",
          rinfo->rank, m+1);
      rinfo->layer_starts[m] = dims[m];
      rinfo->layer_ends[m] = dims[m];
    }
  }
}


static void p_find_my_slices_1d(
  idx_t ** const ssizes,
  idx_t const nmodes,
  idx_t const nnz,
  rank_info * const rinfo)
{
  idx_t const * const dims = rinfo->global_dims;
  /* find start/end slices for my partition */
  for(idx_t m=0; m < nmodes; ++m) {
    /* current processor */
    int currp  = 0;
    idx_t lastn = 0;
    idx_t nnzcnt = 0;

    idx_t pnnz = nnz / rinfo->npes;

    rinfo->layer_starts[m] = 0;
    rinfo->layer_ends[m] = dims[m];

    rinfo->mat_start[m] = 0;
    rinfo->mat_end[m] = dims[m];
    for(idx_t s=0; s < dims[m]; ++s) {
      if(nnzcnt >= lastn + pnnz) {
        /* choose this slice or the previous, whichever is closer */
        if(s > 0) {
          idx_t const thisdist = nnzcnt - (lastn + pnnz);
          idx_t const prevdist = (lastn + pnnz) - (nnzcnt - ssizes[m][s-1]);
          if(prevdist < thisdist) {
            lastn = nnzcnt - ssizes[m][s-1];
          } else {
            lastn = nnzcnt;
          }
        } else {
          lastn = nnzcnt;
        }

        ++currp;

        /* adjust target nnz based on what is left */
        pnnz = (nnz - lastn) / SS_MAX(1, rinfo->npes - currp);

        if(currp == rinfo->rank) {
          rinfo->mat_start[m] = s;
        } else if(currp == rinfo->rank+1 && currp != rinfo->npes) {
          /* only set mat_end if we aren't at the end of the tensor */
          rinfo->mat_end[m] = s;
          break;
        }
      }
      nnzcnt += ssizes[m][s];

      if(rinfo->rank == rinfo->npes-1) {
        assert(rinfo->mat_end[m] == rinfo->global_dims[m]);
      }
    }

    /* it is possible to have a very small dimension and too many ranks */
    if(rinfo->npes > 1 && rinfo->mat_start[m] == 0
        && rinfo->mat_end[m] == dims[m]) {
      fprintf(stderr, "SPLATT: rank: %d too many MPI ranks for mode %lu.\n",
          rinfo->rank, m+1);
      rinfo->mat_start[m] = dims[m];
      rinfo->mat_end[m] = dims[m];
    }
  }
}


/**
* @brief Count the nonzero values in a partition of X.
*
* @param fname The name of the file containing X.
* @param nmodes The number of modes of X.
*
* @return The number of nonzeros in the intersection of all sstarts and sends.
*/
static idx_t p_count_my_nnz(
  char const * const fname,
  idx_t const nmodes,
  idx_t const * const sstarts,
  idx_t const * const sends)
{
  FILE * fin = open_f(fname, "r");

  char * ptr = NULL;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  /* count nnz in my partition */
  idx_t mynnz = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      int mine = 1;
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        if(ind < sstarts[m] || ind >= sends[m]) {
          mine = 0;
        }
      }
      if(mine) {
        ++mynnz;
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
    }
  }
  fclose(fin);

  free(line);

  return mynnz;
}


/**
* @brief Count the nonzero values in a partition of X.
*
* @param fname The name of the file containing X.
* @param nmodes The number of modes of X.
*
* @return The number of nonzeros in the intersection of all sstarts and sends.
*/
static idx_t p_count_my_nnz_1d(
  char const * const fname,
  idx_t const nmodes,
  idx_t const * const sstarts,
  idx_t const * const sends)
{
  FILE * fin = open_f(fname, "r");

  char * ptr = NULL;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  /* count nnz in my partition */
  idx_t mynnz = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      int mine = 0;
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        /* I own the nnz if it falls in any of my slices */
        if(ind >= sstarts[m] && ind < sends[m]) {
          mine = 1;
          break;
        }
      }
      if(mine) {
        ++mynnz;
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
    }
  }
  fclose(fin);

  free(line);

  return mynnz;
}

/**
* @brief Read a partition of X into tt.
*
* @param fname The file containing X.
* @param tt The tensor structure (must be pre-allocated).
* @param sstarts Array of starting slices, inclusive (one for each mode).
* @param sends Array of ending slices, exclusive (one for each mode).
*/
static void p_read_tt_part(
  char const * const fname,
  sptensor_t * const tt,
  idx_t const * const sstarts,
  idx_t const * const sends)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;

  char * ptr = NULL;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  FILE * fin = open_f(fname, "r");
  idx_t nnzread = 0;
  while(nnzread < nnz && (read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      int mine = 1;
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        tt->ind[m][nnzread] = ind;
        if(ind < sstarts[m] || ind >= sends[m]) {
          mine = 0;
        }
      }
      val_t const v = strtod(ptr, &ptr);
      tt->vals[nnzread] = v;
      if(mine) {
        ++nnzread;
      }
    }
  }
  fclose(fin);
  free(line);
}


/**
* @brief Read a partition of X into tt.
*
* @param fname The file containing X.
* @param tt The tensor structure (must be pre-allocated).
* @param sstarts Array of starting slices, inclusive (one for each mode).
* @param sends Array of ending slices, exclusive (one for each mode).
*/
static void p_read_tt_part_1d(
  char const * const fname,
  sptensor_t * const tt,
  idx_t const * const sstarts,
  idx_t const * const sends)
{
  idx_t const nnz = tt->nnz;
  idx_t const nmodes = tt->nmodes;

  char * ptr = NULL;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  FILE * fin = open_f(fname, "r");
  idx_t nnzread = 0;
  while(nnzread < nnz && (read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      int mine = 0;
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10) - 1;
        tt->ind[m][nnzread] = ind;
        if(ind >= sstarts[m] && ind < sends[m]) {
          mine = 1;
        }
      }
      tt->vals[nnzread] = strtod(ptr, &ptr);
      if(mine) {
        ++nnzread;
      }
    }
  }
  fclose(fin);
  free(line);
}


/**
* @brief Read my portion of X from a file.
*
* @param fname The file containing X.
* @param ssizes The nonzero counts in each slice.
* @param nmodes The number of modes in X.
* @param rinfo MPI information (nnz, 3D comm, etc.).
*
* @return My portion of the sparse tensor read from fname.
*/
static sptensor_t * p_read_tt_1d(
  char const * const fname,
  idx_t ** const ssizes,
  idx_t const nmodes,
  rank_info * const rinfo)
{
  int const rank = rinfo->rank;
  idx_t const nnz = rinfo->global_nnz;
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  p_find_my_slices_1d(ssizes, nmodes, nnz, rinfo);

  /* count nnz in my partition and allocate */
  idx_t const mynnz = p_count_my_nnz_1d(fname, nmodes, rinfo->mat_start,
      rinfo->mat_end);
  sptensor_t * tt = tt_alloc(mynnz, nmodes);

  /* now actually load values */
  p_read_tt_part_1d(fname, tt, rinfo->mat_start, rinfo->mat_end);

  return tt;
}


/**
* @brief Read my portion of X from a file.
*
* @param fname The file containing X.
* @param ssizes The nonzero counts in each slice.
* @param nmodes The number of modes in X.
* @param rinfo MPI information (nnz, 3D comm, etc.).
*
* @return My portion of the sparse tensor read from fname.
*/
static sptensor_t * p_read_tt_3d(
  char const * const fname,
  idx_t ** const ssizes,
  idx_t const nmodes,
  rank_info * const rinfo)
{
  int const rank = rinfo->rank_3d;
  idx_t const nnz = rinfo->global_nnz;
  idx_t const * const dims = rinfo->global_dims;

  /* find start/end slices for my partition */
  p_find_my_slices(ssizes, nmodes, nnz, rinfo);

  /* count nnz in my partition and allocate */
  idx_t const mynnz = p_count_my_nnz(fname, nmodes, rinfo->layer_starts,
      rinfo->layer_ends);
  sptensor_t * tt = tt_alloc(mynnz, nmodes);

  /* now actually load values */
  p_read_tt_part(fname, tt, rinfo->layer_starts, rinfo->layer_ends);

  return tt;
}


/**
* @brief Count the nonzeros in each slice of X.
*
* @param tt My subtensor.
* @param ssizes A 2D array for counting slice 'sizes'.
* @param rinfo MPI information (containing global dims, nnz, etc.).
*/
static void p_fill_ssizes(
  sptensor_t const * const tt,
  idx_t ** const ssizes,
  rank_info const * const rinfo)
{
  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t const * const ind = tt->ind[m];
    for(idx_t n=0; n < tt->nnz; ++n) {
      ssizes[m][ind[n]] += 1;
    }

    /* reduce to get total slice counts */
    MPI_Allreduce(MPI_IN_PLACE, ssizes[m], (int) rinfo->global_dims[m],
        SPLATT_MPI_IDX, MPI_SUM, rinfo->comm_3d);
  }
}


/**
* @brief Get dimensions and nonzero count from tensor file.
*
* @param fname Filename of tensor.
* @param outnnz Number of nonzeros found in X.
* @param outnmodes Number of modes found in X.
* @param outdims The dimensions found in X.
*/
static void p_get_dims(
  char const * const fname,
  idx_t * const outnnz,
  idx_t * const outnmodes,
  idx_t * const outdims)
{
  FILE * fin = open_f(fname, "r");

  char * ptr = NULL;
  idx_t nnz = 0;
  char * line = NULL;
  ssize_t read;
  size_t len = 0;

  /* first count modes in tensor */
  idx_t nmodes = 0;
  while((read = getline(&line, &len, fin)) != -1) {
    if(read > 1 && line[0] != '#') {
      /* get nmodes from first nnz line */
      ptr = strtok(line, " \t");
      while(ptr != NULL) {
        ++nmodes;
        ptr = strtok(NULL, " \t");
      }
      break;
    }
  }
  --nmodes;

  for(idx_t m=0; m < nmodes; ++m) {
    outdims[m] = 0;
  }

  /* fill in tensor dimensions */
  rewind(fin);
  while((read = getline(&line, &len, fin)) != -1) {
    /* skip empty and commented lines */
    if(read > 1 && line[0] != '#') {
      ptr = line;
      for(idx_t m=0; m < nmodes; ++m) {
        idx_t ind = strtoull(ptr, &ptr, 10);
        outdims[m] = (ind > outdims[m]) ? ind : outdims[m];
      }
      /* skip over tensor val */
      strtod(ptr, &ptr);
      ++nnz;
    }
  }
  *outnnz = nnz;
  *outnmodes = nmodes;

  free(line);

  fclose(fin);
}


/**
* @brief Return a list of the prime factors (including multiplicatives) of N.
*        The returned list is sorted in increasing order.
*
* @param N The number to factor.
* @param nprimes The number of primes found.
*
* @return The list of primes. This must be deallocated with free().
*/
static int * p_get_primes(
  int N,
  int * nprimes)
{
  int size = 10;
  int * p = (int *) splatt_malloc(size * sizeof(int));
  int np = 0;

  while(N != 1) {
    int i;
    for(i=2; i <= N; ++i) {
      if(N % i == 0) {
        /* found the next prime */
        break;
      }
    }

    /* realloc if necessary */
    if(size == np) {
      p = (int *) realloc(p, size * 2 * sizeof(int));
    }

    p[np++] = i;
    N /= i;
  }

  *nprimes = np;
  return p;
}


/**
* @brief Fill in the best MPI dimensions we can find. The truly optimal
*        solution should involve the tensor's sparsity pattern, but in general
*        this works as good (but usually better) than the hand-tuned dimensions
*        that we tried.
*
* @param rinfo MPI rank information.
*/
static void p_get_best_mpi_dim(
  rank_info * const rinfo)
{
  int nprimes = 0;
  int * primes = p_get_primes(rinfo->npes, &nprimes);

  idx_t total_size = 0;
  for(idx_t m=0; m < rinfo->nmodes; ++m) {
    total_size += rinfo->global_dims[m];

    /* reset mpi dims */
    rinfo->dims_3d[m] = 1;
  }
  int target = total_size / rinfo->npes;

  long diffs[MAX_NMODES];

  /* start from the largest prime */
  for(int p = nprimes-1; p >= 0; --p) {
    int furthest = 0;
    /* find dim furthest from target */
    for(idx_t m=0; m < rinfo->nmodes; ++m) {
      /* distance is current - target */
      diffs[m] = (rinfo->global_dims[m] / rinfo->dims_3d[m]) - target;
      if(diffs[m] > diffs[furthest]) {
        furthest = m;
      }
    }

    /* assign p processes to furthest mode */
    rinfo->dims_3d[furthest] *= primes[p];
  }

  free(primes);
}


/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

sptensor_t * mpi_tt_read(
  char const * const ifname,
  char const * const pfname,
  rank_info * const rinfo)
{
  timer_start(&timers[TIMER_IO]);
  idx_t nmodes;

  if(rinfo->rank == 0) {
    /* get tensor stats */
    p_get_dims(ifname, &(rinfo->global_nnz), &nmodes, rinfo->global_dims);
  }
  MPI_Bcast(&(rinfo->global_nnz), 1, SPLATT_MPI_IDX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nmodes, 1, SPLATT_MPI_IDX, 0, MPI_COMM_WORLD);
  MPI_Bcast(rinfo->global_dims, nmodes, SPLATT_MPI_IDX, 0, MPI_COMM_WORLD);
  rinfo->nmodes = nmodes;

  /* first compute MPI dimension if not specified by the user */
  if(rinfo->decomp == DEFAULT_MPI_DISTRIBUTION) {
    rinfo->decomp = SPLATT_DECOMP_MEDIUM;
    p_get_best_mpi_dim(rinfo);
  }

  mpi_setup_comms(rinfo);

  idx_t * ssizes[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    ssizes[m] = (idx_t *) calloc(rinfo->global_dims[m], sizeof(idx_t));
  }

  /* first naively distribute tensor nonzeros for analysis */
  sptensor_t * ttbuf = mpi_simple_distribute(ifname, rinfo->comm_3d);
  p_fill_ssizes(ttbuf, ssizes, rinfo);

  /* actually parse tensor */
  sptensor_t * tt = NULL;
  switch(rinfo->decomp) {
  case SPLATT_DECOMP_COARSE:
    tt = p_read_tt_1d(ifname, ssizes, nmodes, rinfo);
    /* now fix tt->dims */
    for(idx_t m=0; m < tt->nmodes; ++m) {
      tt->dims[m] = 0;
      for(idx_t n=0; n < tt->nnz; ++n) {
        tt->dims[m] = SS_MAX(tt->dims[m], tt->ind[m][n] + 1);
      }
    }
    break;

  case SPLATT_DECOMP_MEDIUM:
    tt = p_read_tt_3d(ifname, ssizes, nmodes, rinfo);
    /* now map tensor indices to local (layer) coordinates and fill in dims */
    for(idx_t m=0; m < nmodes; ++m) {
      free(ssizes[m]);
      tt->dims[m] = rinfo->layer_ends[m] - rinfo->layer_starts[m];
      for(idx_t n=0; n < tt->nnz; ++n) {
        assert(tt->ind[m][n] >= rinfo->layer_starts[m]);
        assert(tt->ind[m][n] < rinfo->layer_ends[m]);
        tt->ind[m][n] -= rinfo->layer_starts[m];
      }
    }
    break;

  case SPLATT_DECOMP_FINE:
    tt = p_rearrange_fine(ttbuf, pfname, ssizes, rinfo);
    /* now fix tt->dims */
    for(idx_t m=0; m < tt->nmodes; ++m) {
      tt->dims[m] = rinfo->global_dims[m];
      rinfo->layer_ends[m] = tt->dims[m];
    }
    break;
  }

  tt_free(ttbuf);
  timer_stop(&timers[TIMER_IO]);
  return tt;
}


void mpi_filter_tt_1d(
  idx_t const mode,
  sptensor_t const * const tt,
  sptensor_t * const ftt,
  idx_t start,
  idx_t end)
{
  assert(ftt != NULL);

  for(idx_t m=0; m < ftt->nmodes; ++m) {
    ftt->dims[m] = tt->dims[m];
  }

  idx_t const olds = start;
  idx_t const olde = end;
  /* Adjust start and end if tt has been compressed. */
  assert(start != end);
  if(tt->indmap[mode] != NULL) {
    /* TODO: change this linear search into a binary one */
    for(idx_t i=0; i < tt->dims[mode]; ++i) {
      if(tt->indmap[mode][i] == start) {
        start = i;
      }
      if(tt->indmap[mode][i]+1 == end) {
        end = i+1;
        break;
      }
    }
  }

  idx_t nnz = 0;
  for(idx_t n=0; n < tt->nnz; ++n) {
    /* Copy the nonzero if we own the slice. */
    if(tt->ind[mode][n] >= start && tt->ind[mode][n] < end) {
      for(idx_t m=0; m < tt->nmodes; ++m) {
        ftt->ind[m][nnz] = tt->ind[m][n];
      }
      ftt->vals[nnz++] = tt->vals[n];
    }
  }

  /* update ftt dimensions and nnz */
  ftt->nnz = nnz;
  ftt->dims[mode] = end - start;

  /* now map mode coords to [0, end-start) */
  for(idx_t n=0; n < ftt->nnz; ++n) {
    assert(ftt->ind[mode][n] >= start);
    assert(ftt->ind[mode][n] < end);
    ftt->ind[mode][n] -= start;
  }

  /* create new indmap for mode */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    if(tt->indmap[m] == NULL) {
      break;
    }
    ftt->indmap[m] = (idx_t *) realloc(ftt->indmap[m],
        ftt->dims[m] * sizeof(idx_t));

    /* mode indices are shifted. otherwise just copy */
    if(m == mode) {
      for(idx_t i=0; i < ftt->dims[mode]; ++i) {
        ftt->indmap[mode][i] = tt->indmap[mode][i+start];
      }
    } else {
      memcpy(ftt->indmap[m], tt->indmap[m], tt->dims[m] * sizeof(idx_t));
    }
  }

  /* sanity check */
  for(idx_t i=0; i < ftt->dims[mode]; ++i) {
    assert(i + start < end);
  }
  for(idx_t n=0; n < ftt->nnz; ++n) {
    assert(ftt->ind[mode][n] < end - start);
  }
}


void mpi_write_mats(
  matrix_t ** mats,
  permutation_t const * const perm,
  rank_info const * const rinfo,
  char const * const basename,
  idx_t const nmodes)
{
  char * fname;
  idx_t const nfactors = mats[0]->J;

  MPI_Status status;

  idx_t maxdim = 0;
  idx_t maxlocaldim = 0;
  matrix_t * matbuf = NULL;
  val_t * vbuf = NULL;
  idx_t * loc_iperm = NULL;

  for(idx_t m=0; m < nmodes; ++m) {
    maxdim = SS_MAX(maxdim, rinfo->global_dims[m]);
    maxlocaldim = SS_MAX(maxlocaldim, mats[m]->I);
  }

  /* get the largest local dim */
  if(rinfo->rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &maxlocaldim, 1, SPLATT_MPI_IDX, MPI_MAX, 0,
      rinfo->comm_3d);
  } else {
    MPI_Reduce(&maxlocaldim, NULL, 1, SPLATT_MPI_IDX, MPI_MAX, 0,
      rinfo->comm_3d);
  }

  if(rinfo->rank == 0) {
    matbuf = mat_alloc(maxdim, nfactors);
    loc_iperm = (idx_t *) splatt_malloc(maxdim * sizeof(idx_t));
    vbuf = (val_t *) splatt_malloc(maxdim * nfactors * sizeof(val_t));
  }

  for(idx_t m=0; m < nmodes; ++m) {
    /* root handles the writing */
    if(rinfo->rank == 0) {
      asprintf(&fname, "%s%"SPLATT_PF_IDX".mat", basename, m+1);
      matbuf->I = rinfo->global_dims[m];

      /* copy root's matrix to buffer */
      for(idx_t i=0; i < mats[m]->I; ++i) {
        idx_t const gi = rinfo->layer_starts[m] + perm->iperms[m][i];
        for(idx_t f=0; f < nfactors; ++f) {
          matbuf->vals[f + (gi*nfactors)] = mats[m]->vals[f+(i*nfactors)];
        }
      }

      /* receive matrix from each rank */
      for(int p=1; p < rinfo->npes; ++p) {
        idx_t layerstart;
        idx_t nrows;
        MPI_Recv(&layerstart, 1, SPLATT_MPI_IDX, p, 0, rinfo->comm_3d, &status);
        MPI_Recv(&nrows, 1, SPLATT_MPI_IDX, p, 0, rinfo->comm_3d, &status);
        MPI_Recv(vbuf, nrows * nfactors, SPLATT_MPI_VAL, p, 0, rinfo->comm_3d,
            &status);
        MPI_Recv(loc_iperm, nrows, SPLATT_MPI_IDX, p, 0, rinfo->comm_3d, &status);

        /* permute buffer and copy into matbuf */
        for(idx_t i=0; i < nrows; ++i) {
          idx_t const gi = layerstart + loc_iperm[i];
          for(idx_t f=0; f < nfactors; ++f) {
            matbuf->vals[f + (gi*nfactors)] = vbuf[f+(i*nfactors)];
          }
        }
      }

      /* write the factor matrix to disk */
      mat_write(matbuf, fname);

      /* clean up */
      free(fname);
    } else {
      /* send matrix to root */
      MPI_Send(&(rinfo->layer_starts[m]), 1, SPLATT_MPI_IDX, 0, 0, rinfo->comm_3d);
      MPI_Send(&(mats[m]->I), 1, SPLATT_MPI_IDX, 0, 0, rinfo->comm_3d);
      MPI_Send(mats[m]->vals, mats[m]->I * mats[m]->J, SPLATT_MPI_VAL, 0, 0,
          rinfo->comm_3d);
      MPI_Send(perm->iperms[m] + rinfo->mat_start[m], mats[m]->I, SPLATT_MPI_IDX,
          0, 0, rinfo->comm_3d);
    }
  } /* foreach mode */


  if(rinfo->rank == 0) {
    mat_free(matbuf);
    free(vbuf);
    free(loc_iperm);
  }
}


void mpi_write_part(
  sptensor_t const * const tt,
  permutation_t const * const perm,
  rank_info const * const rinfo)
{
  /* file name is <rank>.part */
  char name[256];
  sprintf(name, "%d.part", rinfo->rank);

  FILE * fout = open_f(name, "w");
  for(idx_t n=0; n < tt->nnz; ++n) {
    for(idx_t m=0; m < tt->nmodes; ++m) {
      /* map idx to original global coordinate */
      idx_t idx = tt->ind[m][n];
      if(tt->indmap[m] != NULL) {
        idx = tt->indmap[m][idx];
      }
      if(perm->iperms[m] != NULL) {
        idx = perm->iperms[m][idx];
      }

      /* write index */
      fprintf(fout, "%"SPLATT_PF_IDX" ", 1+idx);
    }
    fprintf(fout, "%"SPLATT_PF_VAL"\n", tt->vals[n]);
  }
  fclose(fout);
}


sptensor_t * mpi_simple_distribute(
  char const * const ifname,
  MPI_Comm comm)
{
  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  idx_t dims[MAX_NMODES];
  idx_t global_nnz;
  idx_t nmodes;

  FILE * fin = NULL;
  if(rank == 0) {
    fin = open_f(ifname, "r");
    /* send dimension info */
    tt_get_dims(fin, &nmodes, &global_nnz, dims);
    rewind(fin);
    MPI_Bcast(&nmodes, 1, SPLATT_MPI_IDX, 0, comm);
    MPI_Bcast(&global_nnz, 1, SPLATT_MPI_IDX, 0, comm);
  } else {
    MPI_Bcast(&nmodes, 1, SPLATT_MPI_IDX, 0, comm);
    MPI_Bcast(&global_nnz, 1, SPLATT_MPI_IDX, 0, comm);
  }

  /* compute my even chunk of nonzeros -- root rank gets the extra amount */
  idx_t const target_nnz = global_nnz / npes;
  idx_t my_nnz = target_nnz;
  if(rank == 0) {
    my_nnz = global_nnz - ((npes-1) * my_nnz);
  }

  sptensor_t * tt = NULL;

  /* read/send all chunks */
  if(rank == 0) {
    sptensor_t * tt_buf = tt_alloc(target_nnz, nmodes);

    /* now send to everyone else */
    for(int p=1; p < npes; ++p) {
      p_fill_tt_nnz(fin, tt_buf, target_nnz);
      for(idx_t m=0; m < tt_buf->nmodes;  ++m) {
        MPI_Send(tt_buf->ind[m], target_nnz, SPLATT_MPI_IDX, p, m, comm);
      }
      MPI_Send(tt_buf->vals, target_nnz, SPLATT_MPI_VAL, p, nmodes, comm);
    }
    tt_free(tt_buf);

    /* load my own */
    tt = tt_alloc(my_nnz, nmodes);
    p_fill_tt_nnz(fin, tt, my_nnz);

    fclose(fin);

  } else {
    MPI_Status status;

    /* receive my chunk */
    tt = tt_alloc(my_nnz, nmodes);
    for(idx_t m=0; m < tt->nmodes;  ++m) {
      MPI_Recv(tt->ind[m], my_nnz, SPLATT_MPI_IDX, 0, m, comm, &status);
    }
    MPI_Recv(tt->vals, my_nnz, SPLATT_MPI_VAL, 0, nmodes, comm, &status);
  }

  return tt;
}



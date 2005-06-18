#include <algorithm>
#include <deque>
#include <set>

#include <omp.h>

#include "reorder.h"
#include "csf.h"

#include "Utils.hpp"

using namespace std;
using namespace SpMP;

idx_t *parallel_max_element(idx_t *begin, idx_t *end)
{
  idx_t local_max_indices[omp_get_max_threads()];
  idx_t *ret = end;

#pragma omp parallel
  {
    int n = end - begin;
    int i_begin, i_end;
    getSimpleThreadPartition(&i_begin, &i_end, n);

    local_max_indices[omp_get_thread_num()] =
      max_element(begin + i_begin, begin + i_end) - begin;

#pragma omp barrier
#pragma omp master
    {
      ret = begin + local_max_indices[0];
      idx_t maximum = *ret;
      for(int i = 1; i < omp_get_num_threads(); ++i) {
        if (begin[local_max_indices[i]] > maximum) {
          ret = begin + local_max_indices[i];
          maximum = *ret;
        }
      }
    }
  }

  //assert(ret == max_element(begin, end));
  return ret;
}

void populate_histogram(idx_t *hist, const splatt_csf *csf, int m1, idx_t s)
{
#pragma omp parallel for
  for(idx_t i = 0; i < csf->dims[m1]; ++i) {
    hist[i] = 0;
  }

  // for each other mode
  for(int m2 = 0; m2 < 3; ++m2) {
    if (m1 == m2) continue;

    csf_sparsity *tile1 = NULL, *tile2 = NULL;
    for(int i = 0; i < 6; ++i) {
      if (csf[i].dim_perm[0] == m1 && csf[i].dim_perm[1] == m2) {
        assert(!tile1);
        tile1 = csf[i].pt;
      }
      if (csf[i].dim_perm[0] == m2 && csf[i].dim_perm[1] == m1) {
        assert(!tile2);
        tile2 = csf[i].pt;
      }
    }
    assert(tile1 && tile2);

    assert(!tile1->fids[0] && !tile2->fids[0]);

    idx_t *sptr1 = tile1->fptr[0];
    idx_t *fids1 = tile1->fids[1];
    idx_t *sptr2 = tile2->fptr[0];
    idx_t *fids2 = tile2->fids[1];

    // for each fiber (s, fids1[f1], *)
#pragma omp parallel for
    for(idx_t f1 = sptr1[s]; f1 < sptr1[s+1]; ++f1) {
      // for each fiber (fids2[f2], fids1[f1], *)
      for(idx_t f2 = sptr2[fids1[f1]]; f2 < sptr2[fids1[f1]+1]; ++f2) {
        assert(fids2[f2] < csf->dims[m1]);
#pragma omp atomic
        hist[fids2[f2]]++;
      }
    }
  } // for each other mode
}

permutation_t *perm_matching(sptensor_t * const tt)
{
  double *opts = splatt_default_opts();
  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLPERMUTE;
  splatt_csf *csf = splatt_csf_alloc(tt, opts);

  idx_t max_dims = 0;
  for(int m=0; m < tt->nmodes; ++m) {
    max_dims = max(max_dims, tt->dims[m]);
  }
  idx_t *hist = new idx_t[max_dims];
  pair<idx_t, idx_t> *sorted_by_connection = new idx_t[max_dims];

  permutation_t *perm = perm_alloc(tt->dims, tt->nmodes);

  for(int m1 = 0; m1 < 3; ++m1) { // for each mode
    printf("\n\n\n--- working on mode %d ---\n\n", m1);

    // (head, first) is the pair with the biggest overlap
    // (head, second) is the pair with the second biggest overlap starting from head
    idx_t first = 0, first_idx = 0, second = 0, second_idx = 0, head = 0;
    double t = omp_get_wtime();
    double histogram_time = 0;
    int histogram_cnt = 0;

    for(idx_t s = 0; s < csf->dims[m1]; ++s) { // for each slice of mode m1
      if (s%max((size_t)(csf->dims[m1]/10.), 1UL) == 0) {
        printf("%ld%% %ld elapsed_time=%f (histogram total %f avg %f)\n", s/max((size_t)(csf->dims[m1]/100.), 1UL), s, omp_get_wtime() - t, histogram_time, histogram_time/histogram_cnt);

        t = omp_get_wtime();
        histogram_time = 0;
        histogram_cnt = 0;
      }

      histogram_time -= omp_get_wtime();
      populate_histogram(hist, csf, m1, s);
      histogram_time += omp_get_wtime();
      ++histogram_cnt;

      hist[s] = 0;

      idx_t local_first_idx = parallel_max_element(hist, hist + csf->dims[m1]) - hist;
      idx_t local_first = hist[local_first_idx];

      if (local_first > first) {
        head = s;

        first = local_first;
        first_idx = local_first_idx;

        hist[first_idx] = 0;
        second_idx = parallel_max_element(hist, hist + csf->dims[m1]) - hist;
        second = hist[second_idx];
        if (second_idx == first_idx) {
          second_idx = csf->dims[m1];
        }
      }

      sorted_by_connection[s] = make_pair(local_first_idx, s);
    } // for each slice of first mode

    t = omp_get_wtime();
    sort(sorted_by_connection, sorted_by_connection + csf->dims[m1]);
    printf("sorting takes %f\n", omp_get_wtime() - t);

    set<idx_t, idx_t> unvisited;
    for(idx_t i = 0; i < csf->dims[m1]; ++i) {
      unvisited[sorted_by_connection[i]] = i;
    }

    deque<idx_t> dq;

    idx_t tail = first_idx;

    dq.push_back(head);
    dq.push_back(tail);

    idx_t head_next = second_idx;
    idx_t tail_next = csf->dims[m1];

    sorted_by_connection[unvisited[head]] = 0;
    sorted_by_connection[unvisited[tail]] = 0;
    sorted_by_connection[unvisited[head_next]] = 0;
    idx_t sort_idx = 0;

    unvisited.erase(head);
    unvisited.erase(tail);
    unvisited.erase(head_next);

    // head_next - head - tail - (tail_next)

    t = omp_get_wtime();
    histogram_time = 0;
    histogram_cnt = 0;

    while (dq.size() < csf->dims[m1]) {
      assert(head_next == csf->dims[m1] ^ tail_next == csf->dims[m1]);
      if (dq.size()%max((size_t)(csf->dims[m1]/10.), 1UL) == 0) {
        printf("%ld%% %ld elapsed_time=%f (histogram total %f avg %f)\n", dq.size()/max((size_t)(csf->dims[m1]/100.), 1UL), dq.size(), omp_get_wtime() - t, histogram_time, histogram_time/histogram_cnt);
        t = omp_get_wtime();
        histogram_time = 0;
        histogram_cnt = 0;
      }

      head = dq.front();
      tail = dq.back();

      idx_t s = head_next == csf->dims[m1] ? head : tail;

      histogram_time -= omp_get_wtime();
      populate_histogram(hist, csf, m1, s);
      histogram_time += omp_get_wtime();
      ++histogram_cnt;

      deque<idx_t>::iterator itr;
      for (itr = dq.begin(); itr != dq.end(); ++itr) {
        hist[*itr] = 0;
      }
      if (head_next == csf->dims[m1]) {
        hist[tail_next] = 0;
      }
      else {
        hist[head_next] = 0;
      }

      idx_t next = parallel_max_element(hist, hist + csf->dims[m1]) - hist;
      if (0 == hist[next] && dq.size() < csf->dims[m1] - 1) {
        for ( ; sorted_by_connection[sort_idx] == 0; sort_idx++);
        next = sort_idx;

        /*first = 0;

        set<idx_t>::iterator itr2;
        double local_histogram_time = 0, other_time = 0;
        int local_histogram_cnt = 0;
        double t = omp_get_wtime();
        for(itr2 = unvisited.begin(); itr2 != unvisited.end(); ++itr2) {

          local_histogram_time -= omp_get_wtime();
          populate_histogram(hist, csf, m1, *itr2);
          local_histogram_time += omp_get_wtime();
          ++local_histogram_cnt;

          hist[*itr2] = 0;

          idx_t local_first_idx = parallel_max_element(hist, hist + csf->dims[m1]) - hist;
          idx_t local_first = hist[local_first_idx];

          if (local_first > first) {
            next = *itr2;
            first = local_first;
          }
        }
        hist[next] = 0;

        printf("s=%ld no connection from here looking at %ld unvisited nodes dq.size() = %ld takes %f (histogram total %f avg %f)\n", s, unvisited.size(), dq.size(), omp_get_wtime() - t, local_histogram_time, local_histogram_time/local_histogram_cnt); fflush(stdout);

        histogram_time += local_histogram_time;
        histogram_cnt += local_histogram_cnt;*/
      }

      if (hist[next] > second) {
        if (head_next == csf->dims[m1]) {
          dq.push_front(next);
        }
        else {
          dq.push_back(next);
        }
      } else {
        if (head_next == csf->dims[m1]) {
          dq.push_back(tail_next);

          tail_next = csf->dims[m1];
          head_next = next;
        }
        else {
          dq.push_front(head_next);

          head_next = csf->dims[m1];
          tail_next = next;
        }
        second = hist[next];
      }
      
      sorted_by_connection[unvisited[next]] = 0;
      unvisited.erase(next);
    }

    deque<idx_t>::iterator itr;
    idx_t i = 0;
    for(itr = dq.begin(); itr != dq.end(); ++itr, ++i) {
      perm->perms[m1][*itr] = i;
      perm->iperms[m1][i] = *itr;
    }

#ifndef NDEBUG
    int *temp_perm = new int[csf->dims[m1]];
    for(int i = 0; i < csf->dims[m1]; ++i) {
      temp_perm[i] = perm->perms[m1][i];
    }
    assert(isPerm(temp_perm, csf->dims[m1]));
#endif
  } // for each mode

  return perm;
}

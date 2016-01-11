#include <stdio.h>
#include "cachesim.h"
#include <string>
#include <set>
#include <algorithm>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#include <pthread.h>
#include <omp.h>

#include "gathertool.h"

#define cachelinesize 64
#define simdw 8
#define uint64 unsigned long long
#define L(x) ((uint64)(x))
#define D(x) ((double)(x))

static int ngather_requests=0;
static int ngathered_elements=0;
static int ndistinct_cachelines=0;
Cache L1(256*1024, 8, 64);
Cache L2(32768/*46080*/*1024, 16/*20*/, 64); // L3 cache of E5-2697 v4
// FA-caches
//Cache L1(32*1024, 32*1024/64, 64);
//Cache L2(512*1024, 512*1024/64, 64);
uint64 cycles=0;

int l1_roi_accesses = 0;
int l1_roi_misses = 0;
int l2_roi_accesses = 0;
int l2_roi_misses = 0;

bool simulateCache = false;
std::vector<int> tidsToSimulate;

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

struct CacheStat
{
  CacheStat(const void *begin, const void *end, const char *name);

  Addr begin, end;
  ULong l1_accesses;
  ULong l1_misses;
  ULong l2_accesses;
  ULong l2_misses;

  std::string name;
};

CacheStat::CacheStat(const void *b, const void *e, const char *n)
 : begin((Addr)b), end((Addr)e), l1_accesses(0), l1_misses(0), l2_accesses(0), l2_misses(0), name(n)
{
}

std::vector<CacheStat> stats;

void RegisterRegion(void *begin, void *end, const char *name)
{
  stats.push_back(CacheStat(begin, end, name));
}

void ClearRegions()
{
  stats.clear();
}

static int Access(Addr addr, bool inROI)
{
  int l1hit = L1.load_line(addr);
  int l2hit = L2.load_line(addr);
  if (inROI) {
    l1_roi_accesses++;
    l2_roi_accesses++;
    if (MISS == l1hit) l1_roi_misses++;
    if (MISS == l2hit) l2_roi_misses++;
  }

  for (int i = 0; i < stats.size(); ++i) {
    if (addr >= stats[i].begin && addr < stats[i].end) {
      stats[i].l1_accesses++;
      stats[i].l2_accesses++;

      if (MISS == l1hit) stats[i].l1_misses++;
      if (MISS == l2hit) stats[i].l2_misses++;
    }
  }

  return l2hit;
}

int *cacheSimPerm = NULL;
int cacheSimN = 192;
#ifdef PRINT_GRIDS
int *cacheSimPerm = NULL;
bool inCacheSimROI = false;
#endif

#if defined(__MIC__) || defined(__AVX512F__)
void Gather(__m512i idx, bool inROI)
{
  Gather(idx, (double *)NULL, inROI);
}

void Gather(__m512i idx, __mmask k, bool inROI)
{
  Gather(idx, k, (double *)NULL, inROI);
}

void Gather(__m512i idx, const double *base, bool inROI /* = false*/)
{
  Gather(idx, 0xffff, base, inROI);
}

void Gather(__m512i idx, __mmask k, const double *base, bool inROI)
{
  __declspec(aligned(64)) int addr[16];
  _mm512_store_epi32(addr, idx);

  Gather(addr, k, base, true, false, inROI);
}

void Gather(int *addr, __mmask k, const double *base, bool cachesim/*=false*/, bool printaddr/*=false*/, bool inROI /*=false*/)
{
  if (!simulateCache ||
    find(
      tidsToSimulate.begin(), tidsToSimulate.end(), omp_get_thread_num()) ==
      tidsToSimulate.end()) {
    return;
  }

  pthread_mutex_lock(&lock);

  __int64 t0=__rdtsc();
  // compress vector of addresses
  int addr_compressed[simdw], ncompressed=0;
  for(int i=0; i < simdw; i++)
  {
     if(k&1) {
       addr_compressed[ncompressed++]=addr[i];
     }
     k = k >> 1;
  }

  if(printaddr)
  {
    for(int v1=0; v1 < ncompressed; v1++)
      printf("%4d", addr_compressed[v1]);
    printf("\n");
  }
  if(cachesim)
  {
    for(int v1=0; v1 < ncompressed; v1++)
    {
      Access(8*((long)(base - (double*)NULL) + addr_compressed[v1]), inROI);
#ifdef PRINT_GRIDS
      if (cacheSimPerm && inCacheSimROI) {
        int offset = cacheSimPerm[addr_compressed[v1]];
        int x = offset%cacheSimN%cacheSimN;
        int y = offset/cacheSimN%cacheSimN;
        int z = offset/cacheSimN/cacheSimN;
        //printf("%s%s(%d, %d, %d) ", (HIT == hit) ? "h" : "m", addr_compressed[v1]%8==0?"!":"", x, y, z);
      }
#endif
    }
#ifdef PRINT_GRIDS
    if (cacheSimPerm && inCacheSimROI) {
      printf("\n");
    }
#endif
  }

  int repeatmask[simdw];
  #pragma simd
  for(int v=0; v < ncompressed; v++)
    repeatmask[v]=1;


  int distinctcl=0;     // distinct cache lines
  for(int v1=0; v1 < ncompressed; v1++)
    if(repeatmask[v1] == 1) {
      distinctcl++;
      for(int v2=v1+1; v2 < ncompressed; v2++)
       if(repeatmask[v2] == 1) {
         uint64 a1 = L(addr_compressed[v1]) / L(cachelinesize);
         uint64 a2 = L(addr_compressed[v2]) / L(cachelinesize);
         if(a1 == a2) {
           repeatmask[v1]=0;
           repeatmask[v2]=0;
         }
       }
    }
  ngather_requests++;
  ngathered_elements += ncompressed;
  ndistinct_cachelines += distinctcl;
  __int64 t1=__rdtsc();
  cycles += L(t1-t0);

  pthread_mutex_unlock(&lock);
}
#endif // __MIC__ | __AVX512F__

void Load(const void *addr, bool inROI /*=false*/)
{
  Load(addr, NULL, inROI);
}

void Load(const void *addr, const void *base, bool inROI /*=false*/)
{
  if (!simulateCache ||
    find(
      tidsToSimulate.begin(), tidsToSimulate.end(), omp_get_thread_num()) ==
      tidsToSimulate.end()) {
    return;
  }

  pthread_mutex_lock(&lock);

  int l2hit = Access((Addr)(L(addr)), inROI);

  pthread_mutex_unlock(&lock);
}

void Analyze()
{
  printf("ngather_requests     = %d\n", ngather_requests);
  printf("ngathered_elements   = %d\n", ngathered_elements);
  printf("ndistinct_cachelines = %d\n", ndistinct_cachelines);

  double l1mr=D(L1.get_misses())/D(L1.get_naccesses());
  printf("%s missrate=%6.2lf%% accesses=%lld misses=%lld\n", L1.get_desc_line(), l1mr*100.0, L1.get_naccesses(), L1.get_misses());
  double l2mr=D(L2.get_misses())/D(L2.get_naccesses());
  printf("%s missrate=%6.2lf%% accesses=%lld misses=%lld\n\n", L2.get_desc_line(), l2mr*100.0, L2.get_naccesses(), L2.get_misses());

  double l1roimr = (double)l1_roi_misses/l1_roi_accesses;
  double l2roimr = (double)l2_roi_misses/l2_roi_accesses;

  printf("%s ROI missrate=%6.2lf%% accesses=%d misses=%d\n", L1.get_desc_line(), l1roimr*100.0, l1_roi_accesses, l1_roi_misses);
  printf("%s ROI missrate=%6.2lf%% accesses=%d misses=%d\n\n", L2.get_desc_line(), l2roimr*100.0, l2_roi_accesses, l2_roi_misses);

  for (int i = 0; i < stats.size(); ++i) {
    l1mr = (double)stats[i].l1_misses/stats[i].l1_accesses;
    l2mr = (double)stats[i].l2_misses/stats[i].l2_accesses;

    printf(
      "%s %s missrate=%6.2lf%% accesses=%lld misses=%lld\n",
      stats[i].name.c_str(), L1.get_desc_line(), l1mr*100.0, stats[i].l1_accesses, stats[i].l1_misses);
    printf(
      "%s %s missrate=%6.2lf%% accesses=%lld misses=%lld\n\n",
      stats[i].name.c_str(), L2.get_desc_line(), l2mr*100.0, stats[i].l2_accesses, stats[i].l2_misses);
  }

  printf("Simulation speed    = %.2lf cycles/gather\n",
          D(cycles)/D(ngather_requests));
}

void Reset()
{
  ngather_requests = 0;
  ngathered_elements = 0;
  ndistinct_cachelines = 0;
  L1.resetstat();
  L2.resetstat();

  l1_roi_accesses = 0;
  l1_roi_misses = 0;
  l2_roi_accesses = 0;
  l2_roi_misses = 0;
}

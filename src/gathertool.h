#pragma once

#include <vector>

void Load(const void *addr, bool inROI = false);
void Load(const void *addr, const void *base, bool inROI = false);

// assume we're accessing doubles, so values in idx will be multiplied by 8
#if defined(__MIC__) || defined(__AVX512F__)
void Gather(__m512i idx, bool inROI = false);
void Gather(__m512i idx, __mmask k, bool inROI = false);
void Gather(__m512i idx, const double *base, bool inROI = false);
void Gather(__m512i idx, __mmask k, const double *base, bool inROI = false);
void Gather(int *addr, __mmask k, const double *base, bool cachesim=false, bool printaddr=false, bool inROI = false);
#endif

void RegisterRegion(void *begin, void *end, const char *name);
void ClearRegions();

void Analyze();
void Reset();

extern bool simulateCache;
extern std::vector<int> tidsToSimulate;

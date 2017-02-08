#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <err.h>
#include "perf_utils.h"
#include "counters.h"


struct gbl_
{
    int mc_rd[NMC];
    int mc_wr[NMC];
    int edc_rd[NEDC];
    int edc_wr[NEDC];
    int edc_hite[NEDC];
    int edc_hitm[NEDC];
    int edc_misse[NEDC];
    int edc_missm[NEDC];
};

static struct gbl_ gbl;

static void
evsetup(const char *ename, int &fd, int event, int umask);

//extern "C" void setup();

void
setup()
{
    int ret;
    char fname[1024];

    // MC RPQ inserts and WPQ inserts (reads & writes)
    for (int mc = 0; mc < NMC; ++mc)
    {
	snprintf(fname, sizeof(fname), "/sys/devices/uncore_imc_%d",mc);
	// RPQ Inserts
	evsetup(fname, gbl.mc_rd[mc], 0x1, 0x1);
	// WPQ Inserts
	evsetup(fname, gbl.mc_wr[mc], 0x2, 0x1);
    }
    // EDC RPQ inserts and WPQ inserts
    for (int edc=0; edc < NEDC; ++edc)
    {
        snprintf(fname, sizeof(fname), "/sys/devices/uncore_edc_eclk_%d",edc);
	// RPQ inserts
        evsetup(fname, gbl.edc_rd[edc], 0x1, 0x1);
	// WPQ inserts
        evsetup(fname, gbl.edc_wr[edc], 0x2, 0x1);
    }
    // EDC HitE, HitM, MissE, MissM
    for (int edc=0; edc < NEDC; ++edc)
    {
	snprintf(fname, sizeof(fname), "/sys/devices/uncore_edc_uclk_%d", edc);
        evsetup(fname, gbl.edc_hite[edc], 0x2, 0x1);
        evsetup(fname, gbl.edc_hitm[edc], 0x2, 0x2);
        evsetup(fname, gbl.edc_misse[edc], 0x2, 0x4);
        evsetup(fname, gbl.edc_missm[edc], 0x2, 0x8);
    }
}

static void
evsetup(const char *ename, int &fd, int event, int umask)
{
    char fname[1024];
    snprintf(fname, sizeof(fname), "%s/type", ename);
    FILE *fp = fopen(fname, "r");
    if (fp == 0)
        err(1, "open %s", fname);
    int type;
    int ret = fscanf(fp, "%d", &type);
    assert(ret == 1);
    fclose(fp);
    //printf("Using PMU type %d from %s\n", type, ename);

    struct perf_event_attr hw = {};
    hw.size = sizeof(hw);
    hw.type = type;
// see /sys/devices/uncore_*/format/*
// All of the events we are interested in are configured the same way, but
// that isn't always true. Proper code would parse the format files
    hw.config = event | (umask << 8);
    //hw.read_format = PERF_FORMAT_GROUP;
    // unfortunately the above only works within a single PMU; might
    // as well just read them one at a time
    int cpu = 0;
    fd = perf_event_open(&hw, -1, cpu, -1, 0);
    if (fd == -1)
	err(1, "CPU %d, box %s, event 0x%lx", cpu, ename, hw.config);
}

static uint64_t
readctr(int fd)
{
    uint64_t data;
    size_t s = read(fd, &data, sizeof(data));
    if (s != sizeof(uint64_t))
	err(1, "read counter %lu", s);
    return data;
}

//extern "C" void readctrs(ctrs *c);
void
readctrs(ctrs *c)
{
    for (int i = 0; i < NMC; ++i)
    {
	c->mcrd[i] = readctr(gbl.mc_rd[i]);
	c->mcwr[i] = readctr(gbl.mc_wr[i]);
    }
    for (int i = 0; i < NEDC; ++i)
    {
	c->edcrd[i] = readctr(gbl.edc_rd[i]);
	c->edcwr[i] = readctr(gbl.edc_wr[i]);
    }
    for (int i = 0; i < NEDC; ++i)
    {
	c->edchite[i] = readctr(gbl.edc_hite[i]);
	c->edchitm[i] = readctr(gbl.edc_hitm[i]);
	c->edcmisse[i] = readctr(gbl.edc_misse[i]);
	c->edcmissm[i] = readctr(gbl.edc_missm[i]);
    }
}

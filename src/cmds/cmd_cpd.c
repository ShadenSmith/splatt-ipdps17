
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../stats.h"
#include "../cpd.h"


/******************************************************************************
 * SPLATT CPD
 *****************************************************************************/
static char cpd_args_doc[] = "TENSOR";
static char cpd_doc[] =
  "splatt-cpd -- Compute the CPD of a sparse tensor.\n"
  "Synchronization methods are:\n"
  "  omp\t\tOpenMP lock\n"
  "  ttas\t\tTest and test-and-set lock with backoff\n"
  "  rtm\t\tRestricted transactional memory\n"
  "  cas\t\tCompare and swap\n"
  "  vcas\t\tEmulation of 64B compare and swap (not functionally correct)\n"
  "  nosync\t\tNo synchronization (just a comparison point)\n";

#define TT_NOWRITE 253
#define TT_TOL 254
#define TT_TILE 255
static struct argp_option cpd_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum number of iterations to use (default: 50)"},
  {"tol", TT_TOL, "TOLERANCE", 0, "minimum change for convergence (default: 1e-5)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 16)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"tile", TT_TILE, 0, 0, "use tiling during SPLATT"},
  {"nowrite", TT_NOWRITE, 0, 0, "do not write output to file"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
  {"tile-depth", 'd', "TILE_DEPTH", 0, "use tiling from the given depth (default: 1)"},
  {"sync", 's', "SYNC", 0, "synchronization option (default: omp)"},
  {"priv-thre", 'p', "PRIV_THRE", 0, "privatization threshold. 0.01 means privatization if dims[m]*nthreads < nnz*0.01 for mode m (default: 0 -> never privatize)"},
  {"alloc", 'a', "ALLOC", 0, "1: one-mode, 2: two-mode, 3+: all-mode (default:all-mode)"},
  { 0 }
};


typedef struct
{
  char * ifname;   /** file that we read the tensor from */
  int write;       /** do we write output to file? */
  double * opts;   /** splatt_cpd options */
  idx_t nfactors;
} cpd_cmd_args;


/**
* @brief Fill a cpd_opts struct with default values.
*
* @param args The cpd_opts struct to fill.
*/
static void default_cpd_opts(
  cpd_cmd_args * args)
{
  args->opts = splatt_default_opts();
  args->ifname    = NULL;
  args->write     = DEFAULT_WRITE;
  args->nfactors  = DEFAULT_NFACTORS;
}



static error_t parse_cpd_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  cpd_cmd_args * args = state->input;
  char * buf;
  int cnt = 0;

  /* -i=50 should also work... */
  if(arg != NULL && arg[0] == '=') {
    ++arg;
  }

  switch(key) {
  case 'i':
    args->opts[SPLATT_OPTION_NITER] = (double) atoi(arg);
    break;
  case TT_TOL:
    args->opts[SPLATT_OPTION_TOLERANCE] = atof(arg);
    break;
  case 't':
    args->opts[SPLATT_OPTION_NTHREADS] = (double) atoi(arg);
    break;
  case 'v':
    timer_inc_verbose();
    args->opts[SPLATT_OPTION_VERBOSITY] += 1;
    break;
  case TT_TILE:
    args->opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
    break;
  case TT_NOWRITE:
    args->write = 0;
    break;
  case 'r':
    args->nfactors = atoi(arg);
    break;
  case 'd':
    args->opts[SPLATT_OPTION_TILEDEPTH] = (double) atoi(arg);
    break;
  case 's':
    if(strcmp(arg, "omp") == 0) {
      args->opts[SPLATT_OPTION_SYNCHRONIZATION] = SPLATT_SYNC_OMP_LOCK;
    } else if(strcmp(arg, "ttas") == 0) {
      args->opts[SPLATT_OPTION_SYNCHRONIZATION] = SPLATT_SYNC_TTAS;
    } else if(strcmp(arg, "rtm") == 0) {
      args->opts[SPLATT_OPTION_SYNCHRONIZATION] = SPLATT_SYNC_RTM;
    } else if(strcmp(arg, "cas") == 0) {
      args->opts[SPLATT_OPTION_SYNCHRONIZATION] = SPLATT_SYNC_CAS;
    } else if(strcmp(arg, "vcas") == 0) {
      args->opts[SPLATT_OPTION_SYNCHRONIZATION] = SPLATT_SYNC_VCAS;
    } else if(strcmp(arg, "nosync") == 0) {
      args->opts[SPLATT_OPTION_SYNCHRONIZATION] = SPLATT_SYNC_NOSYNC;
    } else {
      fprintf(stderr, "SPLATT: unknown synchronization option. Falling back to the default omp lock option\n");
    }
    break;
  case 'p':
    args->opts[SPLATT_OPTION_PRIVATIZATION_THREASHOLD] = atof(arg);
    break;
  case 'a':
    if(atoi(arg) == 1) {
      args->opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
    } else if(atoi(arg) == 2) {
      args->opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_TWOMODE;
    }
    break;

  case ARGP_KEY_ARG:
    if(args->ifname != NULL) {
      argp_usage(state);
      break;
    }
    args->ifname = arg;
    break;
  case ARGP_KEY_END:
    if(args->ifname == NULL) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}

static struct argp cpd_argp =
  {cpd_options, parse_cpd_opt, cpd_args_doc, cpd_doc};


/******************************************************************************
 * SPLATT-CPD
 *****************************************************************************/
extern int splatt_csf_equals(splatt_csf *ct1, splatt_csf *ct2);
void splatt_cpd_cmd(
  int argc,
  char ** argv)
{
  /* assign defaults and parse arguments */
  cpd_cmd_args args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  print_header();

  splatt_verbosity_type which_verb = (splatt_verbosity_type)args.opts[SPLATT_OPTION_VERBOSITY];
  splatt_csf * csf = NULL;
  idx_t nmodes;

  int l = strlen(args.ifname);
  if (l > 4 && !strcmp(args.ifname + l - 4, ".csf")) {
    csf = malloc(sizeof(*csf)*3);
    splatt_csf_read(csf, args.ifname, csf_get_ncopies(args.opts, -1));
    nmodes = csf[0].nmodes;

#ifdef SPLATT_CHECK_CSF_READ
    char buf[256];
    strcpy(buf, args.ifname);
    strcpy(buf + l - 4, ".tns");

    sptensor_t * tt = tt_read(buf);
    if(tt == NULL) {
      return;
    }

    /* print basic tensor stats? */
    if(which_verb >= SPLATT_VERBOSITY_LOW) {
      stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);
    }

    splatt_csf * csf2 = splatt_csf_alloc(tt, args.opts);
    assert(splatt_csf_equals(csf, csf2));
    assert(splatt_csf_equals(csf + 1, csf2 + 1));
#endif
  }
  else {
    sptensor_t * tt = tt_read(args.ifname);
    if(tt == NULL) {
      return;
    }

    /* print basic tensor stats? */
    if(which_verb >= SPLATT_VERBOSITY_LOW) {
      stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);
    }

    csf = splatt_csf_alloc(tt, args.opts);

    nmodes = tt->nmodes;
    tt_free(tt);
  }

  /* print CPD stats? */
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    stats_csf(csf, csf_get_ncopies(args.opts, csf->nmodes), args.opts);
    cpd_stats(csf, args.nfactors, args.opts);
  }

  /* print HBW info */
  printf("SPLATT_HBW_ALLOC=%d\n", SPLATT_HBW_ALLOC);
  printf("SPLATT_MAT_HBW=%d\n", SPLATT_MAT_HBW);
  printf("SPLATT_CSF_FPTR_HBW=%d\n", SPLATT_CSF_FPTR_HBW);
  printf("SPLATT_CSF_FIDS_HBW=%d\n", SPLATT_CSF_FIDS_HBW);
  printf("SPLATT_CSF_VALS_HBW=%d\n", SPLATT_CSF_VALS_HBW);
  printf("SPLATT_NONPERFORM_HBW=%d\n", SPLATT_NONPERFORM_HBW);
  printf("\n");


  /* do the factorization! */
  splatt_kruskal factored;
  int ret = splatt_cpd_als(csf, args.nfactors, args.opts, &factored);
  if(ret != SPLATT_SUCCESS) {
    fprintf(stderr, "splatt_cpd_als returned %d. Aborting.\n", ret);
    exit(1);
  }

  printf("Final fit: %"SPLATT_PF_VAL"\n", factored.fit);

  /* write output */
  if(args.write == 1) {
    vec_write(factored.lambda, args.nfactors, "lambda.mat");

    for(idx_t m=0; m < nmodes; ++m) {
      char * matfname = NULL;
      asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);

      matrix_t tmpmat;
      tmpmat.rowmajor = 1;
      tmpmat.I = csf->dims[m];
      tmpmat.J = args.nfactors;
      tmpmat.vals = factored.factors[m];

      mat_write(&tmpmat, matfname);
      free(matfname);
    }
  }

  /* cleanup */
  splatt_csf_free(csf, args.opts);
  free(args.opts);

  /* free factor matrix allocations */
  splatt_free_kruskal(&factored);

}


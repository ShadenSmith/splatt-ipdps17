#ifndef SPLATT_BASE_H
#define SPLATT_BASE_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../include/splatt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>


/* Memory allocation */
#include "memory.h"


/******************************************************************************
 * DEFINES
 *****************************************************************************/
#define MAX_NMODES SPLATT_MAX_NMODES

/* alias splatt types */
#define val_t splatt_val_t
#define idx_t splatt_idx_t
#define fidx_t splatt_fidx_t
#define storage_val_t splatt_storage_val_t
  /* May want to use narrower types for indices within a factor matrix
   * than indices for accessing non-zeros.
   */

#define SS_MIN(x,y) ((x) < (y) ? (x) : (y))
#define SS_MAX(x,y) ((x) > (y) ? (x) : (y))



/******************************************************************************
 * DEFAULTS
 *****************************************************************************/
static double const DEFAULT_TOL = 1e-5;

static idx_t const DEFAULT_NFACTORS = 16;
static idx_t const DEFAULT_ITS = 50;
static idx_t const DEFAULT_MPI_DISTRIBUTION = MAX_NMODES+1;

#define SPLATT_MPI_FINE (MAX_NMODES + 1)

static int const DEFAULT_WRITE = 0;
static int const DEFAULT_TILE = 0;

#endif

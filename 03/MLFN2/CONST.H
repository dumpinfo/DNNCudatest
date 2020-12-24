// Copyright (c) 1994 John Wiley & Sons, Inc.  All rights reserved.
// Reproduction or translation of this work beyond that permitted in
// section 117 of the 1976 United States Copyright Act without the
// express written permission of the copyright owner is unlawful.
// Requests for further information should be addressed to the
// Permission Department, John Wiley & Sons, Inc.  The purchaser may
// make backup copies for his/her own use only and not for distribution
// or resale.  The publisher assumes no responsibility for errors,
// omissions, or damages, caused by the use of these programs or from
// the use of the information contained herein.

/******************************************************************************/
/*                                                                            */
/* CONST.H - System and program limitation constants                          */
/*           This also contains typedefs, structs, et cetera.                 */
/*                                                                            */
/* See the comment above BAD_COMPILER.                                        */
/*                                                                            */
/* The #if above MALLOC controls whether or not the diagnostic memory         */
/* allocation routines are used.  They only slow things a tiny bit.           */
/*                                                                            */
/* RANDMAX may be system dependent.  See your documentation.                  */
/*                                                                            */
/* TRAIN_BUF_SIZE makes training set memory allocation faster by avoiding     */
/* many calls to realloc.  Users of 32 bit machines with much memory may      */
/* want to increase this considerably.                                        */
/*                                                                            */
/******************************************************************************/

/*
   These interpose memory allocation checking.
*/

#if 0
#define MALLOC memalloc
#define FREE memfree
#define REALLOC memrealloc
#define MEMTEXT memtext
#define MEMCLOSE memclose
#else
#define MALLOC malloc
#define FREE free
#define REALLOC realloc
#define MEMTEXT notext
#define MEMCLOSE nomemclose
#endif

#define PI		3.14159265358979323846

#define KEY_ESCAPE 27

#define VERSION_16_BIT 0

#if VERSION_16_BIT
#define TRAIN_BUF_SIZE 16384     /* Alloc this much tsets mem (max) at a time */
#else
#define TRAIN_BUF_SIZE 65536
#endif

#define RANDMAX 32767            /* rand() returns from 0 through this */
#define CONTROL_LINE_LENGTH 255  /* Max length of user's control input */
#define MAX_CONTROL_FILES 16     /* Nesting of control files */
#define LTAB_LENGTH 500          /* Activation function table length */
#define LTAB_MAX 15.0            /* Maximum in that table */

/*
   The following constants are used in LayerNet::trial_error and gradient
   for normalizing some error types.  LOG_EPS is a limit on how small a
   log error can be.  The other two are derived from it, and must be
   reset if LOG_EPS is changed.
*/

#define LOG_EPS 1.e-5            /* Min for logs in certain errtypes */
#define EPS_OFFSET 11.5129255    /* -log(LOG_EPS) */
#define LOG_FACTOR 12.89922      /* log(4) + EPS_OFFSET */

#define NEURON_ON  0.9           /* Target output activation levels */
#define NEURON_OFF -0.9          /* must not be forced to extremes */

/*
   These are network model codes.  If additional networks are defined, they
   should be appended, leaving existing ones with the same codes, in order
   to avoid disk file incompatibilites.  They must be positive (-1 = unset).
*/

#define NETMOD_REAL 1
#define NETMOD_COMPLEX 2
#define NETMOD_COMPLEX_INPUT 3
#define NETMOD_COMPLEX_HIDDEN 4

/*
   These are network error definitions.  They must be positive.
*/

#define ERRTYPE_MSE 1
#define ERRTYPE_ABS 2
#define ERRTYPE_KK 3
#define ERRTYPE_XENT 4
#define ERRTYPE_MAX 5
#define ERRTYPE_SIXTEENTH 6
#define ERRTYPE_LOGMEAN 7
#define ERRTYPE_MEANLOG 8

/*
   These are output model codes.  If additional outputs are defined, they
   should be appended, leaving existing ones with the same codes, in order
   to avoid disk file incompatibilites.  They must be positive (-1 = unset).
*/

#define OUTMOD_CLASSIFY 1
#define OUTMOD_AUTO 2
#define OUTMOD_GENERAL 3

/*
   These define the learning algorithm to be used
*/

#define METHOD_AN1 1
#define METHOD_AN1_CJ 2
#define METHOD_AN2 3
#define METHOD_AN2_CJ 4
#define METHOD_AN1_LM 5
#define METHOD_AN2_LM 6
#define METHOD_REGRS_CJ 7
#define METHOD_REGRS_LM 8
#define METHOD_SS 9
#define METHOD_SSG 10

/*
   These are for assorted learning parameters
*/

#define ANNEAL_GAUSSIAN 1
#define ANNEAL_CAUCHY 2
#define ANNEAL_REDUCE_EXPONENTIAL 1
#define ANNEAL_REDUCE_FAST 2

enum RandomDensity {
   NormalDensity ,
   CauchyDensity
   } ;

/*
   The annealing parameters have a zero suffix for the value used for finding
   starting weights.  The non-zero parameters are for when annealing is used
   to (hopefully) escape from local minima during learning.   
*/

struct AnnealParams {
   int temps0 ;        // Number of temperatures
   int temps ;
   int iters0 ;        // Iterations per temperature
   int iters ;
   int setback0 ;      // Set back iteration counter if improvement
   int setback ;
   double start0 ;     // Starting temperature
   double start ;
   double stop0 ;      // Stopping temperature
   double stop ;
   double ratio0 ;     // This times std is starting prob factor
   double ratio ;
   int random0 ;       // GAUSSIAN or CAUCHY
   int random ;
   int climb0 ;        // Always accept improvement?
   int climb ;
   int reduc0 ;        // Temp reduction schedule (EXPONENTIAL or FAST)
   int reduc ;
   } ;

struct LearnParams {
   int method ;        // Learning method (METHOD_? above)
   int errtype ;       // Network error definition (ERRTYPE_? above)
   double quit_err ;   // Quit if mean square error fraction of max this low
   int retries ;       // Quit after this many random retries
   int cj_acc ;        // Digits accuracy during retry loop
   int cj_refine ;     // Additional digits for refinement
   int cj_pretries ;   // Number of tries before first refinement
   int cj_progress ;   // Should CJ report progress to screen?
   struct AnnealParams *ap ;
   } ;

/******************************************************************************/
/*                                                                            */
/*  DEFAULTS - Set all defaults                                               */
/*                                                                            */
/* Copyright (c) 1995 Timothy Masters.  All rights reserved.                  */
/* Reproduction or translation of this work beyond that permitted in section  */
/* 117 of the 1976 United States Copyright Act without the express written    */
/* permission of the copyright owner is unlawful.  Requests for further       */
/* information should be addressed to the Permissions Department, John Wiley  */
/* & Sons, Inc.  The purchaser may make backup copies for his/her own use     */
/* only and not for distribution or resale.                                   */
/* Neither the author nor the publisher assumes responsibility for errors,    */
/* omissions, or damages, caused by the use of these programs or from the     */
/* use of the information contained herein.                                   */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

void defaults ( NetParams *net_params , LearnParams *learn_params ,
                MiscParams *misc_params )
{
   AnnealParams *anneal_params ;        // Simulated annealing parameters

   net_params->net_model = NETMOD_PNN ;    // Net model (NETMOD_? in CONST.H)
   net_params->kernel = KERNEL_GAUSS ;     // Parzen kernel
   net_params->domain = DOMAIN_REAL ;      // MLFN layer domains
   net_params->linear = 1 ;                // MLFN linear outputs
   net_params->n_hidden1 = 2 ;             // First hidden layer
   net_params->n_hidden2 = 0 ;             // Second hidden layer

   learn_params->progress = 0 ;       // Show detailed progress?
   learn_params->quit_err = 0.0 ;     // Quit if error drops this low
   learn_params->errtype = ERRTYPE_MSE ; // MLFN mean square error minimized
   learn_params->siglo = 0.01 ;       // Lowest sigma in global search
   learn_params->sighi = 10.0 ;       // And highest
   learn_params->nsigs = 10 ;         // Number of sigmas in global search
   learn_params->method = METHOD_AN1_CJ ;
   learn_params->retries = 32767 ;
   learn_params->pretries = 5 ;
   learn_params->acc = 6 ;
   learn_params->refine = 2 ;

   anneal_params = learn_params->ap ;

   anneal_params->ntempI = 5 ;
   anneal_params->niterI = 50 ;
   anneal_params->sbI = 50 ;
   anneal_params->startI = 3.0 ;
   anneal_params->endI = 0.2 ;
   anneal_params->ratioI = 0.1 ;
   anneal_params->reducI = ANNEAL_REDUCE_EXPONENTIAL ;
   anneal_params->randomI = NormalDensity ;
   anneal_params->climbI = 1 ;
   anneal_params->ntempE = 4 ;
   anneal_params->niterE = 20 ;
   anneal_params->sbE = 20 ;
   anneal_params->startE = 4.0 ;
   anneal_params->endE = 0.1 ;
   anneal_params->ratioE = 0.1 ;
   anneal_params->reducE = ANNEAL_REDUCE_EXPONENTIAL ;
   anneal_params->randomE = NormalDensity ;
   anneal_params->climbE = 1 ;

   misc_params->names = NULL ;           // Signal names from NAME
   misc_params->classif_output = 0 ;     // Current class for CLASSIFY training
   misc_params->threshold = 0.0 ;        // CLASSIFY confusion reject cutoff
   misc_params->classif_prior = 1.0 ;    // Prior probability weight
   misc_params->include = MAXPOSNUM ;    // Include first cases from cumulate
   misc_params->exclude = 0 ;            // Exclude first cases from cumulate
   misc_params->display_range = 0 ;      // Optimal range
   misc_params->display_domain0 = 0 ;    // Offset to first displayed pt
   misc_params->display_domain1 = MAXPOSNUM ; // And last
   misc_params->display_rate = 1.0 ;     // Points per unit time
   misc_params->display_origin = 0.0 ;   // Value corresponding to first point
   misc_params->orthog_type = 1 ;        // Orthogonalization type
   misc_params->orthog_nfac = MAXPOSNUM ;// Orthogonalization max factors
   misc_params->orthog_lim = 1.0 ;       // Orthogonalization variance limit
   misc_params->orthog_std = 1 ;         // Orthogonalization: standardize?
   misc_params->spectrum_window = 1 ;    // SPECTRUM data window = Welch
   misc_params->conf_prob = 0.9 ;        // Confidence interval two-tailed prob
   misc_params->padding = 1 ;            // Filter padding: 0=mean, 1=detrend
}

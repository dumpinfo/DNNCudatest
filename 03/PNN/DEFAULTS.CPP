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
/*  DEFAULTS - Set all defaults                                               */
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
   net_params->net_model = -1 ;       // PNNet model (see NETMOD_? in CONST.H)
   net_params->net_kernel = KERNEL_GAUSS ; // Parzen kernel
   net_params->maxmom = 3 ;           // GCNN max moment
   net_params->out_model = -1 ;       // Output model (see OUTMOD_? in CONST.H)
   net_params->n_inputs = -1 ;        // Number of input neurons
   net_params->n_outputs = -1 ;       // Number of output neurons

   learn_params->quit_err = 0.0 ;     // Quit if error drops this low
   learn_params->quit_tol = 0.0001 ;  // Precision of parameter estimation
   learn_params->siglo = 0.0001 ;     // Lowest sigma in global search
   learn_params->sighi = 10.0 ;       // And highest
   learn_params->nsigs = 10 ;         // Number of sigmas in global search

   misc_params->classif_output = -1 ;    // Current class for CLASSIFY training
   misc_params->out_file[0] = NULL ;     // File for EXECUTE output
   misc_params->threshold = 0.0 ;        // CLASSIFY confusion reject cutoff
   misc_params->classif_prior = 1.0 ;    // Prior probability weight
}

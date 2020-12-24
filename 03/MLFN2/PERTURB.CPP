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
/*  PERTURB - Called by annealing routines to perturb coefficients            */
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

void LayerNet::perturb ( LayerNet *cent , LayerNet *perturbed , double temp ,
                         int reg , enum RandomDensity dens )
{
   int n ;

   if (nhid1 == 0) {                // No hidden layer
      n = nout * nin_n ;
      shake ( n , cent->out_coefs , perturbed->out_coefs , temp , dens ) ;
      limit ( n , perturbed->out_coefs , 20.0 ) ;
      }

   else if (nhid2 == 0) {           // One hidden layer
      n = nhid1 * nin_n ;
      shake ( n , cent->hid1_coefs , perturbed->hid1_coefs , temp , dens ) ;
      limit ( n , perturbed->hid1_coefs , 20.0 ) ;
      if (! reg) {
         n = nout * nhid1_n ;
         shake ( n , cent->out_coefs , perturbed->out_coefs , temp , dens ) ;
         limit ( n , perturbed->out_coefs , 20.0 ) ;
         }
      }

   else {                           // Two hidden layers
      n = nhid1 * nin_n ;
      shake ( n , cent->hid1_coefs , perturbed->hid1_coefs , temp , dens ) ;
      limit ( n , perturbed->hid1_coefs , 20.0 ) ;
      n = nhid2 * nhid1_n ;
      shake ( n , cent->hid2_coefs , perturbed->hid2_coefs , temp , dens ) ;
      limit ( n , perturbed->hid2_coefs , 20.0 ) ;
      if (! reg) {
         n = nout * nhid2_n ;
         shake ( n , cent->out_coefs , perturbed->out_coefs , temp , dens ) ;
         limit ( n , perturbed->out_coefs , 20.0 ) ;
         }
      }
}

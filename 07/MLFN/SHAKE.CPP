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
/*  SHAKE - Randomly perturb a point                                          */
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

void shake ( int nvars , double *center , double *x , double temp )
{
   double r ;

/*
   Recall that the variance of a uniform deviate on 0-1 is 1/12.
   Adding four such random variables multiplies the variance by 4,
   while dividing by 2 divides the variance by 4.
*/

   temp *= 3.464101615 / (2.0 * (double) 0x7FFFFFFFL) ; // SQRT(12)=3.464...

   while (nvars--) {
      r = (double) (flrand()  &  0x7FFFFFFFL) +
          (double) (flrand()  &  0x7FFFFFFFL) -
          (double) (flrand()  &  0x7FFFFFFFL) -
          (double) (flrand()  &  0x7FFFFFFFL) ;
      *x++ = *center++ + temp * r ;
      }
}

/******************************************************************************/
/*                                                                            */
/*  DOTPRODC - Compute dot product of two complex vectors                     */
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

void dotprodc (
   int n ,         // Length of vectors
   double *vec1 ,  // One of the vectors to be dotted
   double *vec2 ,  // The other vector
   double *re ,    // Real part of output
   double *im )    // and imaginary
{
   int k, m ;

   *re = *im = 0.0 ;  // Will cumulate dot product here
   k = n / 4 ;        // Divide vector into this many groups of 4
   m = n % 4 ;        // This is the remainder of that division

   while (k--) {    // Do each group of 4
      *re += *(vec1  ) * *(vec2  )  -  *(vec1+1) * *(vec2+1) ;
      *re += *(vec1+2) * *(vec2+2)  -  *(vec1+3) * *(vec2+3) ;
      *re += *(vec1+4) * *(vec2+4)  -  *(vec1+5) * *(vec2+5) ;
      *re += *(vec1+6) * *(vec2+6)  -  *(vec1+7) * *(vec2+7) ;
      *im += *(vec1  ) * *(vec2+1)  +  *(vec1+1) * *(vec2  ) ;
      *im += *(vec1+2) * *(vec2+3)  +  *(vec1+3) * *(vec2+2) ;
      *im += *(vec1+4) * *(vec2+5)  +  *(vec1+5) * *(vec2+4) ;
      *im += *(vec1+6) * *(vec2+7)  +  *(vec1+7) * *(vec2+6) ;
      vec1 += 8 ;
      vec2 += 8 ;
      }

   while (m--) {     // Do the remainder
      *re += *vec1 * *vec2  -  *(vec1+1) * *(vec2+1) ;
      *im += *vec1 * *(vec2+1)  +  *(vec1+1) * *vec2 ;
      vec1 += 2 ;
      vec2 += 2 ;
      }
}

/*
--------------------------------------------------------------------------------

   DOTPRODCR - This version computes only the real part

--------------------------------------------------------------------------------
*/

double dotprodcr (
   int n ,         // Length of vectors
   double *vec1 ,  // One of the vectors to be dotted
   double *vec2 )  // The other vector
{
   int k, m ;
   double sum ;

   sum = 0.0 ;  // Will cumulate dot product here
   k = n / 4 ;  // Divide vector into this many groups of 4
   m = n % 4 ;  // This is the remainder of that division

   while (k--) {    // Do each group of 4
      sum += *(vec1  ) * *(vec2  )  -  *(vec1+1) * *(vec2+1) ;
      sum += *(vec1+2) * *(vec2+2)  -  *(vec1+3) * *(vec2+3) ;
      sum += *(vec1+4) * *(vec2+4)  -  *(vec1+5) * *(vec2+5) ;
      sum += *(vec1+6) * *(vec2+6)  -  *(vec1+7) * *(vec2+7) ;
      vec1 += 8 ;
      vec2 += 8 ;
      }

   while (m--) {     // Do the remainder
      sum += *vec1 * *vec2  -  *(vec1+1) * *(vec2+1) ;
      vec1 += 2 ;
      vec2 += 2 ;
      }
   return sum ;
}

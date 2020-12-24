/******************************************************************************/
/*                                                                            */
/*  BURG - Use J. P. Burg's version of the Levinson recursion to estimate     */
/*  partial autocorrelations of a series.                                     */
/*  This is adapted from Press "Numerical Recipes"                            */
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
#include "const.h"     // System, limitation constants, typedefs, structs
#include "classes.h"   // Includes all class headers
#include "funcdefs.h"  // Function prototypes

void burg (
   int n ,          // Length of input series, x
   double *x ,      // This is the raw data
   int maxlag ,     // Compute this many partial autocorrelations
   double *pcorr ,  // And output them here
   double *coefs ,  // Work (maxlag long) holds coefs at each recursion
   double *prev ,   // Work vector maxlag long
   double *alpha ,  // Work vector n long
   double *beta )   // Work vector n long
{
   int i, lag ;
   double mean, term1, term2, numer, denom, coef ;

/*
   Center the signal
*/

   mean = 0.0 ;
   for (i=0 ; i<n ; i++)
      mean += x[i] ;
   mean /= n ;

   coef = x[0] - mean ;
   for (i=0 ; i<n-1 ; i++) {
      alpha[i] = coef ;
      coef = x[i+1] - mean ;
      beta[i] = coef ;
      }

/*
   Recursively compute partial autocorrelations
*/

   for (lag=1 ; lag<=maxlag ; lag++) {  // This is the recursion loop

      numer = denom = 0.0 ;             // Compute this partial autocorr
      for (i=0 ; i<n-lag ; i++) {
         term1 = alpha[i] ;
         term2 = beta[i] ;
         numer += term1 * term2 ;
         denom += term1 * term1  +  term2 * term2 ;
         }
      pcorr[lag-1] = coefs[lag-1] = coef = 2.0 * numer / (denom + 1.e-40) ;

      for (i=0 ; i<lag-1 ; i++)   // Adjust previous coefs per this new one
         coefs[i] = prev[i] - coef * prev[lag-i-2] ;

      memcpy ( prev , coefs , lag * sizeof(double) ) ; // Save these for next

      for (i=0 ; i<n-lag-1 ; i++) {
         alpha[i] -= coef * beta[i] ;
         beta[i] = beta[i+1] - coef * alpha[i+1] ;
         }

      } // For all lags
}


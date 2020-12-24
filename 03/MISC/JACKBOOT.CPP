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

#include <math.h>
#include <stdlib.h>

double unifrand () ;
double userstat ( int n , double *data ) ;

/*
--------------------------------------------------------------------------------

   jack - Compute the bias and variance via jackknife

--------------------------------------------------------------------------------
*/

void jack (
   int n ,         // Number of data points
   double *data ,  // The data is here
   double *raw ,   // Raw (uncorrected) statistic for all n data
   double *bias ,  // Output of bias (subtract from raw to unbias)
   double *var     // Output of variance of statistic
   )
{
   int exclude, first ;
   double stat, temp, sum, sumsq ;

   sum = sumsq = 0.0 ;
   exclude = n ;
   first = 1 ;

   while (exclude--) {

      if (first)
         first = 0 ;
      else {               // If not first trial, swap excluded to end
         temp = data[exclude] ;
         data[exclude] = data[n-1] ;
         data[n-1] = temp ;
         }

      stat = userstat ( n-1 , data ) ;

      sum += stat ;
      sumsq += stat * stat ;
      }

   sum /= n ;
   sumsq /= n ;
   sumsq -= sum * sum ;   // Can easily lose precision in this subtraction

   *raw = userstat ( n , data ) ;
   *bias = (n - 1.0) * (sum - *raw) ;
   *var = (n - 1.0) * sumsq ;
}


/*
--------------------------------------------------------------------------------

   boot - Compute the bias and variance via bootstrap

--------------------------------------------------------------------------------
*/

void boot (
   int n ,         // Number of data points
   double *data ,  // The data is here
   double *raw ,   // Raw (uncorrected) statistic for all n data
   double *bias ,  // Output of bias (subtract from raw to unbias)
   double *var     // Output of variance of statistic
   )
{
   int i, rep, sub ;
   int m = 200 ;
   double *x, stat, sum, sumsq ;

   x = (double *) malloc ( n * sizeof(double) ) ;
   sum = sumsq = 0.0 ;

   for (rep=0 ; rep<m ; rep++) {
      for (i=0 ; i<n ; i++) {
         sub = unifrand() * n ;
         if (sub >= n)
            sub = n-1 ;
         x[i] = data[sub] ;
         }
      stat = userstat ( n , x ) ;
      sum += stat ;
      sumsq += stat * stat ;
      }

   sum /= m ;
   sumsq /= m ;
   sumsq -= sum * sum ;   // Can easily lose precision in this subtraction
   *raw = userstat ( n , data ) ;
   *bias = sum - *raw ;
   *var = sumsq * m / (m - 1.0) ;
   free ( x ) ;
}

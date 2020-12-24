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

class Detrender {

public:
   Detrender ( int n , double *model ) ;
   void detrend ( int n , double *input , double *output ) ;
   void retrend ( int n , double *input , double *output ) ;

private:
   double slope, intercept ;
} ;

/*
   Constructor computes and saves slope and intercept
*/

Detrender::Detrender (
   int n ,          // Length of model signal
   double *model    // Use this signal as model
   )
{
   int i ;
   double temp, sig, *sigptr, tmean, sigmean, tss, tsig ;

   if (n < 2) {   // Trivial, but should make provision
      slope = 0.0 ;
      intercept = model[0] ;
      return ;
      }

/*
   Compute the mean of the time variable and the signal
*/

   tmean = 0.5 * (double) (n-1) ;
   sigmean = 0.0 ;
   i = n ;
   sigptr = model ;
   while (i--)
      sigmean += *(sigptr++) ;
   sigmean /= (double) n ;

/*
   Compute the sum of squares and cross-product of
   the time variable and the signal
*/

   tss = tsig = 0.0 ;                  // Time sum squares, crossproduct
   i = n ;                             // This many points
   sigptr = model + n ;                // Will work backwards
   while (i--) {                       // From last point to first
      temp = (double) i - tmean ;      // Time offset from its mean
      sig = *(--sigptr) - sigmean ;    // Ditto for signal
      tss += temp * temp ;             // Time sum of squares
      tsig += temp * sig ;             // Cross product
      }

   slope = tsig / tss ;
   intercept = sigmean - slope * tmean ;
}

/*
   Routines for detrending and retrending
*/

void Detrender::detrend ( int n , double *input , double *output )
{
   while (n--)
      output[n] = input[n] -  slope * n - intercept ;
}


void Detrender::retrend ( int n , double *input , double *output )
{
   while (n--)
      output[n] = input[n] +  slope * n + intercept ;
}

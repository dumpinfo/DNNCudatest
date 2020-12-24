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

/****************************************************************************/
/*                                                                          */
/*  GCNN.C - GCnet and EWnet                                                */
/*                                                                          */
/****************************************************************************/
/*                                                                          */
#include <stdlib.h>
#include <math.h>

class GCnet {

public:
   GCnet ( const int ndata , const double * const data ) ;
   double density ( const double x , const int n_moments ) ;

private:
   double mean, std ;
   double c3, c4, c5, c6, c7, c8 ;
} ;

class EWnet {

public:
   EWnet ( const int ndata , const double * const data ) ;
   double density ( const double x , const int n_moments ) ;

private:
   double mean, std ;
   double c3, c4, c5, c6, c7, c9 ;
} ;

static void hermite ( const double x ) ;
static double h2, h3, h4, h5, h6, h7, h8, h9 ;

/*
--------------------------------------------------------------------------------

   Constructors

--------------------------------------------------------------------------------
*/

GCnet::GCnet (
   const int n ,
   const double * const data
   )
{
   int i ;
   double diff ;

/*
   Compute the mean and standard deviation for standardizing the data
*/

   mean = std = 0.0 ;

   i = n ;
   while (i--)
      mean += data[i] ;
   mean /= n ;

   i = n ;
   while (i--) {
      diff = data[i] - mean ;
      std += diff * diff ;
      }
   std = sqrt ( std / n ) ;

/*
   Compute the GC coefficients via direct projection on the Hermite polynomials
*/

   c3 = c4 = c5 = c6 = c7 = c8 = 0.0 ;
   i = n ;
   while (i--) {
      diff = (data[i] - mean) / std ;
      hermite ( diff ) ;
      c3 += h3 ;
      c4 += h4 ;
      c5 += h5 ;
      c6 += h6 ;
      c7 += h7 ;
      c8 += h8 ;
      }

   c3 /= (n * 6.0) ;
   c4 /= (n * 24.0) ;
   c5 /= (n * 120.0) ;
   c6 /= (n * 720.0) ;
   c7 /= (n * 5040.0) ;
   c8 /= (n * 40320.0) ;
}


EWnet::EWnet (
   const int n ,
   const double *const data
   )
{
   int i ;
   double diff, prod, m3, m4, m5 ;

/*
   Compute the mean and standard deviation for standardizing the data
*/

   mean = std = 0.0 ;

   i = n ;
   while (i--)
      mean += data[i] ;
   mean /= n ;

   i = n ;
   while (i--) {
      diff = data[i] - mean ;
      std += diff * diff ;
      }
   std = sqrt ( std / n ) ;

/*
   Compute the moments
*/

   m3 = m4 = m5 = 0.0 ;
   i = n ;
   while (i--) {
      diff = (data[i] - mean) / std ;
      prod = diff * diff * diff ;
      m3 += prod ;
      prod *= diff ;
      m4 += prod ;
      prod *= diff ;
      m5 += prod ;
      }

   m3 /= n ;
   m4 /= n ;
   m5 /= n ;

/*
   Compute the Edgeworth coefficients
*/

   c3 = m3 / 6.0 ;
   c4 = (m4 - 3.0) / 24.0 ;
   c5 = (m5 - 10.0 * m3) / 120.0 ;
   c6 = m3 * m3 / 72.0 ;
   c7 = (m4 - 3.0) * m3 * 35.0 / 5040.0 ;
   c9 = m3 * m3 * m3 * 280.0 / 362880.0 ;
}

/*
--------------------------------------------------------------------------------

   density

--------------------------------------------------------------------------------
*/

double GCnet::density ( const double x , const int n_moments )
{
   double z, a, b ;

   z = (x - mean) / std ;       // Standardize
   a = exp ( -0.5 * z * z ) ;   // Normal density (times a constant)
   h2 = z * z - 1.0 ;           // Second-degree Hermite polynomial
   h3 = z * (h2 - 2.0) ;        // And third
   b = 1.0  +  c3 * h3 ;        // First extra moment term (3rd moment)
   switch (n_moments) {
      case 2: h4 = z * h3 - 3.0 * h2 ;
              b += c4 * h4 ;
              break ;
      case 3: h4 = z * h3 - 3.0 * h2 ;
              h5 = z * h4 - 4.0 * h3 ;
              b += c4 * h4 + c5 * h5 ;
              break ;
      case 4: h4 = z * h3 - 3.0 * h2 ;
              h5 = z * h4 - 4.0 * h3 ;
              h6 = z * h5 - 5.0 * h4 ;
              b += c4 * h4 + c5 * h5 + c6 * h6 ;
              break ;
      case 5: h4 = z * h3 - 3.0 * h2 ;
              h5 = z * h4 - 4.0 * h3 ;
              h6 = z * h5 - 5.0 * h4 ;
              h7 = z * h6 - 6.0 * h5 ;
              b += c4 * h4 + c5 * h5 + c6 * h6 + c7 * h7 ;
              break ;
      case 6: h4 = z * h3 - 3.0 * h2 ;
              h5 = z * h4 - 4.0 * h3 ;
              h6 = z * h5 - 5.0 * h4 ;
              h7 = z * h6 - 6.0 * h5 ;
              h8 = z * h7 - 7.0 * h6 ;
              b += c4 * h4 + c5 * h5 + c6 * h6 + c7 * h7 + c8 * h8 ;
              break ;
      }

   return a * b ;
}

double EWnet::density ( const double x , const int n_moments )
{
   double z, a, b ;

   z = (x - mean) / std ;       // Standardize
   a = exp ( -0.5 * z * z ) ;   // Normal density (times a constant)
   h2 = z * z - 1.0 ;           // Second-degree Hermite polynomial
   h3 = z * (h2 - 2.0) ;        // And third
   b = 1.0  +  c3 * h3 ;        // First extra moment term (3rd moment)
   switch (n_moments) {
      case 2: h4 = z * h3 - 3.0 * h2 ;
              h5 = z * h4 - 4.0 * h3 ;
              h6 = z * h5 - 5.0 * h4 ;
              b += c4 * h4 + c6 * h6 ;
              break ;
      case 3: h4 = z * h3 - 3.0 * h2 ;
              h5 = z * h4 - 4.0 * h3 ;
              h6 = z * h5 - 5.0 * h4 ;
              h7 = z * h6 - 6.0 * h5 ;
              h8 = z * h7 - 7.0 * h6 ;
              h9 = z * h8 - 8.0 * h7 ;
              b += c4 * h4 + c5 * h5 + c6 * h6 + c7 * h7 + c9 * h9 ;
              break ;
      }

   return a * b ;
}

/*
--------------------------------------------------------------------------------

   Hermite - Compute Hermite polynomials

--------------------------------------------------------------------------------
*/

static void hermite ( const double x )
{
   h2 = x * x - 1.0 ;
   h3 = x * h2 - 2.0 * x ;
   h4 = x * h3 - 3.0 * h2 ;
   h5 = x * h4 - 4.0 * h3 ;
   h6 = x * h5 - 5.0 * h4 ;
   h7 = x * h6 - 6.0 * h5 ;
   h8 = x * h7 - 7.0 * h6 ;
   h9 = x * h8 - 8.0 * h7 ;
}

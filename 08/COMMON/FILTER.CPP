/******************************************************************************/
/*                                                                            */
/*  FILTER - All routines related to the Filter class.                        */
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

/*
--------------------------------------------------------------------------------

   Constructor

   If there is insufficient memory, it leaves public ok=0.
   The user should check for this after allocating with new.
   The flag 'use_qmf' must be set nonzero if 'morlet' or 'qmf' will be called.
   This allocates extra scratch memory for their use.

--------------------------------------------------------------------------------
*/

Filter::Filter (
   int npts ,        // Number of points in series
   double *series ,  // Series to be filtered, not touched
   int pad ,         // Pad with AT LEAST this to power of 2; Default is zero
   int use_qmf       // Extra full-length fft for qmf use?; Default is zero (no)
   )
{
   int i, detrend ;

   ok = 0 ;     // For correct destructor operation
   xr = NULL ;  // Good habit

   if (! npts)  // Careless user?
      return ;

   if (pad < 0) {     // This is a user flag to pad with mean
      pad = -pad ;
      detrend = 0 ;
      }
   else               // Default is endpoint detrending
      detrend = 1 ;

/*
   Push the user's n up to a power of two
*/

   length = npts ;  // Save original length to return correct length
   for (n=2 ; n<MAXPOSNUM/2 ; n*=2) {
      if (n >= npts+pad)
         break ;
      }

   halfn = n / 2 ;

/*
   Allocate the local storage and the FFT object
*/

   MEMTEXT ( "FILTER: xr" ) ;
   xr = (double *) MALLOC ( 3 * n * sizeof(double) ) ;
   if (xr == NULL)
      return ;

   MEMTEXT ( "FILTER: new FFT" ) ;
   fft = new FFT ( halfn , 1 , 1 ) ;

   if ((fft == NULL)  ||  (! fft->ok)) {
      if (fft != NULL) {
         delete fft ;
         fft = NULL ;
         }
      FREE ( xr ) ;
      xr = NULL ;
      return ;
      }

   if (use_qmf) {   // Also includes 'morlet'
      MEMTEXT ( "FILTER: new FFT for qmf" ) ;
      qmf_fft = new FFT ( n , 1 , 1 ) ;
      if ((qmf_fft == NULL)  ||  (! qmf_fft->ok)) {
         if (qmf_fft != NULL) {
            delete qmf_fft ;
            qmf_fft = NULL ;
            }
         FREE ( xr ) ;
         xr = NULL ;
         delete fft ;
         fft = NULL ;
         return ;
         }
      }
   else
      qmf_fft = NULL ;

   ok = 1 ;        // Flag that allocation went ok

   xi = xr + halfn ;
   workr = xi + halfn ;
   worki = workr + n ;

/*
   Compute and subtract the detrend line and pad with zeros (or pad with mean
   by centering and padding with zero).
*/

   if (n > npts) {         // Any padding needed?

      if (detrend) {
         intercept = series[0] ;
         slope = (series[npts-1] - series[0]) / (npts-1) ;
         }
      else {
         intercept = slope = 0.0 ;
         for (i=0 ; i<npts ; i++)
            intercept += series[i] ;
         intercept /= npts ;
         }

      for (i=0 ; i<npts ; i++) {
         if (i % 2)
            xi[i/2] = series[i] - intercept - slope * i ;
         else 
            xr[i/2] = series[i] - intercept - slope * i ;
         }
      for (i=npts ; i<n ; i++) {
         if (i % 2)
            xi[i/2] = 0.0 ;
         else 
            xr[i/2] = 0.0 ;
         }
      }

   else {                           // This would be extremely rare
      intercept = slope = 0.0 ;
      for (i=0 ; i<npts ; i++) {
         if (i % 2)
            xi[i/2] = series[i] ;
         else 
            xr[i/2] = series[i] ;
         }
      }

/*
   Compute the Fourier transform of the padded series
*/

   fft->rv ( xr , xi ) ;

}

Filter::~Filter ()
{
   if (! ok)
      return ;
   if (xr != NULL)
      FREE ( xr ) ;
   if (fft != NULL)
      delete fft ;
   if (qmf_fft != NULL)
      delete qmf_fft ;
}

/*
   Lowpass filter
*/

void Filter::lowpass (
   double freq ,    // Highest frequency before dropoff starts (0-.5)
   double width ,   // Scale factor for exponential (in frequency units)
   double *out      // Filtered output here
   )
{
   int i ;
   double f, dist, wt ;

   for (i=0 ; i<halfn ; i++) {
      f = (double) i / (double) n ;  // This frequency
      if (f <= freq)                 // Flat to here
         wt = 1.0 ;
      else {
         dist = (f - freq) / width ;
         wt = exp ( -dist * dist ) ;
         }
      workr[i] = xr[i] * wt ;
      worki[i] = xi[i] * wt ;
      }

   dist = (0.5 - freq) / width ;     // Also do Nyquist in xi[0]
   worki[0] = xi[0] * exp ( -dist * dist ) ;

   fft->irv ( workr , worki ) ;

   for (i=0 ; i<length ; i++) {
      if (i % 2)
         out[i] = worki[i/2] + intercept + slope * i ;
      else 
         out[i] = workr[i/2] + intercept + slope * i ;
      }
}

/*
   Highpass filter
*/

void Filter::highpass (
   double freq ,    // Lowest frequency before dropoff starts (0-.5)
   double width ,   // Scale factor for exponential (in frequency units)
   double *out      // Filtered output here
   )
{
   int i ;
   double f, dist, wt ;

   for (i=0 ; i<halfn ; i++) {
      f = (double) i / (double) n ;  // This frequency
      if (f >= freq)                 // Flat to here
         wt = 1.0 ;
      else {
         dist = (f - freq) / width ;
         wt = exp ( -dist * dist ) ;
         }
      workr[i] = xr[i] * wt ;
      worki[i] = xi[i] * wt ;
      }

   worki[0] = xi[0] ;  // Nyquist frequency is kept fully

   fft->irv ( workr , worki ) ;

   for (i=0 ; i<length ; i++) {
      if (i % 2)
         out[i] = worki[i/2] ;
      else 
         out[i] = workr[i/2] ;
      }
}

/*
   Bandpass filter
*/

void Filter::bandpass (
   double freq ,    // Center frequency (0-.5)
   double width ,   // Scale factor for exponential (in frequency units)
   double *out      // Filtered output here
   )
{
   int i ;
   double f, dist, wt ;

   for (i=0 ; i<halfn ; i++) {
      f = (double) i / (double) n ;  // This frequency
      dist = (f - freq) / width ;
      wt = exp ( -dist * dist ) ;
      workr[i] = xr[i] * wt ;
      worki[i] = xi[i] * wt ;
      }

   dist = (0.5 - freq) / width ;     // Also do Nyquist in xi[0]
   worki[0] = xi[0] * exp ( -dist * dist ) ;

   fft->irv ( workr , worki ) ;

   for (i=0 ; i<length ; i++) {
      if (i % 2)
         out[i] = worki[i/2] ;
      else 
         out[i] = workr[i/2] ;
      }
}

/*
   Quadrature-mirror filter
*/

void Filter::qmf (
   double freq ,    // Center frequency (0-.5)
   double width ,   // Scale factor for exponential (in frequency units)
   double *real ,   // Filtered real output here
   double *imag     // Filtered imaginary output here
   )
{
   int i ;
   double f, dist, wt ;

   if (qmf_fft == NULL)   // Silly user may forget to warn constructor
      return ;

   for (i=1 ; i<halfn ; i++) {
      f = (double) i / (double) n ;    // This frequency
      dist = (f - freq) / width ;
      wt = exp ( -dist * dist ) ;
      workr[i] = xr[i] * wt ;
      worki[i] = xi[i] * wt ;
      workr[n-i] = worki[n-i] = 0.0 ;  // Causes QMF outputs
      }

   workr[0] = 0.0 ;                    // Force DC to zero for centering output

/*
   When we do the Nyquist (in xi[0]), we must multiply by an additional factor
   of 0.5.  See Equation 3.2 (the power spectrum) for a hint on why this is
   needed.  Recall that the DFT is symmetric, and powers are duplicated on
   both sides of the Nyquist frequency, so they have double influence.
*/

   dist = (0.5 - freq) / width ;       // Nyquist is in xi[0]
   workr[halfn] = 0.5 * xi[0] * exp ( -dist * dist ) ;

   worki[0] = worki[halfn] = 0.0 ;     // By definition of real transform

   qmf_fft->cpx ( workr , worki , -1 ) ;

   for (i=0 ; i<length ; i++) {
      real[i] = workr[i] / halfn ;
      imag[i] = worki[i] / halfn ;
      }
}

/*
   Morlet wavelet filter
*/

void Filter::morlet (
   double freq ,    // Center frequency (0-.5)
   double width ,   // Scale factor for exponential (in frequency units)
   double *real ,   // Filtered real output here
   double *imag     // Filtered imaginary output here
   )
{
   int i ;
   double f, wt, temp, widsq, numer, denom, wtmax ;

   if (qmf_fft == NULL)   // Silly user may forget to warn constructor
      return ;

   widsq = width * width ;
   wtmax = 1.e-30 ;
   for (i=1 ; i<halfn ; i++) {
      f = (double) i / (double) n ;    // This frequency
      temp = 2.0 * freq * f / widsq ;
      numer = exp ( temp )  +  exp ( -temp )  -  2.0 ;
      denom = 2.0 * exp ( (f * f + freq * freq) / widsq ) ;
      wt = numer / denom ;
      if (wt > wtmax)
         wtmax = wt ;
      workr[i] = xr[i] * wt ;
      worki[i] = xi[i] * wt ;
      workr[n-i] = worki[n-i] = 0.0 ;  // Causes QMF outputs
      }

   workr[0] = 0.0 ;                    // Force DC to zero

   temp = 2.0 * freq * 0.5 / widsq ;   // And Nyquist in xi[0]
   numer = exp ( temp )  +  exp ( -temp )  -  2.0 ;
   denom = 2.0 * exp ( (0.5 * 0.5 + freq * freq) / widsq ) ;
   wt = numer / denom ;
   if (wt > wtmax)
      wtmax = wt ;
   workr[halfn] = 0.5 * xi[0] * wt ;

   worki[0] = worki[halfn] = 0.0 ;     // By definition of real transform

   qmf_fft->cpx ( workr , worki , -1 ) ;

   wtmax = 1.0 / (wtmax * halfn) ;
   for (i=0 ; i<length ; i++) {
      real[i] = workr[i] * wtmax ;
      imag[i] = worki[i] * wtmax ;
      }
}

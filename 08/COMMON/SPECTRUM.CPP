/******************************************************************************/
/*                                                                            */
/*  SPECTRUM - Generate one or more new signals as power spectra.             */
/*  Normally, this returns 0.  It returns 1 if insufficient memory.           */
/*  Up to five signals are created: Real, Imag, Power, Phase, Deviation       */
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

int spectrum ( MiscParams *misc , Signal *sig ,int *nsigs ,
               Signal ***signals , double *dmax , double *alpha , char *error )
{
   int i, j, n, ivar, nvars, ncases ;
   double *real, *imag, *temp, *power, *phase, *dev, win, win2, wsum, dsum, wsq;
   double sum, sfac, nfac ;
   Signal **sptr ;
   FFT *fft ;

   ncases = sig->n / 2 + 1 ;

   *dmax = -1.0 ;   // Flag to indicate not computed

/*
   Allocate memory for the generated signal(s) and FFT object.
*/

   MEMTEXT ( "SPECTRUM: 2-4 work, new FFT" ) ;
   if (sig->n % 2)
      n = sig->n ;
   else 
      n = ncases ;
   real = (double *) MALLOC ( n * sizeof(double) ) ;
   imag = (double *) MALLOC ( n * sizeof(double) ) ;
   if ((real == NULL)  ||  (imag == NULL)) {
      if (real != NULL)
         FREE ( real ) ;
      if (imag != NULL)
         FREE ( imag ) ;
      strcpy ( error , "Insufficient memory to create signal" ) ;
      return 1 ;
      }

   if ((misc->names->n > 2)  &&  misc->names->len[2]) {
      power = (double *) MALLOC ( ncases * sizeof(double) ) ;
      if (power == NULL) {
         FREE ( real ) ;
         FREE ( imag ) ;
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }
      }
   else
      power = NULL ;

   if ((misc->names->n > 3)  &&  misc->names->len[3]) {
      phase = (double *) MALLOC ( ncases * sizeof(double) ) ;
      if (phase == NULL) {
         FREE ( real ) ;
         FREE ( imag ) ;
         if (power != NULL)
            FREE ( power ) ;
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }
      }
   else
      phase = NULL ;

   if ((misc->names->n > 4)  &&  misc->names->len[4]) {
      dev = (double *) MALLOC ( ncases * sizeof(double) ) ;
      if (dev == NULL) {
         FREE ( real ) ;
         FREE ( imag ) ;
         if (power != NULL)
            FREE ( power ) ;
         if (phase != NULL)
            FREE ( phase ) ;
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }
      }
   else
      dev = NULL ;

   if (sig->n % 2)                            // If odd signal length
      fft = new FFT ( sig->n , 1 , 1 ) ;      // Must use brute force
   else                                       // But if even, we can
      fft = new FFT ( sig->n / 2 , 1 , 1 ) ;  // Use the compact method

   if ((fft == NULL)  ||  (! fft->ok)) {
      if (fft != NULL)
         delete fft ;
      FREE ( real ) ;
      FREE ( imag ) ;
      if (power != NULL)
         FREE ( power ) ;
      if (phase != NULL)
         FREE ( phase ) ;
      if (dev != NULL)
         FREE ( dev ) ;
      strcpy ( error , "Insufficient memory to create signal" ) ;
      return 1 ;
      }

/*
   Compute the window-weighted mean for precentering
*/

   if (misc->spectrum_window) {
      wsum = dsum = wsq = 0.0 ;
      for (i=0 ; i<sig->n ; i++) {
         win = (i - 0.5 * (sig->n - 1)) / (0.5 * (sig->n + 1)) ;
         win = 1.0 - win * win ;
         wsum += win ;
         dsum += win * sig->sig[i] ;
         wsq += win * win ;
         }
      dsum /= wsum ;
      wsq = 1.0 / sqrt ( sig->n * wsq ) ;
      }
   else {
      dsum = 0.0 ;
      wsq = 1.0 / sig->n ;
      }

/*
   Setup the data for transforming
*/

   if (sig->n % 2) {                     // If the silly user has an odd length
      for (i=0 ; i<sig->n ; i++) {       // We must put it in the real vector
         if (misc->spectrum_window) {    // If a data window is requested
            win = (i - 0.5 * (sig->n - 1)) / (0.5 * (sig->n + 1)) ;
            win = 1.0 - win * win ;      // Welch window
            }
         else 
            win = 1.0 ;
         win *= wsq ;                    // Make it a mean for consistency
         real[i] = win * (sig->sig[i] - dsum) ;   
         imag[i] = 0.0 ;                 // And zero out the imaginary part
         }
      fft->cpx ( real , imag , 1 ) ;  // Then use the full complex version
      real = (double *) REALLOC ( real , ncases * sizeof(double) ) ;
      imag = (double *) REALLOC ( imag , ncases * sizeof(double) ) ;
      }
   else {
      for (i=0 ; i<sig->n/2 ; i++) {
         if (misc->spectrum_window) {    // If a data window is requested
            win = (2*i - 0.5 * (sig->n - 1)) / (0.5 * (sig->n + 1)) ;
            win = 1.0 - win * win ;      // Welch window
            win2 = (2*i+1 - 0.5 * (sig->n - 1)) / (0.5 * (sig->n + 1)) ;
            win2 = 1.0 - win2 * win2 ;
            }
         else
            win = win2 = 1.0 ;
         win *= wsq ;                    // Make it a mean for consistency
         win2 *= wsq ;
         real[i] = win * (sig->sig[2*i] - dsum) ;
         imag[i] = win2 * (sig->sig[2*i+1] - dsum) ;
         }
      fft->rv ( real , imag ) ;          // And use the efficient method
      real[sig->n/2] = imag[0] ;         // The rv routine put real Nyquist in
      imag[0] = 0.0 ;                    // Imaginary zero, which is truly 0.
      imag[sig->n/2] = 0.0 ;             // This is also truly zero
      }

   MEMTEXT ( "SPECTRUM: delete FFT" ) ;
   delete fft ;

/*
   Compute the power, phase, and deviation if needed
*/

   if (power != NULL) {
      for (i=0 ; i<ncases ; i++) {
         power[i] = real[i] * real[i]  +  imag[i] * imag[i] ;
         if (i  &&  (i < (ncases-1)))
            power[i] *= 2.0 ;
         }
      }

   if (phase != NULL) {
      for (i=0 ; i<ncases ; i++) {
         if ((fabs(real[i]) > 1.e-40)  ||  (fabs(imag[i]) > 1.e-40))
            phase[i] = atan2 ( imag[i] , real[i] ) ;
         else
            phase[i] = 0.0 ;
         }
      }

   if (dev != NULL) {
      sum = 0.0 ;
      for (i=1 ; i<ncases ; i++) {
         dev[i] = real[i] * real[i]  +  imag[i] * imag[i] ;
         if (i == (ncases-1))
            dev[i] *= 0.5 ;
         sum += dev[i] ;
         }
      sfac = 1.0 / sum ;
      nfac = 1.0 / (ncases - 1) ;
      sum = *dmax = 0.0 ;
      for (i=1 ; i<ncases-1 ; i++) {
         sum += dev[i] ;
         dev[i] = sum * sfac  -  i * nfac ;
         if (fabs(dev[i]) > *dmax)
            *dmax = fabs(dev[i]) ;
         }
      dev[0] = dev[ncases-1] = 0.0 ;
      }

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   nvars = misc->names->nreal ;
   MEMTEXT ( "SPECTRUM: signals array" ) ;
   if (*nsigs) {                             // If signals already exist
      ivar = *nsigs ;                        // This many signals so far
      sptr = *signals ;                      // Array of pointers to them
      for (i=0 ; i<misc->names->n ; i++) {   // Check every new name
         if (! misc->names->len[i])          // Some may be NULL
            continue ;                       // Obviously skip them
         for (j=*nsigs-1 ; j>=0 ; j--) {     // Check every existing signal
            if (! strcmp ( misc->names->start[i] , sptr[j]->name )) // There?
               break ;                       // If found, quit looking
            }
         if (j < 0)                          // Means not there
            ++ivar ;                         // So count this new entry
         }
      sptr = (Signal **) REALLOC ( sptr , ivar * sizeof(Signal *) ) ;
      }
   else
      sptr = (Signal **) MALLOC ( nvars * sizeof(Signal *) ) ;

   if (sptr == NULL) {
      FREE ( real ) ;
      FREE ( imag ) ;
      if (power != NULL)
         FREE ( power ) ;
      if (phase != NULL)
         FREE ( phase ) ;
      if (dev != NULL)
         FREE ( dev ) ;
      strcpy ( error , "Insufficient memory to create signal" ) ;
      return 1 ;
      }

   *signals = sptr ;

/*
   Now create new signals for each variable.
   If a signal of the same name exists, delete it first.
   Recall that we have provision for up to five outputs in this order:
   Real, Imaginary, Power, Phase, Deviation.
*/

   for (i=0 ; i<misc->names->n ; i++) {  // Check all names
      if (i >= 5)                        // Allow at most 5 outputs
         break ;
      if (! misc->names->len[i]) {       // Some may be NULL
         switch (i) {                    // If not used in a new signal
            case 0: FREE ( real ) ;  break ; // Must free this memory
            case 1: FREE ( imag ) ;  break ;
            }
         continue ;
         }
      for (j=*nsigs-1 ; j>=0 ; j--) {   // Search existing signals for same name
         if (! strcmp ( misc->names->start[i] , sptr[j]->name )) {  // There?
            MEMTEXT ( "SPECTRUM: delete duplicate signal" ) ;
            delete ( sptr[j] ) ;        // If so, delete this signal
            break ;                     // And quit looking
            }
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }

      switch (i) {
         case 0: temp = real ;  break ;
         case 1: temp = imag ;  break ;
         case 2: temp = power ;  break ;
         case 3: temp = phase ;  break ;
         case 4: temp = dev ;  break ;
         }

      MEMTEXT ( "SPECTRUM: new Signal" ) ;
      sptr[j] = new Signal ( misc->names->start[i] , ncases , temp ) ;
      if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
         if (sptr[j] != NULL) {
            delete sptr[j] ;
            sptr[j] = NULL ;
            }
         switch (i) {   // Note that we do NOT want breaks after these cases!
            case 0: FREE ( real ) ;
            case 1: FREE ( imag ) ;
            case 2: if (power != NULL)
                       FREE ( power ) ;
            case 3: if (phase != NULL)
                       FREE ( phase ) ;
            case 4: if (dev != NULL)
                       FREE ( dev ) ;
            }
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }

      if (misc->spectrum_window)
         sptr[j]->source_n = (int) (sig->n / 1.38 + 0.5) ;
      else 
         sptr[j]->source_n = sig->n ;

      if (i == 4) {
         sptr[j]->type = SpectrumDevSignal ;
         *alpha= 2.0 * exp( -2.0 * (sptr[j]->source_n / 2 - 1) * *dmax * *dmax);
         if (*alpha > 1.0)
            *alpha = 1.0 ;
         }
      else 
         sptr[j]->type = SpectrumSignal ;

      } // For all names

   if (misc->names->n == 1) {
      MEMTEXT ( "SPECTRUM: imag" ) ;
      FREE ( imag ) ;
      }

   return 0 ;
}

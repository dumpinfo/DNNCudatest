/******************************************************************************/
/*                                                                            */
/*  MORLET -  Apply a Morlet wavelet filter to a signal.                      */
/*  Normally, this returns 0.  It returns 1 if insufficient memory.           */
/*  Up to four signals are created: Real, Imag, Amp, Phase                    */
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

int morlet ( MiscParams *misc , Signal *sig ,
             double freq , double width ,
             int *nsigs , Signal ***signals , char *error )
{
   int i, j, n, ivar, nvars, pad ;
   double *real, *imag, *temp, *x ;
   double *amp, *phase ;
   Signal **sptr ;
   Filter *filt ;


   nvars = misc->names->nreal ;

   if (! nvars) {
      strcpy ( error , "No signal names specified" ) ;
      return 1 ;
      }

   n = sig->n ;
   x = sig->sig ;

   pad = (int) (0.8 / width) ;   // Good, conservative heuristic

   MEMTEXT ( "MORLET: new Filter" ) ;
   filt = new Filter ( n , x , pad , 1 ) ;
   MEMTEXT ( "MORLET: real, imag" ) ;
   real = (double *) MALLOC ( n * sizeof(double) ) ;
   imag = (double *) MALLOC ( n * sizeof(double) ) ;
   if ((real == NULL) || (imag == NULL) || (filt == NULL) || ! filt->ok){
      if (real != NULL)
         FREE ( real ) ;
      if (imag != NULL)
         FREE ( imag ) ;
      if (filt != NULL)
         delete filt ;
      strcpy ( error , "Insufficient memory to filter signal" ) ;
      return 1 ;
      }


   if ((misc->names->n > 2)  &&  misc->names->len[2]) {
      MEMTEXT ( "MORLET: amp" ) ;
      amp = (double *) MALLOC ( n * sizeof(double) ) ;
      if (amp == NULL) {
         FREE ( real ) ;
         FREE ( imag ) ;
         delete filt ;
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }
      }
   else
      amp = NULL ;

   if ((misc->names->n > 3)  &&  misc->names->len[3]) {
      MEMTEXT ( "MORLET: phase" ) ;
      phase = (double *) MALLOC ( n * sizeof(double) ) ;
      if (phase == NULL) {
         FREE ( real ) ;
         FREE ( imag ) ;
         delete filt ;
         if (amp != NULL)
            FREE ( amp ) ;
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }
      }
   else
      phase = NULL ;

   filt->morlet ( freq , width , real , imag ) ;

   if (amp != NULL) {
      for (i=0 ; i<n ; i++)
         amp[i] = sqrt ( real[i] * real[i]  +  imag[i] * imag[i] ) ;
      }

   if (phase != NULL) {
      for (i=0 ; i<n ; i++) {
         if ((fabs(real[i]) > 1.e-40)  ||  (fabs(imag[i]) > 1.e-40))
            phase[i] = atan2 ( imag[i] , real[i] ) ;
         else
            phase[i] = 0.0 ;
         }
      }

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   MEMTEXT ( "MORLET: signals array" ) ;
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
      if (amp != NULL)
         FREE ( amp ) ;
      if (phase != NULL)
         FREE ( phase ) ;
      strcpy ( error , "Insufficient memory to create signal" ) ;
      return 1 ;
      }

   *signals = sptr ;

/*
   Now create new signals for each variable.
   If a signal of the same name exists, delete it first.
   Recall that we have provision for up to four outputs in this order:
   Real, Imaginary, Amplitude, Phase.
*/

   for (i=0 ; i<misc->names->n ; i++) {  // Check all names
      if (i >= 4)                        // Allow at most 4 outputs
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
         case 2: temp = amp ;  break ;
         case 3: temp = phase ;  break ;
         }

      MEMTEXT ( "SPECTRUM: new Signal" ) ;
      sptr[j] = new Signal ( misc->names->start[i] , n , temp ) ;
      if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
         if (sptr[j] != NULL) {
            delete sptr[j] ;
            sptr[j] = NULL ;
            }
         switch (i) {
            case 0: FREE ( real ) ;
            case 1: FREE ( imag ) ;
            case 2: if (amp != NULL)
                       FREE ( amp ) ;
            case 3: if (phase != NULL)
                       FREE ( phase ) ;
            }
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }

      } // For all names

   if (misc->names->n == 1) {
      MEMTEXT ( "MORLET: imag" ) ;
      FREE ( imag ) ;
      }

   MEMTEXT ( "MORLET: delete Filter" ) ;
   delete filt ;

   return 0 ;
}




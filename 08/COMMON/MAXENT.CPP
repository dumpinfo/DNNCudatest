/******************************************************************************/
/*                                                                            */
/*  MAXENT - Compute the maximum entropy power spectrum of a signal.          */
/*  Normally, this returns 0.  It returns 1 if insufficient memory.           */
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

int maxent ( MiscParams *misc , int ncases , int degree , Signal *sig ,
             int *nsigs , Signal ***signals , char *error )
{
   int i, j, n, ivar, nvars ;
   double *data, *temp, *work, *x, mean, var, diff, *pcorr ;
   double real, imag, theta, maxspec ;
   Signal **sptr ;

   nvars = misc->names->nreal ;

   if (! nvars) {
      strcpy ( error , "No signal names specified" ) ;
      return 1 ;
      }

   n = sig->n ;
   x = sig->sig ;

   MEMTEXT ( "MAXENT: data, work" ) ;
   data = (double *) MALLOC ( ncases * sizeof(double) ) ;
   work = (double *) MALLOC ( (3 * degree + 2 * n) * sizeof(double) ) ;

   if ((data == NULL)  ||  (work == NULL)) {
      strcpy ( error , "Insufficient memory to create signal" ) ;
      return 1 ;
      }

/*
   Compute the spectrum
*/

   burg ( n , x , degree , work+degree , work , work+2*degree ,
          work+3*degree , work+3*degree+sig->n ) ;

   mean = var = 0.0 ;
   for (i=0 ; i<n ; i++)
      mean += x[i] ;
   mean /= n ;

   for (i=0 ; i<n ; i++) {
      diff = x[i] - mean ;
      var += diff * diff ;
      }
   var /= (n * degree) ;

   pcorr = work+degree ;       // Partial autocorrelations
   for (i=0 ; i<degree ; i++)
      var *= (1.0 - pcorr[i] * pcorr[i]) ;

   if (var < 1.e-40)
      var = 1.e-40 ;

   maxspec = 0.0 ;
   for (i=0 ; i<ncases ; i++) {
      theta = PI * (double) i / (double) ncases ;
      real = 1.0 ;
      imag = 0.0 ;
      for (j=0 ; j<degree ; j++) {
         real -= work[j] * cos ( (j+1) * theta ) ;
         imag -= work[j] * sin ( (j+1) * theta ) ;
         }
      data[i] = var / (real * real  +  imag * imag  + 1.e-40) ;
      if (data[i] > maxspec)
         maxspec = data[i] ;
      }

/*
   Optional step:  Normalize so max is almost 1.0 for better display.
*/

   maxspec = 0.99999999999 / maxspec ;
   for (i=0 ; i<ncases ; i++)
      data[i] *= maxspec ;

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   MEMTEXT ( "MAXENT: signals array" ) ;
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
      FREE ( data ) ;
      strcpy ( error , "Insufficient memory to create signal" ) ;
      return 1 ;
      }
   *signals = sptr ;

/*
   Now create new signals for each variable.
   If a signal of the same name exists, delete it first.
*/

   ivar = 0 ;
   for (i=0 ; i<misc->names->n ; i++) { // Check all names
      if (! misc->names->len[i])        // Some may be NULL
         continue ;                     // Obviously skip them
      for (j=*nsigs-1 ; j>=0 ; j--) {   // Search existing signals for same name
         if (! strcmp ( misc->names->start[i] , sptr[j]->name )) {  // There?
            MEMTEXT ( "MAXENT: delete duplicate signal" ) ;
            delete ( sptr[j] ) ;        // If so, delete this signal
            break ;                     // And quit looking
            }
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }

      if (ivar) {   // In this case, must allocate for new signal
         MEMTEXT ( "MAXENT: temp signal" ) ;
         temp = (double *) MALLOC ( ncases * sizeof(double) ) ;
         if (temp == NULL) {
            strcpy ( error , "Insufficient memory to create signal" ) ;
            return 1 ;
            }
         memcpy ( temp , data , ncases * sizeof(double) ) ;
         }
      else
         temp = data ;

      MEMTEXT ( "MAXENT: new Signal" ) ;
      sptr[j] = new Signal ( misc->names->start[i] , ncases , temp ) ;
      if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
         if (sptr[j] != NULL) {
            delete sptr[j] ;
            sptr[j] = NULL ;
            }
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }
      ++ivar ;
      sptr[j]->type = SpectrumSignal ;
      sptr[j]->source_n = n ;
      } // For all names

   MEMTEXT ( "MAXENT: work" ) ;
   FREE ( work ) ;

   return 0 ;
}


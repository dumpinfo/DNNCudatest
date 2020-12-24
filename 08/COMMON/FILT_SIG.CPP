/******************************************************************************/
/*                                                                            */
/*  FILT_SIG - Apply a lowpass, highpass, or bandpass filter to a signal.     */
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

int filt_sig ( MiscParams *misc , int id , Signal *sig ,
               double freq , double width ,
               int *nsigs , Signal ***signals , char *error )
{
   int i, j, n, ivar, nvars, pad ;
   double *data, *temp, *x ;
   Signal **sptr ;
   Filter *filt ;
   enum SignalType save_type ;


   nvars = misc->names->nreal ;

   if (! nvars) {
      strcpy ( error , "No signal names specified" ) ;
      return 1 ;
      }

   n = sig->n ;
   x = sig->sig ;
   save_type = sig->type ;

   pad = (int) (0.8 / width) ;   // Good, conservative heuristic

   MEMTEXT ( "FILT_SIG: new Filter" ) ;
   filt = new Filter ( n , x ,  misc->padding ? pad : -pad , 0 ) ;
   MEMTEXT ( "FILT_SIG: data" ) ;
   data = (double *) MALLOC ( n * sizeof(double) ) ;
   if ((data == NULL) || (filt == NULL) || ! filt->ok){
      if (data != NULL)
         FREE ( data ) ;
      if (filt != NULL)
         delete filt ;
      strcpy ( error , "Insufficient memory to filter signal" ) ;
      return 1 ;
      }

   if (id == ID_PRED_LOWPASS)
      filt->lowpass ( freq , width , data ) ;
   else if (id == ID_PRED_HIGHPASS)
      filt->highpass ( freq , width , data ) ;
   else if (id == ID_PRED_BANDPASS)
      filt->bandpass ( freq , width , data ) ;

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   MEMTEXT ( "FILT_SIG: signals array" ) ;
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
            MEMTEXT ( "FILT_SIG: delete duplicate signal" ) ;
            delete ( sptr[j] ) ;        // If so, delete this signal
            break ;                     // And quit looking
            }
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }

      if (ivar) {   // In this case, must allocate for new signal
         MEMTEXT ( "FILT_SIG: temp signal" ) ;
         temp = (double *) MALLOC ( n * sizeof(double) ) ;
         if (temp == NULL) {
            strcpy ( error , "Insufficient memory to create signal" ) ;
            return 1 ;
            }
         memcpy ( temp , data , n * sizeof(double) ) ;
         }
      else
         temp = data ;

      MEMTEXT ( "FILT_SIG: new Signal" ) ;
      sptr[j] = new Signal ( misc->names->start[i] , n , temp ) ;
      if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
         if (sptr[j] != NULL) {
            delete sptr[j] ;
            sptr[j] = NULL ;
            }
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }
      if (save_type == SpectrumSignal)
         sptr[j]->type = SpectrumSignal ;
      ++ivar ;
      } // For all names

   MEMTEXT ( "FILT_SIG: delete Filter" ) ;
   delete filt ;

   return 0 ;
}



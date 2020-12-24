/******************************************************************************/
/*                                                                            */
/*  COPY - Copy one signal to another                                         */
/*  Normally, this returns 0.  It returns 1 if insufficient memory.           */
/*  If ncases is zero, a straight copy is done.                               */
/*  If positive, that many cases from the beginning are copied.               */
/*  If negative, that many cases from the end are copied.                     */
/*  If too large, zeros are used to fill in the excess.                       */
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

int copy ( MiscParams *misc , int ncases , Signal *sig ,
           int *nsigs , Signal ***signals , char *error )
{
   int i, j, n, ivar, nvars ;
   double *data, *temp ;
   Signal **sptr ;

   nvars = misc->names->nreal ;

   if (! nvars) {
      strcpy ( error , "No signal names specified" ) ;
      return 1 ;
      }

   if (! ncases)
      ncases = sig->n ;

   MEMTEXT ( "COPY: data" ) ;
   data = (double *) MALLOC ( abs(ncases) * sizeof(double) ) ;

   if (data == NULL) {
      strcpy ( error , "Insufficient memory to create signal" ) ;
      return 1 ;
      }

   if (ncases > 0) {    // Copy from beginning of source
      n = (ncases < sig->n)  ?  ncases : sig->n ;
      memcpy ( data , sig->sig , n * sizeof(double) ) ;
      while (n < ncases)
         data[n++] = 0.0 ;
      }

   else {
      ncases = -ncases ;
      n = (ncases < sig->n)  ?  ncases : sig->n ;
      i = sig->n - n ;
      j = ncases - n ;
      memcpy ( data + j , sig->sig + i , n * sizeof(double) ) ;
      while (j--)
         data[j] = 0.0 ;
      }

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   MEMTEXT ( "COPY: signals array" ) ;
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
            MEMTEXT ( "COPY: delete duplicate signal" ) ;
            delete ( sptr[j] ) ;        // If so, delete this signal
            break ;                     // And quit looking
            }
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }

      if (ivar) {   // In this case, must allocate for new signal
         MEMTEXT ( "COPY: temp signal" ) ;
         temp = (double *) MALLOC ( ncases * sizeof(double) ) ;
         if (temp == NULL) {
            strcpy ( error , "Insufficient memory to create signal" ) ;
            return 1 ;
            }
         memcpy ( temp , data , ncases * sizeof(double) ) ;
         }
      else
         temp = data ;

      MEMTEXT ( "COPY: new Signal" ) ;
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
      } // For all names

   return 0 ;
}


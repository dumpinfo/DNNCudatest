/******************************************************************************/
/*                                                                            */
/*  COMBINE - Generate one or more new signals by arithmetic combination      */
/*            of two source signals.                                          */
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

int combine ( MiscParams *misc , int operation , Signal *sig1 , Signal *sig2 ,
              int *nsigs , Signal ***signals , char *error )
{
   int i, j, ivar, nvars, ncases ;
   double *data, *temp, *dptr ;
   Signal **sptr ;

   nvars = misc->names->nreal ;

   if (! nvars) {
      strcpy ( error , "No signal names specified" ) ;
      return 1 ;
      }

   ncases = sig1->n ;
   if (sig2->n < ncases)
      ncases = sig2->n ;

   if (! ncases) {
      strcpy ( error , "Signal length is zero" ) ;
      return 1 ;
      }

   MEMTEXT ( "COMBINE: data" ) ;
   data = (double *) MALLOC ( ncases * sizeof(double) ) ;

   if (data == NULL) {
      strcpy ( error , "Insufficient memory to create signal" ) ;
      return 1 ;
      }

   dptr = data ;

   switch (operation) {
      case ID_PRED_ADD:
         for (i=0 ; i<ncases ; i++)
            *dptr++ = sig1->sig[i] + sig2->sig[i] ;
         break ;
      case ID_PRED_SUBTRACT:
         for (i=0 ; i<ncases ; i++)
            *dptr++ = sig1->sig[i] - sig2->sig[i] ;
         break ;
      case ID_PRED_MULTIPLY:
         for (i=0 ; i<ncases ; i++)
            *dptr++ = sig1->sig[i] * sig2->sig[i] ;
         break ;
      case ID_PRED_DIVIDE:
         for (i=0 ; i<ncases ; i++)
            *dptr++ = (sig2->sig[i] != 0.0) ? sig1->sig[i] / sig2->sig[i] : 0.0;
         break ;
      }

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   MEMTEXT ( "COMBINE: signals array" ) ;
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
            MEMTEXT ( "COMBINE: delete duplicate signal" ) ;
            delete ( sptr[j] ) ;        // If so, delete this signal
            break ;                     // And quit looking
            }
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }

      if (ivar) {   // In this case, must allocate for new signal
         MEMTEXT ( "COMBINE: temp signal" ) ;
         temp = (double *) MALLOC ( ncases * sizeof(double) ) ;
         if (temp == NULL) {
            strcpy ( error , "Insufficient memory to create signal" ) ;
            return 1 ;
            }
         memcpy ( temp , data , ncases * sizeof(double) ) ;
         }
      else
         temp = data ;

      MEMTEXT ( "COMBINE: new Signal" ) ;
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
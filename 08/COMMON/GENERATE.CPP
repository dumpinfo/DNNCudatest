/******************************************************************************/
/*                                                                            */
/*  GENERATE - Generate one or more signals                                   */
/*                                                                            */
/*  Normally, this returns 0.  It returns 1 if an error occurred.             */
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

#define DEBUG 0

int generate ( MiscParams *misc , char *rest , int *nsigs ,
               Signal ***signals , char *error )
{
   int i, j, n, type, nparams, ivar, nvars, ncases ;
#if DEBUG
   char msg[256] ;
#endif
   double *data, *temp, *dptr, x, y, z, params[6], factor, phase ;
   Signal **sptr ;

   nvars = misc->names->nreal ;

   if (! nvars) {
      strcpy ( error , "No signal names specified" ) ;
      return 1 ;
      }

/*
   The format is GENERATE = #cases  type (optional params).
   Parse that.
*/

   n = sscanf ( rest , "%d" , &ncases ) ;
   if ((! n)  ||  (ncases < 1)) {
      strcpy ( error , "Illegal generated signal length specified" ) ;
      return 1 ;
      }

   while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
      ++rest ;  // Skip number of cases

   if (! *rest) {
      strcpy ( error , "No generated type specified" ) ;
      return 1 ;
      }

   if (! strncmp ( rest , "UNIFORM" , 7 )) {
      type = 1 ;
      rest += 7 ;
      nparams = 2 ;
      }
   else if (! strncmp ( rest , "NORMAL" , 6 )) {
      type = 2 ;
      rest += 6 ;
      nparams = 2 ;
      }
   else if (! strncmp ( rest , "SINE" , 4 )) {
      type = 3 ;
      rest += 4 ;
      nparams = 3 ;
      }
   else if (! strncmp ( rest , "RAMP" , 4 )) {
      type = 4 ;
      rest += 4 ;
      nparams = 2 ;
      }
   else if (! strncmp ( rest , "ARMA" , 4 )) {
      type = 5 ;
      rest += 4 ;
      nparams = 6 ;
      }
   else {
      sprintf ( error , "Illegal generator type (%s)" , rest ) ;
      return 1 ;
      }

   for (i=0 ; i<nparams ; i++)
      params[i] = ParseDouble ( &rest ) ;


   MEMTEXT ( "GENERATE: data" ) ;
   if (type == 5)
      data = (double *) MALLOC ( 2 * ncases * sizeof(double) ) ;
   else 
      data = (double *) MALLOC ( ncases * sizeof(double) ) ;

   if (data == NULL) {
      strcpy ( error , "Insufficient memory to generate signal" ) ;
      return 1 ;
      }

   dptr = data ;

   switch (type) {
      case 1:
         for (i=0 ; i<ncases ; i++)
            *dptr++ = unifrand() * (params[1] - params[0])  +  params[0] ;
         break ;
      case 2:
         for (i=0 ; i<ncases ; i++) {
            if (! (i%2)) {
               normal_pair ( &x , &y ) ;
               *dptr++ = params[1] * x + params[0] ;
               }
            else 
               *dptr++ = params[1] * y + params[0] ;
            }
         break ;
      case 3:
         factor = 2.0 * PI / (double) params[1] ;
         phase = params[2] * PI / 180.0 ;
         for (i=0 ; i<ncases ; i++)
            *dptr++ = params[0] * sin ( i * factor + phase ) ;
         break ;
      case 4:
         factor = params[0] / params[1] ;
         for (i=0 ; i<ncases ; i++)
            *dptr++ = factor * (i % (int) (params[1])) ;
         break ;
      case 5:
         dptr = data + ncases ;
         for (i=0 ; i<ncases ; i++)
            if (! (i%2)) {
               normal_pair ( &x , &y ) ;
               dptr[i] = x ;
               }
            else 
               dptr[i] = y ;
         for (i=0 ; i<ncases ; i++) {
            x = (i >= 1) ?  dptr[i-1]  :  0.0 ;
            y = (i >= 2)  ?  dptr[i-2]  :  0.0 ;
            z = (i >= 10)  ?  dptr[i-10]  :  0.0 ;
            data[i] = dptr[i] + params[3] * x + params[4] * y + params[5] * z ;
            x = (i >= 1) ?  data[i-1]  :  0.0 ;
            y = (i >= 2)  ?  data[i-2]  :  0.0 ;
            z = (i >= 10)  ?  data[i-10]  :  0.0 ;
            data[i] += params[0] * x + params[1] * y + params[2] * z ;
            }
         break ;
      }

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   MEMTEXT ( "GENERATE: signals array" ) ;
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
      strcpy ( error , "Insufficient memory to generate signal" ) ;
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
            MEMTEXT ( "GENERATE: delete duplicate signal" ) ;
            delete ( sptr[j] ) ;        // If so, delete this signal
            break ;                     // And quit looking
            }
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }

      if (ivar) {   // In this case, must allocate for new signal
         MEMTEXT ( "GENERATE: temp signal" ) ;
         temp = (double *) MALLOC ( ncases * sizeof(double) ) ;
         memcpy ( temp , data , ncases * sizeof(double) ) ;
         }
      else
         temp = data ;

      MEMTEXT ( "GENERATE: new Signal" ) ;
      sptr[j] = new Signal ( misc->names->start[i] , ncases , temp ) ;
      if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
         if (sptr[j] != NULL) {
            delete sptr[j] ;
            sptr[j] = NULL ;
            }
         strcpy ( error , "Insufficient memory to generate signal" ) ;
         return 1 ;
         }
      ++ivar ;
      } // For all names

   return 0 ;
}

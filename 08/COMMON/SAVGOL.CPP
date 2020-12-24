/******************************************************************************/
/*                                                                            */
/*  SAVGOL - Apply a Savitzky-Golay filter to a signal (usually a spectrum).  */
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

int savgol ( MiscParams *misc , int halflen , int degree , Signal *sig ,
             int *nsigs , Signal ***signals , char *error )
{
   int i, j, k, ij, limit, iband, n, ivar, nvars ;
   double *data, *temp, *coefs, *x, *soln, band, factor, sum ;
   Signal **sptr ;
   SingularValueDecomp *svdptr ;
   enum SignalType save_type ;


   nvars = misc->names->nreal ;

   if (! nvars) {
      strcpy ( error , "No signal names specified" ) ;
      return 1 ;
      }

   n = sig->n ;
   x = sig->sig ;
   save_type = sig->type ;

   MEMTEXT ( "SAVGOL: new svd, data, work" ) ;
   svdptr = new SingularValueDecomp ( degree+1 , degree+1 , 0 ) ;
   data = (double *) MALLOC ( n * sizeof(double) ) ;
   coefs = (double *) MALLOC ( (degree + halflen + 2) * sizeof(double) ) ;
   if ((data == NULL) || (coefs == NULL) || (svdptr == NULL) || !svdptr->ok){
      if (data != NULL)
         FREE ( data ) ;
      if (coefs != NULL)
         FREE ( coefs ) ;
      if (svdptr != NULL)
         delete svdptr ;
      strcpy ( error , "Insufficient memory to filter signal" ) ;
      return 1 ;
      }

   soln = coefs + halflen + 1 ;

/*
   Compute the A'A matrix.  It is banded, so make use of that fact.
   A'A(i,j) = sum from k = -degree to degree of k ^ (i+j) for i,j=0 to degree.
*/

   for (ij=0 ; ij<=2*degree ; ij++) {      // These are the bands (i+j)
      band = ij  ?  0.0  :  1.0 ;          // 0 to the 0 power is 1 here!
      for (k=1 ; k<=halflen ; k++)         // Sum across filter span
         band += pow ( (double) k , (double) ij ) +
                 pow ( (double) -k , (double) ij ) ;
      limit = 2 * degree - ij ;
      if (ij < limit)
         limit = ij ;
      for (iband=-limit ; iband<=limit ; iband+=2) {
         i = (ij + iband) / 2 ;
         j = ij - i ;
         svdptr->a [ i * (degree+1) + j ] = band ;
         }
      }

/*
   Solve for the coefficients
*/

   svdptr->svdcmp () ;

   svdptr->b[0] = 1.0 ;
   for (i=1 ; i<=degree ; i++)
      svdptr->b[i] = 0.0 ;

   svdptr->backsub ( 1.e-14 , soln ) ;

   for (i=0 ; i<=halflen ; i++) {
      factor = 1.0 ;
      sum = soln[0] ;
      for (j=1 ; j<=degree ; j++) {
         factor *= i ;
         sum += factor * soln[j] ;
         }
      coefs[i] = sum ;
      }

/*
   Apply the filter.
*/

   for (i=0 ; i<n ; i++) {
      sum = coefs[0] * x[i] ;
      for (j=1 ; j<=halflen ; j++) {
         k = i + j ;
         if (k < n)
            sum += coefs[j] * x[k] ;
         k = i - j ;
         if (k >= 0)
            sum += coefs[j] * x[k] ;
         }
      if (sum < 0.0)
         sum = 0.0 ;
      data[i] = sum ;
      }

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   MEMTEXT ( "SAVGOL: signals array" ) ;
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
            MEMTEXT ( "SAVGOL: delete duplicate signal" ) ;
            delete ( sptr[j] ) ;        // If so, delete this signal
            break ;                     // And quit looking
            }
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }

      if (ivar) {   // In this case, must allocate for new signal
         MEMTEXT ( "SAVGOL: temp signal" ) ;
         temp = (double *) MALLOC ( n * sizeof(double) ) ;
         if (temp == NULL) {
            strcpy ( error , "Insufficient memory to create signal" ) ;
            return 1 ;
            }
         memcpy ( temp , data , n * sizeof(double) ) ;
         }
      else
         temp = data ;

      MEMTEXT ( "SAVGOL: new Signal" ) ;
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

   MEMTEXT ( "SAVGOL: coefs, delete svdptr" ) ;
   FREE ( coefs ) ;
   delete svdptr ;

   return 0 ;
}


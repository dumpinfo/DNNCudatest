/******************************************************************************/
/*                                                                            */
/*  AUTOCORR - Generate one or more new signals by auto or cross correlation. */
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

static int pcc ( int n , int lag , double *x , double *y , double *coefs ) ;

int autocorr ( MiscParams *misc , int operation , int ncases ,
               Signal *sig1 , Signal *sig2 , int *nsigs ,
               Signal ***signals , char *error , double **corrs )
{
   int i, j, lag, n, n1, n2, ivar, nvars ;
   double *data, *temp, *dptr1, *dptr2, *work, r ;
   double mean1, mean2, std1, std2, diff, diff2, sum, factor ;
   double *coefs, *prev, *alpha, *beta ;
   char msg[84] ;
   Signal **sptr ;

/*
   Allocate memory for the generated signal.
   If PARTIAL, allocate work memory for recursive reduction.
*/

   MEMTEXT ( "autocorr: data" ) ;
   data = (double *) MALLOC ( ncases * sizeof(double) ) ;
   if ((operation == ID_PRED_P_AUTOCORR) || (operation == ID_PRED_P_CROSSCORR)){
      MEMTEXT ( "autocorr: work" ) ;
      if (operation == ID_PRED_P_AUTOCORR)
         work = (double *) MALLOC ( 2 * (sig1->n + ncases) * sizeof(double) ) ;
      else if (operation == ID_PRED_P_CROSSCORR)
         work = (double *) MALLOC ( (ncases+1) * sizeof(double) ) ;
      if (work == NULL) {
         if (data != NULL) {
            FREE ( data ) ;
            data = NULL ;
            }
         }
      }
   else
      work = NULL ;

   if (data == NULL) {
      if (work != NULL)
         FREE ( work ) ;
      strcpy ( error , "Insufficient memory to create signal" ) ;
      return 1 ;
      }

   make_progress_window ( "Lagged correlation" ) ;

/*
   Compute means and standard deviations
*/

   if ((operation == ID_PRED_AUTOCORR)  ||  (operation == ID_PRED_CROSSCORR)
    || (operation == ID_PRED_P_CROSSCORR)) {

      dptr1 = sig1->sig ;            // Point to first signal
      n1 = sig1->n ;                 // Its length
      mean1 = std1 = 0.0 ;
      for (i=0 ; i<n1 ; i++)         // Compute its mean
         mean1 += dptr1[i] ;
      mean1 /= n1 ;
      for (i=0 ; i<n1 ; i++) {       // And its standard deviation
         diff = dptr1[i] - mean1 ;
         std1 += diff * diff ;
         }
      std1 = sqrt ( std1 / n1 ) ;

      if (operation == ID_PRED_AUTOCORR) {  // If AUTO, both signals same
         dptr2 = dptr1 ;
         n = n2 = n1 ;
         mean2 = mean1 ;
         std2 = std1 ;
         }
      else {                         // Else ditto for second
         dptr2 = sig2->sig ;         // Point to second signal
         n2 = sig2->n ;              // Its length
         n = (n1 < n2) ? n1 : n2 ;   // Grand length is shorter
         mean2 = std2 = 0.0 ;
         for (i=0 ; i<n2 ; i++)      // Compute its mean
            mean2 += dptr2[i] ;
         mean2 /= n2 ;
         for (i=0 ; i<n2 ; i++) {    // And standard deviation
            diff = dptr2[i] - mean2 ;
            std2 += diff * diff ;
            }
         std2 = sqrt ( std2 / n2 ) ;
         }
      }

/*
   The means and standard deviations have been computed (if needed).
   Compute the correlations.
*/

   if ((operation == ID_PRED_AUTOCORR)  ||  (operation == ID_PRED_CROSSCORR)) {
      factor = 1.0 / (std1 * std2 + 1.e-40) ; // Correlation normalizer
      for (lag=1 ; lag<=ncases ; lag++) {
         sum = 0.0 ;
         for (j=lag ; j<n ; j++) {
            diff = dptr1[j] - mean1 ;
            diff2 = dptr2[j-lag] - mean2 ;
            sum += diff * diff2 ;
            }
         r = factor * sum / n ;      // Less bias than n-lag
         if (r > 1.0)                // Safety for display only
            r = 1.0 ;
         if (r < -1.0)
            r = -1.0 ;
         data[lag-1] = r ;
         }
      }

/*
   Compute PARTIAL AUTOCORR
*/

   else if (operation == ID_PRED_P_AUTOCORR) {
      coefs = work ;
      prev = coefs + ncases ;
      alpha = prev + ncases ;
      beta = alpha + sig1->n ;
      n = sig1->n ;  // Also needed later!
      burg ( n , sig1->sig , ncases , data , coefs , prev , alpha , beta);
      }

/*
   Compute partial crosscorrelation
*/

   else if (operation == ID_PRED_P_CROSSCORR) {
      MEMTEXT ( "AUTOCORR: pcc starting" ) ;
      n = (sig1->n < sig2->n) ? sig1->n : sig2->n ; // Grand length is shorter
      for (lag=1 ; lag<=ncases ; lag++) {
         sprintf ( msg , "Lag = %d", lag ) ;
         write_non_progress ( msg ) ;
         if (user_pressed_escape ()) {
            strcpy ( error , "User aborted" ) ;
            FREE ( data ) ;
            FREE ( work ) ;
            destroy_progress_window () ;
            return 1 ;
            }
         if (pcc ( n , lag , sig2->sig , sig1->sig , work )) {
            strcpy ( error , "Insufficient memory to create signal" ) ;
            FREE ( data ) ;
            FREE ( work ) ;
            destroy_progress_window () ;
            return 1 ;
            }
         data[lag-1] = work[lag-1] * std2 / (std1 + 1.e-40) ;
         }
      MEMTEXT ( "AUTOCORR: pcc complete" ) ;
      }

   destroy_progress_window () ;

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   nvars = misc->names->nreal ;
   MEMTEXT ( "AUTOCORR: signals array" ) ;
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
            MEMTEXT ( "AUTOCORR: delete duplicate signal" ) ;
            delete ( sptr[j] ) ;        // If so, delete this signal
            break ;                     // And quit looking
            }
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }

      if (ivar) {   // In this case, must allocate for new signal
         MEMTEXT ( "AUTOCORR: temp signal" ) ;
         temp = (double *) MALLOC ( ncases * sizeof(double) ) ;
         memcpy ( temp , data , ncases * sizeof(double) ) ;
         }
      else
         temp = data ;

      MEMTEXT ( "AUTOCORR: new Signal" ) ;
      sptr[j] = new Signal ( misc->names->start[i] , ncases , temp ) ;
      if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
         if (sptr[j] != NULL) {
            delete sptr[j] ;
            sptr[j] = NULL ;
            }
         strcpy ( error , "Insufficient memory to create signal" ) ;
         return 1 ;
         }
      sptr[j]->type = CorrelationSignal ;
      sptr[j]->source_n = n ;
      ++ivar ;
      } // For all names

   if (work != NULL)
      FREE ( work ) ;

   *corrs = data ;  // This lets caller see the results if interested
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   Local routine to crudely compute partial cross-correlations

--------------------------------------------------------------------------------
*/

static int pcc ( int n , int lag , double *x , double *y , double *coefs )
{
   int i, j ;
   double *aptr, *bptr ;
   SingularValueDecomp *svdptr ;

   svdptr = new SingularValueDecomp ( n-lag , lag+1 , 0 ) ;
   if ((svdptr == NULL)  || ! svdptr->ok) {
      if (svdptr != NULL)
         delete svdptr ;
      return 1 ;
      }

   aptr = svdptr->a ;
   bptr = svdptr->b ;

   for (i=lag ; i<n ; i++) {
      for (j=1 ; j<=lag ; j++)
         *aptr++ = x[i-j] ;
      *aptr++ = 1.0 ;
      *bptr++ = y[i] ;
      }

   svdptr->svdcmp () ;
   svdptr->backsub ( 1.e-8 , coefs ) ;

   delete svdptr ;
   return 0 ;
}

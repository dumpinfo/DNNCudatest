/******************************************************************************/
/*                                                                            */
/*  ARMAPRED - Predict using an ARMA model                                    */
/*                                                                            */
/*  Normally, this returns 0.  It returns -1 if insufficient memory,          */
/*  1 if user pressed ESCape.                                                 */
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

int armapred (
   int npred ,            // Number of cases to generate
   ARMA *arma ,           // Use this ARMA
   MiscParams *misc ,     // Mainly for generated signal names
   int n_ins_outs ,       // Length of next array
   InputOutput **in_out , // Input/output signal list
   int *nsigs ,           // Number of signals currently existing
   Signal ***signals      // This is them
   )
{
   int i, j, k, n, ivar, nouts ;
   int ncases, shocks_known ;
   double **outputs, *outvars ;
   Signal **sptr, *sigptr ;

/*
   Allocate memory for the signals that will be predicted.
   There is a little inefficiency here.  The Signal constructor requires
   a valid malloc'd pointer, while arma->predict requires one contiguous
   array for all signals.  We must allocate both styles!
   Note that we actually compute ncases = npred - maxlag cases.  The
   first maxlag cases cannot be computed.
   Start by seeing if the user is providing known shocks.  If so, or if
   there are MA terms, we must allocate double for shock storage.
*/

   shocks_known = 0 ;
   for (i=0 ; i<n_ins_outs ; i++) {
      if ((! in_out[i]->is_input)  &&  (in_out[i]->shock >= 0)) {
         shocks_known = 1 ;
         break ;
         }
      }

   ncases = npred - arma->maxlag ;    // This many cases predicted
   nouts = misc->names->nreal ;       // This many signals predicted

   MEMTEXT ( "ARMAPRED: outputs, signal arrays" ) ;

   outputs = (double **) MALLOC ( nouts * sizeof(double *) ) ;
   if (arma->nma  ||  shocks_known)
      outvars = (double *) MALLOC ( 2 * npred * arma->nout * sizeof(double) ) ;
   else 
      outvars = (double *) MALLOC ( npred * arma->nout * sizeof(double) ) ;

   if ((outputs == NULL)  ||  (outvars == NULL)) {
      if (outputs != NULL)
         FREE ( outputs ) ;
      if (outvars != NULL)
         FREE ( outvars ) ;
      return -1 ;
      }

   for (i=0 ; i<nouts ; i++) {               // For each predicted signal
      outputs[i] = (double *) MALLOC ( npred * sizeof(double) ) ; // Goes here
      if (outputs[i] == NULL) {
         for (j=0 ; j<i ; j++)
            FREE ( outputs[j] ) ;
         FREE ( outputs ) ;
         FREE ( outvars ) ;
         return -1 ;
         }
      }

/*
   Count how many of these signals have names not already in use.
   Allocate additional memory for their pointers.
*/

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
      sptr = (Signal **) MALLOC ( nouts * sizeof(Signal *) ) ;

   if (sptr == NULL) {
      for (i=0 ; i<nouts ; i++)
         FREE ( outputs[i] ) ;
      FREE ( outputs ) ;
      FREE ( outvars ) ;
      return -1 ;
      }
   *signals = sptr ;

   arma->get_outputs ( n_ins_outs , in_out , *signals , npred ,
                       1 , 1 , (arma->nma || shocks_known) , outvars ) ;

   make_progress_window ( "ARMA prediction" ) ;

   (void) arma->predict ( n_ins_outs , in_out , *signals , ncases ,
                          shocks_known , outvars ) ;

   destroy_progress_window () ;

/*
   The final step is to create a new signal for each output.
   If a predicted output is also a model output, copy the latter to
   the former so that only the points past the original end are changed.
   This facilitates certain charting operations, and makes this compatible
   with network prediction.
*/

   ivar = 0 ;
   for (i=0 ; i<misc->names->n ; i++) { // Check all names
      if (! misc->names->len[i])        // Some may be NULL
         continue ;                     // Obviously skip them
      sigptr = NULL ;                   // Init that this output not also input
      for (j=*nsigs-1 ; j>=0 ; j--) {   // Search existing signals for same name
         if (! strcmp ( misc->names->start[i] , sptr[j]->name )) {  // There?
            for (k=0 ; k<n_ins_outs ; k++) { // Check all ins and outs
               if ((! in_out[k]->is_input)  &&  (in_out[k]->which == j)) {
                  sigptr = sptr[j] ;    // Flag to copy known signal
                  break ;               // to predicted output
                  }
               }
            break ;                     // Quit looking
            }
         }

      memcpy ( outputs[ivar] , outvars + i*npred ,
               npred * sizeof(double) ) ; // Get this prediction

      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         MEMTEXT ( "ARMAPRED: new Signal" ) ;
         sptr[j] = new Signal ( misc->names->start[i] , npred , outputs[ivar] ) ;
         if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
            if (sptr[j] != NULL) {
               delete sptr[j] ;            // This frees the signal
               sptr[j] = NULL ;
               }
            else
               FREE ( outputs[ivar] ) ;    // Manually free it
            for (j=ivar+1 ; j<nouts ; j++) // Free the rest
               FREE ( outputs[j] ) ;
            FREE ( outvars ) ;
            return -1 ;
            }
         }
      else {                            // This output signal already exists
         MEMTEXT ( "ARMAPRED: replace signal" ) ;
         if (sigptr != NULL) {          // Means this outvar is also an output
            n = sigptr->n ;             // So we copy the known output
            if (n > npred)              // To the predicted output,
               n = npred ;              // Leaving the prediction at the end
            sigptr->replace ( npred , n , outputs[ivar] ) ;
            }
         else 
            sptr[j]->replace ( npred , 0 , outputs[ivar] ) ;
         }

      ++ivar ;
      } // For all names


   MEMTEXT ( "ARMAPRED: outputs, outvars" ) ;
   FREE ( outputs ) ;
   FREE ( outvars ) ;

   return 0 ;
}


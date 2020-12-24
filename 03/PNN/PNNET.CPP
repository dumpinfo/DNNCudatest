// Copyright (c) 1994 John Wiley & Sons, Inc.  All rights reserved.
// Reproduction or translation of this work beyond that permitted in
// section 117 of the 1976 United States Copyright Act without the
// express written permission of the copyright owner is unlawful.
// Requests for further information should be addressed to the
// Permission Department, John Wiley & Sons, Inc.  The purchaser may
// make backup copies for his/her own use only and not for distribution
// or resale.  The publisher assumes no responsibility for errors,
// omissions, or damages, caused by the use of these programs or from
// the use of the information contained herein.

/******************************************************************************/
/*                                                                            */
/*  PNNET - All principal routines for PNNet processing                       */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

/*
--------------------------------------------------------------------------------

   Constructor and destructor (for PNNet, the base class)

   Failure (due to insufficient memory) is indicated in two ways.
   The 'ok' flag will be set to zero to indicate to child classes that
   they should abort and to the caller that there was failure.
   Also, 'out' will be set to NULL.  In case this constructor succeeds but
   the child fails, a real possibility, the child will set 'ok' to zero
   but 'out' here will be valid.  That tells this base class destructor to
   free its memory.

--------------------------------------------------------------------------------
*/

PNNet::PNNet (
   int net_kernel , 
   int out_model ,
   int n_inputs ,
   int n_outputs
   )
{
   kernel = net_kernel ;
   outmod = out_model ;
   nin = n_inputs ;
   nout = n_outputs ;
   errtype = 0 ;
   neterr = 1.e30 ;
   confusion = NULL ;

   ok = 0 ;   // Indicates failure of malloc (What a pessimist!)

   MEMTEXT ( "PNNET constructor" ) ;
   if ((out = (double *) MALLOC ( nout * sizeof(double))) == NULL)
      return ;

   if (outmod == OUTMOD_CLASSIFICATION) { // One for each class, plus reject
      if ((confusion = (int *) MALLOC ( (nout+1) * sizeof(int))) == NULL) {
         FREE ( out ) ;
         out = NULL ;
         return ;
         }
      memset ( confusion , 0 , (nout+1) * sizeof(int) ) ;
      }

   ok = 1 ;            // Indicate to caller that all mallocs succeeded
}

PNNet::~PNNet ()
{
   if (out == NULL)  // If constructor's mallocs failed
      return ;       // there is nothing to free

   MEMTEXT ( "PNNET destructor" ) ;
   FREE ( out ) ;
   if (confusion != NULL)
      FREE ( confusion ) ;
}

/*
--------------------------------------------------------------------------------

   trial_error - Compute the mean square error for the entire training set

      This regularly calls user_pressed_escape, and it returns a negative error
      if the user pressed ESCape.

--------------------------------------------------------------------------------
*/

double PNNet::trial_error ( TrainingSet *tptr , int find_deriv )
{
   int i, nsig, tclass, oclass, exclude, first, user_quit ;
   double err, tot_err, temp, *dptr, *exclude_ptr, diff ;

   tot_err = 0.0 ;       // Total error will be cumulated here

   if (find_deriv) {
      nsig = (model == NETMOD_SEPCLASS)  ?  nin*nout : nin ;
      for (i=0 ; i<nsig ; i++) {
         deriv[i] = 0.0 ;   // Will sum here across all cases
         deriv2[i] = 0.0 ;  // Ditto
         }
      }

/*
   We will use cross validation, so "exclude" tells which one we ignore
   and we must reduce the training set size by one.
   "dptr" will permanently point to the last slot in the training set,
   which is where the currently excluded case resides.
*/

   user_quit = 0 ;
   exclude = tptr->ntrain ;
   --tptr->ntrain ;
   dptr = tptr->data + tptr->size * tptr->ntrain ;

   first = 1 ;      // For first trial no swapping needed

   while (exclude--) {   // Exclude each training case

      if ((user_quit = user_pressed_escape ()) != 0)
         break ;

      if (! first) {          // If not first trial, swap excluded to end
         exclude_ptr = tptr->data + tptr->size * exclude ;
         for (i=0 ; i<tptr->size ; i++) {
            temp = exclude_ptr[i] ;
            exclude_ptr[i] = dptr[i] ;
            dptr[i] = temp ;
            }
         }

      first = 0 ;                // Flag that we swap from now on
      err = 0.0 ;                // Will sum this case's error here

      if (outmod == OUTMOD_AUTOASSOCIATION) {        // If this is AUTOASSOCIATIVE
         if (find_deriv)
            trial_deriv ( dptr , tclass , dptr ) ; // tclass ignored
         else 
            trial ( dptr ) ;              // Return value ignored
         for (i=0 ; i<nout ; i++) {       // then the expected outputs
            diff = dptr[i] - out[i] ;     // are just the inputs
            err += diff * diff ;
            } // For all outputs
         } // If OUTMOD_AUTOASSOCIATION

      else if (outmod == OUTMOD_CLASSIFICATION) {   // If this is Classification
         tclass = (int) dptr[tptr->nin] - 1 ; // class is stored after inputs
         if (find_deriv)
            oclass = trial_deriv ( dptr , tclass , dptr ) ; // 2'nd dptr ignored
         else 
            oclass = trial ( dptr ) ;
         for (i=0 ; i<nout ; i++) {
            if (i == tclass) {
               diff = 1.0 - out[i] ;
               err += diff * diff ;
               }
            else
               err += out[i] * out[i] ;
            }
         } // If OUTMOD_CLASSIFICATION

      else if (outmod == OUTMOD_MAPPING) {  // If this is GENERAL output
         if (find_deriv)
            trial_deriv ( dptr , tclass , dptr + tptr->nin ) ; // tclass ignored
         else 
            trial ( dptr ) ;              // Return value ignored
         for (i=0 ; i<nout ; i++) {       // Outputs stored after inputs
            diff = dptr[tptr->nin+i] - out[i] ;
            err += diff * diff ;
            } // For all outputs
         } // If OUTMOD_MAPPING

      tot_err += err ;
      } // for all excluded

   ++tptr->ntrain ;  // Undo shrinking for jackknife

/*
   Find the mean per presentation.  Also, compensate for nout if that was
   not done implicitly in the error computation.
*/

   if (user_quit)
      tot_err = -1.e30 ;

   neterr = tot_err / (double) tptr->ntrain ; // Mean per presentation
   if (find_deriv) {
      for (i=0 ; i<nsig ; i++) {
         deriv[i] /= (double) tptr->ntrain ;     // Ditto for derivative
         deriv2[i] /= (double) tptr->ntrain ;
         }
      }

   if ((outmod == OUTMOD_AUTOASSOCIATION)  ||  (outmod == OUTMOD_MAPPING)) {
      neterr /= (double) nout ;        // Make it per output neuron
      if (find_deriv) {
         for (i=0 ; i<nin ; i++) {
            deriv[i] /= (double) nout ;   // Ditto for derivatives
            deriv2[i] /= (double) nout ;
            }
         }
      }
   return neterr ;
}

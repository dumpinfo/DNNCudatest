/******************************************************************************/
/*                                                                            */
/*  PNNET - All principal routines for PNNet processing                       */
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

PNNet::PNNet ( char *netname , NetParams *net_params )
   : Network ( netname , net_params )
{
   MEMTEXT ( "PNNet constructor" ) ;

   if (! ok)    // Did the parent constructor fail?
      return ;  // If so, nothing to do here

   tdata = NULL ;          // No training data here
   kernel = net_params->kernel ;
}

PNNet::~PNNet ()
{
   MEMTEXT ( "PNNet destructor" ) ;

   if (tdata != NULL) {
      MEMTEXT ( "PNNet destructor deleting tset" ) ;
      delete tdata ;
      }
}

/*
--------------------------------------------------------------------------------

   trial_error - Compute the mean square error for the entire training set

      This regularly calls user_pressed_escape, and it returns a negative error
      if the user pressed ESCape.

      Note that this operates in two different manners, depending on the
      context.  This routine is nearly always called by a training
      algorithm, in which case the training set passed as its parameter
      is the same set used by the network.  With that in mind, this routine
      uses cross validation by swapping each test case to the end, leaving
      the others up front for the classification.  In the (rare) situation
      that the user directly calls this routine with a test set, this
      swapping has the effect of reordering the test set.  This is of no
      consequence (or shouldn't be), but is a waste of a little time.

      Note that when this is called by a training algorithm, the returned
      error will exceed that when it is called with a training set that
      is identical to but independent of the network's set.  This is
      because each test case has a representative in the network set!

--------------------------------------------------------------------------------
*/

double PNNet::trial_error ( TrainingSet *tptr )
{
   return trial_error ( tptr , 0 ) ;
}

double PNNet::trial_error ( TrainingSet *tptr , int find_deriv )
{
   int i, nsig, tclass, exclude, first, user_quit ;
   double err, tot_err, temp, *dptr, *exclude_ptr, diff ;
#if DEBUG
   char msg[256] ;
#endif


   tot_err = 0.0 ;       // Total error will be cumulated here

   if (find_deriv) {
      nsig = (model == NETMOD_SEPCLASS)  ?  n_inputs*n_outputs : n_inputs ;
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

      if (output_mode == OUTMOD_CLASSIFICATION) {  // If this is Classification
         tclass = (int) dptr[tptr->n_inputs] - 1 ; // class is stored after inputs
         if (find_deriv)
            trial_deriv ( dptr , tclass , dptr ) ; // 2'nd dptr ignored
         else 
            trial ( dptr ) ;
         for (i=0 ; i<n_outputs ; i++) {
            if (i == tclass) {
               diff = 1.0 - out[i] ;
               err += diff * diff ;
               }
            else
               err += out[i] * out[i] ;
            }
#if DEBUG
         sprintf ( msg , "exclude=%d  class=%d  out=(%lf %lf)  err=%lf",
            exclude, tclass, out[0], out[1], err) ;
         MEMTEXT ( msg ) ;
#endif
         } // If OUTMOD_CLASSIFICATION

      else if (output_mode == OUTMOD_MAPPING) {  // If this is MAPPING mode
         if (find_deriv)
            trial_deriv ( dptr , tclass , dptr + tptr->n_inputs ) ; // tclass ignored
         else 
            trial ( dptr ) ;              // Return value ignored
         for (i=0 ; i<n_outputs ; i++) {  // Outputs stored after inputs
            diff = dptr[tptr->n_inputs+i] - out[i] ;
            err += diff * diff ;
            } // For all outputs
         } // If OUTMOD_MAPPING

      tot_err += err ;
      } // for all excluded

   ++tptr->ntrain ;  // Undo shrinking for jackknife

/*
   Find the mean per presentation.  Also, compensate for n_outputs if that was
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

   neterr /= (double) n_outputs ;        // Make it per output neuron
   if (find_deriv) {
      for (i=0 ; i<n_inputs ; i++) {
         deriv[i] /= (double) n_outputs ;   // Ditto for derivatives
         deriv2[i] /= (double) n_outputs ;
         }
      }

#if DEBUG
   sprintf ( msg , "Grand err=%lf", neterr ) ;
   MEMTEXT ( msg ) ;
#endif
   return neterr ;
}

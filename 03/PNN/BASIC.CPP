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
/*  BASIC - All principal routines for PNNbasic processing                    */
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

   Constructor

   In case of malloc failure, we set 'ok' to zero so the user knows about it.
   Also, we always leave unallocated pointers set to NULL.  There is no
   hard reason for doing this; calling programs should always know enough not
   to reference them.  However, it is simply good style.  Most compilers are
   much better at producing code that intercepts NULL pointer references than
   just wild pointers.  An ounce of prevention...

--------------------------------------------------------------------------------
*/

PNNbasic::PNNbasic (
   int net_kernel , 
   int out_model ,
   int n_inputs ,
   int n_outputs
   )
   : PNNet ( net_kernel , out_model , n_inputs , n_outputs )
{
   if (! ok)    // Did the parent constructor fail?
      return ;  // If so, nothing to do here

   model = NETMOD_BASIC ;  // Seems silly in C++, but helps sometimes
   tdata = NULL ;          // No training data here

   MEMTEXT ( "PNNbasic constructor" ) ;
   deriv = (double *) MALLOC ( sizeof(double) ) ;
   deriv2 = (double *) MALLOC ( sizeof(double) ) ;
   if ((deriv == NULL)  ||  (deriv2 == NULL)) {
      if (deriv != NULL)
         FREE ( deriv ) ;
      if (deriv2 != NULL)
         FREE ( deriv2 ) ;
      ok = 0 ;
      }
}


/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

PNNbasic::~PNNbasic()
{
   if (! ok)    // If constructor's mallocs failed
      return ;  // there is nothing to free

   MEMTEXT ( "PNNbasic destructor (2)" ) ;
   FREE ( deriv ) ;
   FREE ( deriv2 ) ;

   if ((tdata != NULL)  &&  ! tdata_shared) {
      MEMTEXT ( "PNNbasic destructor deleting tset" ) ;
      delete ( tdata ) ;
      }
}


/*
--------------------------------------------------------------------------------

   privatize_tset () - Copy the currently shared training set into a private
      copy so that the shared copy can be destroyed or modified without
      harming this trained network.
      This returns 1 if insufficient memory, else 0.

--------------------------------------------------------------------------------
*/

int PNNbasic::privatize_tset ()
{
   TrainingSet *shared ;

   if (! tdata_shared)   // If it is already private
      return 0 ;         // Nothing to do

   shared = tdata ;      // This is the shared tset

   MEMTEXT ( "PNNbasic privatize_tset new, copy tset" ) ;
   tdata = new TrainingSet ( outmod , nin , nout ) ; // Make a new tset
   if (tdata == NULL)
      return 1 ;

   *tdata = *shared ;     // Invoke assignment operator to duplicate it

   if (tdata->ntrain) {   // If all went well
      tdata_shared = 0 ;  // The tset is no longer shared
      return 0 ;          // Return normal exit flag
      }

   delete tdata ;         // Went bad, so abort
   return 1 ;
}


/*
--------------------------------------------------------------------------------

   trial - Compute the output for a given input by evaluating the network
           This also returns the subscript of the maximum output.

--------------------------------------------------------------------------------
*/

int PNNbasic::trial ( double *input )
{
   int tset, pop, ivar, ibest ;
   double *dptr, diff, dist, width, best, psum ;

   width = 1.0 / (sigma * sigma) ; // Multiplies Euclidean distances

   for (pop=0 ; pop<nout ; pop++) // For each population
      out[pop] = 0.0 ;            // will sum kernels here
   psum = 0.0 ;                   // Denominator sum if AUTO or GENERAL

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case

      dist = 0.0 ;                          // Will sum distance here
      for (ivar=0 ; ivar<nin ; ivar++) {    // All variables in this case
         diff = input[ivar] - dptr[ivar] ;  // Input minus case
         dist += diff * diff ;              // Cumulate Euclidean distance
         }
      dist *= width ;                       // Divide dist by sigma squared

      if (kernel == KERNEL_GAUSS)           // Apply the kernel function
         dist = exp ( -dist ) ;
      else if (kernel == KERNEL_RECIP)
         dist = 1.0 / ( 1.0 + dist ) ;

      if (dist < 1.e-30)                    // If this case is far from all
         dist = 1.e-30 ;                    // prevent zero density

      if (outmod == OUTMOD_CLASSIFICATION) { // If this is Classification
         pop = (int) dptr[nin] - 1 ;        // class stored after inputs
         out[pop] += dist ;                 // Cumulate this pop's density
         }
      else if (outmod == OUTMOD_AUTOASSOCIATION) { // If this is autoassociative
         for (ivar=0 ; ivar<nin ; ivar++)   // Outputs same as inputs
            out[ivar] += dist * dptr[ivar] ;// Cumulate numerator
         psum += dist ;                     // Cumulate denominator
         }
      else if (outmod == OUTMOD_MAPPING) {  // If this is general mapping
         dptr += tdata->nin ;               // Outputs stored after inputs
         for (ivar=0 ; ivar<nout ; ivar++)  // For every output variable
            out[ivar] += dist * dptr[ivar] ;// Cumulate numerator
         psum += dist ;                     // Cumulate denominator
         }

      } // For all training cases

/*
   If we are in CLASSIFY mode, return the class having highest output.
   Also, deal with class count normalization and prior probabilities.

   For other output models, divide by the density sum to get predicted outputs.
   Note that we limited 'dist' away from zero for each case.  That prevents
   us from dividing by zero now.  More importantly, if the case is far from
   all training samples, it causes us to automatically return the mean
   dependent variable as the prediction, a good practice.
*/

   if (outmod == OUTMOD_CLASSIFICATION) {     // If this is Classification
      psum = 0.0 ;
      for (pop=0 ; pop<nout ; pop++) {
         if (tdata->priors[pop] >=  0.0)
            out[pop] *= tdata->priors[pop] / tdata->nper[pop] ;
         psum += out[pop] ;
         }

      if (psum < 1.e-30)                // If this test case is far from all
         psum = 1.e-30 ;                // prevent division by zero

      for (pop=0 ; pop<nout ; pop++)
         out[pop] /= psum ;

      best = -1.0 ;                     // Keep track of max across pops
      for (pop=0 ; pop<nout ; pop++) {  // For each population
         if (out[pop] > best) {         // find the highest activation
            best = out[pop] ;
            ibest = pop ;
            }
         }
      return ibest ;
      } // If CLASSIFY output mode

   else if (outmod == OUTMOD_AUTOASSOCIATION) {     // If this is autoassociative
      for (ivar=0 ; ivar<nin ; ivar++)   // Outputs same as inputs
         out[ivar] /= psum ;
      }

   else if (outmod == OUTMOD_MAPPING) {  // If this is general mapping
      for (ivar=0 ; ivar<nout ; ivar++)
         out[ivar] /= psum ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   trial_deriv - We do not use derivative method if one sigma

--------------------------------------------------------------------------------
*/

int PNNbasic::trial_deriv ( double *input , int , double * )
{
   int n ;
   n = trial ( input ) ;
   deriv[0] = 0.0 ;
   deriv2[0] = 0.0 ;
   return n ;
}


/*
--------------------------------------------------------------------------------

   learn - Compute the optimal sigma

   This uses a seemingly roundabout and confusing call flow to provide
   good generality of algorithms.  Everything would be simple if we
   embedded the optimization routines "glob_min" and "brentmin"
   in the network class.  These routines could then directly reference
   all of the data that they need.  But this would mean that the routines
   would have to be custom-coded for each different use.  We take the more
   satisfying route of writing general-purpose optimizing routines, and
   passing to them a pointer to the criterion function.  Unfortunately, we
   then need a way for the criterion function to get the class data that
   it needs without having to take the ugly route of passing that data
   through the call list of the general optimizing routine.  We do that
   by defining static variables here:  one for the training set, and one
   for the network.  These statics are set before the optimizing routines
   are called, then they are referenced by the local criterion subroutine.
   This appears confusing, but it is a price worth paying.

--------------------------------------------------------------------------------
*/

static double basic_crit ( double sig ) ; // Local criterion to optimize
static TrainingSet *local_tptr ;  // These two statics pass class data
static PNNbasic *local_netptr ;   // to the local criterion routine

void PNNbasic::learn ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int k ;
   double x1, y1, x2, y2, x3, y3 ;
   char msg[84] ;

   make_progress_window () ;

   if ((tdata != NULL)  &&  ! tdata_shared) {
      MEMTEXT ( "PNNbasic learn deleting tset" ) ;
      delete ( tdata ) ;
      }

   tdata = tptr ;         // Save memory by sharing the training set
   tdata_shared = 1 ;     // Flag that it is shared
   local_netptr = this ;
   local_tptr = tdata ;

   if (errtype) { // If the network is already trained (errtype != 0) use sigma
      k = glob_min ( 0.9 * sigma , 1.1 * sigma , lptr->nsigs , 1 ,
            lptr->quit_err , basic_crit , &x1 , &y1 , &x2 , &y2 , &x3 , &y3 ) ;
      sprintf ( msg , "Global err at %.5lf = %.5lf", x2, y2 ) ;
      normal_message ( msg ) ;
      }
   else {
      k = glob_min ( lptr->siglo , lptr->sighi , lptr->nsigs , 1 ,
            lptr->quit_err , basic_crit , &x1 , &y1 , &x2 , &y2 , &x3 , &y3 ) ;
      sprintf ( msg , "Global err at %.5lf = %.5lf", x2, y2 ) ;
      normal_message ( msg ) ;
      }

   if (! k) { // If global was not interrupted by user ESCape before trio
      y2 = brentmin ( 50 , lptr->quit_err , 1.e-12 , lptr->quit_tol ,
                      basic_crit , &x1 , &x2 , &x3 , y2 ) ;
      sprintf ( msg , "  Refined err at %.5lf = %.5lf", x2, y2 ) ;
      normal_message ( msg ) ;
      }

   sigma = x2 ;
   neterr = fabs ( y2 ) ; // Brentmin returned neg if ESCape
   errtype = 1 ;          // Tell other routines net is trained
   destroy_progress_window () ;
   return ;
}

static double basic_crit ( double sig )
{
   local_netptr->sigma = sig ;
   return local_netptr->trial_error ( local_tptr , 0 ) ;
}

/*
--------------------------------------------------------------------------------

   wt_save - Save network to disk (called from WT_SAVE.CPP)
   wt_restore - Restore network from disk (called from WT_SAVE.CPP)

--------------------------------------------------------------------------------
*/

int PNNbasic::wt_save ( FILE *fp )
{
   fwrite ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fwrite ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   if (outmod == OUTMOD_CLASSIFICATION) {
      fwrite ( tdata->nper , nout * sizeof(unsigned) , 1 , fp ) ;
      fwrite ( tdata->priors , nout * sizeof(double) , 1 , fp ) ;
      }
   fwrite ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fwrite ( &sigma , sizeof(double) , 1 , fp ) ;
   if (ferror ( fp ))
      return 1 ;
   return 0 ;
}

void PNNbasic::wt_restore ( FILE *fp )
{
   MEMTEXT ( "PNNbasic wt_restore new tset" ) ;
   tdata = new TrainingSet ( outmod , nin , nout ) ; // Make a new tset
   if (tdata == NULL) {
      ok = 0 ;
      return ;
      }

   fread ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fread ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   if (outmod == OUTMOD_CLASSIFICATION) {
      fread ( tdata->nper , nout * sizeof(unsigned) , 1 , fp ) ;
      fread ( tdata->priors , nout * sizeof(double) , 1 , fp ) ;
      }

   MEMTEXT ( "PNNbasic wt_restore alloc for data" ) ;
   tdata->data= (double *) MALLOC(tdata->ntrain * tdata->size * sizeof(double));
   if (tdata->data == NULL) {   // If insufficient memory
      ok = 0 ;
      delete tdata ;
      tdata = NULL ;
      memory_message ( "to read network" ) ;
      return ;
      }

   fread ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fread ( &sigma , sizeof(double) , 1 , fp ) ;
   tdata_shared = 0 ;
   if (ferror ( fp ))
      ok = 0 ;
   return ;
}

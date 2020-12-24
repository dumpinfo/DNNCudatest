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
/*  SEPVAR - All principal routines for PNNsevpar processing                  */
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

--------------------------------------------------------------------------------
*/

PNNsepvar::PNNsepvar (
   int net_kernel , 
   int out_model ,
   int n_inputs ,
   int n_outputs
   )
   : PNNet ( net_kernel , out_model , n_inputs , n_outputs )
{
   if (! ok)    // Did the parent constructor fail?
      return ;  // If so, nothing to do here

   model = NETMOD_SEPVAR ; // Seems silly in C++, but helps sometimes
   tdata = NULL ;          // No training data here

   MEMTEXT ( "PNNsepvar constructor" ) ;
   sigma = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   deriv = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   deriv2 = (double *) MALLOC ( n_inputs * sizeof(double) ) ;

   if ((sigma == NULL)  ||  (deriv == NULL)  ||  (deriv2 == NULL)) {
      if (sigma != NULL)
         FREE ( sigma ) ;
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

PNNsepvar::~PNNsepvar()
{
   if (! ok)    // If constructor's mallocs failed
      return ;  // there is nothing to free

   MEMTEXT ( "PNNsepvar destructor (3)" ) ;
   FREE ( sigma ) ;
   FREE ( deriv ) ;
   FREE ( deriv2 ) ;

   if ((tdata != NULL)  &&  ! tdata_shared) {
      MEMTEXT ( "PNNsepvar destructor deleting tset" ) ;
      delete tdata ;
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

int PNNsepvar::privatize_tset ()
{
   TrainingSet *shared ;

   if (! tdata_shared)   // If it is already private
      return 0 ;         // Nothing to do

   shared = tdata ;      // This is the shared tset

   MEMTEXT ( "PNNsepvar privatize_tset new, copy tset" ) ;
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

int PNNsepvar::trial ( double *input )
{
   int tset, pop, ivar, ibest ;
   double *dptr, diff, dist, best, psum ;

   for (pop=0 ; pop<nout ; pop++) // For each population
      out[pop] = 0.0 ;            // will sum kernels here
   psum = 0.0 ;                   // Denominator sum if AUTO or GENERAL

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case

      dist = 0.0 ;                          // Will sum distance here
      for (ivar=0 ; ivar<nin ; ivar++) {    // All variables in this case
         diff = input[ivar] - dptr[ivar] ;  // Input minus case
         diff /= sigma[ivar] ;              // Scale per sigma
         dist += diff * diff ;              // Cumulate Euclidean distance
         }

      if (kernel == KERNEL_GAUSS)           // Apply the kernel function
         dist = exp ( -dist ) ;
      else if (kernel == KERNEL_RECIP)
         dist = 1.0 / ( 1.0 + dist ) ;

      if (dist < 1.e-40)                    // If this case is far from all
         dist = 1.e-40 ;                    // prevent zero density

      if (outmod == OUTMOD_CLASSIFICATION) {      // If this is Classification
         pop = (int) dptr[nin] - 1 ;        // class stored after inputs
         out[pop] += dist ;                 // Cumulate this pop's density
         }
      else if (outmod == OUTMOD_AUTOASSOCIATION) {     // If this is autoassociative
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

      if (psum < 1.e-40)                // If this test case is far from all
         psum = 1.e-40 ;                // prevent division by zero

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

   trial_deriv - Just like 'trial', but also cumulates derivative

   Note that in order to compute the derivative, we need the target
   value.  In CLASSIFICATION mode that will be "tclass", and "target"
   will be ignored.  In other modes it will be "target" and "tclass"
   is ignored.

--------------------------------------------------------------------------------
*/

int PNNsepvar::trial_deriv ( double *input , int tclass , double *target )
{
   int tset, pop, ivar, ibest, outvar ;
   double *dptr, diff, dist, truedist, best, *vptr, *wptr, vtot, wtot ;
   double temp, der1, der2, psum, *vsptr, *wsptr ;

   for (pop=0 ; pop<nout ; pop++) {  // For each population
      out[pop] = 0.0 ;               // Will sum kernels here
      for (ivar=0 ; ivar<nin ; ivar++) {
         v[pop*nin+ivar] = 0.0 ;     // Scratch for derivative stuff
         w[pop*nin+ivar] = 0.0 ;     // Ditto
         }
      }

   psum = 0.0 ;                   // Denominator sum if AUTO or GENERAL

/*
   If we are in CLASSIFY mode, then VSUM and WSUM are the simple sums
   of v and w.  But for general regression they are not simple sums.
   So LEARN allocated an extra vector at the end of v and w where these
   are stored.
*/

   if (outmod != OUTMOD_CLASSIFICATION) {
      vsptr = v + nout * nin ;   // Will cumulate vsum and wsum here
      wsptr = w + nout * nin ;   // at the end of the v and w matrices
      for (ivar=0 ; ivar<nin ; ivar++) { // One for each sigma
         vsptr[ivar] = 0.0 ;
         wsptr[ivar] = 0.0 ;
         }
      }

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case

      dist = 0.0 ;                          // Will sum distance here
      for (ivar=0 ; ivar<nin ; ivar++) {    // All variables in this case
         diff = input[ivar] - dptr[ivar] ;  // Input minus case
         diff /= sigma[ivar] ;              // Scale per sigma
         dsqr[ivar] = diff * diff ;         // Squared weighted distance
         dist += dsqr[ivar] ;               // Cumulate for all vars
         }

      if (kernel == KERNEL_GAUSS)           // Apply the kernel function
         dist = exp ( -dist ) ;
      else if (kernel == KERNEL_RECIP)
         dist = 1.0 / ( 1.0 + dist ) ;

      truedist = dist ;                     // Need this for derivatives
      if (dist < 1.e-40)                    // If this case is far from all
         dist = 1.e-40 ;                    // prevent zero density

      if (outmod == OUTMOD_CLASSIFICATION) { // If this is Classification
         pop = (int) dptr[nin] - 1 ;        // Class stored after inputs
         out[pop] += dist ;                 // Cumulate this class' density
         vptr = v + pop * nin ;             // Point to this row in v
         wptr = w + pop * nin ;             // And w
         for (ivar=0 ; ivar<nin ; ivar++) { // All variables in this case
            temp = truedist * dsqr[ivar] ;
            vptr[ivar] += temp ;
            wptr[ivar] += temp * (2.0 * dsqr[ivar] - 3.0) ;
            }
         } // If outmod=CLASSIFY

      else if (outmod == OUTMOD_AUTOASSOCIATION) {     // If this is autoassociative
         for (ivar=0 ; ivar<nin ; ivar++) { // Outputs same as inputs
            out[ivar] += dist * dptr[ivar] ;// Cumulate gen reg numerator
            temp = truedist * dsqr[ivar] ;  // This is for vsum and wsum
            vsptr[ivar] += temp ;           // Cumulate these sums
            wsptr[ivar] += temp * (2.0 * dsqr[ivar] - 3.0) ;
            }
         vptr = v ;                         // Cumulate v and w
         wptr = w ;
         for (outvar=0 ; outvar<nout ; outvar++) {
            for (ivar=0 ; ivar<nin ; ivar++) {
               temp = truedist * dsqr[ivar] * dptr[outvar] ;
               *vptr++ += temp ;
               *wptr++ += temp * (2.0 * dsqr[ivar] - 3.0) ;
               }
            }
         psum += dist ;                     // Cumulate denominator
         }
      else if (outmod == OUTMOD_MAPPING) {  // If this is general mapping
         dptr += tdata->nin ;               // Outputs stored after inputs
         for (ivar=0 ; ivar<nout ; ivar++)  // Cumulate gen reg numerator
            out[ivar] += dist * dptr[ivar] ;
         vptr = v ;                         // Cumulate v and w
         wptr = w ;
         for (outvar=0 ; outvar<nout ; outvar++) {
            for (ivar=0 ; ivar<nin ; ivar++) {
               temp = truedist * dsqr[ivar] * dptr[outvar] ;
               *vptr++ += temp ;
               *wptr++ += temp * (2.0 * dsqr[ivar] - 3.0) ;
               }
            }
         for (ivar=0 ; ivar<nin ; ivar++) { // Cumulate vsum and wsum
            temp = truedist * dsqr[ivar] ;
            vsptr[ivar] += temp ;
            wsptr[ivar] += temp * (2.0 * dsqr[ivar] - 3.0) ;
            }
         psum += dist ;                     // Cumulate denominator
         }
      } // For all training cases

/*
   If we are in CLASSIFY mode, deal with class count normalization and
   prior probabilities.

   For other output models, divide by the density sum to get predicted outputs.
   Note that we limited 'dist' away from zero for each case.  That prevents
   us from dividing by zero now.  More importantly, if the case is far from
   all training samples, it causes us to automatically return the mean
   dependent variable as the prediction, a good practice.
*/

   if (outmod == OUTMOD_CLASSIFICATION) {
      psum = 0.0 ;
      for (pop=0 ; pop<nout ; pop++) {
         if (tdata->priors[pop] >=  0.0)
            out[pop] *= tdata->priors[pop] / tdata->nper[pop] ;
         psum += out[pop] ;
         }
   
      if (psum < 1.e-40)    // Even though we kept dist away from 0 above
         psum = 1.e-40 ;    // Pathological priors can still cause problems
      } // If CLASSIFY

   for (pop=0 ; pop<nout ; pop++)
      out[pop] /= psum ;

/*
   Compute the derivatives.  If this is CLASSIFY mode we must also
   worry about priors.  Also recall that in CLASSIFY mode vsum and wsum
   are the simple sums of v and w.  Do that summing here.  But in other
   modes they were computed as we passed through the training set.  All
   we need to do now is complete their compution per common factors.
*/

   for (ivar=0 ; ivar<nin ; ivar++) {

      if (outmod == OUTMOD_CLASSIFICATION)  // If this is Classification
         vtot = wtot = 0.0 ;          // We will be summing these
      else {                          // Otherwise they were already computed
         vtot = vsptr[ivar] * 2.0 / (psum * sigma[ivar]) ;
         wtot = wsptr[ivar] * 2.0 / (psum * sigma[ivar] * sigma[ivar]) ;
         }

      for (outvar=0 ; outvar<nout ; outvar++) {  // Cumulate vsum and wsum
         if ((outmod == OUTMOD_CLASSIFICATION)  &&
             (tdata->priors[outvar] >=  0.0)) {
            v[outvar*nin+ivar] *= tdata->priors[outvar] / tdata->nper[outvar] ;
            w[outvar*nin+ivar] *= tdata->priors[outvar] / tdata->nper[outvar] ;
            }
         v[outvar*nin+ivar] *= 2.0 / (psum * sigma[ivar]) ; // Common factors
         w[outvar*nin+ivar] *= 2.0 / (psum * sigma[ivar] * sigma[ivar]) ;
         if (outmod == OUTMOD_CLASSIFICATION) {      // If this is Classification
            vtot += v[outvar*nin+ivar] ;       // These are simple sums
            wtot += w[outvar*nin+ivar] ;       // Otherwise in vsptr and wsptr
            }
         }

      for (outvar=0 ; outvar<nout ; outvar++) { // Derivatives computed here
         der1 = v[outvar*nin+ivar] - out[outvar] * vtot ;
         der2 = w[outvar*nin+ivar] + 2.0 * out[outvar] * vtot * vtot -
                2.0 * v[outvar*nin+ivar] * vtot - out[outvar] * wtot ;
         if (outmod == OUTMOD_CLASSIFICATION) {      // If this is Classification
            if (outvar == tclass)
               temp = 2.0 * (out[outvar] - 1.0) ;
            else 
               temp = 2.0 * out[outvar] ;
            }
         else
            temp = 2.0 * (out[outvar] - target[outvar]) ;
         deriv[ivar] += temp * der1 ;
         deriv2[ivar] += temp * der2  +  2.0 * der1 * der1 ;
         }
      }

/*
   If we are in CLASSIFY mode, return the class having highest output.
*/

   if (outmod == OUTMOD_CLASSIFICATION) {     // If this is Classification
      best = -1.0 ;                     // Keep track of max across pops
      for (pop=0 ; pop<nout ; pop++) {  // For each population
         if (out[pop] > best) {         // find the highest activation
            best = out[pop] ;
            ibest = pop ;
            }
         }
      return ibest ;
      } // If CLASSIFY output mode

   return 0 ;  // Return value ignored in modes other than CLASSIFY
}


/*
--------------------------------------------------------------------------------

   learn - Compute the optimal sigma

   This uses a seemingly roundabout and confusing call flow to provide
   good generality of algorithms.  Everything would be simple if we
   embedded the optimization routines "glob_min" and "dermin"
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

static double sepvar_crit0 ( double sig ) ;   // Local criterion to optimize
static double sepvar_crit1 ( double *sigs , double *der1 , double *der2 ,
                             int der ) ;
static TrainingSet *local_tptr ;   // These two statics pass class data
static PNNsepvar *local_netptr ;   // to the local criterion routine

void PNNsepvar::learn ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int ivar, k ;
   double x1, y1, x2, y2, x3, y3 ;
   double *x, *base, *direc, *g, *h, *dwk2 ;
   char msg[84] ;

/*
   Allocate work areas.  Some of these (v, w, dsqr) are private class members
   to simplify use across subroutines.  If we are not in CLASSIFY mode then
   allocate one extra vector at the end of v and w to store the general
   alternatives to vsum and wsum (which are no longer just sums).
*/

   if (outmod == OUTMOD_CLASSIFICATION) // If this is Classification
      k = nout ;                  // We just need nin by nout
   else                           // But in general case
      k = nout + 1 ;              // Keep VSUM and WSUM at end of matrix

   MEMTEXT ( "PNNsepvar::learn works (8)" ) ;
   dsqr = (double *) MALLOC ( nin * sizeof(double) ) ;
   v = (double *) MALLOC ( nin * k * sizeof(double) ) ;
   w = (double *) MALLOC ( nin * k * sizeof(double) ) ;
   x = (double *) MALLOC ( nin * sizeof(double) ) ;
   base = (double *) MALLOC ( nin * sizeof(double) ) ;
   direc = (double *) MALLOC ( nin * sizeof(double) ) ;
   g = (double *) MALLOC ( nin * sizeof(double) ) ;
   h = (double *) MALLOC ( nin * sizeof(double) ) ;
   dwk2 = (double *) MALLOC ( nin * sizeof(double) ) ;

   if ((dsqr == NULL)  ||  (v == NULL)  ||  (w == NULL)  ||  (x == NULL)  ||
       (base == NULL)  ||  (direc == NULL) || (dwk2 == NULL) ||
       (g == NULL) || (h == NULL)) {
      if (dsqr != NULL)
         FREE ( dsqr ) ;
      if (v != NULL)
         FREE ( v ) ;
      if (w != NULL)
         FREE ( w ) ;
      if (x != NULL)
         FREE ( x ) ;
      if (base != NULL)
         FREE ( base ) ;
      if (direc != NULL)
         FREE ( direc ) ;
      if (g != NULL)
         FREE ( g ) ;
      if (h != NULL)
         FREE ( h ) ;
      if (dwk2 != NULL)
         FREE ( dwk2 ) ;
      printf ( "\nInsufficient memory to learn" ) ;
      return ;
      }

/*
   If there is a local training set copy it may no longer be valid.
   Take no chances.  Delete it and share with the current set.
*/

   if ((tdata != NULL)  &&  ! tdata_shared) {
      MEMTEXT ( "PNNsepvar learn deleting tset" ) ;
      delete ( tdata ) ;
      }

   tdata = tptr ;         // Save memory by sharing the training set
   tdata_shared = 1 ;     // Flag that it is shared

   make_progress_window () ;

   local_netptr = this ;  // Passes this information
   local_tptr = tdata ;   // To criterion routines

/*
   Compute a crude global minimum using a single sigma, common to all variables.
   Glob_min will return this somewhat optimal sigma in "x2".
   Its function return value is nonzero iff the user pressed ESCape.
*/

   if (errtype) { // If the network is already trained (errtype != 0)
      k = 0 ;     // Then skip global initialization
      for (ivar=0 ; ivar<nin ; ivar++)  // retrieve sigma 
         x[ivar] = sigma[ivar] ;
      y2 = 1.e30 ;
      }
   else {
      k = glob_min ( lptr->siglo , lptr->sighi , lptr->nsigs , 1 ,
           lptr->quit_err , sepvar_crit0 , &x1 , &y1 , &x2 , &y2 , &x3 , &y3 ) ;
      sprintf ( msg , "Global err at %.6lf = %.6lf", x2, y2 ) ;
      normal_message ( msg ) ;
      for (ivar=0 ; ivar<nin ; ivar++)  // Set x for derivmin
         x[ivar] = x2 ;   // Do outside next 'if' so sigma gets it if user quit
      }

   if (k) { // If global was interrupted by user ESCape before trio
      y2 = 1.e30 ;
      normal_message ( "Learning aborted by user" ) ;
      }
   else {
      y2 = dermin ( 32767 , lptr->quit_err , 1.e-8 , lptr->quit_tol ,
            sepvar_crit1 , nin , x , y2 , base , direc , g , h , dwk2 ) ;
      sprintf ( msg , "Refined err = %.6lf", fabs ( y2 ) ) ;
      normal_message ( msg ) ;
      }

/*
   That's it.  Copy the optimal values to sigma and do other cleanup.
*/

   for (ivar=0 ; ivar<nin ; ivar++)  // Set sigma 
      sigma[ivar] = x[ivar] ;

   neterr = fabs ( y2 ) ; // Derivmin returned neg if ESCape
   errtype = 1 ;          // Tell other routines net is trained

   for (ivar=0 ; ivar<nin ; ivar++) {
      sprintf ( msg , "%2d: %.4le", ivar, sigma[ivar] ) ;
      write_progress ( msg ) ;
      }

   destroy_progress_window () ;

   MEMTEXT ( "PNNsepvar::learn works (9)" ) ;
   FREE ( dsqr ) ;
   FREE ( v ) ;
   FREE ( w ) ;
   FREE ( x ) ;
   FREE ( base ) ;
   FREE ( direc ) ;
   FREE ( g ) ;
   FREE ( h ) ;
   FREE ( dwk2 ) ;
   return ;
}

static double sepvar_crit0 ( double sig )
{
   int ivar ;

   for (ivar=0 ; ivar<local_netptr->nin ; ivar++)
      (local_netptr->sigma)[ivar] = sig ;

   return local_netptr->trial_error ( local_tptr , 0 ) ;
}

static double sepvar_crit1 ( double *x , double *der1 , double *der2 , int der )
{
   int ivar ;
   double err ;

   for (ivar=0 ; ivar<local_netptr->nin ; ivar++)
      (local_netptr->sigma)[ivar] = x[ivar] ;

   if (! der)
      return local_netptr->trial_error ( local_tptr , 0 ) ;

   err = local_netptr->trial_error ( local_tptr , 1 ) ;

   for (ivar=0 ; ivar<local_netptr->nin ; ivar++) {
      der1[ivar] = local_netptr->deriv[ivar] ;
      der2[ivar] = local_netptr->deriv2[ivar] ;
      }

   return err ;
}

/*
--------------------------------------------------------------------------------

   wt_save - Save network to disk (called from WT_SAVE.CPP)
   wt_restore - Restore network from disk (called from WT_SAVE.CPP)

--------------------------------------------------------------------------------
*/

int PNNsepvar::wt_save ( FILE *fp )
{
   fwrite ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fwrite ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   if (outmod == OUTMOD_CLASSIFICATION) {
      fwrite ( tdata->nper , nout * sizeof(unsigned) , 1 , fp ) ;
      fwrite ( tdata->priors , nout * sizeof(double) , 1 , fp ) ;
      }
   fwrite ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fwrite ( sigma , sizeof(double) , nin , fp ) ;
   if (ferror ( fp ))
      return 1 ;
   return 0 ;
}

void PNNsepvar::wt_restore ( FILE *fp )
{

   MEMTEXT ( "PNNsepvar wt_restore new tset" ) ;
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

   MEMTEXT ( "PNNsepvar wt_restore alloc for data" ) ;
   tdata->data= (double *) MALLOC(tdata->ntrain * tdata->size * sizeof(double));
   if (tdata->data == NULL) {   // If insufficient memory
      ok = 0 ;
      delete tdata ;
      tdata = NULL ;
      memory_message ( "to read network" ) ;
      return ;
      }

   fread ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fread ( sigma , sizeof(double) , nin , fp ) ;
   tdata_shared = 0 ;
   if (ferror ( fp ))
      ok = 0 ;
   return ;
}


/******************************************************************************/
/*                                                                            */
/*  SEPVAR - All principal routines for PNNsevpar processing                  */
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

#define DEBUG_DERIV 0
#define FULL_DEBUG 0

#define EPS1 1.e-180
#define EPS2 1.e-190

/*
   This is to avoid Borland's silly overflow exception trapping
*/

static double max_exp = log ( 1.e190 ) ;
inline double safe_exp ( double x )
{
   if (x <= max_exp)
      return exp ( x ) ;
   return 1.e190 ;
}

/*
--------------------------------------------------------------------------------

   Constructor

   In case of malloc failure, we set 'ok' to zero so the user knows about it.

--------------------------------------------------------------------------------
*/

PNNsepvar::PNNsepvar ( char *netname , NetParams *net_params )
   : PNNet ( netname , net_params )
{
   if (! ok)    // Did the parent constructor fail?
      return ;  // If so, nothing to do here

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

   for (pop=0 ; pop<n_outputs ; pop++) // For each population
      out[pop] = 0.0 ;            // will sum kernels here
   psum = 0.0 ;                   // Denominator sum if AUTO or GENERAL

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case

      dist = 0.0 ;                          // Will sum distance here
      for (ivar=0 ; ivar<n_inputs ; ivar++) {    // All variables in this case
         diff = input[ivar] - dptr[ivar] ;  // Input minus case
         diff /= sigma[ivar] ;              // Scale per sigma
         dist += diff * diff ;              // Cumulate Euclidean distance
         }

      dist = safe_exp ( -dist ) ;           // Apply the kernel function

      if (dist < EPS1)                      // If this case is far from all
         dist = EPS1 ;                      // prevent zero density

      if (output_mode == OUTMOD_CLASSIFICATION) { // If this is Classification
         pop = (int) dptr[n_inputs] - 1 ;        // class stored after inputs
         out[pop] += dist ;                 // Cumulate this pop's density
         }
      else if (output_mode == OUTMOD_MAPPING) {  // If this is general mapping
         dptr += tdata->n_inputs ;               // Outputs stored after inputs
         for (ivar=0 ; ivar<n_outputs ; ivar++)  // For every output variable
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

   if (output_mode == OUTMOD_CLASSIFICATION) {     // If this is Classification
      psum = 0.0 ;
      for (pop=0 ; pop<n_outputs ; pop++) {
         if (tdata->priors[pop] >=  0.0)
            out[pop] *= tdata->priors[pop] / tdata->nper[pop] ;
         psum += out[pop] ;
         }

      if (psum < EPS2)                  // If this test case is far from all
         psum = EPS2 ;                  // prevent division by zero

      for (pop=0 ; pop<n_outputs ; pop++)
         out[pop] /= psum ;

      best = -1.0 ;                     // Keep track of max across pops
      for (pop=0 ; pop<n_outputs ; pop++) {  // For each population
         if (out[pop] > best) {         // find the highest activation
            best = out[pop] ;
            ibest = pop ;
            }
         }
      return ibest ;
      } // If CLASSIFY output mode

   else if (output_mode == OUTMOD_MAPPING) {  // If this is general mapping
      for (ivar=0 ; ivar<n_outputs ; ivar++)
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

   for (pop=0 ; pop<n_outputs ; pop++) {  // For each population
      out[pop] = 0.0 ;               // Will sum kernels here
      for (ivar=0 ; ivar<n_inputs ; ivar++) {
         v[pop*n_inputs+ivar] = 0.0 ;     // Scratch for derivative stuff
         w[pop*n_inputs+ivar] = 0.0 ;     // Ditto
         }
      }

   psum = 0.0 ;                   // Denominator sum if AUTO or GENERAL

/*
   If we are in CLASSIFY mode, then VSUM and WSUM are the simple sums
   of v and w.  But for general regression they are not simple sums.
   So LEARN allocated an extra vector at the end of v and w where these
   are stored.
*/

   if (output_mode != OUTMOD_CLASSIFICATION) {
      vsptr = v + n_outputs * n_inputs ;   // Will cumulate vsum and wsum here
      wsptr = w + n_outputs * n_inputs ;   // at the end of the v and w matrices
      for (ivar=0 ; ivar<n_inputs ; ivar++) { // One for each sigma
         vsptr[ivar] = 0.0 ;
         wsptr[ivar] = 0.0 ;
         }
      }

#if FULL_DEBUG
   printf ( "\nCalling trial_deriv input %lf %lf  Target %lf %lf",
      input[0], input[1], target[0], target[1] ) ;
#endif

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case

      dist = 0.0 ;                          // Will sum distance here
      for (ivar=0 ; ivar<n_inputs ; ivar++) {    // All variables in this case
         diff = input[ivar] - dptr[ivar] ;  // Input minus case
         diff /= sigma[ivar] ;              // Scale per sigma
         dsqr[ivar] = diff * diff ;         // Squared weighted distance
         dist += dsqr[ivar] ;               // Cumulate for all vars
         }

      dist = safe_exp ( -dist ) ;           // Apply the kernel function

#if FULL_DEBUG
      printf ( "\n  %d: dist=%le  dsqr=%lf %lf", tset, dist, dsqr[0], dsqr[1] ) ;
#endif

      truedist = dist ;                     // Need this for derivatives
      if (dist < EPS1)                      // If this test case is far from all
         dist = EPS1 ;                      // prevent division by zero

      if (output_mode == OUTMOD_CLASSIFICATION) { // If this is Classification
         pop = (int) dptr[n_inputs] - 1 ;        // Class stored after inputs
         out[pop] += dist ;                 // Cumulate this class' density
         vptr = v + pop * n_inputs ;             // Point to this row in v
         wptr = w + pop * n_inputs ;             // And w
         for (ivar=0 ; ivar<n_inputs ; ivar++) { // All variables in this case
            temp = truedist * dsqr[ivar] ;
            vptr[ivar] += temp ;
            wptr[ivar] += temp * (2.0 * dsqr[ivar] - 3.0) ;
            }
         } // If output_mode=CLASSIFY

      else if (output_mode == OUTMOD_MAPPING) {  // If this is general mapping
         dptr += tdata->n_inputs ;               // Outputs stored after inputs
         for (ivar=0 ; ivar<n_outputs ; ivar++)  // Cumulate gen reg numerator
            out[ivar] += dist * dptr[ivar] ;
         vptr = v ;                         // Cumulate v and w
         wptr = w ;
         for (outvar=0 ; outvar<n_outputs ; outvar++) {
            for (ivar=0 ; ivar<n_inputs ; ivar++) {
               temp = truedist * dsqr[ivar] * dptr[outvar] ;
               *vptr++ += temp ;
               *wptr++ += temp * (2.0 * dsqr[ivar] - 3.0) ;
               }
            }
         for (ivar=0 ; ivar<n_inputs ; ivar++) { // Cumulate vsum and wsum
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

   if (output_mode == OUTMOD_CLASSIFICATION) {
      psum = 0.0 ;
      for (pop=0 ; pop<n_outputs ; pop++) {
         if (tdata->priors[pop] >=  0.0)
            out[pop] *= tdata->priors[pop] / tdata->nper[pop] ;
         psum += out[pop] ;
         }
   
      if (psum < EPS2)    // Even though we kept dist away from 0 above
         psum = EPS2 ;    // Pathological priors can still cause problems
      } // If CLASSIFY

   for (pop=0 ; pop<n_outputs ; pop++)
      out[pop] /= psum ;

/*
   Compute the derivatives.  If this is CLASSIFY mode we must also
   worry about priors.  Also recall that in CLASSIFY mode vsum and wsum
   are the simple sums of v and w.  Do that summing here.  But in other
   modes they were computed as we passed through the training set.  All
   we need to do now is complete their compution per common factors.
*/

   for (ivar=0 ; ivar<n_inputs ; ivar++) {

      if (output_mode == OUTMOD_CLASSIFICATION)  // If this is Classification
         vtot = wtot = 0.0 ;          // We will be summing these
      else {                          // Otherwise they were already computed
         vtot = vsptr[ivar] * 2.0 / (psum * sigma[ivar]) ;
         wtot = wsptr[ivar] * 2.0 / (psum * sigma[ivar] * sigma[ivar]) ;
         }

      for (outvar=0 ; outvar<n_outputs ; outvar++) {  // Cumulate vsum and wsum
         if ((output_mode == OUTMOD_CLASSIFICATION)  &&
             (tdata->priors[outvar] >=  0.0)) {
            v[outvar*n_inputs+ivar] *= tdata->priors[outvar] / tdata->nper[outvar] ;
            w[outvar*n_inputs+ivar] *= tdata->priors[outvar] / tdata->nper[outvar] ;
            }
         v[outvar*n_inputs+ivar] *= 2.0 / (psum * sigma[ivar]) ; // Common factors
         w[outvar*n_inputs+ivar] *= 2.0 / (psum * sigma[ivar] * sigma[ivar]) ;
         if (output_mode == OUTMOD_CLASSIFICATION) {      // If this is Classification
            vtot += v[outvar*n_inputs+ivar] ;       // These are simple sums
            wtot += w[outvar*n_inputs+ivar] ;       // Otherwise in vsptr and wsptr
            }
         }

#if FULL_DEBUG
      printf ( "\n  Var %d  vtot=%lf  wtot=%lf", ivar, vtot, wtot ) ;
#endif

      for (outvar=0 ; outvar<n_outputs ; outvar++) { // Derivatives computed here
         der1 = v[outvar*n_inputs+ivar] - out[outvar] * vtot ;
         der2 = w[outvar*n_inputs+ivar] + 2.0 * out[outvar] * vtot * vtot -
                2.0 * v[outvar*n_inputs+ivar] * vtot - out[outvar] * wtot ;
         if (output_mode == OUTMOD_CLASSIFICATION) {      // If this is Classification
            if (outvar == tclass)
               temp = 2.0 * (out[outvar] - 1.0) ;
            else 
               temp = 2.0 * out[outvar] ;
            }
         else
            temp = 2.0 * (out[outvar] - target[outvar]) ;
#if FULL_DEBUG
         printf ( "\n  temp=%lf  der1=%lf  der2=%lf", temp, der1, der2 ) ;
#endif
         deriv[ivar] += temp * der1 ;
         deriv2[ivar] += temp * der2  +  2.0 * der1 * der1 ;
         }
      }

/*
   If we are in CLASSIFY mode, return the class having highest output.
*/

   if (output_mode == OUTMOD_CLASSIFICATION) {     // If this is Classification
      best = -1.0 ;                     // Keep track of max across pops
      for (pop=0 ; pop<n_outputs ; pop++) {  // For each population
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

   This normally returns 0.  It returns 1 if user pressed ESCape (before or
   after some weights were found) and -1 if insufficient memory.

--------------------------------------------------------------------------------
*/

static double sepvar_crit0 ( double sig ) ;   // Local criterion to optimize
static double sepvar_crit1 ( double *sigs , int der , double *der1 ,
                             double *der2 ) ;
static double sepvar_crit2 ( double sig ) ;   // Local criterion to optimize
static int local_ivar ;
static TrainingSet *local_tptr ;   // These two statics pass class data
static PNNsepvar *local_netptr ;   // to the local criterion routine

int PNNsepvar::learn ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int ivar, k ;
   double x1, y1, x2, y2, x3, y3, accuracy ;
   double *x, *base, *direc, *g, *h, *dwk2, best ;
   char msg[84] ;

   memcpy ( lags , tptr->lags , n_inputs*sizeof(unsigned) ) ;
   if (output_mode == OUTMOD_MAPPING)
      memcpy ( leads , tptr->leads , n_outputs*sizeof(unsigned) ) ;

/*
   Allocate work areas.  Some of these (v, w, dsqr) are private class members
   to simplify use across subroutines.  If we are not in CLASSIFY mode then
   allocate one extra vector at the end of v and w to store the general
   alternatives to vsum and wsum (which are no longer just sums).
*/

   if (output_mode == OUTMOD_CLASSIFICATION) // If this is Classification
      k = n_outputs ;                  // We just need n_inputs by n_outputs
   else                           // But in general case
      k = n_outputs + 1 ;              // Keep VSUM and WSUM at end of matrix

   MEMTEXT ( "PNNsepvar::learn works (8)" ) ;
   dsqr = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   v = (double *) MALLOC ( n_inputs * k * sizeof(double) ) ;
   w = (double *) MALLOC ( n_inputs * k * sizeof(double) ) ;
   x = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   base = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   direc = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   g = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   h = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   dwk2 = (double *) MALLOC ( n_inputs * sizeof(double) ) ;

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
      return -1 ;
      }

   if (tdata != NULL) {
      MEMTEXT ( "SEPVAR learn deleting tset" ) ;
      delete ( tdata ) ;
      tdata = NULL ;
      }

   MEMTEXT ( "SEPVAR new, copy tset" ) ;
   tdata = new TrainingSet ( output_mode , n_inputs , n_outputs , 0 , NULL ) ;
   if (tdata == NULL)
      return -1 ;

   *tdata = *tptr ;       // Invoke assignment operator to duplicate it

   if (! tdata->ntrain) { // Insufficient memory to copy?
      delete tdata ;
      tdata = NULL ;
      return -1 ;         // Return error flag
      }

   local_netptr = this ;  // Passes this information
   local_tptr = tdata ;   // To criterion routines

   make_progress_window ( "PNN (SEPVAR) learning" ) ;

/*
   Compute a crude global minimum using a single sigma, common to all variables.
   Glob_min will return this somewhat optimal sigma in "x2".
   Its function return value is nonzero iff the user pressed ESCape.
*/

   if (errtype) { // If the network is already trained (errtype != 0)
      k = 0 ;     // Then skip global initialization
      for (ivar=0 ; ivar<n_inputs ; ivar++)  // retrieve sigma 
         x[ivar] = log ( sigma[ivar] ) ;
      y2 = 1.e30 ; // This must be huge (not neterr) in case different tset
      x2 = x[0] ;  // This is a flag to avoid multivariate min if sigma huge
      }
   else {
      k = glob_min ( log(lptr->siglo) , log(lptr->sighi) , lptr->nsigs , 0 ,
           lptr->quit_err , sepvar_crit0 , &x1 , &y1 , &x2 , &y2 ,
           &x3 , &y3 , lptr->progress ) ;
      if (k) {
         strcpy ( msg , "Interrupted by user" ) ;
         x2 = log(lptr->siglo) ;
         }
      else 
         sprintf ( msg , "Global err at %.6lf = %.6lf", x2, y2 ) ;
      if (lptr->progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;
      for (ivar=0 ; ivar<n_inputs ; ivar++)  // Set x for derivmin
         x[ivar] = x2 ;   // Do outside next 'if' so sigma gets it if user quit

      if (fabs(x2) > 15.0) {  // If univariate init failed, try single
         sprintf ( msg , "Switching to stepwise..." ) ;
         if (lptr->progress)
            write_progress ( msg ) ;
         else 
            write_non_progress ( msg ) ;
         for (ivar=0 ; ivar<n_inputs ; ivar++)
            x[ivar] = 15.0 ;
         for (local_ivar=0 ; local_ivar<n_inputs ; local_ivar++) {
            best = y2 ;
            k = glob_min ( log(lptr->siglo) , log(lptr->sighi) , lptr->nsigs, 0,
                 lptr->quit_err , sepvar_crit2 , &x1 , &y1 , &x2 , &y2 ,
                 &x3 , &y3 , lptr->progress ) ;
            if (k) {
               strcpy ( msg , "Interrupted by user" ) ;
               break ;
               }
            if ((best < y2)  ||  (x2 > 15.0)) {
               x2 = 15.0 ;
               y2 = best ;
               }
            x[local_ivar] = x2 ;
            sprintf ( msg , "Variable %d err at %.6lf = %.6lf",
                      local_ivar+1, x2, y2 ) ;
            if (lptr->progress)
               write_progress ( msg ) ;
            else 
               write_non_progress ( msg ) ;
            }
         for (ivar=0 ; ivar<n_inputs ; ivar++) {      // Make x2 be smallest for
            if ((ivar == 0)  ||  (x[ivar] < x2)) // dermin refinement check
               x2 = x[ivar] ;
            }
         }
      }

   if (k) { // If global was interrupted by user ESCape before trio
      y2 = -1.e30 ;
      write_progress ( "Learning aborted by user" ) ;
      }
   else if (fabs(x2) > 15.1 ) {  // Safety: avoid stupid refinement
      if (errtype) {             // If already trained to huge sigma
         for (ivar=0 ; ivar<n_inputs ; ivar++)
            sigma[ivar] = safe_exp ( x2 ) ;
         y2 = trial_error ( tptr , 0 ) ; // May be new tset, so neterr invalid
         }
      else 
         errtype = 1 ;          // Tell other routines net is trained
      }
   else {
      accuracy = pow ( 10.0 , -lptr->acc - lptr->refine ) ;
      y2 = dermin ( 32767 , lptr->quit_err , accuracy ,
           sepvar_crit1 , n_inputs , x , y2 , base , direc , g , h ,
           dwk2 , lptr->progress ) ;
      errtype = 1 ;          // Tell other routines net is trained
      }

/*
   That's it.  Copy the optimal values to sigma and do other cleanup.
*/

   for (ivar=0 ; ivar<n_inputs ; ivar++)  // Set sigma 
      sigma[ivar] = safe_exp ( x[ivar] ) ;

   neterr = fabs ( y2 ) ; // Dermin returned neg if ESCape

   if (errtype  &&  lptr->progress) {
      for (ivar=0 ; ivar<n_inputs ; ivar++) {
         sprintf ( msg , "%2d: %.4le", ivar, sigma[ivar] ) ;
         write_progress ( msg ) ;
         }
      }

   if (errtype) {
      sprintf ( msg , "Final error = %.6lf", neterr ) ;
      if (lptr->progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;
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
   if ((! errtype)  ||  (y2 < 0))
      return 1 ;
   return 0 ;
}

static double sepvar_crit0 ( double sig )
{
#if DEBUG_DERIV
   int ivar ;
   double err, f1, f2, d1, d2 ;
   double d = 0.0001 * fabs(sig) ;
   double der[100] ;
   double der2[100] ;

   for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++)
      (local_netptr->sigma)[ivar] = safe_exp ( sig ) ;

   err = local_netptr->trial_error ( local_tptr , 1 ) ;

   for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++) {
      (local_netptr->sigma)[ivar] = safe_exp (sig + d) ;
      f1 = local_netptr->trial_error ( local_tptr , 0 ) ;
      d1 = (f1 - err) / d ;
      (local_netptr->sigma)[ivar] = safe_exp (sig - d) ;
      f2 = local_netptr->trial_error ( local_tptr , 0 ) ;
      d2 = (err - f2) / d ;
      der[ivar] = (f1 - f2) / (2.0 * d) ;
      der2[ivar] = (d1 - d2) / d ;
      (local_netptr->sigma)[ivar] = safe_exp ( sig ) ;
      }

   printf ( "\nSigma=%lf  Err=%lf", sig, err ) ;
   for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++)
      printf ( " (%lf %lf) [%lf %lf]",
         safe_exp(sig) * local_netptr->deriv[ivar],
         der[ivar],
         safe_exp(sig) * local_netptr->deriv[ivar] +
            safe_exp(2*sig) * local_netptr->deriv2[ivar],
         der2[ivar] ) ;
   getch () ;
   return err ;
#else
   int ivar ;
#define C0SIGLIM 40.0

   if (sig > C0SIGLIM) {
      for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++)
         (local_netptr->sigma)[ivar] = safe_exp ( C0SIGLIM ) ;
      return local_netptr->trial_error ( local_tptr , 0 ) +
             10.0 * (sig - C0SIGLIM) ;
      }
   else if (sig < -C0SIGLIM) {
      for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++)
         (local_netptr->sigma)[ivar] = safe_exp ( -C0SIGLIM ) ;
      return local_netptr->trial_error ( local_tptr , 0 ) +
             10.0 * (-sig - C0SIGLIM) ;
      }
   else {
      for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++)
         (local_netptr->sigma)[ivar] = safe_exp ( sig ) ;
      return local_netptr->trial_error ( local_tptr , 0 ) ;
      }
#endif
}

#define C1SIGLIM 20.0
static double sepvar_crit1 ( double *x , int der , double *der1 , double *der2 )
{
   int ivar ;
   double err ;

   err = 0.0 ;

   for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++) {
      if (x[ivar] > C1SIGLIM) {
         (local_netptr->sigma)[ivar] = safe_exp ( C1SIGLIM ) ;
         err += 10.0 * (x[ivar] - C1SIGLIM) ;
         }
      else if (x[ivar] < -C1SIGLIM) {
         (local_netptr->sigma)[ivar] = safe_exp ( -C1SIGLIM ) ;
         err += 10.0 * (-x[ivar] - C1SIGLIM) ;
         }
      else 
         (local_netptr->sigma)[ivar] = safe_exp ( x[ivar] ) ;
      }

   if (! der)
      return err + local_netptr->trial_error ( local_tptr , 0 ) ;

   err += local_netptr->trial_error ( local_tptr , 1 ) ;

   for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++) {
      der1[ivar] = (local_netptr->sigma)[ivar] * local_netptr->deriv[ivar] ;
      der2[ivar] = der1[ivar] + (local_netptr->sigma)[ivar] *
                   (local_netptr->sigma)[ivar] * local_netptr->deriv2[ivar] ;
      }

   return err ;
}

#define C2SIGLIM 20.0
static double sepvar_crit2 ( double sig )
{
   if (sig > C2SIGLIM) {
      (local_netptr->sigma)[local_ivar] = safe_exp ( C2SIGLIM ) ;
      return local_netptr->trial_error ( local_tptr , 0 ) +
             10.0 * (sig - C2SIGLIM) ;
      }
   else if (sig < -C2SIGLIM) {
      (local_netptr->sigma)[local_ivar] = safe_exp ( -C2SIGLIM ) ;
      return local_netptr->trial_error ( local_tptr , 0 ) +
             10.0 * (-sig - C2SIGLIM) ;
      }
   else {
      (local_netptr->sigma)[local_ivar] = safe_exp ( sig ) ;
      return local_netptr->trial_error ( local_tptr , 0 ) ;
      }
}

/*
--------------------------------------------------------------------------------

   wt_print - Print weights as ASCII to file
     Returns:
         0 - Normal
         1 - Unable to open file
         2 - Unable to write file

   wt_save - Save network to disk (called from WT_SAVE.CPP)
   wt_restore - Restore network from disk (called from WT_SAVE.CPP)

--------------------------------------------------------------------------------
*/

int PNNsepvar::wt_print ( char *name )
{
   int i ;
   FILE *fp ;

   if ((fp = fopen ( name , "wt" )) == NULL)
      return 1 ;

   fprintf ( fp , "SEPVAR ASCII weight file" ) ;

   for (i=0 ; i<n_inputs ; i++)
      fprintf ( fp , "\n%3d : %.5le", i+1, sigma[i] ) ;

   if (ferror ( fp )) {
      fclose ( fp ) ;
      return 2 ;
      }
   fclose ( fp ) ;
   return 0 ;
}

int PNNsepvar::wt_save ( FILE *fp )
{
   fwrite ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fwrite ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   if (output_mode == OUTMOD_CLASSIFICATION) {
      fwrite ( tdata->nper , n_outputs * sizeof(unsigned) , 1 , fp ) ;
      fwrite ( tdata->priors , n_outputs * sizeof(double) , 1 , fp ) ;
      }
   fwrite ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fwrite ( sigma , sizeof(double) , n_inputs , fp ) ;
   if (ferror ( fp ))
      return 1 ;
   return 0 ;
}

int PNNsepvar::wt_restore ( FILE *fp )
{

   MEMTEXT ( "PNNsepvar wt_restore new tset" ) ;
   tdata = new TrainingSet ( output_mode , n_inputs , n_outputs , 0 , NULL ) ;
   if (tdata == NULL)
      return 4 ;

   fread ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fread ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   if (output_mode == OUTMOD_CLASSIFICATION) {
      fread ( tdata->nper , n_outputs * sizeof(unsigned) , 1 , fp ) ;
      fread ( tdata->priors , n_outputs * sizeof(double) , 1 , fp ) ;
      }

   if (ferror ( fp )) {
      delete tdata ;
      tdata = NULL ;
      return 2 ;
      }

   MEMTEXT ( "PNNsepvar wt_restore alloc for data" ) ;
   tdata->data= (double *) MALLOC(tdata->ntrain * tdata->size * sizeof(double));
   if (tdata->data == NULL) {   // If insufficient memory
      delete tdata ;
      tdata = NULL ;
      return 4 ;
      }

   fread ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fread ( sigma , sizeof(double) , n_inputs , fp ) ;
   if (ferror ( fp )) {
      delete tdata ;
      tdata = NULL ;
      return 2 ;
      }

   return 0 ;
}

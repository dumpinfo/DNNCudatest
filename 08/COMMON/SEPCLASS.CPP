/******************************************************************************/
/*                                                                            */
/*  SEPCLASS - All principal routines for PNNsepclass processing              */
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
#define OVFL 1.e190

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

PNNsepclass::PNNsepclass ( char *netname , NetParams *net_params )
   : PNNet ( netname , net_params )
{
   if (! ok)    // Did the parent constructor fail?
      return ;  // If so, nothing to do here

   MEMTEXT ( "PNNsepclass constructor" ) ;
   sigma = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;
   deriv = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;
   deriv2 = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;

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

PNNsepclass::~PNNsepclass()
{
   if (! ok)    // If constructor's mallocs failed
      return ;  // there is nothing to free

   MEMTEXT ( "PNNsepclass destructor (3)" ) ;
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

int PNNsepclass::trial ( double *input )
{
   int tset, pop, ivar, ibest, overflow ;
   double *dptr, diff, dist, best, psum, temp ;

   overflow = 0 ;                 // Flags serious overflow

   for (pop=0 ; pop<n_outputs ; pop++) // For each population
      out[pop] = 0.0 ;            // will sum kernels here

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case
      pop = (int) dptr[n_inputs] - 1 ;           // class stored after inputs

      dist = 0.0 ;                            // Will sum distance here
      for (ivar=0 ; ivar<n_inputs ; ivar++) { // All variables in this case
         diff = input[ivar] - dptr[ivar] ;    // Input minus case
         diff /= sigma[pop*n_inputs+ivar] ;   // Scale per sigma
         dist += diff * diff ;                // Cumulate Euclidean distance
         }

      dist = safe_exp ( -dist ) ;
      if (dist < EPS1)                      // If this case is far from all
         dist = EPS1 ;                      // prevent zero density

      out[pop] += dist ;                    // Cumulate this pop's density

      } // For all training cases

/*
   Deal with division by the sigmas for this class, class count normalization,
   and prior probabilities.  It is vital to check for overflow here.
   If the sigmas are tiny and there are a lot of input variables, we can
   easily get overflow.  If so, set a flag that will cause all outputs to
   be zeroed upon return.  This is primitive, but should normally be fine.
   Return the class having highest output.
*/

   psum = 0.0 ;
   for (pop=0 ; pop<n_outputs ; pop++) {

      dptr = sigma + pop * n_inputs ;   // Point to the sigmas for this class
      temp = 1.0 ;                      // Will cumulate the
      for (ivar=0 ; ivar<n_inputs ; ivar++)  // product of all sigmas
         temp *= dptr[ivar] ;           // for the 'pop' class
      if (temp < 1.0 / OVFL) {
         temp = 1.0 / OVFL ;
         overflow = 1 ;
         }
      out[pop] /= temp ;                // Scale outputs per sigmas

      if (tdata->priors[pop] >=  0.0)   // If user specified priors
         out[pop] *= tdata->priors[pop] / tdata->nper[pop] ; // Use them
      psum += out[pop] ;
      }

   if (psum < EPS2)                  // If this test case is far from all
      psum = EPS2 ;                  // prevent division by zero

   if (overflow) {
      for (pop=0 ; pop<n_outputs ; pop++)
         out[pop] = 0.0 ;
      }
   else {
      for (pop=0 ; pop<n_outputs ; pop++)
         out[pop] /= psum ;
      }

   best = -1.0 ;                     // Keep track of max across pops
   for (pop=0 ; pop<n_outputs ; pop++) {  // For each population
      if (out[pop] > best) {         // find the highest activation
         best = out[pop] ;
         ibest = pop ;
         }
      }
   return ibest ;
}

/*
--------------------------------------------------------------------------------

   trial_deriv - Just like 'trial', but also cumulates derivative

   Note that in order to compute the derivative, we need the target
   value.  Since the sepclass model is always in CLASSIFICATION mode,
   that will be "tclass", and "target" will be ignored.
   We just need to include the useless "target" because this is a virtual
   function in the base class.

--------------------------------------------------------------------------------
*/

int PNNsepclass::trial_deriv ( double *input , int tclass , double * )
{
   int i, tset, pop, ivar, ibest, outvar, overflow ;
   double *dptr, diff, dist, truedist, best, *vptr, *wptr, vij, wij ;
   double temp, der1, der2, psum ;

   overflow = 0 ;                    // Flags serious overflow

   for (pop=0 ; pop<n_outputs ; pop++) {  // For each population
      out[pop] = 0.0 ;               // Will sum kernels here
      for (ivar=0 ; ivar<n_inputs ; ivar++) {
         v[pop*n_inputs+ivar] = 0.0 ;     // Scratch for derivative stuff
         w[pop*n_inputs+ivar] = 0.0 ;     // Ditto
         }
      }

#if FULL_DEBUG
   printf ( "\nCalling trial_deriv input %lf %lf  Class %d",
      input[0], input[1], tclass ) ;
#endif

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case
      pop = (int) dptr[n_inputs] - 1 ;           // Class stored after inputs

      dist = 0.0 ;                          // Will sum distance here
      for (ivar=0 ; ivar<n_inputs ; ivar++) {    // All variables in this case
         diff = input[ivar] - dptr[ivar] ;  // Input minus case
         diff /= sigma[pop*n_inputs+ivar] ;      // Scale per sigma
         dsqr[ivar] = diff * diff ;         // Squared weighted distance
         dist += dsqr[ivar] ;               // Cumulate for all vars
         }

      dist = safe_exp ( -dist ) ;

#if FULL_DEBUG
      printf ( "\n  %d (%d): dist=%le  dsqr=%lf %lf",
            tset, pop, dist, dsqr[0], dsqr[1] ) ;
#endif

      truedist = dist ;                  // Need this for derivatives
      if (dist < EPS1)                   // If this case is far from all
         dist = EPS1 ;                   // prevent zero density

      out[pop] += dist ;                 // Cumulate this class's density
      vptr = v + pop * n_inputs ;             // Point to this row in v
      wptr = w + pop * n_inputs ;             // And w
      for (ivar=0 ; ivar<n_inputs ; ivar++) { // All variables in this case
         temp = truedist * dsqr[ivar] ;
         vptr[ivar] += temp ;
         wptr[ivar] += temp * (2.0 * dsqr[ivar] - 3.0) ;
         }

      } // For all training cases

/*
   Scale the outputs per the sigmas.
   Make the v's and w's be the actual derivatives of the activations.
*/

   for (outvar=0 ; outvar<n_outputs ; outvar++) {  // i in sigma[ij], v[kij]
      temp = 1.0 ;                            // Will cumulate the
      for (ivar=0 ; ivar<n_inputs ; ivar++)        // product of all sigmas
         temp *= sigma[outvar*n_inputs+ivar] ;     // for the outvar class
      if (temp < 1.0 / OVFL) {
         temp = 1.0 / OVFL ;
         overflow = 1 ;
         }
      out[outvar] /= temp ;                   // Scale outputs per sigmas
      for (ivar=0 ; ivar<n_inputs ; ivar++) { // j in sigma[ij], v[kij]
         v[outvar*n_inputs+ivar] *= 2.0 / sigma[outvar*n_inputs+ivar] ; // Common factors
         w[outvar*n_inputs+ivar] *=
                      2.0 / (sigma[outvar*n_inputs+ivar] * sigma[outvar*n_inputs+ivar]) ;
         // At this point, v and w are derivatives of activation before scaling
         v[outvar*n_inputs+ivar] /= temp ;         // Also scale first and
         w[outvar*n_inputs+ivar] /= temp ;         // second derivatives
         // Apply Equations !!! and !!! to compute revised derivatives
         w[outvar*n_inputs+ivar] += 2.0 / sigma[outvar*n_inputs+ivar] * 
             (out[outvar] / sigma[outvar*n_inputs+ivar] - v[outvar*n_inputs+ivar] ) ;
         v[outvar*n_inputs+ivar] -= out[outvar] / sigma[outvar*n_inputs+ivar] ;
         }
      }


/*
   Deal with class count normalization and prior probabilities.
*/

   psum = 0.0 ;
   for (pop=0 ; pop<n_outputs ; pop++) {
      if (tdata->priors[pop] >=  0.0)
         out[pop] *= tdata->priors[pop] / tdata->nper[pop] ;
      psum += out[pop] ;
      }
   
   if (psum < EPS2)    // Even though we kept dist away from 0 above
      psum = EPS2 ;    // Pathological priors can still cause problems

   for (pop=0 ; pop<n_outputs ; pop++)
      out[pop] /= psum ;

/*
   Compute the derivatives.  Since this is CLASSIFY mode, we must also
   worry about priors.
*/

   for (ivar=0 ; ivar<n_inputs ; ivar++) {  // j in sigma[ij], v[kij]

      for (outvar=0 ; outvar<n_outputs ; outvar++) {  // Apply priors to derivs
         if (tdata->priors[outvar] >=  0.0) {
            v[outvar*n_inputs+ivar] *= tdata->priors[outvar] / tdata->nper[outvar] ;
            w[outvar*n_inputs+ivar] *= tdata->priors[outvar] / tdata->nper[outvar] ;
            }
         v[outvar*n_inputs+ivar] /= psum ;  // Doing this now
         w[outvar*n_inputs+ivar] /= psum ;  // Saves a little time later
         }

      for (outvar=0 ; outvar<n_outputs ; outvar++) {
         if (outvar == tclass)
            temp = 2.0 * (out[outvar] - 1.0) ;
         else 
            temp = 2.0 * out[outvar] ;

         for (i=0 ; i<n_outputs ; i++) {   // i in sigma[ij], v[kij]
            vij = v[i*n_inputs+ivar] ;
            wij = w[i*n_inputs+ivar] ;
            if (i == outvar) {
               der1 = vij * (1.0 - out[outvar]) ;
               der2 = wij * (1.0 - out[outvar]) +
                      2.0 * vij * vij * (out[outvar] - 1.0) ;
               }
            else {
               der1 = -out[outvar] * vij ;
               der2 = out[outvar] * (2.0 * vij * vij - wij) ;
               }
            deriv[i*n_inputs+ivar] += temp * der1 ;
            deriv2[i*n_inputs+ivar] += temp * der2  +  2.0 * der1 * der1 ;
            } // For i to n_outputs
         }  // For outvar (k in sigma[kij])
      }  // For ivar (j in sigma[kij])

#if FULL_DEBUG
   for (ivar=0 ; ivar<n_inputs ; ivar++) {  // j in sigma(kij)
      for (i=0 ; i<n_outputs ; i++) {   // i in sigma(kij)
         printf ( "\n  pop=%d  var=%d  out=%lf  sigma=%lf  der1=%lf  der2=%lf",
                         i, ivar, out[i], sigma[i*n_inputs+ivar],
                         deriv[i*n_inputs+ivar], deriv2[i*n_inputs+ivar] ) ;
         }
      }
#endif

/*
   Return the class having highest output.
   Also set all outputs to zero in case there was overflow.
*/

   if (overflow) {
      for (pop=0 ; pop<n_outputs ; pop++)
         out[pop] = 0.0 ;
      }

   best = -1.0 ;                     // Keep track of max across pops
   for (pop=0 ; pop<n_outputs ; pop++) {  // For each population
      if (out[pop] > best) {         // find the highest activation
         best = out[pop] ;
         ibest = pop ;
         }
      }
   return ibest ;
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

static double sepclass_crit0 ( double sig ) ;   // Local criterion to optimize
static double sepclass_crit1 ( double *sigs , int der , double *der1 ,
                               double *der2 ) ;
static TrainingSet *local_tptr ;    // These two statics pass class data
static PNNsepclass *local_netptr ;  // to the local criterion routine

int PNNsepclass::learn ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int ivar, k, pop ;
   double x1, y1, x2, y2, x3, y3, accuracy ;
   double *x, *base, *direc, *g, *h, *dwk2 ;
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

   MEMTEXT ( "PNNsepclass::learn works (9)" ) ;
   dsqr = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   v = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;
   w = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;
   x = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;
   base = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;
   direc = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;
   g = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;
   h = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;
   dwk2 = (double *) MALLOC ( n_inputs * n_outputs * sizeof(double) ) ;

   if ((dsqr == NULL)  ||  (v == NULL)  ||  (w == NULL)  ||  (x == NULL)  ||
       (base == NULL)  ||  (direc == NULL) || (g == NULL) ||
       (h == NULL)  ||  (dwk2 == NULL)){
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
      MEMTEXT ( "SEPCLASS learn deleting tset" ) ;
      delete ( tdata ) ;
      tdata = NULL ;
      }

   MEMTEXT ( "SEPCLASS new, copy tset" ) ;
   tdata = new TrainingSet ( output_mode , n_inputs , n_outputs , 0 , NULL ) ;
   if (tdata == NULL)
      return -1 ;

   *tdata = *tptr ;       // Invoke assignment operator to duplicate it

   if (! tdata->ntrain) { // Insufficient memory to copy?
      delete tdata ;
      tdata = NULL ;
      return -1 ;         // Return error flag
      }

   local_netptr = this ;  // Passes this infor
   local_tptr = tdata ;   // To criterion routines

   make_progress_window ( "PNN (SEPCLASS) learning" ) ;

/*
   Compute a crude global minimum using a single sigma, common to all variables.
   Glob_min will return this somewhat optimal sigma in "x2".
   Its function return value is nonzero iff the user pressed ESCape.
*/

   if (errtype) { // If the network is already trained (errtype != 0)
      k = 0 ;     // Then skip global initialization
      for (pop=0 ; pop<n_outputs ; pop++) {
         for (ivar=0 ; ivar<n_inputs ; ivar++)  // retrieve sigma 
            x[pop*n_inputs+ivar] = log ( sigma[pop*n_inputs+ivar] ) ;
         }
      y2 = 1.e30 ; // This must be huge (not neterr) in case different tset
      x2 = x[0] ;  // This is a flag to avoid multivariate min if sigma huge
      }
   else {
      k = glob_min ( log(lptr->siglo) , log(lptr->sighi) , lptr->nsigs , 0 ,
           lptr->quit_err , sepclass_crit0 , &x1 , &y1 , &x2 , &y2 ,
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
      for (pop=0 ; pop<n_outputs ; pop++) {
         for (ivar=0 ; ivar<n_inputs ; ivar++)  // Set x for derivmin
            x[pop*n_inputs+ivar] = x2 ; // outside 'if' so sigma gets it if user quit
         }
      }

   if (k) { // If global was interrupted by user ESCape before trio
      y2 = -1.e30 ;
      write_progress ( "Learning aborted by user" ) ;
      }
   else if (fabs(x2) > max_exp/2.0) { // Safety: avoid stupid refinement
      if (errtype) {                  // If already trained to huge sigma
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
            sepclass_crit1 , n_inputs*n_outputs , x , y2 , base , direc , g , h ,
            dwk2 , lptr->progress ) ;
      errtype = 1 ;          // Tell other routines net is trained
      }

/*
   That's it.  Copy the optimal values to sigma and do other cleanup.
*/

   for (pop=0 ; pop<n_outputs ; pop++) {
      for (ivar=0 ; ivar<n_inputs ; ivar++)  // Set sigma 
         sigma[pop*n_inputs+ivar] = safe_exp ( x[pop*n_inputs+ivar] ) ;
      }

   neterr = fabs ( y2 ) ; // Derivmin returned neg if ESCape

   if (errtype  &&  lptr->progress) {
      for (pop=0 ; pop<n_outputs ; pop++) {
         for (ivar=0 ; ivar<n_inputs ; ivar++) {
            sprintf ( msg , "pop %2d  var %2d: %.6le",
               pop, ivar, sigma[pop*n_inputs+ivar] ) ;
            write_progress ( msg ) ;
            }
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

   MEMTEXT ( "PNNsepclass::learn works (8)" ) ;
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

static double sepclass_crit0 ( double sig )
{
#if DEBUG_DERIV
   int ivar, pop ;
   double err, f1, f2, d1, d2 ;
   double d = 0.00001 * fabs(sig) ;
   double der, der2 ;

   for (pop=0 ; pop<local_netptr->n_outputs ; pop++) {
      for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++)
         (local_netptr->sigma)[pop*local_netptr->n_inputs+ivar] = safe_exp ( sig ) ;
      }

   err = local_netptr->trial_error ( local_tptr , 1 ) ;

   printf ( "\nSigma=%lf  Err=%.18le", sig, err ) ;
   for (pop=0 ; pop<local_netptr->n_outputs ; pop++) {
      for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++) {
         (local_netptr->sigma)[pop*local_netptr->n_inputs+ivar] = safe_exp ( sig + d ) ;
         f1 = local_netptr->trial_error ( local_tptr , 0 ) ;
         d1 = (f1 - err) / d ;
         (local_netptr->sigma)[pop*local_netptr->n_inputs+ivar] = safe_exp ( sig - d ) ;
         f2 = local_netptr->trial_error ( local_tptr , 0 ) ;
         d2 = (err - f2) / d ;
         der = (f1 - f2) / (2.0 * d) ;
         der2 = (d1 - d2) / d ;
         (local_netptr->sigma)[pop*local_netptr->n_inputs+ivar] = safe_exp ( sig ) ;
         printf ( " %d %d: (%lf %lf) [%lf %lf]", pop, ivar,
            safe_exp(sig) * local_netptr->deriv[pop*local_netptr->n_inputs+ivar],
            der,
            safe_exp(sig) * local_netptr->deriv[pop*local_netptr->n_inputs+ivar] +
            safe_exp(2*sig) * local_netptr->deriv2[pop*local_netptr->n_inputs+ivar],
            der2 ) ;
         }
      }

   getch () ;
   return err ;
#else
   int ivar, pop ;
   double retval ;

   for (pop=0 ; pop<local_netptr->n_outputs ; pop++) {
      for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++)
         (local_netptr->sigma)[pop*local_netptr->n_inputs+ivar] = safe_exp ( sig ) ;
      }

   retval = local_netptr->trial_error ( local_tptr , 0 ) ;

   if (sig > max_exp)   // Prevent wildness
      retval += sig - max_exp ;
   return retval ;
#endif
}

static double sepclass_crit1 ( double *x , int der , double *der1 ,
                               double *der2 )
{
   int ivar, pop ;
   double err ;

   for (pop=0 ; pop<local_netptr->n_outputs ; pop++) {
      for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++)
         (local_netptr->sigma)[pop*local_netptr->n_inputs+ivar] =
                       safe_exp ( x[pop*local_netptr->n_inputs+ivar] ) ;
      }

   if (! der)
      return local_netptr->trial_error ( local_tptr , 0 ) ;

   err = local_netptr->trial_error ( local_tptr , 1 ) ;

   for (pop=0 ; pop<local_netptr->n_outputs ; pop++) {
      for (ivar=0 ; ivar<local_netptr->n_inputs ; ivar++) {
         der1[pop*local_netptr->n_inputs+ivar] =
                       (local_netptr->sigma)[pop*local_netptr->n_inputs+ivar] *
                        local_netptr->deriv[pop*local_netptr->n_inputs+ivar] ;
         der2[pop*local_netptr->n_inputs+ivar] =
            der1[pop*local_netptr->n_inputs+ivar] +
            (local_netptr->sigma)[pop*local_netptr->n_inputs+ivar] *
            (local_netptr->sigma)[pop*local_netptr->n_inputs+ivar] *
            local_netptr->deriv2[pop*local_netptr->n_inputs+ivar] ;
         }
      }

   return err ;
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

int PNNsepclass::wt_print ( char *name )
{
   int i, j ;
   FILE *fp ;

   if ((fp = fopen ( name , "wt" )) == NULL)
      return 1 ;

   fprintf ( fp , "SEPCLASS ASCII weight file" ) ;

   for (i=0 ; i<n_outputs ; i++) {
      for (j=0 ; j<n_inputs ; j++)
         fprintf ( fp , "\nClass %d  Var %d : %.5le", i+1, j+1, sigma[i*n_inputs+j]);
      }

   if (ferror ( fp )) {
      fclose ( fp ) ;
      return 2 ;
      }
   fclose ( fp ) ;
   return 0 ;
}

int PNNsepclass::wt_save ( FILE *fp )
{
   fwrite ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fwrite ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   fwrite ( tdata->nper , n_outputs * sizeof(unsigned) , 1 , fp ) ;
   fwrite ( tdata->priors , n_outputs * sizeof(double) , 1 , fp ) ;
   fwrite ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fwrite ( sigma , sizeof(double) , n_inputs*n_outputs , fp ) ;
   if (ferror ( fp ))
      return 1 ;
   return 0 ;
}

int PNNsepclass::wt_restore ( FILE *fp )
{

   MEMTEXT ( "PNNsepclass wt_restore new tset" ) ;
   tdata = new TrainingSet ( output_mode , n_inputs , n_outputs , 0 , NULL ) ;
   if (tdata == NULL)
      return 4 ;

   fread ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fread ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   fread ( tdata->nper , n_outputs * sizeof(unsigned) , 1 , fp ) ;
   fread ( tdata->priors , n_outputs * sizeof(double) , 1 , fp ) ;

   if (ferror ( fp )) {
      delete tdata ;
      tdata = NULL ;
      return 2 ;
      }

   MEMTEXT ( "PNNsepclass wt_restore alloc for data" ) ;
   tdata->data= (double *) MALLOC(tdata->ntrain * tdata->size * sizeof(double));
   if (tdata->data == NULL) {   // If insufficient memory
      delete tdata ;
      tdata = NULL ;
      return 4 ;
      }

   fread ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fread ( sigma , sizeof(double) , n_inputs*n_outputs , fp ) ;
   if (ferror ( fp )) {
      delete tdata ;
      tdata = NULL ;
      return 2 ;
      }
   return 0 ;
}

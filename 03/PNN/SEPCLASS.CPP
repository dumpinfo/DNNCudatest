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
/*  SEPCLASS - All principal routines for PNNsepclass processing              */
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

#define EPS1 1.e-180
#define EPS2 1.e-190
#define OVFL 1.e190

/*
--------------------------------------------------------------------------------

   Constructor

   In case of malloc failure, we set 'ok' to zero so the user knows about it.

--------------------------------------------------------------------------------
*/

PNNsepclass::PNNsepclass (
   int net_kernel , 
   int out_model ,
   int n_inputs ,
   int n_outputs
   )
   : PNNet ( net_kernel , out_model , n_inputs , n_outputs )
{
   if (! ok)    // Did the parent constructor fail?
      return ;  // If so, nothing to do here

   model = NETMOD_SEPCLASS ; // Seems silly in C++, but helps sometimes
   tdata = NULL ;            // No training data here

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

   if ((tdata != NULL)  &&  ! tdata_shared) {
      MEMTEXT ( "PNNsepclass destructor deleting tset" ) ;
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

int PNNsepclass::privatize_tset ()
{
   TrainingSet *shared ;

   if (! tdata_shared)   // If it is already private
      return 0 ;         // Nothing to do

   shared = tdata ;      // This is the shared tset

   MEMTEXT ( "PNNsepclass privatize_tset new, copy tset" ) ;
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

int PNNsepclass::trial ( double *input )
{
   int tset, pop, ivar, ibest, overflow ;
   double *dptr, diff, dist, best, psum, temp ;

   overflow = 0 ;                 // Flags serious overflow

   for (pop=0 ; pop<nout ; pop++) // For each population
      out[pop] = 0.0 ;            // will sum kernels here

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case
      pop = (int) dptr[nin] - 1 ;           // class stored after inputs

      dist = 0.0 ;                          // Will sum distance here
      for (ivar=0 ; ivar<nin ; ivar++) {    // All variables in this case
         diff = input[ivar] - dptr[ivar] ;  // Input minus case
         diff /= sigma[pop*nin+ivar] ;      // Scale per sigma
         dist += diff * diff ;              // Cumulate Euclidean distance
         }

      dist = exp ( -dist ) ;
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
   for (pop=0 ; pop<nout ; pop++) {

      dptr = sigma + pop * nin ;        // Point to the sigmas for this class
      temp = 1.0 ;                      // Will cumulate the
      for (ivar=0 ; ivar<nin ; ivar++)  // product of all sigmas
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
      for (pop=0 ; pop<nout ; pop++)
         out[pop] = 0.0 ;
      }
   else {
      for (pop=0 ; pop<nout ; pop++)
         out[pop] /= psum ;
      }

   best = -1.0 ;                     // Keep track of max across pops
   for (pop=0 ; pop<nout ; pop++) {  // For each population
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

   for (pop=0 ; pop<nout ; pop++) {  // For each population
      out[pop] = 0.0 ;               // Will sum kernels here
      for (ivar=0 ; ivar<nin ; ivar++) {
         v[pop*nin+ivar] = 0.0 ;     // Scratch for derivative stuff
         w[pop*nin+ivar] = 0.0 ;     // Ditto
         }
      }

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case
      pop = (int) dptr[nin] - 1 ;           // Class stored after inputs

      dist = 0.0 ;                          // Will sum distance here
      for (ivar=0 ; ivar<nin ; ivar++) {    // All variables in this case
         diff = input[ivar] - dptr[ivar] ;  // Input minus case
         diff /= sigma[pop*nin+ivar] ;      // Scale per sigma
         dsqr[ivar] = diff * diff ;         // Squared weighted distance
         dist += dsqr[ivar] ;               // Cumulate for all vars
         }

      dist = exp ( -dist ) ;

      truedist = dist ;                  // Need this for derivatives
      if (dist < EPS1)                   // If this case is far from all
         dist = EPS1 ;                   // prevent zero density

      out[pop] += dist ;                 // Cumulate this class's density
      vptr = v + pop * nin ;             // Point to this row in v
      wptr = w + pop * nin ;             // And w
      for (ivar=0 ; ivar<nin ; ivar++) { // All variables in this case
         temp = truedist * dsqr[ivar] ;
         vptr[ivar] += temp ;
         wptr[ivar] += temp * (2.0 * dsqr[ivar] - 3.0) ;
         }

      } // For all training cases

/*
   Scale the outputs per the sigmas.
   Make the v's and w's be the actual derivatives of the activations.
*/

   for (outvar=0 ; outvar<nout ; outvar++) {  // i in sigma[ij], v[kij]
      temp = 1.0 ;                            // Will cumulate the
      for (ivar=0 ; ivar<nin ; ivar++)        // product of all sigmas
         temp *= sigma[outvar*nin+ivar] ;     // for the outvar class
      if (temp < 1.0 / OVFL) {
         temp = 1.0 / OVFL ;
         overflow = 1 ;
         }
      out[outvar] /= temp ;                   // Scale outputs per sigmas
      for (ivar=0 ; ivar<nin ; ivar++) {      // j in sigma[ij], v[kij]
         v[outvar*nin+ivar] *= 2.0 / sigma[outvar*nin+ivar] ; // Common factors
         w[outvar*nin+ivar] *=
                      2.0 / (sigma[outvar*nin+ivar] * sigma[outvar*nin+ivar]) ;
         // At this point, v and w are derivatives of activation before scaling
         v[outvar*nin+ivar] /= temp ;         // Also scale first and
         w[outvar*nin+ivar] /= temp ;         // second derivatives
         // Apply Equations !!! and !!! to compute revised derivatives
         w[outvar*nin+ivar] += 2.0 / sigma[outvar*nin+ivar] * 
             (out[outvar] / sigma[outvar*nin+ivar] - v[outvar*nin+ivar] ) ;
         v[outvar*nin+ivar] -= out[outvar] / sigma[outvar*nin+ivar] ;
         }
      }


/*
   Deal with class count normalization and prior probabilities.
*/

   psum = 0.0 ;
   for (pop=0 ; pop<nout ; pop++) {
      if (tdata->priors[pop] >=  0.0)
         out[pop] *= tdata->priors[pop] / tdata->nper[pop] ;
      psum += out[pop] ;
      }
   
   if (psum < EPS2)    // Even though we kept dist away from 0 above
      psum = EPS2 ;    // Pathological priors can still cause problems

   for (pop=0 ; pop<nout ; pop++)
      out[pop] /= psum ;

/*
   Compute the derivatives.  Since this is CLASSIFY mode, we must also
   worry about priors.
*/

   for (ivar=0 ; ivar<nin ; ivar++) {  // j in sigma[ij], v[kij]

      for (outvar=0 ; outvar<nout ; outvar++) {  // Apply priors to derivs
         if (tdata->priors[outvar] >=  0.0) {
            v[outvar*nin+ivar] *= tdata->priors[outvar] / tdata->nper[outvar] ;
            w[outvar*nin+ivar] *= tdata->priors[outvar] / tdata->nper[outvar] ;
            }
         v[outvar*nin+ivar] /= psum ;  // Doing this now
         w[outvar*nin+ivar] /= psum ;  // Saves a little time later
         }

      for (outvar=0 ; outvar<nout ; outvar++) {
         if (outvar == tclass)
            temp = 2.0 * (out[outvar] - 1.0) ;
         else 
            temp = 2.0 * out[outvar] ;

         for (i=0 ; i<nout ; i++) {   // i in sigma[ij], v[kij]
            vij = v[i*nin+ivar] ;
            wij = w[i*nin+ivar] ;
            if (i == outvar) {
               der1 = vij * (1.0 - out[outvar]) ;
               der2 = wij * (1.0 - out[outvar]) +
                      2.0 * vij * vij * (out[outvar] - 1.0) ;
               }
            else {
               der1 = -out[outvar] * vij ;
               der2 = out[outvar] * (2.0 * vij * vij - wij) ;
               }
            deriv[i*nin+ivar] += temp * der1 ;
            deriv2[i*nin+ivar] += temp * der2  +  2.0 * der1 * der1 ;
            } // For i to nout
         }  // For outvar (k in sigma[kij])
      }  // For ivar (j in sigma[kij])

/*
   Return the class having highest output.
   Also set all outputs to zero in case there was overflow.
*/

   if (overflow) {
      for (pop=0 ; pop<nout ; pop++)
         out[pop] = 0.0 ;
      }

   best = -1.0 ;                     // Keep track of max across pops
   for (pop=0 ; pop<nout ; pop++) {  // For each population
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

--------------------------------------------------------------------------------
*/

static double sepclass_crit0 ( double sig ) ;   // Local criterion to optimize
static double sepclass_crit1 ( double *sigs , double *der1 , double *der2 ,
                               int der ) ;
static TrainingSet *local_tptr ;    // These two statics pass class data
static PNNsepclass *local_netptr ;  // to the local criterion routine

void PNNsepclass::learn ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int ivar, k, pop ;
   double x1, y1, x2, y2, x3, y3 ;
   double *x, *base, *direc, *g, *h, *dwk2 ;
   char msg[84] ;

/*
   Allocate work areas.  Some of these (v, w, dsqr) are private class members
   to simplify use across subroutines.  If we are not in CLASSIFY mode then
   allocate one extra vector at the end of v and w to store the general
   alternatives to vsum and wsum (which are no longer just sums).
*/

   MEMTEXT ( "PNNsepclass::learn works (9)" ) ;
   dsqr = (double *) MALLOC ( nin * sizeof(double) ) ;
   v = (double *) MALLOC ( nin * nout * sizeof(double) ) ;
   w = (double *) MALLOC ( nin * nout * sizeof(double) ) ;
   x = (double *) MALLOC ( nin * nout * sizeof(double) ) ;
   base = (double *) MALLOC ( nin * nout * sizeof(double) ) ;
   direc = (double *) MALLOC ( nin * nout * sizeof(double) ) ;
   g = (double *) MALLOC ( nin * nout * sizeof(double) ) ;
   h = (double *) MALLOC ( nin * nout * sizeof(double) ) ;
   dwk2 = (double *) MALLOC ( nin * nout * sizeof(double) ) ;

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
      printf ( "\nInsufficient memory to learn" ) ;
      return ;
      }

/*
   If there is a local training set copy it may no longer be valid.
   Take no chances.  Delete it and share with the current set.
*/

   if ((tdata != NULL)  &&  ! tdata_shared) {
      MEMTEXT ( "PNNsepclass learn deleting tset" ) ;
      delete ( tdata ) ;
      }

   tdata = tptr ;         // Save memory by sharing the training set
   tdata_shared = 1 ;     // Flag that it is shared

   make_progress_window () ;

   local_netptr = this ;  // Passes this infor
   local_tptr = tdata ;   // To criterion routines

/*
   Compute a crude global minimum using a single sigma, common to all variables.
   Glob_min will return this somewhat optimal sigma in "x2".
   Its function return value is nonzero iff the user pressed ESCape.
*/

   if (errtype) { // If the network is already trained (errtype != 0)
      k = 0 ;     // Then skip global initialization
      for (pop=0 ; pop<nout ; pop++) {
         for (ivar=0 ; ivar<nin ; ivar++)  // retrieve sigma 
            x[pop*nin+ivar] = sigma[pop*nin+ivar] ;
         }
      y2 = 1.e30 ;
      }
   else {
      k = glob_min ( lptr->siglo , lptr->sighi , lptr->nsigs , 1 ,
           lptr->quit_err , sepclass_crit0 , &x1 , &y1 , &x2 , &y2 , &x3 , &y3 ) ;
      sprintf ( msg , "Global err at %.6lf = %.6lf", x2, y2 ) ;
      normal_message ( msg ) ;
      for (pop=0 ; pop<nout ; pop++) {
         for (ivar=0 ; ivar<nin ; ivar++)  // Set x for derivmin
            x[pop*nin+ivar] = x2 ; // outside 'if' so sigma gets it if user quit
         }
      }

   if (k) { // If global was interrupted by user ESCape before trio
      y2 = 1.e30 ;
      normal_message ( "Learning aborted by user" ) ;
      }
   else {
      y2 = dermin ( 32767 , lptr->quit_err , 1.e-8 , lptr->quit_tol ,
            sepclass_crit1 , nin*nout , x , y2 , base , direc , g , h , dwk2 ) ;
      sprintf ( msg , "Refined err = %.6lf", fabs ( y2 ) ) ;
      normal_message ( msg ) ;
      }

/*
   That's it.  Copy the optimal values to sigma and do other cleanup.
*/

   for (pop=0 ; pop<nout ; pop++) {
      for (ivar=0 ; ivar<nin ; ivar++)  // Set sigma 
         sigma[pop*nin+ivar] = x[pop*nin+ivar] ;
      }

   neterr = fabs ( y2 ) ; // Derivmin returned neg if ESCape
   errtype = 1 ;          // Tell other routines net is trained

   for (pop=0 ; pop<nout ; pop++) {
      for (ivar=0 ; ivar<nin ; ivar++) {
         sprintf ( msg , "pop %2d  var %2d: %.6le",
            pop, ivar, sigma[pop*nin+ivar] ) ;
         write_progress ( msg ) ;
         }
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
   return ;
}

static double sepclass_crit0 ( double sig )
{
   int ivar, pop ;

   for (pop=0 ; pop<local_netptr->nout ; pop++) {
      for (ivar=0 ; ivar<local_netptr->nin ; ivar++)
         (local_netptr->sigma)[pop*local_netptr->nin+ivar] = sig ;
      }

   return local_netptr->trial_error ( local_tptr , 0 ) ;
}

static double sepclass_crit1 ( double *x , double *der1 , double *der2 , int der )
{
   int ivar, pop ;
   double err ;

   for (pop=0 ; pop<local_netptr->nout ; pop++) {
      for (ivar=0 ; ivar<local_netptr->nin ; ivar++)
         (local_netptr->sigma)[pop*local_netptr->nin+ivar] =
                                                x[pop*local_netptr->nin+ivar] ;
      }

   if (! der)
      return local_netptr->trial_error ( local_tptr , 0 ) ;

   err = local_netptr->trial_error ( local_tptr , 1 ) ;

   for (pop=0 ; pop<local_netptr->nout ; pop++) {
      for (ivar=0 ; ivar<local_netptr->nin ; ivar++) {
         der1[pop*local_netptr->nin+ivar] =
                              local_netptr->deriv[pop*local_netptr->nin+ivar] ;
         der2[pop*local_netptr->nin+ivar] =
                              local_netptr->deriv2[pop*local_netptr->nin+ivar] ;
         }
      }

   return err ;
}

/*
--------------------------------------------------------------------------------

   wt_save - Save network to disk (called from WT_SAVE.CPP)
   wt_restore - Restore network from disk (called from WT_SAVE.CPP)

--------------------------------------------------------------------------------
*/

int PNNsepclass::wt_save ( FILE *fp )
{
   fwrite ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fwrite ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   fwrite ( tdata->nper , nout * sizeof(unsigned) , 1 , fp ) ;
   fwrite ( tdata->priors , nout * sizeof(double) , 1 , fp ) ;
   fwrite ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fwrite ( sigma , sizeof(double) , nin*nout , fp ) ;
   if (ferror ( fp ))
      return 1 ;
   return 0 ;
}

void PNNsepclass::wt_restore ( FILE *fp )
{

   MEMTEXT ( "PNNsepclass wt_restore new tset" ) ;
   tdata = new TrainingSet ( outmod , nin , nout ) ; // Make a new tset
   if (tdata == NULL) {
      ok = 0 ;
      return ;
      }

   fread ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fread ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   fread ( tdata->nper , nout * sizeof(unsigned) , 1 , fp ) ;
   fread ( tdata->priors , nout * sizeof(double) , 1 , fp ) ;

   MEMTEXT ( "PNNsepclass wt_restore alloc for data" ) ;
   tdata->data= (double *) MALLOC(tdata->ntrain * tdata->size * sizeof(double));
   if (tdata->data == NULL) {   // If insufficient memory
      ok = 0 ;
      delete tdata ;
      tdata = NULL ;
      memory_message ( "to read network" ) ;
      return ;
      }

   fread ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fread ( sigma , sizeof(double) , nin*nout , fp ) ;
   tdata_shared = 0 ;
   if (ferror ( fp ))
      ok = 0 ;
   return ;
}


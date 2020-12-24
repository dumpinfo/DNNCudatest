/******************************************************************************/
/*                                                                            */
/*  PNNBASIC - All principal routines for PNNbasic processing                 */
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

#define EPS1 1.e-180
#define EPS2 1.e-190


/*
--------------------------------------------------------------------------------

   Constructor

   In case of error (such as insufficient memory), 'ok' is returned zero.

--------------------------------------------------------------------------------
*/

PNNbasic::PNNbasic ( char *netname , NetParams *net_params )
   : PNNet ( netname , net_params )
{
   if (! ok)    // Did the parent constructor fail?
      return ;  // If so, nothing to do here

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
#if DEBUG
   double temp ;
   char msg[256] ;
#endif

   width = 1.0 / (sigma * sigma) ; // Multiplies Euclidean distances

   for (pop=0 ; pop<n_outputs ; pop++) // For each population
      out[pop] = 0.0 ;            // will sum kernels here
   psum = 0.0 ;                   // Denominator sum if GENERAL

   for (tset=0 ; tset<tdata->ntrain ; tset++) {  // Do all training cases

      dptr = tdata->data + tdata->size * tset ;  // Point to this case

      dist = 0.0 ;                          // Will sum distance here
      for (ivar=0 ; ivar<n_inputs ; ivar++) { // All variables in this case
         diff = input[ivar] - dptr[ivar] ;  // Input minus case
         dist += diff * diff ;              // Cumulate Euclidean distance
         }
      dist *= width ;                       // Divide dist by sigma squared
#if DEBUG
      temp = dist ;
#endif

      if (kernel == KERNEL_GAUSS)           // Apply the kernel function
         dist = exp ( -dist ) ;
      else if (kernel == KERNEL_RECIP)
         dist = 1.0 / ( 1.0 + dist ) ;

      if (dist < EPS1)                      // If this case is far from all
         dist = EPS1 ;                      // prevent zero density

      if (output_mode == OUTMOD_CLASSIFICATION) { // If this is Classification
         pop = (int) dptr[n_inputs] - 1 ;        // class stored after inputs
         out[pop] += dist ;                 // Cumulate this pop's density
         }
      else if (output_mode == OUTMOD_MAPPING) {  // If this is general mapping
         dptr += n_inputs ;                      // Outputs stored after inputs
         for (ivar=0 ; ivar<n_outputs ; ivar++)  // For every output variable
            out[ivar] += dist * dptr[ivar] ;// Cumulate numerator
         psum += dist ;                     // Cumulate denominator
         }
#if DEBUG
      sprintf ( msg , "tset=%d  pop=%d  in = (%lf %lf)  dptr=(%lf %lf)",
         tset, pop, input[0], input[1], dptr[0], dptr[1] ) ;
      MEMTEXT ( msg ) ;
      sprintf ( msg , "wdsq=%lf  dist=%lf (%lf %lf)  sigma=%lf",
         temp, dist, out[0], out[1], sigma ) ;
      MEMTEXT ( msg ) ;
#endif
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
#if DEBUG
      sprintf ( msg , "priors=(%lf %lf)  nper=(%d %d)  psum=%lf",
         tdata->priors[0], tdata->priors[1], tdata->nper[0], tdata->nper[1], psum ) ;
      MEMTEXT ( msg ) ;
#endif

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

   This normally returns 0.  It returns 1 if user pressed ESCape (before or
   after some weights were found) and -1 if insufficient memory.

--------------------------------------------------------------------------------
*/

static double basic_crit ( double sig ) ; // Local criterion to optimize
static TrainingSet *local_tptr ;  // These two statics pass class data
static PNNbasic *local_netptr ;   // to the local criterion routine

int PNNbasic::learn ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int k ;
   double x1, y1, x2, y2, x3, y3, accuracy ;
   char msg[84] ;

   memcpy ( lags , tptr->lags , n_inputs*sizeof(unsigned) ) ;
   if (output_mode == OUTMOD_MAPPING)
      memcpy ( leads , tptr->leads , n_outputs*sizeof(unsigned) ) ;

   if (tdata != NULL) {
      MEMTEXT ( "PNNbasic learn deleting tset" ) ;
      delete ( tdata ) ;
      tdata = NULL ;
      }

   MEMTEXT ( "PNNbasic new, copy tset" ) ;
   tdata = new TrainingSet ( output_mode , n_inputs , n_outputs , 0 , NULL ) ;
   if (tdata == NULL)
      return -1 ;

   *tdata = *tptr ;       // Invoke assignment operator to duplicate it

   if (! tdata->ntrain) { // Insufficient memory to copy?
      delete tdata ;
      tdata = NULL ;
      return -1 ;         // Return error flag
      }

   local_netptr = this ;
   local_tptr = tdata ;

   make_progress_window ( "PNN (BASIC) learning" ) ;

   if (errtype) { // If the network is already trained (errtype != 0) use sigma
      k = glob_min ( 0.9 * sigma , 1.1 * sigma , lptr->nsigs , 1 ,
            lptr->quit_err , basic_crit , &x1 , &y1 , &x2 , &y2 ,
            &x3 , &y3 , lptr->progress ) ;
      if (k)
         strcpy ( msg , "Interrupted by user" ) ;
      else 
         sprintf ( msg , "Global err at %.5lf = %.5lf", x2, y2 ) ;
      if (lptr->progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;
      }
   else {
      k = glob_min ( lptr->siglo , lptr->sighi , lptr->nsigs , 1 ,
            lptr->quit_err , basic_crit , &x1 , &y1 , &x2 , &y2 ,
            &x3 , &y3 , lptr->progress ) ;
      if (k)
         strcpy ( msg , "Interrupted by user" ) ;
      else 
         sprintf ( msg , "Global err at %.5lf = %.5lf", x2, y2 ) ;
      if (lptr->progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;
      }

   if (! k) { // If global was not interrupted by user ESCape before trio
      accuracy = pow ( 10.0 , -lptr->acc - lptr->refine ) ;
      y2 = brentmin ( 50 , lptr->quit_err , 1.e-12 , accuracy ,
                      basic_crit , &x1 , &x2 , &x3 , y2 , lptr->progress ) ;
      errtype = 1 ;          // Tell other routines net is trained
      }

   sigma = x2 ;
   neterr = fabs ( y2 ) ; // Brentmin returned neg if ESCape
   if (errtype) {
      sprintf ( msg , "Final error at %.6lf = %.6lf", sigma, neterr ) ;
      if (lptr->progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;
      }
   destroy_progress_window () ;
   if ((! errtype)  ||  (y2 < 0))
      return 1 ;
   return 0 ;
}

static double basic_crit ( double sig )
{
   local_netptr->sigma = sig ;
   return local_netptr->trial_error ( local_tptr , 0 ) ;
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

int PNNbasic::wt_print ( char *name )
{
   FILE *fp ;

   if ((fp = fopen ( name , "wt" )) == NULL)
      return 1 ;

   fprintf ( fp , "PNN ASCII weight file, sigma = %.5le", sigma ) ;

   if (ferror ( fp )) {
      fclose ( fp ) ;
      return 2 ;
      }
   fclose ( fp ) ;
   return 0 ;
}

int PNNbasic::wt_save ( FILE *fp )
{
   fwrite ( &tdata->ntrain , sizeof(unsigned) , 1 , fp ) ;
   fwrite ( &tdata->size , sizeof(unsigned) , 1 , fp ) ;
   if (output_mode == OUTMOD_CLASSIFICATION) {
      fwrite ( tdata->nper , n_outputs * sizeof(unsigned) , 1 , fp ) ;
      fwrite ( tdata->priors , n_outputs * sizeof(double) , 1 , fp ) ;
      }
   fwrite ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fwrite ( &sigma , sizeof(double) , 1 , fp ) ;
   if (ferror ( fp ))
      return 1 ;
   return 0 ;
}

int PNNbasic::wt_restore ( FILE *fp )
{
   MEMTEXT ( "PNNbasic wt_restore new tset" ) ;
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

   MEMTEXT ( "PNNbasic wt_restore alloc for data" ) ;
   tdata->data= (double *) MALLOC(tdata->ntrain * tdata->size * sizeof(double));
   if (tdata->data == NULL) {   // If insufficient memory
      delete tdata ;
      tdata = NULL ;
      return 4 ;
      }

   fread ( tdata->data , tdata->size * sizeof(double) , tdata->ntrain , fp ) ;
   fread ( &sigma , sizeof(double) , 1 , fp ) ;
   if (ferror ( fp )) {
      delete tdata ;
      tdata = NULL ;
      return 2 ;
      }
   return 0 ;
}

/******************************************************************************/
/*                                                                            */
/*  TESTNET - Test a trained neural network (MSE, confusion)                  */
/*                                                                            */
/*  This returns -1 if insufficient memory, 1 if user aborted, else zero.     */
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

static void jack ( int n , double *data , double *work ,
                   double *raw , double *bias , double *std ) ;
static void boot ( int n , double *data , double *work ,
                   double *raw , double *bias , double *std ) ;
static double userstat ( int n , double *data , double *work ) ;

int Network::testnet (
   TrainingSet *tptr ,  // Test set
   double threshold ,   // Threshold (0-1) for confusion
   int *confusion ,     // (nout+1) square confusion matrix (last r,c is reject)
   double *confuse ,    // Returns fraction (0-1) misclassified
   int extended ,       // Compute extended (jack/boot) statistics?
   TestNetResults *res  // All other test results for each output variable
   )
{
   int i, size, tset, tclass, ioutmax, ncases, user_quit ;
   double neuron_on, neuron_off, *dptr, t, diff, outmax, *x, val, *xptr ;
   double *work ;
   char msg[84] ;

   MEMTEXT ( "TESTNET: x" ) ;
   x = (double *) MALLOC ( tptr->ntrain * tptr->n_outputs * sizeof(double) ) ;
   if (x == NULL)
      return -1 ;

   if (extended) {
      MEMTEXT ( "TESTNET: work" ) ;
      work = (double *) MALLOC ( tptr->ntrain * sizeof(double) ) ;
      if (work == NULL) {
         FREE ( x ) ;
         return -1 ;
         }
      }

   if ((output_mode == OUTMOD_CLASSIFICATION)  &&  (confusion != NULL))
      memset ( confusion , 0 , (tptr->n_outputs+1) * (tptr->n_outputs+1) * sizeof(int)) ;
   *confuse = 0.0 ;  // Will return fraction error

   for (i=0 ; i<tptr->n_outputs ; i++)
      res[i].mse = res[i].mae = res[i].stderr_mse = res[i].stderr_mae = 0.0 ;

   if ((model == NETMOD_MLFN)  &&  ! ((MLFN *) this) -> outlin) {
      neuron_on = 0.9 * NEURON_ON ;
      neuron_off = 0.9 * NEURON_OFF ;
      }
   else {
      neuron_on = NEURON_ON ;
      neuron_off = NEURON_OFF ;
      }

   if (output_mode == OUTMOD_CLASSIFICATION) // Compute size of each case
      size = tptr->n_inputs + 1 ;
   else if (output_mode == OUTMOD_MAPPING)
      size = tptr->n_inputs + tptr->n_outputs ;

   user_quit = 0 ;
   ncases = tptr->ntrain ;

   make_progress_window ( "Network testing" ) ;

   for (tset=0 ; tset<ncases ; tset++) {  // Do all samples

      dptr = tptr->data + size * tset ;     // Point to this sample
      trial ( dptr ) ;                      // Evaluate network for it
      if (user_pressed_escape ()) {
         user_quit = 1 ;
         goto FINISH ;
         }

      if (output_mode == OUTMOD_CLASSIFICATION) {  // If this is Classification
         tclass = (int) dptr[tptr->n_inputs] - 1 ; // class is after inputs
         outmax = -1.0 ;
         ioutmax = 0 ;
         for (i=0 ; i<tptr->n_outputs ; i++) { // If COMPLEX, twice this->n_outputs!
            if (out[i] > outmax) {          // Keep track of max output
               outmax = out[i] ;            // For computing confusion matrix
               ioutmax = i ;                // This is which one
               }
            if (tclass == i)                // This the neuron for true class?
               t = neuron_on ;              // If so, ask for full activation
            else                            // For the other
               t = neuron_off ;             // Ask for no activation
            diff = out[i] - t ;
            x[i*ncases+tset] = diff ;       // Preserve error
            res[i].mse += diff * diff ;
            res[i].mae += fabs ( diff ) ;
            }

         t = (outmax - neuron_off) / (neuron_on - neuron_off) ; // Frac activ
         if (t < threshold)   // If activation does not meet user's expectation
            ioutmax = tptr->n_outputs ;  // This goes to reject category

         if (tclass < 0)            // If the true class is reject
            tclass = tptr->n_outputs ;   // Point to reject row

         if (tclass != ioutmax)
            *confuse += 1.0 ;

         if (confusion != NULL)
            ++confusion[tclass*(tptr->n_outputs+1)+ioutmax] ;
         }


      else if (output_mode == OUTMOD_MAPPING) {  // If this is MAPPING output
         dptr += tptr->n_inputs ;                // outputs stored after inputs
         for (i=0 ; i<tptr->n_outputs ; i++) {
            diff = out[i] - *dptr++ ;
            x[i*ncases+tset] = diff ;       // Preserve error
            res[i].mse += diff * diff ;
            res[i].mae += fabs ( diff ) ;
            }
         }

      if (! (tset % (1 + ncases / 10))) {
         sprintf ( msg , "%.2lf percent complete", 100.0 * tset / ncases ) ;
         write_non_progress ( msg ) ;
         }

      } // For all training cases

/*
--------------------------------------------------------------------------------

   All information from passing through the test set has been cumulated.
   Now compute the error measures.

--------------------------------------------------------------------------------
*/

   for (i=0 ; i<tptr->n_outputs ; i++) {
      res[i].mse /= ((double) ncases) ;
      res[i].mae /= ((double) ncases) ;
      }

   for (tset=0 ; tset<ncases ; tset++) {
      for (i=0 ; i<tptr->n_outputs ; i++) {
         val = x[i*ncases+tset] ;
         diff = val * val - res[i].mse ;
         res[i].stderr_mse += diff * diff ;
         diff = fabs ( val ) - res[i].mae ;
         res[i].stderr_mae += diff * diff ;
         }
      }

   for (i=0 ; i<tptr->n_outputs ; i++) {
      res[i].stderr_mse /= (((double) ncases) * ((double) (ncases-1))) ;
      res[i].stderr_mae /= (((double) ncases) * ((double) (ncases-1))) ;
      res[i].stderr_mse = sqrt ( res[i].stderr_mse ) ;
      res[i].stderr_mae = sqrt ( res[i].stderr_mae ) ;
      }

   for (i=0 ; i<tptr->n_outputs ; i++) {
      xptr = x+i*ncases ;
      qsort ( 0 , ncases-1 , xptr ) ;
      res[i].p5 = xptr[5 * ncases / 100] ;
      res[i].p10 = xptr[10 * ncases / 100] ;
      res[i].p25 = xptr[25 * ncases / 100] ;
      res[i].p75 = xptr[ncases - 25 * ncases / 100 - 1] ;
      res[i].p90 = xptr[ncases - 10 * ncases / 100 - 1] ;
      res[i].p95 = xptr[ncases - 5 * ncases / 100 - 1] ;
      if (extended) {
         jack ( ncases , xptr , work , &res[i].abs90 , &res[i].jackbias ,
                &res[i].jackstd ) ;
         boot ( ncases , xptr , work , &res[i].abs90 , &res[i].bootbias ,
                &res[i].bootstd ) ;
         }
      }

FINISH:
   destroy_progress_window () ;
   MEMTEXT ( "TESTNET: x" ) ;
   FREE ( x ) ;
   if (extended) {
      MEMTEXT ( "TESTNET: work" ) ;
      FREE ( work ) ;
      }

   *confuse /= ncases ;

   if (user_quit)
      return 1 ;
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   userstat - This computes the user's statistic
              As supplied, this returns the 90'th percentile.
              Users can substitute any statistic that depends on the
              distribution (but not on the data order!)

--------------------------------------------------------------------------------
*/

double userstat ( int n , double *data , double *work )
{
   int i ;

   if (work != data) // If caller cares that we mess with data, work area given
      memcpy ( work , data , n * sizeof(double) ) ;
   for (i=0 ; i<n ; i++)
      work[i] = fabs ( work[i] ) ;
   qsort ( 0 , n-1 , work ) ;
   return work[(int) (9 * (n-0.5) / 10)] ;
}


/*
--------------------------------------------------------------------------------

   jack - Compute the bias and standard deviation via jackknife

--------------------------------------------------------------------------------
*/

void jack (
   int n ,         // Number of data points
   double *data ,  // The data is here
   double *work ,  // Work vector (passed to userstat only)
   double *raw ,   // Raw (uncorrected) statistic for all n data
   double *bias ,  // Output of bias (subtract from raw to unbias)
   double *std     // Output of standard deviation of statistic
   )
{
   int exclude, first ;
   double stat, temp, sum, sumsq ;

   sum = sumsq = 0.0 ;
   exclude = n ;
   first = 1 ;

   while (exclude--) {
      if (first)
         first = 0 ;
      else {               // If not first trial, swap excluded to end
         temp = data[exclude] ;
         data[exclude] = data[n-1] ;
         data[n-1] = temp ;
         }

      stat = userstat ( n-1 , data , work ) ;

      sum += stat ;
      sumsq += stat * stat ;
      }

   sum /= n ;
   sumsq /= n ;
   sumsq -= sum * sum ;   // Can easily lose precision in this subtraction

   *raw = userstat ( n , data , work ) ;
   *bias = (n - 1.0) * (sum - *raw) ;
   if (sumsq > 0.0)
      *std = sqrt ( (n - 1.0) * sumsq ) ;
   else
      *std = 0.0 ;
}



/*
--------------------------------------------------------------------------------

   boot - Compute the bias and standard deviation via bootstrap

--------------------------------------------------------------------------------
*/

void boot (
   int n ,         // Number of data points
   double *data ,  // The data is here
   double *work ,  // Work vector
   double *raw ,   // Raw (uncorrected) statistic for all n data
   double *bias ,  // Output of bias (subtract from raw to unbias)
   double *std     // Output of standard deviation of statistic
   )
{
   int i, rep, sub ;
   int m = 200 ;
   double stat, sum, sumsq ;

   sum = sumsq = 0.0 ;

   for (rep=0 ; rep<m ; rep++) {
      for (i=0 ; i<n ; i++) {
         sub = unifrand() * n ;
         if (sub >= n)
            sub = n-1 ;
         work[i] = data[sub] ;
         }
      stat = userstat ( n , work , work ) ;
      sum += stat ;
      sumsq += stat * stat ;
      }

   sum /= m ;
   sumsq /= m ;
   sumsq -= sum * sum ;   // Can easily lose precision in this subtraction
   *raw = userstat ( n , data , work ) ;
   *bias = sum - *raw ;
   if (sumsq > 0.0)
      *std = sqrt ( sumsq * m / (m - 1.0) ) ;
   else
      *std = 0.0 ;
}

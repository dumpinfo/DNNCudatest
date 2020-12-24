/******************************************************************************/
/*                                                                            */
/*  MLFN - All principal routines for MLFN processing                         */
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

static void free_non_null ( void **p ) ;

/*
--------------------------------------------------------------------------------

   Constructor

   The public 'ok' is returned nonzero if all went well.  A value of zero
   tells the caller that there was insufficient memory.

   Note that we allocate only ONE array for ALL of the weights.
   The others (if any) are picked up at the appropriate place.
   The order is: HID1, HID2, OUTPUT.

--------------------------------------------------------------------------------
*/

MLFN::MLFN (
   char *netname ,
   NetParams *net_params ,
   int zero                 // Zero all weights? (Optional, default=1)
   )
   : Network ( netname , net_params )
{
   int n1, n2, n3 ;

   if (! ok)    // Did the parent constructor fail?
      return ;  // If so, nothing to do here

   MEMTEXT ( "MLFN constructor" ) ;

   outlin = net_params->linear ;
   domain = net_params->domain ;
   nhid1 = net_params->n_hidden1 ;
   nhid2 = net_params->n_hidden2 ;

   if (domain == DOMAIN_REAL) {
      nin_w = n_inputs ;
      nin_n = n_inputs + 1 ;
      nhid1_w = nhid1 ;
      nhid1_n = nhid1  ?  (nhid1 + 1)  :  0 ;
      nhid2_w = nhid2 ;
      nhid2_n = nhid2  ?  (nhid2 + 1)  :  0 ;
      nout_n = n_outputs ;
      }
   else if (domain == DOMAIN_COMPLEX) {
      n_inputs /= 2 ;  // Constructor was called with n of inputs, not neurons
      n_outputs /= 2 ; // Ditto for outputs
      nin_w = 2 * n_inputs ;
      nin_n = 2 * n_inputs + 2 ;
      nhid1_w = 2 * nhid1 ;
      nhid1_n = nhid1  ?  (2 * nhid1 + 2)  :  0 ;
      nhid2_w = 2 * nhid2 ;
      nhid2_n = nhid2  ?  (2 * nhid2 + 2)  :  0 ;
      nout_n = 2 * n_outputs ;
      }
   else if (domain == DOMAIN_COMPLEX_INPUT) {
      n_inputs /= 2 ;  // Constructor was called with n of inputs, not neurons
      nin_w = 2 * n_inputs ;
      nin_n = 2 * n_inputs + 2 ;
      nhid1_w = nhid1 ;
      nhid1_n = nhid1  ?  (nhid1 + 1)  :  0 ;
      nhid2_w = nhid2 ;
      nhid2_n = nhid2  ?  (nhid2 + 1)  :  0 ;
      nout_n = n_outputs ;
      }
   else if (domain == DOMAIN_COMPLEX_HIDDEN) {
      n_inputs /= 2 ;  // Constructor was called with n of inputs, not neurons
      nin_w = 2 * n_inputs ;
      nin_n = 2 * n_inputs + 2 ;
      nhid1_w = 2 * nhid1 ;
      nhid1_n = nhid1  ?  (2 * nhid1 + 2)  :  0 ;
      nhid2_w = 2 * nhid2 ;
      nhid2_n = nhid2  ?  (2 * nhid2 + 2)  :  0 ;
      nout_n = n_outputs ;
      }

   hid1_coefs = hid2_coefs = out_coefs = hid1 = hid2 = NULL ;

   ok = 0 ;   // Indicates failure of malloc (What a pessimist!)

   if (nhid1 == 0) {                // No hidden layer
      n1 = n_outputs * nin_n ;
      if ((out_coefs = (double *) MALLOC ( n1 * sizeof(double) )) == NULL) {
         free_non_null ( (void **) &out_coefs ) ;
         return ;
         }
      all_weights = out_coefs ;
      nhid = 0 ;
      ntot = n1 ;
      if (zero) {
         while (n1--)
            out_coefs[n1] = 0.0 ;
         }
      }

   else if (nhid2 == 0) {           // One hidden layer
      n1 = n_outputs * nhid1_n ;         // Number of output weights
      n2 = nhid1 * nin_n ;          // And hidden layer weights
      if (((hid1_coefs = (double *) MALLOC ((n1+n2) * sizeof(double))) == NULL)
       || ((hid1 = (double *) MALLOC ( nhid1_w * sizeof(double) ))==NULL)) {
         free_non_null ( (void **) &hid1_coefs ) ;
         free_non_null ( (void **) &hid1 ) ;
         return ;
         }
      all_weights = hid1_coefs ;
      out_coefs = hid1_coefs + n2 ;
      nhid = n2 ;
      ntot = n1 + nhid ;
      if (zero) {
         while (n1--)
            out_coefs[n1] = 0.0 ;
         while (n2--)
            hid1_coefs[n2] = 0.0 ;
         }
      }

   else {                           // Two hidden layers
      n1 = n_outputs * nhid2_n ;    // Number of output layer weights
      n2 = nhid2 * nhid1_n ;        // And second hidden layer
      n3 = nhid1 * nin_n ;          // And first hidden layer
      if (((hid1_coefs = (double *) MALLOC((n1+n2+n3) * sizeof(double)))== NULL)
       || ((hid1 = (double *) MALLOC ( nhid1_w * sizeof(double) ))==NULL)
       || ((hid2 = (double *) MALLOC ( nhid2_w * sizeof(double) ))==NULL)) {
         free_non_null ( (void **) &hid1_coefs ) ;
         free_non_null ( (void **) &hid1 ) ;
         free_non_null ( (void **) &hid2 ) ;
         return ;
         }
      all_weights = hid1_coefs ;
      hid2_coefs = hid1_coefs + n3 ;
      out_coefs = hid2_coefs + n2 ;
      nhid = n2 + n3 ;
      ntot = n1 + nhid ;
      if (zero) {
         while (n1--)
            out_coefs[n1] = 0.0 ;
         while (n2--)
            hid2_coefs[n2] = 0.0 ;
         while (n3--)
            hid1_coefs[n3] = 0.0 ;
         }
      }

   ok = 1 ;            // Indicate to caller that all mallocs succeeded
}

/*
   Free memory if not a NULL pointer
*/

static void free_non_null ( void **p )
{
   if (*p) {
      FREE ( *p ) ;
      *p = NULL ;
      }
}


/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

MLFN::~MLFN()
{
   MEMTEXT ( "MLFN destructor" ) ;
   if (! ok)    // If constructor's mallocs failed
      return ;  // there is nothing to free

   if (all_weights != NULL)
      FREE ( all_weights ) ;
   if (hid1 != NULL)
      FREE ( hid1 ) ;
   if (hid2 != NULL)
      FREE ( hid2 ) ;
}

/*
--------------------------------------------------------------------------------

   trial - Compute the output for a given input by evaluating network

--------------------------------------------------------------------------------
*/

int MLFN::trial ( double *input )
{
   int i ;
   
   if (nhid1 == 0) {                // No hidden layer
      switch (domain) {
         case DOMAIN_REAL:
            for (i=0 ; i<n_outputs ; i++)
               activity_rr ( input , out_coefs+i*nin_n , out+i , n_inputs , outlin );
            break ;
         case DOMAIN_COMPLEX:
            for (i=0 ; i<n_outputs ; i++)
               activity_cc ( input , out_coefs+i*nin_n , out+2*i , n_inputs, outlin);
            break ;
         case DOMAIN_COMPLEX_INPUT:
            for (i=0 ; i<n_outputs ; i++)
               activity_cr ( input , out_coefs+i*nin_n , out+i , n_inputs , outlin );
            break ;
         } // Switch on domain
      }

   else if (nhid2 == 0) {           // One hidden layer
      switch (domain) {
         case DOMAIN_REAL:
            for (i=0 ; i<nhid1 ; i++)
               activity_rr ( input , hid1_coefs+i*nin_n , hid1+i , n_inputs , 0 ) ;
            for (i=0 ; i<n_outputs ; i++)
               activity_rr ( hid1 , out_coefs+i*nhid1_n , out+i, nhid1, outlin);
            break ;
         case DOMAIN_COMPLEX:
            for (i=0 ; i<nhid1 ; i++)
               activity_cc ( input , hid1_coefs+i*nin_n , hid1+2*i , n_inputs , 0 ) ;
            for (i=0 ; i<n_outputs ; i++)
               activity_cc ( hid1, out_coefs+i*nhid1_n, out+2*i, nhid1, outlin);
            break ;
         case DOMAIN_COMPLEX_INPUT:
            for (i=0 ; i<nhid1 ; i++)
               activity_cr ( input , hid1_coefs+i*nin_n , hid1+i , n_inputs , 0 ) ;
            for (i=0 ; i<n_outputs ; i++)
               activity_rr ( hid1 , out_coefs+i*nhid1_n , out+i, nhid1, outlin);
            break ;
         case DOMAIN_COMPLEX_HIDDEN:
            for (i=0 ; i<nhid1 ; i++)
               activity_cc ( input , hid1_coefs+i*nin_n , hid1+2*i , n_inputs , 0 ) ;
            for (i=0 ; i<n_outputs ; i++)
               activity_cr ( hid1 , out_coefs+i*nhid1_n , out+i, nhid1, outlin);
            break ;
         } // Switch on domain
      }

   else {                           // Two hidden layers
      switch (domain) {
         case DOMAIN_REAL:
            for (i=0 ; i<nhid1 ; i++)
               activity_rr ( input , hid1_coefs+i*nin_n , hid1+i , n_inputs , 0 ) ;
            for (i=0 ; i<nhid2 ; i++)
               activity_rr ( hid1 , hid2_coefs+i*nhid1_n , hid2+i , nhid1 , 0 );
            for (i=0 ; i<n_outputs ; i++)
               activity_rr ( hid2 , out_coefs+i*nhid2_n , out+i, nhid2, outlin);
            break ;
         case DOMAIN_COMPLEX:
            for (i=0 ; i<nhid1 ; i++)
               activity_cc ( input , hid1_coefs+i*nin_n , hid1+2*i , n_inputs , 0 ) ;
            for (i=0 ; i<nhid2 ; i++)
               activity_cc ( hid1 , hid2_coefs+i*nhid1_n , hid2+2*i , nhid1, 0);
            for (i=0 ; i<n_outputs ; i++)
               activity_cc ( hid2, out_coefs+i*nhid2_n, out+2*i, nhid2, outlin);
            break ;
         case DOMAIN_COMPLEX_INPUT:
            for (i=0 ; i<nhid1 ; i++)
               activity_cr ( input , hid1_coefs+i*nin_n , hid1+i , n_inputs , 0 ) ;
            for (i=0 ; i<nhid2 ; i++)
               activity_rr ( hid1 , hid2_coefs+i*nhid1_n , hid2+i , nhid1 , 0 );
            for (i=0 ; i<n_outputs ; i++)
               activity_rr ( hid2 , out_coefs+i*nhid2_n , out+i, nhid2, outlin);
            break ;
         case DOMAIN_COMPLEX_HIDDEN:
            for (i=0 ; i<nhid1 ; i++)
               activity_cc ( input , hid1_coefs+i*nin_n , hid1+2*i , n_inputs , 0 ) ;
            for (i=0 ; i<nhid2 ; i++)
               activity_cc ( hid1 , hid2_coefs+i*nhid1_n , hid2+2*i , nhid1, 0);
            for (i=0 ; i<n_outputs ; i++)
               activity_cr ( hid2 , out_coefs+i*nhid2_n , out+i, nhid2, outlin);
            break ;
         } // Switch on domain
      }
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   trial_error - Compute the mean square error for the entire training set

--------------------------------------------------------------------------------
*/

double MLFN::trial_error ( TrainingSet *tptr )
{
   int i, size, casenum, true_class ;
   double err, tot_err, *inptr, diff, dsq, prev, denom, t, x, xx ;
   double neuron_on, neuron_off ;

   if (outlin  &&  (errtype != ERRTYPE_XENT)  &&  (errtype != ERRTYPE_KK)) {
      neuron_on = NEURON_ON ;
      neuron_off = NEURON_OFF ;
      }
   else {
      neuron_on = 0.9 * NEURON_ON ;
      neuron_off = 0.9 * NEURON_OFF ;
      }

   if (output_mode == OUTMOD_CLASSIFICATION) // Compute size of each training sample
      size = tptr->n_inputs + 1 ;
   else if (output_mode == OUTMOD_MAPPING)
      size = tptr->n_inputs + tptr->n_outputs ;

   tot_err = 0.0 ;  // Total error will be cumulated here
   prev = 0.0 ;     // Shuts up compilers about 'use before defined'

   for (casenum=0 ; casenum<tptr->ntrain ; casenum++) {  // Do all samples

      inptr = tptr->data + size * casenum ;  // This case in training set set
      trial ( inptr ) ;                      // Execute network
      err = 0.0 ;

      if (output_mode == OUTMOD_CLASSIFICATION) {  // Classification output mode?
         true_class = (int) inptr[tptr->n_inputs] - 1 ; // Class is after inputs
         for (i=0 ; i<n_outputs ; i++) {
            if (true_class == i)
               t = neuron_on ;
            else
               t = neuron_off ;
            diff = t - out[i] ;
            switch (errtype) {
               case ERRTYPE_MSE:            // Sum for both types
                  dsq = diff * diff ;
                  err += dsq ;
                  break ;
               case ERRTYPE_ABS:
                  err += fabs(diff) ;
                  break ;
               case ERRTYPE_KK:
                  dsq = diff * diff ;
                  denom = 1.0 - out[i] * out[i] ; // SHOULD never be negative
                  if (denom < 1.e-10)
                     denom = 1.e-10 ;
                  err += dsq / denom ;
                  break ;
               case ERRTYPE_XENT:
                  xx = out[i] ;
                  if (xx < -.99999999)
                     xx = -.99999999 ;
                  if (xx > .99999999)
                     xx = .99999999 ;
                  t = 0.5 * (t + 1.0) ;
                  x = 0.5 * (xx + 1.0) ;
                  err += t * log ( t/x ) + (1.0 - t) * log ((1.0-t) / (1.0-x));
                  break ;
               } // Switch errtype
            } // For all outputs
         } // If OUTMOD_CLASSIFICATION

      else if (output_mode == OUTMOD_MAPPING) {  // MAPPING output mode?
         inptr += tptr->n_inputs ;               // Outputs after inputs
         for (i=0 ; i<nout_n ; i++) {
            diff = *inptr++ - out[i] ;
            switch (errtype) {
               case ERRTYPE_MSE:            // Sum for both types
                  dsq = diff * diff ;
                  err += dsq ;
                  break ;
               case ERRTYPE_ABS:
                  if (domain == DOMAIN_COMPLEX) { // Sqrt ( r*r + i*i )
                     dsq = diff * diff ;
                     if (i % 2)
                        err += sqrt ( dsq + prev ) ;
                     else
                        prev = dsq ;
                     }
                  else
                     err += fabs(diff) ;
                  break ;
               case ERRTYPE_KK:
                  dsq = diff * diff ;
                  if (domain == DOMAIN_COMPLEX) {
                     if (i % 2) {
                        denom = 1.0 - out[i] * out[i] - out[i-1] * out[i-1] ;
                        if (denom < 1.e-10) // SHOULD never be negative
                           denom = 1.e-10 ;
                        err += (dsq + prev) / denom ;
                        }
                     else
                        prev = dsq ;
                     }
                  else {
                     denom = 1.0 - out[i] * out[i] ; // SHOULD never be negative
                     if (denom < 1.e-10)
                        denom = 1.e-10 ;
                     err += dsq / denom ;
                     }
                  break ;
               } // Switch errtype
            } // For all outputs
         } // If OUTMOD_MAPPING

      tot_err += err ;
      } // for all casenums

/*
   Find the mean per presentation.  Also, compensate for n_outputs.
*/

   neterr = tot_err / ((double) tptr->ntrain  *  (double) nout_n) ;
   return neterr ;
}

/*
--------------------------------------------------------------------------------

   learn

--------------------------------------------------------------------------------
*/

int MLFN::learn ( TrainingSet *tptr , struct LearnParams *lptr )
{

   memcpy ( lags , tptr->lags , tptr->n_inputs*sizeof(unsigned) ) ;
   if (output_mode == OUTMOD_MAPPING)
      memcpy ( leads , tptr->leads , tptr->n_outputs*sizeof(unsigned) ) ;

   this->errtype = lptr->errtype ;  // Tell net routines what our error def is
   if ((lptr->method == METHOD_AN1)  ||  (lptr->method == METHOD_AN2))
      return anx ( tptr , lptr ) ;
   else if (lptr->method == METHOD_SS)
      return ssg ( tptr , lptr , 0 ) ;
   else if (lptr->method == METHOD_SSG)
      return ssg ( tptr , lptr , 1 ) ;
   else if ((lptr->method == METHOD_AN1_CJ) || (lptr->method == METHOD_AN2_CJ)
         || (lptr->method == METHOD_AN1_LM) || (lptr->method == METHOD_AN2_LM))
      return anx_dd ( tptr , lptr ) ;
   else if ((lptr->method==METHOD_REGRS_CJ) || (lptr->method==METHOD_REGRS_LM))
      return regrs_dd ( tptr , lptr ) ;

   return 0 ;
}

/*
--------------------------------------------------------------------------------

   wt_print - Print weights as ASCII to file
     Returns:
         0 - Normal
         1 - Unable to open file
         2 - Unable to write file

   wt_save - Save weights to disk (called from WT_SAVE.CPP)
   wt_restore - Restore weights from disk (called from WT_SAVE.CPP)

--------------------------------------------------------------------------------
*/

int MLFN::wt_print ( char *name )
{
   int i, j, k ;
   FILE *fp ;

   if ((fp = fopen ( name , "wt" )) == NULL)
      return 1 ;

   fprintf ( fp , "MLFN ASCII weight file" ) ;

   if (nhid1 == 0) {                // No hidden layer
      k = 0 ;
      for (i=0 ; i<n_outputs ; i++) {
         fprintf ( fp , "\nInput-to-Output neuron %d weights, bias last:", i+1);
         if (domain == DOMAIN_REAL) {
            for (j=0 ; j<=n_inputs ; j++)
               fprintf ( fp , "\n%2d: %lf", j+1, out_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=n_inputs ; j++) {
               fprintf ( fp , "\n%2d: (%lf %lf)", j+1,
                              out_coefs[k], out_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      }

   else if (nhid2 == 0) {           // One hidden layer
      k = 0 ;
      for (i=0 ; i<nhid1 ; i++) {
         fprintf ( fp , "\nInput-to-Hid1 neuron %d weights, bias last:", i+1 ) ;
         if (domain == DOMAIN_REAL) {
            for (j=0 ; j<=n_inputs ; j++)
               fprintf ( fp , "\n%2d: %lf", j+1, hid1_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=n_inputs ; j++) {
               fprintf ( fp , "\n%2d: (%lf %lf)",
                              j+1, hid1_coefs[k], hid1_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      k = 0 ;
      for (i=0 ; i<n_outputs ; i++) {
         fprintf ( fp , "\nHid1-to-Output neuron %d weights, bias last:", i+1 );
         if ((domain == DOMAIN_REAL)  ||  (domain == DOMAIN_COMPLEX_INPUT)) {
            for (j=0 ; j<=nhid1 ; j++)
               fprintf ( fp , "\n%2d: %lf", j+1, out_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=nhid1 ; j++) {
               fprintf ( fp , "\n%2d: (%lf %lf)",
                              j+1, out_coefs[k], out_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      }

   else {                           // Two hidden layers
      k = 0 ;
      for (i=0 ; i<nhid1 ; i++) {
         fprintf ( fp , "\nInput-to-Hid1 neuron %d weights, bias last:", i+1 ) ;
         if (domain == DOMAIN_REAL) {
            for (j=0 ; j<=n_inputs ; j++)
               fprintf ( fp , "\n%2d: %lf", j+1, hid1_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=n_inputs ; j++) {
               fprintf ( fp , "\n%2d: (%lf %lf)",
                              j+1, hid1_coefs[k], hid1_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      k = 0 ;
      for (i=0 ; i<nhid2 ; i++) {
         fprintf ( fp , "\nHid1-to-Hid2 neuron %d weights, bias last:", i+1 ) ;
         if ((domain == DOMAIN_REAL)  ||  (domain == DOMAIN_COMPLEX_INPUT)) {
            for (j=0 ; j<=nhid1 ; j++)
               fprintf ( fp , "\n%2d: %lf", j+1, hid2_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=nhid1 ; j++) {
               fprintf ( fp , "\n%2d: (%lf %lf)",
                              j+1, hid2_coefs[k], hid2_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      k = 0 ;
      for (i=0 ; i<n_outputs ; i++) {
         fprintf ( fp , "\nHid2-to-Output neuron %d weights, bias last:", i+1 );
         if ((domain == DOMAIN_REAL)  ||  (domain == DOMAIN_COMPLEX_INPUT)) {
            for (j=0 ; j<=nhid2 ; j++)
               fprintf ( fp , "\n%2d: %lf", j+1, out_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=nhid2 ; j++) {
               fprintf ( fp , "\n%2d: (%lf %lf)",
                              j+1, out_coefs[k], out_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      }

   if (ferror ( fp )) {
      fclose ( fp ) ;
      return 2 ;
      }

   fclose ( fp ) ;
   return 0 ;
}

int MLFN::wt_save ( FILE *fp )
{
   fwrite ( all_weights , ntot * sizeof(double) , 1 , fp ) ;

   if (ferror ( fp ))
      return 1 ;

   return 0 ;
}

int MLFN::wt_restore ( FILE *fp )
{
   fread ( all_weights , ntot * sizeof(double) , 1 , fp ) ;

   if (ferror ( fp ))
      return 2 ;
   return 0 ;
}

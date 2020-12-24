/******************************************************************************/
/*                                                                            */
/*  REGRESS - Use linear regression to compute MLFN output weights            */
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
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

double MLFN::regress (
   TrainingSet *tptr ,
   SingularValueDecomp *sptr
   )

{
   int i, out, casenum, size, nvars, is_complex ;
   double *aptr, *bptr, *inptr, err, diff, cdiff[2] ;
   double neuron_on, neuron_off ;

   if (outlin  &&  (errtype != ERRTYPE_XENT)  &&  (errtype != ERRTYPE_KK)) {
      neuron_on = NEURON_ON ;
      neuron_off = NEURON_OFF ;
      }
   else {
      neuron_on = 0.9 * NEURON_ON ;
      neuron_off = 0.9 * NEURON_OFF ;
      }

/*
   The variable is_complex is 0 if both input and output are real,
   one if input complex and output real, two if both complex.
*/

   if (domain == DOMAIN_REAL)
      is_complex = 0 ;
   else if (domain == DOMAIN_COMPLEX_INPUT)
      is_complex = nhid1  ?  0 : 1 ;
   else if (domain == DOMAIN_COMPLEX_HIDDEN)
      is_complex = 1 ;
   else 
      is_complex = 2 ;

/*
   Find the number of variables in each case.
*/

   if (output_mode == OUTMOD_CLASSIFICATION)
      size = tptr->n_inputs + 1 ;
   else if (output_mode == OUTMOD_MAPPING)
      size = tptr->n_inputs + tptr->n_outputs ;

/*
   Find the number of weights to compute.
*/

   if (nhid1 == 0)         // No hidden layer
      nvars = nin_n ;
   else if (nhid2 == 0)    // One hidden layer
      nvars = nhid1_n ;
   else                    // Two hidden layers
      nvars = nhid2_n ;

/*
   Traverse the training set, building the design matrix of activations
   prior to the output layer, then find its singular value decomposition.
   We need to preserve 'a' so that we can compute the error at the end.
   If we are mapping real-to-real, the rows are straight copies of activations.
   Otherwise, recall that (a+bi) * (w+vi) = (aw-bv) + (av+bw) i.
   Thus, if mapping complex-to-real, we switch the sign of every other input
   to take care of the real part of the product, and we ignore the imaginary.
   If complex-to-complex, we must also take care of the imaginary part by
   adding another equation.
*/

   aptr = sptr->a ;                     // Will build matrix here

   for (casenum=0 ; casenum<tptr->ntrain ; casenum++) { // Do all cases

      inptr = tptr->data + casenum * size ; // Point to this sample

      if (nhid1 == 0) {                 // No hidden layer, so matrix is inputs
         if (is_complex == 0) {
            for (i=0 ; i<tptr->n_inputs ; i++)
               *aptr++ = *inptr++ ;
            *aptr++ = 1.0 ;             // Bias goes in last column
            }
         else {
            for (i=0 ; i<tptr->n_inputs ; i+=2) {  // Predicting real part
               *aptr++ = *inptr++ ;
               *aptr++ = -*inptr++ ;
               }
            *aptr++ = 1.0 ;
            *aptr++ = 0.0 ;
            if (is_complex == 2) {      // Must also predict imaginary part
               inptr -= tptr->n_inputs ;
               for (i=0 ; i<tptr->n_inputs ; i+=2) {
                  *(aptr+1) = *inptr++ ;
                  *aptr = *inptr++ ;
                  aptr += 2 ;
                  }
               *aptr++ = 0.0 ;
               *aptr++ = 1.0 ;
               }
            }
         }

      else if (nhid2 == 0) {            // One hidden layer
         switch (domain) {
            case DOMAIN_REAL:
               for (i=0 ; i<nhid1 ; i++) { // so matrix is hidden1 activations
                  activity_rr ( inptr , hid1_coefs+i*nin_n , aptr , n_inputs , 0 ) ;
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;             // Bias term is last column of matrix
               break ;
            case DOMAIN_COMPLEX_INPUT:
               for (i=0 ; i<nhid1 ; i++) {
                  activity_cr ( inptr , hid1_coefs+i*nin_n , aptr , n_inputs , 0 ) ;
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;
               break ;
            case DOMAIN_COMPLEX_HIDDEN:
               for (i=0 ; i<nhid1 ; i++) {
                  activity_cc ( inptr , hid1_coefs+i*nin_n , aptr , n_inputs , 0 ) ;
                  ++aptr ;
                  *aptr = -*aptr ;
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;
               *aptr++ = 0.0 ;
               break ;
            case DOMAIN_COMPLEX:
               for (i=0 ; i<nhid1 ; i++) {
                  activity_cc ( inptr , hid1_coefs+i*nin_n , aptr , n_inputs , 0 ) ;
                  *(aptr+nvars) = *(aptr+1) ; // Next equation which predicts
                  *(aptr+nvars+1) = *aptr ;   // the imaginary part
                  ++aptr ;                    // This is the real part
                  *aptr = -*aptr ;            // of the prediction
                  ++aptr ;
                  }
               *(aptr+nvars) = 0.0 ;          // Bias terms for both equations
               *aptr++ = 1.0 ;
               *(aptr+nvars) = 1.0 ;
               *aptr++ = 0.0 ;
               aptr += nvars ;                // Bypass imaginary equation
               break ;
            } // Switch on domain
         }

      else {                            // Two hidden layers
         switch (domain) {
            case DOMAIN_REAL:
               for (i=0 ; i<nhid1 ; i++)
                  activity_rr ( inptr , hid1_coefs+i*nin_n , hid1+i , n_inputs , 0 );
               for (i=0 ; i<nhid2 ; i++) {
                  activity_rr ( hid1 , hid2_coefs+i*nhid1_n , aptr , nhid1 , 0);
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;          // Bias term is last column of matrix
               break ;
            case DOMAIN_COMPLEX_INPUT:
               for (i=0 ; i<nhid1 ; i++)
                  activity_cr ( inptr , hid1_coefs+i*nin_n , hid1+i , n_inputs , 0 );
               for (i=0 ; i<nhid2 ; i++) {
                  activity_rr ( hid1 , hid2_coefs+i*nhid1_n , aptr , nhid1 , 0);
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;          // Bias term is last column of matrix
               break ;
            case DOMAIN_COMPLEX_HIDDEN:
               for (i=0 ; i<nhid1 ; i++)
                  activity_cc ( inptr , hid1_coefs+i*nin_n , hid1+2*i , n_inputs, 0);
               for (i=0 ; i<nhid2 ; i++) {
                  activity_cc ( hid1 , hid2_coefs+i*nhid1_n , aptr , nhid1 , 0);
                  ++aptr ;
                  *aptr = -*aptr ;
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;
               *aptr++ = 0.0 ;
               break ;
            case DOMAIN_COMPLEX:
               for (i=0 ; i<nhid1 ; i++)
                  activity_cc ( inptr , hid1_coefs+i*nin_n , hid1+2*i , n_inputs, 0);
               for (i=0 ; i<nhid2 ; i++) {
                  activity_cc ( hid1 , hid2_coefs+i*nhid1_n , aptr , nhid1 , 0);
                  *(aptr+nvars) = *(aptr+1) ; // Next equation which predicts
                  *(aptr+nvars+1) = *aptr ;   // the imaginary part
                  ++aptr ;                    // This is the real part
                  *aptr = -*aptr ;            // of the prediction
                  ++aptr ;
                  }
               *(aptr+nvars) = 0.0 ;          // Bias terms for both equations
               *aptr++ = 1.0 ;
               *(aptr+nvars) = 1.0 ;
               *aptr++ = 0.0 ;
               aptr += nvars ;                // Bypass imaginary equation
               break ;
            } // Switch on domain
         } // Two layer net
      } // For each training sample


/*
   Compute the SVD of the design matrix a.
   Call backsub for each output to find the weights.
   Also compute the error for that optimal output.
*/

   sptr->svdcmp () ;

   err = 0.0 ;

   for (out=0 ; out<nout_n ; out++) {  // For each output neuron

      if (out % 2  &&  (domain == DOMAIN_COMPLEX)) // We did the imaginary part
         continue ;                                // when we did the real part

      bptr = sptr->b ;               // Backsub requires RHS to be placed here

/*
   This pass through the training set computes the desired net inputs to
   the output layer by applying the inverse activation function to the
   target outputs.
*/

      for (casenum=0 ; casenum<tptr->ntrain ; casenum++) {

         inptr = tptr->data + casenum * size ;     // This case

         if (output_mode == OUTMOD_CLASSIFICATION) {    // If this is Classification
            if ((int) inptr[tptr->n_inputs] == out+1) { // class ID past inputs
               if (outlin)
                  *bptr++ = neuron_on ;
               else 
                  *bptr++ = inverse_act ( neuron_on ) ;
               }
            else {
               if (outlin)
                  *bptr++ = neuron_off ;
               else 
                  *bptr++ = inverse_act ( neuron_off ) ;
               }
            }

         else if (output_mode == OUTMOD_MAPPING) { // If this is MAPPING output
            if (domain == DOMAIN_COMPLEX) {
               if (outlin) {
                  bptr[0] = inptr[tptr->n_inputs+out] ;
                  bptr[1] = inptr[tptr->n_inputs+out+1] ;
                  }
               else 
                  inverse_act_cc ( inptr+tptr->n_inputs+out , bptr ) ;
               bptr += 2 ;
               }
            else {
               if (outlin)
                  *bptr++ = inptr[tptr->n_inputs+out] ;
               else 
                  *bptr++ = inverse_act ( inptr[tptr->n_inputs+out] ) ;
               }
            }
         } // For all training samples

/*
   Find the output weights by back-substitution
*/

      if (domain == DOMAIN_COMPLEX)
         bptr = out_coefs + out * nvars / 2 ; // Out goes to 2*n_outputs if complex
      else 
         bptr = out_coefs + out * nvars ;

      sptr->backsub ( 1.e-8 , bptr ) ;

/*
   We have just computed the (linearly) optimal weights for this output.
   Cumulate the epoch error for this output.
   We must also reverse the signs of the imaginary parts of the activations
   because we reversed them before storing.  If this is a full complex model,
   every other row of the activation matrix is for the imaginary part of the
   prediction, and can be ignored because it is redundant.
   If the error type is anything but MSE we cannot do this.  Just skip it
   and call trial_error.
*/

      if (errtype != ERRTYPE_MSE)
         continue ;

      for (casenum=0 ; casenum<tptr->ntrain ; casenum++) {// Epoch for this output

         inptr = tptr->data + size * casenum ;    // This case

         if (is_complex == 2)                 // Point to inputs to output layer
            aptr = sptr->a + 2 * casenum * nvars ; // Imaginary row is redundant
         else 
            aptr = sptr->a + casenum * nvars ;

         if (is_complex  &&  ! out) {         // Get original activations
            for (i=0 ; i<nvars ; i++) {       // by flipping imaginary part
               if (i % 2)
                  aptr[i] = -aptr[i] ;
               }
            }

         if (is_complex == 0)
            activity_rr ( aptr , bptr , &diff , nvars-1 , outlin ) ;
         else if (is_complex == 1)
            activity_cr ( aptr , bptr , &diff , (nvars-2)/2 , outlin ) ;
         else
            activity_cc ( aptr , bptr , cdiff , (nvars-2)/2 , outlin ) ;

         if (output_mode == OUTMOD_CLASSIFICATION) {   // If this is Classification
            if ((int) inptr[tptr->n_inputs] == out+1)  // class ID past inputs
               diff -= neuron_on ;
            else 
               diff -= neuron_off ;
            err += diff * diff ;
            }

         else if (output_mode == OUTMOD_MAPPING) {    // If MAPPING output past input
            if (is_complex == 2) {
               cdiff[0] -= inptr[tptr->n_inputs+out] ;   // Real part of output
               cdiff[1] -= inptr[tptr->n_inputs+out+1] ; // And imaginary
               err += cdiff[0] * cdiff[0]  +  cdiff[1] * cdiff[1] ;
               }
            else {
               diff -= inptr[tptr->n_inputs+out] ;
               err += diff * diff ;
               }
            }
         } // For all casenums
      } // For each output

   if (errtype != ERRTYPE_MSE)
      neterr = trial_error ( tptr ) ; // Did we skip above error computation?
   else 
      neterr = err / ((double) tptr->ntrain * (double) nout_n) ;

   return neterr ;
}


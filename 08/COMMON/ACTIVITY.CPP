/******************************************************************************/
/*                                                                            */
/*  ACTIVITY - Evaluate the activity of a single MLFN neuron                  */
/*                                                                            */
/*  This module also contains routines which compute the partial derivatives  */
/*  of the activations with respect to real and imaginary parts.              */
/*  It also has a routine which computes the inverse of a complex-valued      */
/*  activation function.                                                      */
/*                                                                            */
/*  This contains several versions, according to whether the inputs and       */
/*  outputs are real or complex.                                              */
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

void activity_rr ( double *input , double *coefs , double *output , int ninputs,
                   int linear )
{
   double sum ;

   sum = dotprod ( ninputs , input , coefs ) ;
   sum += coefs[ninputs] ;      // Bias term

   if (linear)
      *output = sum ;
   else 
      *output = act_func ( sum ) ;
}

void activity_cr ( double *input , double *coefs , double *output , int ninputs,
                   int linear )
{
   double sum ;

   sum = dotprodcr ( ninputs , input , coefs ) ;
   sum += coefs[2*ninputs] ;      // Bias term (ignore imaginary part of bias)

   if (linear)
      *output = sum ;
   else 
      *output = act_func ( sum ) ;
}

void activity_cc ( double *input , double *coefs , double *output , int ninputs,
                   int linear )
{
   double rsum, isum, raw_length, squashed_length, f ;

   dotprodc ( ninputs , input , coefs , &rsum , &isum ) ;
   rsum += coefs[2*ninputs] ;      // Bias term
   isum += coefs[2*ninputs+1] ;

   if (linear) {
      *output = rsum ;
      *(output+1) = isum ;
      return ;
      }

   raw_length = sqrt ( rsum*rsum + isum*isum ) + 1.e-30 ;
   squashed_length = act_func ( raw_length ) ;
   f = squashed_length / raw_length ;

   *output = rsum * f ;
   *(output+1) = isum * f ;
}

/*
--------------------------------------------------------------------------------

   This is the inverse of the above function

--------------------------------------------------------------------------------
*/

void inverse_act_cc ( double *out , double *net )
{
   double raw_length, squashed_length, f ;

   squashed_length = sqrt ( out[0] * out[0]  +  out[1] * out[1] ) + 1.e-30 ;
   raw_length = inverse_act ( squashed_length ) ;
   f = raw_length / squashed_length ;

   net[0] = f * out[0] ;
   net[1] = f * out[1] ;
}

/*
--------------------------------------------------------------------------------

   These are the partial derivatives of the above functions with respect to
   the real and imaginary parts.
   This routine also returns the activation function value, as it is most
   efficient to do so because of the duplication of effort.

--------------------------------------------------------------------------------
*/

void partial_cr ( double *input , double *coefs , double *output ,
                 int ninputs , double *deriv_re , double *deriv_im , int linear)
                  
{
   double sum ;

   sum = dotprodcr ( ninputs , input , coefs ) ;
   sum += coefs[2*ninputs] ;      // Bias term (ignore imaginary part of bias)

   if (linear) {
      *output = sum ;
      *deriv_re = 1.0 ;
      *deriv_im = 0.0 ;
      }
   else {
      *output = act_func ( sum ) ;
      *deriv_re = actderiv ( *output ) ;
      *deriv_im = 0.0 ;
      }
}

void partial_cc ( double *input , double *coefs , double *output , int ninputs ,
                  double *deriv_rr , double *deriv_ri ,
                  double *deriv_ir , double *deriv_ii , int linear )
{
   double rsum, isum, raw_length, squashed_length, ratio, d, l2, temp ;

   dotprodc ( ninputs , input , coefs , &rsum , &isum ) ;
   rsum += coefs[2*ninputs] ;      // Bias term
   isum += coefs[2*ninputs+1] ;

   if (linear) {
      *output = rsum ;
      *(output+1) = isum ;
      *deriv_rr = *deriv_ii = 1.0 ;
      *deriv_ri = *deriv_ir = 0.0 ;
      return ;
      }

   raw_length = sqrt ( rsum*rsum + isum*isum ) + 1.e-30 ;
   squashed_length = act_func ( raw_length ) ; // Activation function
   d = actderiv ( squashed_length ) ;          // Its derivative
   ratio = squashed_length / raw_length ;

   *output = rsum * ratio ;
   *(output+1) = isum * ratio ;

   l2 = raw_length * raw_length + 1.e-30 ;
   temp = (d - ratio) / l2 ;

   *deriv_rr = ratio  +  rsum * rsum * temp ;
   *deriv_ii = ratio  +  isum * isum * temp ;
   *deriv_ri = *deriv_ir = rsum * isum * temp ;
}


/******************************************************************************/
/*                                                                            */
/*  ACT_FUNC - Compute the activation function                                */
/*                                                                            */
/*  The default used here is f(x) = TanH ( 1.5 x )                            */
/*                                                                            */
/*  This also contains the routine actderiv which computes the derivative     */
/*  as a function of the activation level.                                    */
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

double act_func ( double x )
{
   return tanh ( 1.5 * x ) ;
}

/*
   ACTDERIV - Compute the derivative of the activation function.
   Note that it is written as a function of the output activation,
   f(net), rather than net itself. To make this dependent on the net
   would necessitate storing the nets, which is otherwise unnecessary.
*/

double actderiv ( double f )
{
   return 1.5 * (1.0 - f * f) ;
}

/*
   INVERSE_ACT - Compute the inverse of the activation function.
*/

double inverse_act ( double f )
{
   if (f < -0.999999)
      f = -0.999999 ;
   if (f > 0.999999)
      f = 0.999999 ;

   return 0.3333333 * log ( (1.0 + f) / (1.0 - f) ) ;
}


/******************************************************************************/
/*                                                                            */
/*  SHAKE - Randomly perturb a point                                          */
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

void shake ( int nvars , double *center , double *x , double temp ,
             enum RandomDensity dens )
{
   switch (dens) {
      case NormalDensity:
         while (nvars--)
            *x++ = *center++ + temp * normal () ;
         break ;
      case CauchyDensity:
         cauchy ( nvars , temp , x ) ;
         while (nvars--)
            *x++ += *center++ ;
         break ;
      }
}

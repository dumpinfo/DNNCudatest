/******************************************************************************/
/*                                                                            */
/*  PARSDUBL - ParseDouble routine to parse a double from a string            */
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

inline int digit ( int character )
{
   character &= 255 ;
   return (character >= '0'  &&  character <= '9') ;
}

double ParseDouble ( char **str )
{
   double num ;

/*
   Skip nonnumeric stuff.  If we run into a comma, that means missing data.
*/

   while (! ( digit ( **str ) || (**str == '-') || (**str == '.'))) {
      if (**str == ',') {
         ++(*str) ;
         return MISSING ;  // In CONST.H
         }
      if (**str)
         ++(*str) ;
      else
         return MISSING ;
      }

/*
   Read the number, then pass it by
*/

   num = atof ( *str ) ;

   while (digit ( **str )  ||  (**str == '-')  ||  (**str == '.'))
      ++(*str) ;

/*
   Skip a single comma that may be a delimiter
*/

   while ((**str == ' ')  ||  (**str == ',')) {
      if (**str == ',') {
         ++(*str) ;
         break ;
         }
      ++(*str) ;
      }

   return num ;
}


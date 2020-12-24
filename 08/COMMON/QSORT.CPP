/****************************************************************************/
/*                                                                          */
/*  QSORT - Quick sort a double array.                                      */
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
/*                                                                          */
/****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"     // System, limitation constants, typedefs, structs
#include "classes.h"   // Includes all class headers
#include "funcdefs.h"  // Function prototypes

void qsort ( int first , int last , double data[] )
{
   int bottom, top ;
   double ftemp, split ;

   split = data[(first+last)/2] ;
   bottom = first ;
   top = last ;

   do {
      while ( split > data[bottom] )
         ++bottom ;
      while ( split < data[top] )
         --top ;
      if (bottom == top) {
         ++bottom ;
         --top ;
         }
      else if (bottom < top) {
         ftemp = data[bottom] ;
         data[bottom++] = data[top] ;
         data[top--] = ftemp ;
         }
      } while ( bottom <= top ) ;

   if (first < top)
      qsort ( first , top , data ) ;
   if (bottom < last)
      qsort ( bottom , last , data ) ;
}


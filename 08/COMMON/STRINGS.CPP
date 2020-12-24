/******************************************************************************/
/*                                                                            */
/*  STRINGS - Implement Strings class (arrays of strings)                     */
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

Strings::Strings ( char *all )
{
   int length ;
   char *cptr ;

   MEMTEXT ( "data for new Strings (3+2)" ) ;

   length = strlen ( all ) + 1 ;
   data = (char *) MALLOC ( length ) ;
   len = (int *) MALLOC ( length * sizeof(int) ) ;
   start = (char **) MALLOC ( length * sizeof(char *) ) ;

   if ((! data)  ||  (! len)  ||  (! start)) {
      if (! data)
         FREE ( data ) ;
      if (! len)
         FREE ( len ) ;
      if (! start)
         FREE ( start ) ;
      n = 0 ;
      return ;
      }

   nreal = 0 ;
   cptr = data ;
   for (n=0 ; *all ;) {
      start[n] = cptr ;                   // This upcoming string starts here
      length = 0 ;                        // Its length computed here
      while (*all  &&  (*all == ' '))     // Remove leading blanks
         ++all ;
      while (*all  &&  (*all != ',')) {   // Copy until delimitter found
         *cptr++ = *all++ ;
         ++length ;
         }
      while (*(cptr-1) == ' ') {          // Remove trailing blanks
         --cptr ;
         --length ;
         }
      *cptr++ = 0 ;                       // Terminate this string
      if (*all == ',')                    // Bypass delimitter
         ++all ;
      len[n++] = length ;                 // Record length of this string
      if (length)                         // If this is not a NULL string
         ++nreal ;                        // Count it
      }

/*
   Change embedded blanks to underscores
*/

   while (--cptr != data) {
      if (*cptr == ' ')
         *cptr = '_' ;
      }

/*
   These two arrays were initially allocated much too large
*/

   len = (int *) REALLOC ( len , n * sizeof(int) ) ;
   start = (char **) REALLOC ( start , n * sizeof(char *) ) ;
}

Strings::~Strings ()
{
   if (n) {
      MEMTEXT ( "delete Strings (3)" ) ;
      FREE ( data ) ;
      FREE ( len ) ;
      FREE ( start ) ;
      }
}

int Strings::is_duplication ()
{
   int i, j ;

   for (i=1 ; i<n ; i++) {
      if (! len[i])
         continue ;
      for (j=0 ; j<i ; j++) {
         if (! len[j])
            continue ;
         if (! strcmp ( start[i] , start[j] ))
            return 1 ;
         }
      }
   return 0 ;
}

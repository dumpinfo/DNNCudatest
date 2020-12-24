// Copyright (c) 1994 John Wiley & Sons, Inc.  All rights reserved.
// Reproduction or translation of this work beyond that permitted in
// section 117 of the 1976 United States Copyright Act without the
// express written permission of the copyright owner is unlawful.
// Requests for further information should be addressed to the
// Permission Department, John Wiley & Sons, Inc.  The purchaser may
// make backup copies for his/her own use only and not for distribution
// or resale.  The publisher assumes no responsibility for errors,
// omissions, or damages, caused by the use of these programs or from
// the use of the information contained herein.

/******************************************************************************/
/*                                                                            */
/*  MESSAGES - All routines for issuing messages to user                      */
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

void error_message ( char *msg )
{
   printf ( "\nERROR: %s", msg ) ;
}

void warning_message ( char *msg )
{
   printf ( "\nWARNING: %s", msg ) ;
}

void normal_message ( char *msg )
{
   printf ( "\n%s", msg ) ;
}

void progress_message ( char *msg )
{
   printf ( "%s", msg ) ;
}

void memory_message ( char *msg )
{
   printf ( "\nInsufficient memory %s", msg ) ;
}

int user_pressed_escape ()
{
   int key ;

   if (kbhit()) {          // Was a key pressed?
      key = getch () ;     // Read it if so
      while (kbhit())      // Flush key buffer in case function key
         getch () ;        // or key was held down
      if (key == KEY_ESCAPE)
         return 1 ;
      }
   return 0 ;
}

/******************************************************************************/
/*                                                                            */
/*  PROG_WIN - Routines related to a progress window                          */
/*                                                                            */
/*    make_progress_window                                                    */
/*    destroy_progress_window                                                 */
/*    write_progress                                                          */
/*    user_pressed_escape                                                     */
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
#include "..\common\const.h"  // System, limitation constants, typedefs, structs
#include "..\common\classes.h"  // Includes all class headers
#include "..\common\funcdefs.h" // Function prototypes

extern int show_progress ;

void make_progress_window ( char * )
{
   return ;
}

void destroy_progress_window ()
{
   return ;
}

void write_progress ( char *msg )
{
   if (msg == NULL)
      return ;
   printf ( "\n%s", msg ) ;
}

void write_non_progress ( char * )
{
   return ;
}

int user_pressed_escape ()
{
   int key ;

   if (kbhit()) {         // Was a key pressed?
      key = getch () ;    // Read it if so
      while (kbhit())     // Flush key buffer in case function key
         getch () ;       // or key was held down
      return (key == KEY_ESCAPE) ;
      }
   else
      return 0 ;
}

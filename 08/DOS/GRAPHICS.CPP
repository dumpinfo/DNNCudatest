/******************************************************************************/
/*                                                                            */
/*  GRAPHICS - These are Flash Graphics interface routines.                   */
/*             All routines that are specific to that library are here.       */
/*             Other routines call routines here to do graphics operations.   */
/*                                                                            */
/*  This also includes Symantec text-mode save and restore routines.          */
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
#include <disp.h>
#include <dos.h>
#include "..\common\const.h"  // System, limitation constants, typedefs, structs
#include "..\common\classes.h"  // Includes all class headers
#include "..\common\funcdefs.h" // Function prototypes
#include "fg.h"                 // FlashGraphics

static int mode ;           // Display mode
static int drows, dcols ;   // Dimensions in pixels
static int crows, ccols ;   // Size of a character
static double aspect ;      // Aspect ratio


/*
--------------------------------------------------------------------------------

   Call this routine at the start of the program.
   It returns 0 if all went well, 1 if the adapter is not supported.

--------------------------------------------------------------------------------
*/

int init_graphics ()
{
   mode = fg_get_first () ;
   while (mode) {
      if (mode == FG_VGA12) {
         drows = 480 ;
         dcols = 640 ;
         fg_set_8x16_font () ;
         crows = fg.charbox[FG_Y2] + 1 ;
         ccols = fg.charbox[FG_X2] + 1 ;
         aspect = 3.0 / 4.0 ;
         return 0 ;   // Normal return
         }
      mode = fg_get_next () ;
      }

   mode = fg_get_first () ;
   while (mode) {
      if (mode == FG_VESA3) {
         drows = 600 ;
         dcols = 800 ;
         fg_set_8x16_font () ;
         crows = fg.charbox[FG_Y2] + 1 ;
         ccols = fg.charbox[FG_X2] + 1 ;
         aspect = 3.0 / 4.0 ;
         return 0 ;   // Normal return
         }
      mode = fg_get_next () ;
      }

   drows = -1 ; // Flag error
   return 1 ;   // And tell caller
}

/*
--------------------------------------------------------------------------------

   Switch to graphics mode.

--------------------------------------------------------------------------------
*/

void goto_graphics ( int *nrows , int *ncols , int *chrows , int *chcols ,
                     double *aspect_ratio )
{
   if (drows <= 0)
      return ;

   fg_init_mode ( mode ) ;

   *nrows = drows ;
   *ncols = dcols ;
   *aspect_ratio = aspect ;
   *chrows = crows ;
   *chcols = ccols ;
}

/*
--------------------------------------------------------------------------------

   Return to text mode

--------------------------------------------------------------------------------
*/

void exit_graphics ()
{
   if (drows <= 0)
      return ;

   fg_term () ;
}

/*
--------------------------------------------------------------------------------

   Draw a line

--------------------------------------------------------------------------------
*/

void drawline ( int x1 , int y1 , int x2 , int y2 , int color )
{
   fg_line_t line ;

   if (drows <= 0)
      return ;

   line[FG_X1] = x1 ;
   line[FG_X2] = x2 ;
   line[FG_Y1] = y1 ;
   line[FG_Y2] = y2 ;
   fg_drawline ( color , FG_MODE_SET , ~0 , FG_LINE_SOLID , line ) ;
}

void write_graphics_text ( int row , int col , char *text , int color )
{
   fg_box_t box ;

   if (drows <= 0)
      return ;

   fg_make_box ( box , col , row , col + strlen ( text ) * ccols - 1 ,
                 row + crows - 1 ) ;
   fg_box_normalize ( box ) ;
   fg_fillbox ( 0 , FG_MODE_SET , ~0 , box ) ;
   fg_puts ( color , FG_MODE_SET , ~0 , FG_ROT0 , col ,
             row , text , fg.displaybox ) ;
}

/*
--------------------------------------------------------------------------------

   These are for Symantic text-mode operations (not Flash Graphics)

--------------------------------------------------------------------------------
*/

static unsigned short *screen = NULL ;
static unsigned short cursor_row, cursor_col ;

void init_textmode ()
{
   if (disp_getmode () != 3)
      disp_setmode ( 3 ) ;
   disp_open () ;
}

void close_textmode ()
{
   disp_close () ;
}

int save_screen ()
{
   union REGS r ;
   MEMTEXT ( "GRAPHICS: screen" ) ;
   screen = MALLOC ( 25 * 80 * 2 ) ;
   if (screen == NULL)
      return 1 ;

   r.h.ah = 3 ;    // Read cursor
   r.h.bh = 0 ;    // Page
   int86 ( 0x10 , &r , &r ) ;
   cursor_row = r.h.dh ;
   cursor_col = r.h.dl ;
   disp_peekbox ( screen , 0 , 0 , 24 , 79 ) ;
   return 0 ;
}

void restore_screen ()
{
   union REGS r ;
   if (screen == NULL)
      return ;
   disp_pokebox ( screen , 0 , 0 , 24 , 79 ) ;
   MEMTEXT ( "GRAPHICS: screen" ) ;
   FREE ( screen ) ;
   r.h.ah = 2 ;    // Set cursor
   r.h.bh = 0 ;    // Page
   r.h.dh = cursor_row ;
   r.h.dl = cursor_col ;
   int86 ( 0x10 , &r , &r ) ;
}

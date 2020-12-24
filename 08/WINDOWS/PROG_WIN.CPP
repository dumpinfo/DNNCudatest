/******************************************************************************/
/*                                                                            */
/*  PROG_WIN - Routines related to a progress window                          */
/*                                                                            */
/*    make_progress_window                                                    */
/*    destroy_progress_window                                                 */
/*    write_progress                                                          */
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

#include <owl\owlpch.h>
#include <\owl\listbox.h>
#include <\owl\button.h>
#include <\owl\static.h>
#include <\owl\validate.h>
#include <\owl\groupbox.h>
#include <owl\inputdia.h>
#include <owl\dialog.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#pragma hdrstop

#include "..\common\const.h"  // System, limitation constants, typedefs, structs
#include "..\common\classes.h"  // Includes all class headers
#include "..\common\funcdefs.h" // Function prototypes

#include "textlist.h"   // Things in TEXTLIST.CPP
#include "prog_win.h"   // Things in PROG_WIN.CPP
#include "prdctapp.rh"

static TDecoratedMDIFrame *fr ;    // Main calls init_progress to set this
static ProgressWindow *pw = NULL ; // make_progress_window creates new one here
static TextList *lb ;              // user_pressed_escape filters lb messages
static int escape_key_pressed = 0 ;
static char abort_string[] =
  {"Click the Abort button or press ESCape to terminate this operation"} ;
static int label_x, label_y ;

/*
--------------------------------------------------------------------------------

   Everything in this first section is local to this module.
   They define the ProgressWindow class.

--------------------------------------------------------------------------------
*/

#define ID_ABORT_BUTTON 102

DEFINE_RESPONSE_TABLE1 ( ProgressWindow , TFrameWindow )
  EV_COMMAND ( ID_ABORT_BUTTON , HandleAbortButton ) ,
END_RESPONSE_TABLE ;

ProgressWindow::ProgressWindow (	char *title	)
	: TFrameWindow ( fr , title )
{
	int rows, cols, cxborder, cycaption ;
	int win_x, win_y, win_h, win_w, x, y, w, h, lb_bot ;
	char msg[256] ;
	TButton *but1 ;

	cols = GetSystemMetrics ( SM_CXSCREEN ) ;
	rows = GetSystemMetrics ( SM_CYSCREEN ) ;
	cxborder = GetSystemMetrics ( SM_CXBORDER ) ;
	cycaption = GetSystemMetrics ( SM_CYCAPTION ) ;
	
	sprintf ( msg , "new ProgressWindow in %d by %d", rows, cols ) ;
	MEMTEXT ( msg ) ;

	SetIcon ( GetApplication() , IDI_PROGRESS);
   SetBkgndColor ( RGB ( 255 , 255 , 0 )) ;
	
	win_w = 0.97 * cols ;
	win_h = 0.48 * rows ;
	win_x = 0.5 * (cols - win_w - 2 * cxborder ) ;
	win_y = rows - win_h - 10 ;

	Attr.Style = WS_POPUP | WS_CAPTION ;
	Attr.Style |= WS_MINIMIZEBOX | WS_VISIBLE | WS_CLIPCHILDREN ;
	Attr.X = win_x ;
	Attr.Y = win_y ;
	Attr.W = win_w ;
	Attr.H = win_h ;

	win_h -= cycaption ;       // Window height excluding caption

	w = 0.98 * win_w ;
	x = 0.5 * (win_w - w - 2 * cxborder) ;
	h = 0.85 * win_h ;
	y = 0.03 * win_h ;
	lb = new TextList ( this , -1 , x , y , w , h ) ;
	lb_bot = y + h ;

	w = 0.1 * win_w ;
	h = 0.1 * win_h ;
	y = lb_bot + 0.5 * (win_h - lb_bot - h) ;
	but1 = new TButton ( this , ID_ABORT_BUTTON , "Abort" , x , y , w , h ) ;
	but1->EnableAutoCreate() ;

	label_x = x + w + 10 ;
	label_y = y + 3 ;

	return ;
}

ProgressWindow::~ProgressWindow ()
{
	MEMTEXT ( "ProgressWindow destructor" ) ;
}

void ProgressWindow::SetupWindow ()
{
	TFrameWindow::SetupWindow() ;  // Creates child controls
}


void ProgressWindow::AppendLine ( char *str )
{
   lb->AppendTextLine ( str ) ;
}

void ProgressWindow::HandleAbortButton ()
{
	escape_key_pressed = 1 ;
}

/*
   Paint routine writes static text.  (A Windows static does not let you
   set its background color!)
*/

void ProgressWindow::Paint ( TDC& dc , BOOL , TRect& )
{
	dc.SetBkColor ( RGB ( 255 , 255 , 0 ) ) ;
	dc.SetTextColor ( RGB ( 255 , 0 , 0 ) ) ;
   dc.SetTextAlign ( TA_TOP | TA_LEFT ) ;
   dc.TextOut ( label_x , label_y , abort_string ) ;
}


/*
--------------------------------------------------------------------------------

	These are the routines that are called from elsewhere

--------------------------------------------------------------------------------
*/

void init_progress ( TDecoratedMDIFrame *frame )
{
	fr = frame ;
}

void make_progress_window ( char *title )
{
	char msg[256] ;
   MEMTEXT ( "make_progress_window" ) ;
	pw = new ProgressWindow ( title ) ;
	sprintf ( msg , "make_progress_window = %d", (int) pw ) ;
	MEMTEXT ( msg ) ;
	if (pw->HWindow)
		pw->Destroy() ;
	pw->Create() ;
	escape_key_pressed = 0 ;
   fr->AssignMenu(NULL_MENU) ;
   user_pressed_escape () ;  // Spin the message loop so subwindows appear
}

void destroy_progress_window ()
{
   if (pw != NULL) {
      MEMTEXT ( "destroy_progress_window" ) ;
	   delete pw ;
      pw = NULL ;
      }
	escape_key_pressed = 0 ;
   fr->AssignMenu(MDI_MENU) ;
}

void write_progress ( char *msg )
{
	if (pw == NULL  ||  msg == NULL)
		return ;
	pw->AppendLine ( msg ) ;
}

void write_non_progress ( char *msg )
{
	if (pw == NULL  ||  msg == NULL)
		return ;
	pw->AppendLine ( msg ) ;
}

int user_pressed_escape ()
{
	MSG msg ;

/*
   This is a safety check.  We never want to enter the PeekMessage loop
   unless there is a progress window.  Otherwise, the user might click on
   a menu item and cause dangerous recursion.
*/

	if ((pw == NULL)  ||  (pw->HWindow == NULL))
      return 0 ;

	while (PeekMessage ( &msg , 0 , 0 , 0 , PM_REMOVE )) {
      if (msg.message == WM_CHAR) {
         if ((TCHAR) msg.wParam == VK_ESCAPE) // Check for user pressing ESCape
            escape_key_pressed = 1 ;
			}
		TranslateMessage ( &msg ) ;
		DispatchMessage ( &msg ) ;
		}

	return escape_key_pressed ;
}


/******************************************************************************/
/*                                                                            */
/*  COM_FILE - Routines related to reading/executing command control file     */
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

#include <owl\opensave.h>

#include "..\common\const.h"  // System, limitation constants, typedefs, structs
#include "..\common\classes.h"  // Includes all class headers
#include "..\common\funcdefs.h" // Function prototypes

#include "textlist.h"   // Things in TEXTLIST.CPP
#include "com_file.h"
#include "prdctapp.rh"

static char abort_string[] =
  {"Click the Abort button or press ESCape to skip remainder of command file"} ;
static int label_x, label_y ;

/*
--------------------------------------------------------------------------------

   ComfileWindow class stuff

--------------------------------------------------------------------------------
*/

#define ID_ABORT_BUTTON 102

DEFINE_RESPONSE_TABLE1 ( ComfileWindow , TFrameWindow )
  EV_COMMAND ( ID_ABORT_BUTTON , HandleAbortButton ) ,
END_RESPONSE_TABLE ;

ComfileWindow::ComfileWindow ( TDecoratedMDIFrame *fr , char *title	)
	: TFrameWindow ( fr , title )
{
	int rows, cols, cxborder, cycaption ;
	int win_x, win_y, win_h, win_w, x, y, w, h ;
	char msg[256] ;
	TButton *but1 ;

	cols = GetSystemMetrics ( SM_CXSCREEN ) ;
	rows = GetSystemMetrics ( SM_CYSCREEN ) ;
	cxborder = GetSystemMetrics ( SM_CXBORDER ) ;
	cycaption = GetSystemMetrics ( SM_CYCAPTION ) ;
	
	sprintf ( msg , "new ComfileWindow in %d by %d", rows, cols ) ;
	MEMTEXT ( msg ) ;

	SetIcon ( GetApplication() , IDI_COMMAND ) ;
   SetBkgndColor ( RGB ( 0 , 255 , 255 )) ;
	
	win_w = 0.97 * cols ;
	win_h = 0.48 * rows ;
	win_x = 0.5 * (cols - win_w - 2 * cxborder ) ;
	win_y = 10 ;

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
	y = 0.13 * win_h ;
	lb = new TextList ( this , -1 , x , y , w , h ) ;

	w = 0.1 * win_w ;
	h = 0.1 * win_h ;
	y = 0.5 * (y - h) ;
	but1 = new TButton ( this , ID_ABORT_BUTTON , "Abort" , x , y , w , h ) ;
	but1->EnableAutoCreate() ;

	label_x = x + w + 10 ;
	label_y = y + 3 ;

	escape_key_pressed = 0 ;
	return ;
}

ComfileWindow::~ComfileWindow ()
{
	MEMTEXT ( "delete ComfileWindow" ) ;
}

void ComfileWindow::SetupWindow ()
{
	TFrameWindow::SetupWindow() ;  // Creates child controls
}


void ComfileWindow::AppendLine ( char *str )
{
   lb->AppendTextLine ( str ) ;
}

void ComfileWindow::HandleAbortButton ()
{
	escape_key_pressed = 1 ;
}


int ComfileWindow::user_pressed_escape ()
{
	MSG msg ;

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


/*
   Paint routine writes static text.  (A Windows static does not let you
   set its background color!)
*/

void ComfileWindow::Paint ( TDC& dc , BOOL , TRect& )
{
	dc.SetBkColor ( RGB ( 0 , 255 , 255 ) ) ;
	dc.SetTextColor ( RGB ( 255 , 0 , 0 ) ) ;
   dc.SetTextAlign ( TA_TOP | TA_LEFT ) ;
   dc.TextOut ( label_x , label_y , abort_string ) ;
}

/*
--------------------------------------------------------------------------------

	This the the (only) external routine, called from the main module

--------------------------------------------------------------------------------
*/

void read_command_file ( TDecoratedMDIFrame *fr )
{
	int id ;
	char *rest, line[256], results[256] ;
	char msg[256] ;
	ControlData cbuf ;
	ComfileWindow *cfw ;
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("Command files (*.CON)|*.CON|All Files (*.*)|*.*|") ;
	*FileData.FileName = 0 ;

	id = TFileOpenDialog ( fr , FileData ).Execute() ;
	if (id != IDOK)
		return ;

	fr->AssignMenu(NULL_MENU);
	cfw = new ComfileWindow ( fr , "Reading command file" ) ;
	if (cfw->HWindow) {
		cfw->Destroy() ;
      MEMTEXT ( "!!! Had to PreDestroy ComfileWindow !!!" ) ;
      }
	cfw->Create() ;

/*
	Tell 'process' to read the command file.
	Then, as long as the command buffer is not empty, extract and execute
	the commands in it.
*/

	cbuf.n = 0 ;
	cbuf.next = 0 ;
	cbuf.buf = NULL ;

	process ( ID_PRED_COMMANDFILE , FileData.FileName , &cbuf , line , results );

	while ( cbuf.n ) {
		get_control_line ( &cbuf , line ) ;
		cfw->AppendLine ( line ) ;
		if (cfw->user_pressed_escape ())
			goto COMMAND_FINISH ;
		id = interpret_control_line ( line , &rest ) ;
		if (! id)       // No command (blank or comment)
			continue ;
		if (id < 0) {   // Unknown command
			delete cfw ;
			cfw = NULL ;
			fr->MessageBox ( line , "UNKNOWN COMMAND" ) ;
			break ;
			}

#if 0
		sprintf ( msg , "ID=%d  (%s)", id, rest ) ;
		MEMTEXT ( msg ) ;
#endif

		id = process ( id , rest , &cbuf , msg , results ) ;
      if (strlen ( results ))
         cfw->AppendLine ( results ) ;
		if (id) {  // >0 means BYE, <0 means error
			if (id < 0)
				fr->MessageBox ( msg , "ERROR" ) ;
			delete cfw ;
			cfw = NULL ;
			break ;
			}
		} // Endless command loop until buffer empty

COMMAND_FINISH:
	if (cbuf.n) {  // Only happens if QUIT in control file before end
		MEMTEXT ( "PNN: control buffer" ) ;
		FREE ( cbuf.buf ) ;
		cbuf.buf = NULL ;
		cbuf.n = 0 ;
		}
	if (cfw)
		delete cfw ;
	fr->AssignMenu(MDI_MENU);
}

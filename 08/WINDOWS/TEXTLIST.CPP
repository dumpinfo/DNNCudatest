/******************************************************************************/
/*                                                                            */
/*  TEXTLIST - Derived from TListBox for scrolled display of text             */
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

#include "textlist.h"

DEFINE_RESPONSE_TABLE1 (TextList , TListBox )
  EV_WM_LBUTTONDOWN,
  EV_WM_RBUTTONDOWN,
  EV_WM_CHAR,
  EV_WM_KEYDOWN,
END_RESPONSE_TABLE ;

TextList::TextList ( TWindow *parent , int Id , int x , int y , int w , int h )
	: TListBox ( parent , Id , x , y , w , h )
{
	Attr.Style &= ~(WS_HSCROLL | WS_VSCROLL | LBS_SORT | LBS_NOTIFY) ;
	EnableAutoCreate() ; // Probably not necessary
}

TextList::~TextList ()
{
}

void TextList::AppendTextLine ( char *str )
{
#define TL_MAX_STRINGS 40
	int last = GetCount() ;      // This many strings in ListBox so far
	if (last > TL_MAX_STRINGS) { // If there are getting to be a lot
		DeleteString( 0 ) ;       // Delete the first, no longer visible
		last = TL_MAX_STRINGS ;   // Index (org 0) of string to be added
		}
	AddString ( str ) ;          // Append new string to end of list
	SetCaretIndex ( last , 1 ) ; // Make sure it's visible (scrolls)
	UpdateWindow () ;            // Force repainting
}

/*
	The purpose of these routines is to prevent the user from selecting
	a line in the TListBox.
*/
	
void TextList::EvLButtonDown ( UINT , TPoint& )
{
}

void TextList::EvRButtonDown ( UINT , TPoint& )
{
}

void TextList::EvChar ( UINT , UINT , UINT )
{
}

void TextList::EvKeyDown ( UINT , UINT , UINT )
{
}


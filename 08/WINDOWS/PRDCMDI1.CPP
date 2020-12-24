/******************************************************************************/
/*                                                                            */
/*  PRDCMDI1 - Control routines for MDI children showing univariate series    */
/*                                                                            */
/*    npredictMDIChild - Constructor                                          */
/*    ~npredictMDIChild - Destructor                                          */
/*    Paint is in DISPLAY.CPP.                                                */
/*                                                                            */
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

#include "prdctapp.h"
#include "prdcmdi1.h"

DEFINE_RESPONSE_TABLE1 (npredictMDIChild , TMDIChild )
  EV_WM_SIZE,
END_RESPONSE_TABLE ;

/*
--------------------------------------------------------------------------------

   Constructor

--------------------------------------------------------------------------------
*/

npredictMDIChild::npredictMDIChild (
	TMDIClient *parent ,
   const char far *title ,
   TWindow *clientWnd ,
   BOOL shrinkToClient ,
   TModule *module )
	: TMDIChild ( *parent , title , clientWnd , shrinkToClient , module )
{
	 MEMTEXT ( "--->npredictMDIChild constructor" ) ;
	// Associate ICON w/ this child window.
	SetIcon ( GetApplication(), IDI_DOC ) ;

	// If the current active MDI child is maximize then this one should be also.
	npredictMDIChild *curChild=(npredictMDIChild *) parent->GetActiveMDIChild();
	if (curChild && (curChild->GetWindowLong(GWL_STYLE) & WS_MAXIMIZE))
		 Attr.Style |= WS_MAXIMIZE;
}

/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

npredictMDIChild::~npredictMDIChild ()
{
	 MEMTEXT ( "--->npredictMDIChild destructor" ) ;
    purge_display ( this ) ;
	 Destroy();
}


void npredictMDIChild::EvSize ( UINT sizeType , TSize& size )
{
	MEMTEXT ( "--->Child EvSize" ) ;
	TMDIChild::EvSize ( sizeType , size ) ;
	Invalidate () ;
}


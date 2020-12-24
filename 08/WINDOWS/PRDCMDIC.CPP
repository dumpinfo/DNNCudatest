/******************************************************************************/
/*                                                                            */
/*  PRDCMDIC - Client window routines                                         */
/*                                                                            */
/*    npredictMDIClient - Constructor                                         */
/*    ~npredictMDIClient - Destructor                                         */
/*    SetupWindow                                                             */
/*       calls TMDIClient::SetupWindow ()                                     */
/*       Sets file filters for open and save as                               */
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
#include "prdcmdic.h"
#include "prdcmdi1.h"

/*
--------------------------------------------------------------------------------

   Constructor

--------------------------------------------------------------------------------
*/

npredictMDIClient::npredictMDIClient ()
 : TMDIClient ()
{
    // Change the window's background color
    SetBkgndColor(RGB(0xff, 0xff, 0xff));

    ChildCount = 0;

   MEMTEXT ( "--->Client constructor" ) ;
}


/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

npredictMDIClient::~npredictMDIClient ()
{
   MEMTEXT ( "--->Client destructor starting" ) ;
   Destroy() ;
   MEMTEXT ( "--->Client destructor called Destroy" ) ;
}

void npredictMDIClient::SetupWindow ()
{
   MEMTEXT ( "--->Client SetUpWindow" ) ;
}

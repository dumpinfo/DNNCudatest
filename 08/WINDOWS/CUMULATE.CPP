/******************************************************************************/
/*                                                                            */
/*  CUMULATE - All commands related to cumulating training and test sets      */
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

#include "cumulate.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( CumulateDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_CUMULATE_TRAINING , EvTraining ) ,
  EV_COMMAND ( IDC_CUMULATE_TEST , EvTest ) ,
END_RESPONSE_TABLE ;

CumulateDialog::CumulateDialog ( TWindow *parent )
	: TDialog ( parent , IDD_CUMULATE ), TWindow ( parent )
{
	training = new TRadioButton ( this , IDC_CUMULATE_TRAINING ) ;
	test = new TRadioButton ( this , IDC_CUMULATE_TEST ) ;

	include = new TEdit ( this , IDC_CUMULATE_INCLUDE ) ;
   include_valid = new TRangeValidator ( 1 , 9999999 ) ;
   include->SetValidator ( include_valid ) ;
	exclude = new TEdit ( this , IDC_CUMULATE_EXCLUDE ) ;
   exclude_valid = new TRangeValidator ( 0 , 999999 ) ;
   exclude->SetValidator ( exclude_valid ) ;
}

CumulateDialog::~CumulateDialog ()
{
}

void CumulateDialog::SetupWindow ()
{
   TDialog::SetupWindow () ;

   include->SetText ( "999999" ) ;
   exclude->SetText ( "0" ) ;

   parm_type = ID_PRED_CUMULATE_TRAINING ;
   training->Check () ;
}

void CumulateDialog::CmOk ()
{
   char msg[84] ;

   include->GetText ( msg , 83 ) ;
   parm_include = atoi ( msg ) ;
   exclude->GetText ( msg , 83 ) ;
   parm_exclude = atoi ( msg ) ;

   TDialog::CmOk () ;
}

void CumulateDialog::EvTraining ()
{
   parm_type = ID_PRED_CUMULATE_TRAINING ;
}

void CumulateDialog::EvTest ()
{
   parm_type = ID_PRED_CUMULATE_TEST ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void cumulate ( TWindow *parent )
{
	char error[256], rest[256] ;
	CumulateDialog *cm ;

	cm = new CumulateDialog ( parent ) ;

	if (cm->Execute() == IDOK) {
      sprintf ( rest , "%d", cm->parm_include ) ;
      if (process ( ID_PRED_CUMULATE_INCLUDE , rest , NULL , error, NULL ) < 0){
         parent->MessageBox ( error , "ERROR" ) ;
         delete cm ;
         return ;
         }
      sprintf ( rest , "%d", cm->parm_exclude ) ;
      if (process ( ID_PRED_CUMULATE_EXCLUDE , rest , NULL , error, NULL ) < 0){
         parent->MessageBox ( error , "ERROR" ) ;
         delete cm ;
         return ;
         }
      if (process ( cm->parm_type , rest , NULL , error , NULL ) < 0)
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete cm ;
}

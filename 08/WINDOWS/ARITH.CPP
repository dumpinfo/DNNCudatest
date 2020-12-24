/******************************************************************************/
/*                                                                            */
/*  ARITH - Arithmetic combination of signals                                 */
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

#include "arith.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( ArithDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_ARITH_ADD , EvAdd ) ,
  EV_COMMAND ( IDC_ARITH_SUBTRACT , EvSubtract ) ,
  EV_COMMAND ( IDC_ARITH_MULTIPLY , EvMultiply ) ,
  EV_COMMAND ( IDC_ARITH_DIVIDE , EvDivide ) ,
END_RESPONSE_TABLE ;

ArithDialog::ArithDialog ( TWindow *parent ,
                           int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_ARITH ), TWindow ( parent )
{
   n_signals = p_nsignals ;
   signals = p_signals ;

	add = new TRadioButton ( this , IDC_ARITH_ADD ) ;
	subtract = new TRadioButton ( this , IDC_ARITH_SUBTRACT ) ;
	multiply = new TRadioButton ( this , IDC_ARITH_MULTIPLY ) ;
	divide = new TRadioButton ( this , IDC_ARITH_DIVIDE ) ;

   source = new TListBox ( this , IDC_ARITH_SOURCE ) ;
   source2 = new TListBox ( this , IDC_ARITH_SOURCE2 ) ;
	dest = new TComboBox ( this , IDC_ARITH_DEST ) ;
}

ArithDialog::~ArithDialog ()
{
}

void ArithDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   strcpy ( source_name , "" ) ;
   strcpy ( source2_name , "" ) ;
   strcpy ( dest_name , "" ) ;

   arith_type = ID_PRED_ADD ;
   add->Check () ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      source->AddString ( sigptr->name ) ;
      source2->AddString ( sigptr->name ) ;
      dest->AddString ( sigptr->name ) ;
      }
   source->SetSelIndex ( 0 ) ;
   source2->SetSelIndex ( 0 ) ;
   dest->SetSelIndex ( -1 ) ;
}

void ArithDialog::CmOk ()
{
   source->GetSelString ( source_name , 255 ) ;
   source2->GetSelString ( source2_name , 255 ) ;
   dest->GetText ( dest_name , 255 ) ;

   strupr ( source_name ) ;
   strupr ( source2_name ) ;
   strupr ( dest_name ) ;

	TDialog::CmOk () ;
}

void ArithDialog::EvAdd ()
{
   arith_type = ID_PRED_ADD ;
}

void ArithDialog::EvSubtract ()
{
   arith_type = ID_PRED_SUBTRACT ;
}

void ArithDialog::EvMultiply ()
{
   arith_type = ID_PRED_MULTIPLY ;
}

void ArithDialog::EvDivide ()
{
   arith_type = ID_PRED_DIVIDE ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void arith ( TWindow *parent )
{
   int nsigs ;
	char error[256], rest[256] ;
	ArithDialog *ad ;
   Signal **signals ;

   nsigs = get_signals ( &signals ) ;
	ad = new ArithDialog ( parent , nsigs , signals ) ;

	if (ad->Execute() == IDOK) {
      if (process ( ID_PRED_NAME , ad->dest_name , NULL , error , NULL ) < 0){
         parent->MessageBox ( error , "ERROR" ) ;
         delete ad ;
         return ;
         }
      sprintf ( rest , "%s AND %s", ad->source_name, ad->source2_name ) ;
      if (process ( ad->arith_type , rest , NULL , error , NULL ) < 0)
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete ad ;
}


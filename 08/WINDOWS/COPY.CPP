/******************************************************************************/
/*                                                                            */
/*  COPY - Copy a signal                                                      */
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

#include "copy.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( CopyDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_COPY_LENGTH , EvLength ) ,
  EV_COMMAND ( IDC_COPY_START , EvStart ) ,
  EV_COMMAND ( IDC_COPY_END , EvEnd ) ,
END_RESPONSE_TABLE ;

CopyDialog::CopyDialog ( TWindow *parent ,
                         int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_COPY ), TWindow ( parent )
{
   n_signals = p_nsignals ;
   signals = p_signals ;

	length = new TButton ( this , IDC_COPY_LENGTH ) ;
	start = new TRadioButton ( this , IDC_COPY_START ) ;
	end = new TRadioButton ( this , IDC_COPY_END ) ;
	points = new TEdit ( this , IDC_COPY_POINTS ) ;
   points_valid = new TRangeValidator ( 1 , 9999999 ) ;
   points->SetValidator ( points_valid ) ;

	source = new TListBox ( this , IDC_COPY_SOURCE ) ;
	dest = new TComboBox ( this , IDC_COPY_DEST ) ;
}

CopyDialog::~CopyDialog ()
{
}

void CopyDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   points->SetText ( "0" ) ;

   parm_start = 1 ;
   start->Check () ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      source->AddString ( sigptr->name ) ;
      dest->AddString ( sigptr->name ) ;
      }
   source->SetSelIndex ( 0 ) ;
   dest->SetSelIndex ( -1 ) ;
}

void CopyDialog::CmOk ()
{
   char msg[84] ;

   points->GetText ( msg , 83 ) ;
   parm_points = atoi ( msg ) ;
   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;

	TDialog::CmOk () ;
}

void CopyDialog::EvLength ()
{
   int i ;
   char msg[84] ;
   Signal *sigptr ;

   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      if (! strcmp ( sigptr->name , source_name )) {
         sprintf ( msg , "%d" , sigptr->n ) ;
         points->SetText ( msg ) ;
         return ;
         }
      }
}

void CopyDialog::EvStart ()
{
   parm_start = 1 ;
}

void CopyDialog::EvEnd ()
{
   parm_start = 0 ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void do_copy ( TWindow *parent )
{
   int nsigs, npts ;
	char error[256], rest[256] ;
	CopyDialog *cpy ;
   Signal **signals ;

   nsigs = get_signals ( &signals ) ;
	cpy = new CopyDialog ( parent , nsigs , signals ) ;

	if (cpy->Execute() == IDOK) {
      npts = cpy->parm_points ;
      if (cpy->parm_start == 0)
         npts = -npts ;
      if (process ( ID_PRED_NAME , cpy->dest_name , NULL , error , NULL ) < 0){
         parent->MessageBox ( error , "ERROR" ) ;
         delete cpy ;
         return ;
         }
      sprintf ( rest , "%d %s", npts , cpy->source_name ) ;
      if (process ( ID_PRED_COPY , rest , NULL , error , NULL ) < 0)
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete cpy ;
}


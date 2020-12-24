/******************************************************************************/
/*                                                                            */
/*  CONFCOMP - Build confidence compensation list                             */
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

#include "confcomp.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( ConfCompDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_CONFCOMP_CLEAR , EvClear ) ,
  EV_COMMAND ( IDC_CONFCOMP_CENTER , EvCenter ) ,
  EV_COMMAND ( IDC_CONFCOMP_DETREND , EvDetrend ) ,
  EV_COMMAND ( IDC_CONFCOMP_OFFSET , EvOffset ) ,
  EV_COMMAND ( IDC_CONFCOMP_SCALE , EvScale ) ,
  EV_COMMAND ( IDC_CONFCOMP_STD , EvStd ) ,
  EV_COMMAND ( IDC_CONFCOMP_LOG , EvLog ) ,
  EV_COMMAND ( IDC_CONFCOMP_DIFF , EvDiff ) ,
  EV_COMMAND ( IDC_CONFCOMP_SEASONAL , EvSeasonal ) ,
END_RESPONSE_TABLE ;

ConfCompDialog::ConfCompDialog ( TWindow *parent ,
                                 int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_CONFCOMP ), TWindow ( parent )
{
   n_signals = p_nsignals ;
   signals = p_signals ;

   clear = new TCheckBox ( this , IDC_CONFCOMP_CLEAR ) ;

	center = new TRadioButton ( this , IDC_CONFCOMP_CENTER ) ;
	detrend = new TRadioButton ( this , IDC_CONFCOMP_DETREND ) ;
	offset = new TRadioButton ( this , IDC_CONFCOMP_OFFSET ) ;
	scale = new TRadioButton ( this , IDC_CONFCOMP_SCALE ) ;
	std = new TRadioButton ( this , IDC_CONFCOMP_STD ) ;
	log = new TRadioButton ( this , IDC_CONFCOMP_LOG ) ;
	diff = new TRadioButton ( this , IDC_CONFCOMP_DIFF ) ;
	seasonal = new TRadioButton ( this , IDC_CONFCOMP_SEASONAL ) ;

   sig = new TListBox ( this , IDC_CONFCOMP_SIG ) ;
   per_label = new TStatic ( this , IDC_CONFCOMP_PER_LABEL ) ;
   per = new TComboBox ( this , IDC_CONFCOMP_PER ) ;
}

ConfCompDialog::~ConfCompDialog ()
{
}

void ConfCompDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   parm_clear = 0 ;
   clear->SetCheck ( BF_UNCHECKED ) ;

   parm_type = CCcenter ;
   center->Check () ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      sig->AddString ( sigptr->name ) ;
      per->AddString ( sigptr->name ) ;
      }

   sig->SetSelIndex ( 0 ) ;
   per->SetSelIndex ( -1 ) ;

   per_label->Show ( true ) ;
   per->Show ( true ) ;

   strcpy ( parm_sig , "" ) ;
   strcpy ( parm_per , "" ) ;
}

void ConfCompDialog::CmOk ()
{

   sig->GetSelString ( parm_sig , 255 ) ;
   strupr ( parm_sig ) ;
   per->GetText ( parm_per , 255 ) ;
   strupr ( parm_per ) ;

	TDialog::CmOk () ;
}

void ConfCompDialog::EvClear ()
{
   if (clear->GetCheck() == BF_CHECKED)
      parm_clear = 1 ;
   else 
      parm_clear = 0 ;
}

void ConfCompDialog::EvCenter ()
{
   parm_type = CCcenter ;
   per_label->Show ( true ) ;
   per->Show ( true ) ;
}

void ConfCompDialog::EvDetrend ()
{
   parm_type = CCdetrend ;
   per_label->Show ( true ) ;
   per->Show ( true ) ;
}

void ConfCompDialog::EvOffset ()
{
   parm_type = CCoffset ;
   per_label->Show ( true ) ;
   per->Show ( true ) ;
}

void ConfCompDialog::EvScale ()
{
   parm_type = CCscale ;
   per_label->Show ( true ) ;
   per->Show ( true ) ;
}

void ConfCompDialog::EvStd ()
{
   parm_type = CCstandardize ;
   per_label->Show ( true ) ;
   per->Show ( true ) ;
}

void ConfCompDialog::EvLog ()
{
   parm_type = CClog ;
   per_label->Show ( false ) ;
   per->Show ( false ) ;
}

void ConfCompDialog::EvDiff ()
{
   parm_type = CCdifference ;
   per_label->Show ( true ) ;
   per->Show ( true ) ;
}

void ConfCompDialog::EvSeasonal ()
{
   parm_type = CCseasonal ;
   per_label->Show ( true ) ;
   per->Show ( true ) ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void set_conf_comp ( TWindow *parent )
{
   int nsigs, type ;
	char error[256], rest[256] ;
	ConfCompDialog *ccd ;
   Signal **signals ;

   nsigs = get_signals ( &signals ) ;
	ccd = new ConfCompDialog ( parent , nsigs , signals ) ;

	if (ccd->Execute() == IDOK) {
      if (ccd->parm_clear)
         process ( ID_PRED_CLEAR_CONFIDENCE , NULL , NULL , error , NULL ) ;

      strcpy ( rest , ccd->parm_sig ) ;
      if (strlen ( ccd->parm_per)) {
         strcat ( rest , " PER " ) ;
         strcat ( rest , ccd->parm_per ) ;
         }

      switch (ccd->parm_type) {
         case CCcenter: type = ID_PRED_CONFIDENCE_CENTER ; break ;
         case CCdetrend: type = ID_PRED_CONFIDENCE_DETREND ; break ;
         case CCoffset: type = ID_PRED_CONFIDENCE_OFFSET ; break ;
         case CCscale: type = ID_PRED_CONFIDENCE_SCALE ; break ;
         case CCstandardize: type = ID_PRED_CONFIDENCE_STANDARDIZE ; break ;
         case CClog: type = ID_PRED_CONFIDENCE_LOG ; break ;
         case CCdifference: type = ID_PRED_CONFIDENCE_DIFFERENCE ; break ;
         case CCseasonal: type = ID_PRED_CONFIDENCE_SEASONAL ; break ;
         }
      if (process ( type , rest , NULL , error , NULL ) < 0)
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete ccd ;
}


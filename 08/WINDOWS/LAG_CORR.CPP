/******************************************************************************/
/*                                                                            */
/*  LAGCORR - LagCorr a signal                                                  */
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

#include "lag_corr.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( LagCorrDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_LAGCORR_AUTO , EvAuto ) ,
  EV_COMMAND ( IDC_LAGCORR_PARTAUTO , EvPartAuto ) ,
  EV_COMMAND ( IDC_LAGCORR_CROSS , EvCross ) ,
  EV_COMMAND ( IDC_LAGCORR_PARTCROSS , EvPartCross ) ,
END_RESPONSE_TABLE ;

LagCorrDialog::LagCorrDialog ( TWindow *parent ,
                               int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_LAGCORR ), TWindow ( parent )
{
   n_signals = p_nsignals ;
   signals = p_signals ;

	autoc = new TRadioButton ( this , IDC_LAGCORR_AUTO ) ;
	part_autoc = new TRadioButton ( this , IDC_LAGCORR_PARTAUTO ) ;
	cross = new TRadioButton ( this , IDC_LAGCORR_CROSS ) ;
	part_cross = new TRadioButton ( this , IDC_LAGCORR_PARTCROSS ) ;

   source = new TListBox ( this , IDC_LAGCORR_SOURCE ) ;
   source2 = new TListBox ( this , IDC_LAGCORR_SOURCE2 ) ;
	source2_label = new TStatic ( this , IDC_LAGCORR_SOURCE2_LABEL ) ;
	dest = new TComboBox ( this , IDC_LAGCORR_DEST ) ;
	maxlag = new TEdit ( this , IDC_LAGCORR_MAXLAG ) ;
   maxlag_valid = new TRangeValidator ( 1 , 999999 ) ;
   maxlag->SetValidator ( maxlag_valid ) ;
}

LagCorrDialog::~LagCorrDialog ()
{
}

void LagCorrDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   strcpy ( source_name , "" ) ;
   strcpy ( source2_name , "" ) ;
   strcpy ( dest_name , "" ) ;

   maxlag->SetText ( "10" ) ;

   corr_type = ID_PRED_AUTOCORR ;
   autoc->Check () ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      source->AddString ( sigptr->name ) ;
      source2->AddString ( sigptr->name ) ;
      dest->AddString ( sigptr->name ) ;
      }
   source->SetSelIndex ( 0 ) ;
   source2->SetSelIndex ( 0 ) ;
   dest->SetSelIndex ( -1 ) ;
   source2_label->Show ( false ) ;
   source2->Show ( false ) ;
}

void LagCorrDialog::CmOk ()
{
   char msg[84] ;

   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
   source2->GetSelString ( source2_name , 255 ) ;
   strupr ( source2_name ) ;
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;
   maxlag->GetText ( msg , 83 ) ;
   parm_maxlag = atoi ( msg ) ;

	TDialog::CmOk () ;
}

void LagCorrDialog::EvAuto ()
{
   corr_type = ID_PRED_AUTOCORR ;
   source2_label->Show ( false ) ;
   source2->Show ( false ) ;
}

void LagCorrDialog::EvPartAuto ()
{
   corr_type = ID_PRED_P_AUTOCORR ;
   source2_label->Show ( false ) ;
   source2->Show ( false ) ;
}

void LagCorrDialog::EvCross ()
{
   corr_type = ID_PRED_CROSSCORR ;
   source2_label->Show ( true ) ;
   source2->Show ( true ) ;
}

void LagCorrDialog::EvPartCross ()
{
   corr_type = ID_PRED_P_CROSSCORR ;
   source2_label->Show ( true ) ;
   source2->Show ( true ) ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void lag_corr ( TWindow *parent )
{
   int nsigs, type ;
	char error[256], rest[256] ;
	LagCorrDialog *lc ;
   Signal **signals ;

   nsigs = get_signals ( &signals ) ;
	lc = new LagCorrDialog ( parent , nsigs , signals ) ;

	if (lc->Execute() == IDOK) {
      type = lc->corr_type ;
      if (process ( ID_PRED_NAME , lc->dest_name , NULL , error , NULL ) < 0){
         parent->MessageBox ( error , "ERROR" ) ;
         delete lc ;
         return ;
         }
      if ((type == ID_PRED_AUTOCORR)  ||  (type == ID_PRED_P_AUTOCORR)) {
         sprintf ( rest , "%d %s", lc->parm_maxlag, lc->source_name ) ;
         if (process ( type , rest , NULL , error , NULL ) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      else if ((type == ID_PRED_CROSSCORR)  ||  (type == ID_PRED_P_CROSSCORR)) {
         sprintf ( rest , "%d %s AND %s", lc->parm_maxlag,
                   lc->source_name, lc->source2_name ) ;
         if (process ( type , rest , NULL , error , NULL ) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      }

	delete lc ;
}


/******************************************************************************/
/*                                                                            */
/*  MODIFY - Modify (and UNDO) a signal                                       */
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

#include "modify.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( ModifyDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_MODIFY_UNDO , EvUndo ) ,
  EV_COMMAND ( IDC_MODIFY_MEDIAN , EvMedian ) ,
  EV_COMMAND ( IDC_MODIFY_CENTER , EvCenter ) ,
  EV_COMMAND ( IDC_MODIFY_DETREND , EvDetrend ) ,
  EV_COMMAND ( IDC_MODIFY_OFFSET , EvOffset ) ,
  EV_COMMAND ( IDC_MODIFY_SCALE , EvScale ) ,
  EV_COMMAND ( IDC_MODIFY_STDIZE , EvStdize ) ,
  EV_COMMAND ( IDC_MODIFY_DIFF , EvDiff ) ,
  EV_COMMAND ( IDC_MODIFY_SEASONAL , EvSeasonal ) ,
  EV_COMMAND ( IDC_MODIFY_LOG , EvLog ) ,
  EV_COMMAND ( IDC_MODIFY_EXP , EvExp ) ,
  EV_COMMAND ( IDC_MODIFY_INTEGRATE , EvIntegrate ) ,
END_RESPONSE_TABLE ;

ModifyDialog::ModifyDialog ( TWindow *parent , int p_nsignals ,
                             Signal **p_signals )
	: TDialog ( parent , IDD_MODIFY ), TWindow ( parent )
{
   n_signals = p_nsignals ;
   signals = p_signals ;

	undo = new TCheckBox ( this , IDC_MODIFY_UNDO ) ;
	median = new TCheckBox ( this , IDC_MODIFY_MEDIAN ) ;

   center = new TRadioButton ( this , IDC_MODIFY_CENTER ) ;
   detrend = new TRadioButton ( this , IDC_MODIFY_DETREND ) ;
   offset = new TRadioButton ( this , IDC_MODIFY_OFFSET ) ;
	offset_parm = new TEdit ( this , IDC_MODIFY_OFFSET_PARM ) ;
   offset_valid = new TFilterValidator ( "0-9.-" );
   offset_parm->SetValidator ( offset_valid ) ;
   scale = new TRadioButton ( this , IDC_MODIFY_SCALE ) ;
	scale_parm = new TEdit ( this , IDC_MODIFY_SCALE_PARM ) ;
   scale_valid = new TFilterValidator ( "0-9." );
   scale_parm->SetValidator ( scale_valid ) ;
   stdize = new TRadioButton ( this , IDC_MODIFY_STDIZE ) ;
   diff = new TRadioButton ( this , IDC_MODIFY_DIFF ) ;
	diff_parm = new TEdit ( this , IDC_MODIFY_DIFF_PARM ) ;
   diff_valid = new TRangeValidator ( 1 , 3 );
   diff_parm->SetValidator ( diff_valid ) ;
   seasonal = new TRadioButton ( this , IDC_MODIFY_SEASONAL ) ;
	seasonal_parm = new TEdit ( this , IDC_MODIFY_SEASONAL_PARM ) ;
   seasonal_valid = new TRangeValidator ( 1 , 99999 );
   seasonal_parm->SetValidator ( seasonal_valid ) ;
   log = new TRadioButton ( this , IDC_MODIFY_LOG ) ;
   exp = new TRadioButton ( this , IDC_MODIFY_EXP ) ;
   integrate = new TRadioButton ( this , IDC_MODIFY_INTEGRATE ) ;
	integrate_parm = new TEdit ( this , IDC_MODIFY_INTEGRATE_PARM ) ;
   integrate_valid = new TRangeValidator ( 1 , 5 );
   integrate_parm->SetValidator ( integrate_valid ) ;

	source_label = new TStatic ( this , IDC_MODIFY_SOURCE_LABEL ) ;
	source = new TListBox ( this , IDC_MODIFY_SOURCE ) ;
	per_label = new TStatic ( this , IDC_MODIFY_PER_LABEL ) ;
	per = new TListBox ( this , IDC_MODIFY_PER ) ;
}

ModifyDialog::~ModifyDialog ()
{
}

void ModifyDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   parm_median = 0 ;
   median->SetCheck ( BF_UNCHECKED ) ;
   parm_undo = 0 ;
   undo->SetCheck ( BF_UNCHECKED ) ;
   undo->Show ( true ) ;

   parm_offset = 0.0 ;
   offset_parm->SetText ( "0.0" ) ;
   parm_scale = 1.0 ;
   scale_parm->SetText ( "1.0" ) ;
   parm_diff = 1 ;
   diff_parm->SetText ( "1" ) ;
   parm_seasonal = 1 ;
   seasonal_parm->SetText ( "1" ) ;
   parm_integrate = 1 ;
   integrate_parm->SetText ( "1" ) ;

   parm_type = ID_PRED_CENTER ;
   center->Check () ;
   offset_parm->Show ( false ) ;
   scale_parm->Show ( false ) ;
   diff_parm->Show ( false ) ;
   seasonal_parm->Show ( false ) ;
   integrate_parm->Show ( false ) ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      source->AddString ( sigptr->name ) ;
      per->AddString ( sigptr->name ) ;
      }
   source->SetSelIndex ( 0 ) ;
   per->SetSelIndex ( -1 ) ;

   per_label->Show ( false ) ;
   per->Show ( false ) ;
}

void ModifyDialog::CmOk ()
{
   char msg[84] ;
   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
   if (per->GetSelIndex () >= 0) {
      per->GetSelString ( per_name , 255 ) ;
      strupr ( per_name ) ;
      }
   else
      strcpy ( per_name , "" ) ;

   offset_parm->GetText ( msg , 83 ) ;
   parm_offset = atof ( msg ) ;
   scale_parm->GetText ( msg , 83 ) ;
   parm_scale = atof ( msg ) ;
   diff_parm->GetText ( msg , 83 ) ;
   parm_diff = atoi ( msg ) ;
   seasonal_parm->GetText ( msg , 83 ) ;
   parm_seasonal = atoi ( msg ) ;
   integrate_parm->GetText ( msg , 83 ) ;
   parm_integrate = atoi ( msg ) ;

	TDialog::CmOk () ;
}

void ModifyDialog::EvUndo ()
{
   if (undo->GetCheck() == BF_CHECKED) {
      parm_undo = 1 ;
      per_label->Show ( true ) ;
      per->Show ( true ) ;
      parm_median = 0 ;
      median->SetCheck ( BF_UNCHECKED ) ;
      median->Show ( false ) ;
      offset_parm->Show ( false ) ;
      scale_parm->Show ( false ) ;
      diff_parm->Show ( false ) ;
      seasonal_parm->Show ( false ) ;
      integrate_parm->Show ( false ) ;
      }
   else {
      parm_undo = 0 ;
      per_label->Show ( false ) ;
      per->Show ( false ) ;
      if (parm_type == ID_PRED_CENTER)
         median->Show ( true ) ;
      if (parm_type == ID_PRED_OFFSET)
         offset_parm->Show ( true ) ;
      if (parm_type == ID_PRED_SCALE)
         scale_parm->Show ( true ) ;
      if (parm_type == ID_PRED_DIFFERENCE)
         diff_parm->Show ( true ) ;
      if (parm_type == ID_PRED_SEASONAL_DIFFERENCE)
         seasonal_parm->Show ( true ) ;
      if (parm_type == ID_PRED_INTEGRATE)
         integrate_parm->Show ( true ) ;
      }
}

void ModifyDialog::EvMedian ()
{
   if (median->GetCheck() == BF_CHECKED) {
      parm_median = 1 ;
      parm_undo = 0 ;
      undo->SetCheck ( BF_UNCHECKED ) ;
      undo->Show ( false ) ;
      per_label->Show ( false ) ;
      per->Show ( false ) ;
      }
   else {
      parm_median = 0 ;
      undo->Show ( true ) ;
      }
}

void ModifyDialog::EvCenter ()
{
   parm_type = ID_PRED_CENTER ;
   if (parm_undo)
      median->Show ( false ) ;
   else 
      median->Show ( true ) ;
   offset_parm->Show ( false ) ;
   scale_parm->Show ( false ) ;
   diff_parm->Show ( false ) ;
   seasonal_parm->Show ( false ) ;
   integrate_parm->Show ( false ) ;
   if (parm_median) {
      parm_undo = 0 ;
      undo->SetCheck ( BF_UNCHECKED ) ;
      undo->Show ( false ) ;
      per_label->Show ( false ) ;
      per->Show ( false ) ;
      }
   else 
      undo->Show ( true ) ;
}

void ModifyDialog::EvDetrend ()
{
   parm_type = ID_PRED_DETREND ;
   median->Show ( false ) ;
   offset_parm->Show ( false ) ;
   scale_parm->Show ( false ) ;
   diff_parm->Show ( false ) ;
   seasonal_parm->Show ( false ) ;
   integrate_parm->Show ( false ) ;
   undo->Show ( true ) ;
}

void ModifyDialog::EvOffset ()
{
   parm_type = ID_PRED_OFFSET ;
   median->Show ( false ) ;
   if (parm_undo)
      offset_parm->Show ( false ) ;
   else 
      offset_parm->Show ( true ) ;
   scale_parm->Show ( false ) ;
   diff_parm->Show ( false ) ;
   seasonal_parm->Show ( false ) ;
   integrate_parm->Show ( false ) ;
   undo->Show ( true ) ;
}

void ModifyDialog::EvScale ()
{
   parm_type = ID_PRED_SCALE ;
   median->Show ( false ) ;
   offset_parm->Show ( false ) ;
   if (parm_undo)
      scale_parm->Show ( false ) ;
   else 
      scale_parm->Show ( true ) ;
   diff_parm->Show ( false ) ;
   seasonal_parm->Show ( false ) ;
   integrate_parm->Show ( false ) ;
   undo->Show ( true ) ;
}

void ModifyDialog::EvStdize ()
{
   parm_type = ID_PRED_STANDARDIZE ;
   median->Show ( false ) ;
   offset_parm->Show ( false ) ;
   scale_parm->Show ( false ) ;
   diff_parm->Show ( false ) ;
   seasonal_parm->Show ( false ) ;
   integrate_parm->Show ( false ) ;
   undo->Show ( true ) ;
}

void ModifyDialog::EvDiff ()
{
   parm_type = ID_PRED_DIFFERENCE ;
   median->Show ( false ) ;
   offset_parm->Show ( false ) ;
   scale_parm->Show ( false ) ;
   if (parm_undo)
      diff_parm->Show ( false ) ;
   else 
      diff_parm->Show ( true ) ;
   seasonal_parm->Show ( false ) ;
   integrate_parm->Show ( false ) ;
   undo->Show ( true ) ;
}

void ModifyDialog::EvSeasonal ()
{
   parm_type = ID_PRED_SEASONAL_DIFFERENCE ;
   median->Show ( false ) ;
   offset_parm->Show ( false ) ;
   scale_parm->Show ( false ) ;
   diff_parm->Show ( false ) ;
   if (parm_undo)
      seasonal_parm->Show ( false ) ;
   else 
      seasonal_parm->Show ( true ) ;
   integrate_parm->Show ( false ) ;
   undo->Show ( true ) ;
}

void ModifyDialog::EvLog ()
{
   parm_type = ID_PRED_LOG ;
   median->Show ( false ) ;
   offset_parm->Show ( false ) ;
   scale_parm->Show ( false ) ;
   diff_parm->Show ( false ) ;
   seasonal_parm->Show ( false ) ;
   integrate_parm->Show ( false ) ;
   parm_undo = 0 ;
   undo->SetCheck ( BF_UNCHECKED ) ;
   undo->Show ( false ) ;
   per_label->Show ( false ) ;
   per->Show ( false ) ;
}

void ModifyDialog::EvExp ()
{
   parm_type = ID_PRED_EXP ;
   median->Show ( false ) ;
   offset_parm->Show ( false ) ;
   scale_parm->Show ( false ) ;
   diff_parm->Show ( false ) ;
   seasonal_parm->Show ( false ) ;
   integrate_parm->Show ( false ) ;
   parm_undo = 0 ;
   undo->SetCheck ( BF_UNCHECKED ) ;
   undo->Show ( false ) ;
   per_label->Show ( false ) ;
   per->Show ( false ) ;
}

void ModifyDialog::EvIntegrate ()
{
   parm_type = ID_PRED_INTEGRATE ;
   median->Show ( false ) ;
   offset_parm->Show ( false ) ;
   scale_parm->Show ( false ) ;
   diff_parm->Show ( false ) ;
   seasonal_parm->Show ( false ) ;
   if (parm_undo)
      integrate_parm->Show ( false ) ;
   else 
      integrate_parm->Show ( true ) ;
   parm_undo = 0 ;
   undo->SetCheck ( BF_UNCHECKED ) ;
   undo->Show ( false ) ;
   per_label->Show ( false ) ;
   per->Show ( false ) ;
}

/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void do_modify ( TWindow *parent )
{
   int nsigs, type ;
	char error[256], rest[256] ;
	ModifyDialog *mod ;
   Signal **signals ;

   nsigs = get_signals ( &signals ) ;
	mod = new ModifyDialog ( parent , nsigs , signals ) ;

	if (mod->Execute() == IDOK) {

      if (mod->parm_type == ID_PRED_CENTER) {
         if (mod->parm_undo)
            type = ID_PRED_UNDO_CENTER ;
         else if (mod->parm_median)
            type = ID_PRED_MEDIAN_CENTER ;
         else 
            type = ID_PRED_CENTER ;
         strcpy ( rest , mod->source_name ) ;
         }

      else if (mod->parm_type == ID_PRED_DETREND) {
         if (mod->parm_undo)
            type = ID_PRED_UNDO_DETREND ;
         else 
            type = ID_PRED_DETREND ;
         strcpy ( rest , mod->source_name ) ;
         }

      else if (mod->parm_type == ID_PRED_OFFSET) {
         if (mod->parm_undo) {
            type = ID_PRED_UNDO_OFFSET ;
            strcpy ( rest , mod->source_name ) ;
            }
         else {
            type = ID_PRED_OFFSET ;
            sprintf ( rest , "%lf %s", mod->parm_offset, mod->source_name ) ;
            }
         }

      else if (mod->parm_type == ID_PRED_SCALE) {
         if (mod->parm_undo) {
            type = ID_PRED_UNDO_SCALE ;
            strcpy ( rest , mod->source_name ) ;
            }
         else {
            type = ID_PRED_SCALE ;
            sprintf ( rest , "%lf %s", mod->parm_scale, mod->source_name ) ;
            }
         }

      else if (mod->parm_type == ID_PRED_STANDARDIZE) {
         if (mod->parm_undo)
            type = ID_PRED_UNDO_STANDARDIZE ;
         else 
            type = ID_PRED_STANDARDIZE ;
         strcpy ( rest , mod->source_name ) ;
         }

      else if (mod->parm_type == ID_PRED_DIFFERENCE) {
         if (mod->parm_undo) {
            type = ID_PRED_UNDO_DIFFERENCE ;
            strcpy ( rest , mod->source_name ) ;
            }
         else {
            type = ID_PRED_DIFFERENCE ;
            sprintf ( rest , "%d %s", mod->parm_diff, mod->source_name ) ;
            }
         }

      else if (mod->parm_type == ID_PRED_SEASONAL_DIFFERENCE) {
         if (mod->parm_undo) {
            type = ID_PRED_UNDO_SEASONAL_DIFFERENCE ;
            strcpy ( rest , mod->source_name ) ;
            }
         else {
            type = ID_PRED_SEASONAL_DIFFERENCE ;
            sprintf ( rest , "%d %s", mod->parm_seasonal, mod->source_name ) ;
            }
         }

      else if (mod->parm_type == ID_PRED_LOG) {
         type = ID_PRED_LOG ;
         strcpy ( rest , mod->source_name ) ;
         }

      else if (mod->parm_type == ID_PRED_EXP) {
         type = ID_PRED_EXP ;
         strcpy ( rest , mod->source_name ) ;
         }

      else if (mod->parm_type == ID_PRED_INTEGRATE) {
         type = ID_PRED_INTEGRATE ;
         sprintf ( rest , "%d %s", mod->parm_integrate, mod->source_name ) ;
         }

      if (mod->parm_undo  &&  strlen ( mod->per_name )) {
         strcat ( rest , " PER " ) ;
         strcat ( rest , mod->per_name ) ;
         }

      if (process ( type , rest , NULL , error , NULL ) < 0 )
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete mod ;
}


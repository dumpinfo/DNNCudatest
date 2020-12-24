/******************************************************************************/
/*                                                                            */
/*  FILTER - Filter a signal                                                  */
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

#include "filter.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( FilterDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_FILTER_LOWPASS , EvLowpass ) ,
  EV_COMMAND ( IDC_FILTER_HIGHPASS , EvHighpass ) ,
  EV_COMMAND ( IDC_FILTER_BANDPASS , EvBandpass ) ,
  EV_COMMAND ( IDC_FILTER_SAVGOL , EvSavgol ) ,
  EV_COMMAND ( IDC_FILTER_MA , EvMA ) ,
  EV_COMMAND ( IDC_FILTER_QMF , EvQMF ) ,
  EV_COMMAND ( IDC_FILTER_MORLET , EvMorlet ) ,
  EV_COMMAND ( IDC_FILTER_DETREND , EvDetrend ) ,
END_RESPONSE_TABLE ;

FilterDialog::FilterDialog ( TWindow *parent , int p_detrend ,
                             int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_FILTER ), TWindow ( parent )
{
   use_detrend = p_detrend ;
   n_signals = p_nsignals ;
   signals = p_signals ;

	lowpass = new TRadioButton ( this , IDC_FILTER_LOWPASS ) ;
	highpass = new TRadioButton ( this , IDC_FILTER_HIGHPASS ) ;
	bandpass = new TRadioButton ( this , IDC_FILTER_BANDPASS ) ;
	savgol = new TRadioButton ( this , IDC_FILTER_SAVGOL ) ;
	ma = new TRadioButton ( this , IDC_FILTER_MA ) ;
	qmf = new TRadioButton ( this , IDC_FILTER_QMF ) ;
	morlet = new TRadioButton ( this , IDC_FILTER_MORLET ) ;
	source = new TListBox ( this , IDC_FILTER_SOURCE ) ;
	dest_label = new TStatic ( this , IDC_FILTER_DEST_LABEL ) ;
	dest = new TComboBox ( this , IDC_FILTER_DEST ) ;
	freq_label = new TStatic ( this , IDC_FILTER_FREQ_LABEL ) ;
	freq = new TEdit ( this , IDC_FILTER_FREQ ) ;
   freq_valid = new TFilterValidator ( "0-9." );
   freq->SetValidator ( freq_valid ) ;
	width_label = new TStatic ( this , IDC_FILTER_WIDTH_LABEL ) ;
	width = new TEdit ( this , IDC_FILTER_WIDTH ) ;
   width_valid = new TFilterValidator ( "0-9." );
   width->SetValidator ( width_valid ) ;
	points_label = new TStatic ( this , IDC_FILTER_POINTS_LABEL ) ;
	points = new TEdit ( this , IDC_FILTER_POINTS ) ;
   points_valid = new TRangeValidator ( 2 , 9999 ) ;
   points->SetValidator ( points_valid ) ;
	order_label = new TStatic ( this , IDC_FILTER_ORDER_LABEL ) ;
	order = new TEdit ( this , IDC_FILTER_ORDER ) ;
   order_valid = new TRangeValidator ( 2 , 20 ) ;
   order->SetValidator ( order_valid ) ;
	detrend = new TCheckBox ( this , IDC_FILTER_DETREND ) ;
}

FilterDialog::~FilterDialog ()
{
}

void FilterDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   freq->SetText ( "0.0" ) ;
   width->SetText ( "0.05" ) ;
   points->SetText ( "10" ) ;
   order->SetText ( "2" ) ;

   if (use_detrend)
      detrend->Check () ;
   else 
      detrend->SetCheck ( BF_UNCHECKED ) ;

   filter_type = ID_PRED_LOWPASS ;
   lowpass->Check () ;
   freq_label->Show ( true ) ;
   freq->Show ( true ) ;
   width_label->Show ( true ) ;
   width->Show ( true ) ;
   points_label->Show ( false ) ;
   points->Show ( false ) ;
   order_label->Show ( false ) ;
   order->Show ( false ) ;
   detrend->Show ( true ) ;
   dest_label->Show ( false ) ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      source->AddString ( sigptr->name ) ;
      dest->AddString ( sigptr->name ) ;
      }
   source->SetSelIndex ( 0 ) ;
   dest->SetSelIndex ( 0 ) ;
}

void FilterDialog::CmOk ()
{
   char msg[84] ;
   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;
   freq->GetText ( msg , 83 ) ;
   parm_freq = atof ( msg ) ;
   width->GetText ( msg , 83 ) ;
   parm_width = atof ( msg ) ;
   points->GetText ( msg , 83 ) ;
   parm_points = atoi ( msg ) ;
   order->GetText ( msg , 83 ) ;
   parm_order = atoi ( msg ) ;

	TDialog::CmOk () ;
}

void FilterDialog::EvLowpass ()
{
   filter_type = ID_PRED_LOWPASS ;
   freq_label->Show ( true ) ;
   freq->Show ( true ) ;
   width_label->Show ( true ) ;
   width->Show ( true ) ;
   points_label->Show ( false ) ;
   points->Show ( false ) ;
   order_label->Show ( false ) ;
   order->Show ( false ) ;
   detrend->Show ( true ) ;
   dest_label->Show ( false ) ;
}

void FilterDialog::EvHighpass ()
{
   filter_type = ID_PRED_HIGHPASS ;
   freq_label->Show ( true ) ;
   freq->Show ( true ) ;
   width_label->Show ( true ) ;
   width->Show ( true ) ;
   points_label->Show ( false ) ;
   points->Show ( false ) ;
   order_label->Show ( false ) ;
   order->Show ( false ) ;
   detrend->Show ( true ) ;
   dest_label->Show ( false ) ;
}

void FilterDialog::EvBandpass ()
{
   filter_type = ID_PRED_BANDPASS ;
   freq_label->Show ( true ) ;
   freq->Show ( true ) ;
   width_label->Show ( true ) ;
   width->Show ( true ) ;
   points_label->Show ( false ) ;
   points->Show ( false ) ;
   order_label->Show ( false ) ;
   order->Show ( false ) ;
   detrend->Show ( true ) ;
   dest_label->Show ( false ) ;
}

void FilterDialog::EvSavgol ()
{
   filter_type = ID_PRED_SAVGOL ;
   freq_label->Show ( false ) ;
   freq->Show ( false ) ;
   width_label->Show ( false ) ;
   width->Show ( false ) ;
   points_label->Show ( true ) ;
   points->Show ( true ) ;
   order_label->Show ( true ) ;
   order->Show ( true ) ;
   detrend->Show ( false ) ;
   dest_label->Show ( false ) ;
}

void FilterDialog::EvMA ()
{
   filter_type = ID_PRED_MOV_AVG ;
   freq_label->Show ( false ) ;
   freq->Show ( false ) ;
   width_label->Show ( false ) ;
   width->Show ( false ) ;
   points_label->Show ( true ) ;
   points->Show ( true ) ;
   order_label->Show ( false ) ;
   order->Show ( false ) ;
   detrend->Show ( false ) ;
   dest_label->Show ( false ) ;
}

void FilterDialog::EvQMF ()
{
   filter_type = ID_PRED_QMF ;
   freq_label->Show ( true ) ;
   freq->Show ( true ) ;
   width_label->Show ( true ) ;
   width->Show ( true ) ;
   points_label->Show ( false ) ;
   points->Show ( false ) ;
   order_label->Show ( false ) ;
   order->Show ( false ) ;
   detrend->Show ( true ) ;
   dest_label->Show ( true ) ;
}

void FilterDialog::EvMorlet ()
{
   filter_type = ID_PRED_MORLET ;
   freq_label->Show ( true ) ;
   freq->Show ( true ) ;
   width_label->Show ( true ) ;
   width->Show ( true ) ;
   points_label->Show ( false ) ;
   points->Show ( false ) ;
   order_label->Show ( false ) ;
   order->Show ( false ) ;
   detrend->Show ( true ) ;
   dest_label->Show ( true ) ;
}

void FilterDialog::EvDetrend ()
{
   if (detrend->GetCheck() == BF_CHECKED)
      use_detrend = 1 ;
   else 
      use_detrend = 0 ;
}

/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void do_filter ( TWindow *parent )
{
   int nsigs, type ;
	char error[256], rest[256] ;
	FilterDialog *filt ;
   Signal **signals ;
   MiscParams *mp ;

   mp = get_misc_params () ;
   nsigs = get_signals ( &signals ) ;
	filt = new FilterDialog ( parent , mp->padding , nsigs , signals ) ;

	if (filt->Execute() == IDOK) {
      type = filt->filter_type ;
      if (process ( ID_PRED_NAME , filt->dest_name , NULL , error , NULL ) < 0){
         parent->MessageBox ( error , "ERROR" ) ;
         delete filt ;
         return ;
         }
      if ((type == ID_PRED_LOWPASS)  ||  (type == ID_PRED_HIGHPASS)
       || (type == ID_PRED_BANDPASS)  ||  (type == ID_PRED_QMF)
       || (type == ID_PRED_MORLET)) {
         if (filt->use_detrend)
            strcpy ( rest , "DETREND" ) ;
         else 
            strcpy ( rest , "MEAN" ) ;
         if (process ( ID_PRED_PADDING , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete filt ;
            return ;
            }
         sprintf ( rest , "%lf %lf %s", filt->parm_freq , filt->parm_width ,
                   filt->source_name ) ;
         if (process ( type , rest , NULL , error , NULL ) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      else if (type == ID_PRED_SAVGOL) {
         sprintf ( rest , "%d %d %s", filt->parm_points , filt->parm_order ,
                   filt->source_name ) ;
         if (process ( type , rest , NULL , error , NULL ) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      else if (type == ID_PRED_MOV_AVG) {
         sprintf ( rest , "%d %s", filt->parm_points , filt->source_name ) ;
         if (process ( type , rest , NULL , error , NULL ) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      }

	delete filt ;
}


/******************************************************************************/
/*                                                                            */
/*  SPECTRUM - Compute spectra                                                */
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

#include "spectrum.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( SpectrumDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_SPECTRUM_DFT , EvDft ) ,
  EV_COMMAND ( IDC_SPECTRUM_MAXENT , EvMaxent ) ,
  EV_COMMAND ( IDC_SPECTRUM_NOWINDOW , EvNoWindow ) ,
  EV_COMMAND ( IDC_SPECTRUM_WELCH , EvWelch ) ,
END_RESPONSE_TABLE ;

SpectrumDialog::SpectrumDialog ( TWindow *parent , int p_window ,
                                 int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_SPECTRUM ), TWindow ( parent )
{
   parm_window = p_window ;
   n_signals = p_nsignals ;
   signals = p_signals ;

	dft = new TRadioButton ( this , IDC_SPECTRUM_DFT ) ;
	maxent = new TRadioButton ( this , IDC_SPECTRUM_MAXENT ) ;
	nowindow = new TRadioButton ( this , IDC_SPECTRUM_NOWINDOW ) ;
	welch = new TRadioButton ( this , IDC_SPECTRUM_WELCH ) ;

   source = new TListBox ( this , IDC_SPECTRUM_SOURCE ) ;
	dest = new TComboBox ( this , IDC_SPECTRUM_DEST ) ;

	points_label = new TStatic ( this , IDC_SPECTRUM_POINTS_LABEL ) ;
	points = new TEdit ( this , IDC_SPECTRUM_POINTS ) ;
   points_valid = new TRangeValidator ( 2 , 999999 ) ;
   points->SetValidator ( points_valid ) ;
	order_label = new TStatic ( this , IDC_SPECTRUM_ORDER_LABEL ) ;
	order = new TEdit ( this , IDC_SPECTRUM_ORDER ) ;
   order_valid = new TRangeValidator ( 2 , 9999 ) ;
   order->SetValidator ( order_valid ) ;
	dest_label = new TStatic ( this , IDC_SPECTRUM_DEST_LABEL ) ;
}

SpectrumDialog::~SpectrumDialog ()
{
}

void SpectrumDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   points->SetText ( "1000" ) ;
   order->SetText ( "50" ) ;

   if (parm_window)
      welch->Check () ;
   else 
      nowindow->Check () ;

   parm_type = ID_PRED_SPECTRUM ;
   dft->Check () ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      source->AddString ( sigptr->name ) ;
      dest->AddString ( sigptr->name ) ;
      }
   source->SetSelIndex ( 0 ) ;
   dest->SetSelIndex ( -1 ) ;

   points_label->Show ( false ) ;
   points->Show ( false ) ;
   order_label->Show ( false ) ;
   order->Show ( false ) ;
   nowindow->Show ( true ) ;
   welch->Show ( true ) ;
   dest_label->Show ( true ) ;
}

void SpectrumDialog::CmOk ()
{
   char msg[84] ;
   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;
   points->GetText ( msg , 83 ) ;
   parm_points = atoi ( msg ) ;
   order->GetText ( msg , 83 ) ;
   parm_order = atoi ( msg ) ;

	TDialog::CmOk () ;
}

void SpectrumDialog::EvDft ()
{
   parm_type = ID_PRED_SPECTRUM ;
   points_label->Show ( false ) ;
   points->Show ( false ) ;
   order_label->Show ( false ) ;
   order->Show ( false ) ;
   nowindow->Show ( true ) ;
   welch->Show ( true ) ;
   dest_label->Show ( true ) ;
}

void SpectrumDialog::EvMaxent ()
{
   parm_type = ID_PRED_MAXENT ;
   points_label->Show ( true ) ;
   points->Show ( true ) ;
   order_label->Show ( true ) ;
   order->Show ( true ) ;
   nowindow->Show ( false ) ;
   welch->Show ( false ) ;
   dest_label->Show ( false ) ;
}

void SpectrumDialog::EvNoWindow ()
{
   parm_window = 0 ;
}

void SpectrumDialog::EvWelch ()
{
   parm_window = 1 ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void do_spectrum ( TWindow *parent )
{
   int nsigs ;
	char error[256], rest[256] ;
	SpectrumDialog *spec ;
   Signal **signals ;
   MiscParams *mp ;

   mp = get_misc_params () ;
   nsigs = get_signals ( &signals ) ;
	spec = new SpectrumDialog ( parent , mp->spectrum_window , nsigs , signals );

	if (spec->Execute() == IDOK) {
      if (process ( ID_PRED_NAME , spec->dest_name , NULL , error , NULL ) < 0){
         parent->MessageBox ( error , "ERROR" ) ;
         delete spec ;
         return ;
         }
      if (spec->parm_type == ID_PRED_SPECTRUM) {
         if (spec->parm_window)
            strcpy ( rest , "WELCH" ) ;
         else 
            strcpy ( rest , "NONE" ) ;
         if (process ( ID_PRED_SPECTRUM_WINDOW , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete spec ;
            return ;
            }
         if (process ( spec->parm_type , spec->source_name , NULL , error , NULL ) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      else if (spec->parm_type == ID_PRED_MAXENT) {
         sprintf ( rest , "%d %d %s", spec->parm_points , spec->parm_order ,
                   spec->source_name ) ;
         if (process ( spec->parm_type , rest , NULL , error , NULL ) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      }

	delete spec ;
}


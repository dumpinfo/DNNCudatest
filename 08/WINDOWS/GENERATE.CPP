/******************************************************************************/
/*                                                                            */
/*  GENERATE - Generate a signal                                              */
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

#include "generate.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( GenerateDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_GENERATE_ARMA , EvArma ) ,
  EV_COMMAND ( IDC_GENERATE_NORMAL , EvNormal ) ,
  EV_COMMAND ( IDC_GENERATE_UNIFORM , EvUniform ) ,
  EV_COMMAND ( IDC_GENERATE_RAMP , EvRamp ) ,
  EV_COMMAND ( IDC_GENERATE_SINE , EvSine ) ,
END_RESPONSE_TABLE ;

GenerateDialog::GenerateDialog ( TWindow *parent ,
                                 int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_GENERATE ), TWindow ( parent )
{
   n_signals = p_nsignals ;
   signals = p_signals ;

	Arma = new TRadioButton ( this , IDC_GENERATE_ARMA ) ;
	Normal = new TRadioButton ( this , IDC_GENERATE_NORMAL ) ;
	Uniform = new TRadioButton ( this , IDC_GENERATE_UNIFORM ) ;
	Ramp = new TRadioButton ( this , IDC_GENERATE_RAMP ) ;
	Sine = new TRadioButton ( this , IDC_GENERATE_SINE ) ;

	ar1_label = new TStatic ( this , IDC_GENERATE_AR1_LABEL ) ;
	ar1 = new TEdit ( this , IDC_GENERATE_AR1 ) ;
   ar1_valid = new TFilterValidator ( "0-9.-" );
   ar1->SetValidator ( ar1_valid ) ;

	ar2_label = new TStatic ( this , IDC_GENERATE_AR2_LABEL ) ;
	ar2 = new TEdit ( this , IDC_GENERATE_AR2 ) ;
   ar2_valid = new TFilterValidator ( "0-9.-" );
   ar2->SetValidator ( ar2_valid ) ;

	ar10_label = new TStatic ( this , IDC_GENERATE_AR10_LABEL ) ;
	ar10 = new TEdit ( this , IDC_GENERATE_AR10 ) ;
   ar10_valid = new TFilterValidator ( "0-9.-" );
   ar10->SetValidator ( ar10_valid ) ;

	ma1_label = new TStatic ( this , IDC_GENERATE_MA1_LABEL ) ;
	ma1 = new TEdit ( this , IDC_GENERATE_MA1 ) ;
   ma1_valid = new TFilterValidator ( "0-9.-" );
   ma1->SetValidator ( ma1_valid ) ;

	ma2_label = new TStatic ( this , IDC_GENERATE_MA2_LABEL ) ;
	ma2 = new TEdit ( this , IDC_GENERATE_MA2 ) ;
   ma2_valid = new TFilterValidator ( "0-9.-" );
   ma2->SetValidator ( ma2_valid ) ;

	ma10_label = new TStatic ( this , IDC_GENERATE_MA10_LABEL ) ;
	ma10 = new TEdit ( this , IDC_GENERATE_MA10 ) ;
   ma10_valid = new TFilterValidator ( "0-9.-" );
   ma10->SetValidator ( ma10_valid ) ;

	mean_label = new TStatic ( this , IDC_GENERATE_MEAN_LABEL ) ;
	mean = new TEdit ( this , IDC_GENERATE_MEAN ) ;
   mean_valid = new TFilterValidator ( "0-9.-" );
   mean->SetValidator ( mean_valid ) ;

	std_label = new TStatic ( this , IDC_GENERATE_STD_LABEL ) ;
	std = new TEdit ( this , IDC_GENERATE_STD ) ;
   std_valid = new TFilterValidator ( "0-9." );
   std->SetValidator ( std_valid ) ;

	low_label = new TStatic ( this , IDC_GENERATE_LOW_LABEL ) ;
	low = new TEdit ( this , IDC_GENERATE_LOW ) ;
   low_valid = new TFilterValidator ( "0-9.-" );
   low->SetValidator ( low_valid ) ;

	high_label = new TStatic ( this , IDC_GENERATE_HIGH_LABEL ) ;
	high = new TEdit ( this , IDC_GENERATE_HIGH ) ;
   high_valid = new TFilterValidator ( "0-9.-" );
   high->SetValidator ( high_valid ) ;

	height_label = new TStatic ( this , IDC_GENERATE_HEIGHT_LABEL ) ;
	height = new TEdit ( this , IDC_GENERATE_HEIGHT ) ;
   height_valid = new TFilterValidator ( "0-9." );
   height->SetValidator ( height_valid ) ;

	rperiod_label = new TStatic ( this , IDC_GENERATE_RPERIOD_LABEL ) ;
	rperiod = new TEdit ( this , IDC_GENERATE_RPERIOD ) ;
   rperiod_valid = new TFilterValidator ( "0-9." );
   rperiod->SetValidator ( rperiod_valid ) ;

	amp_label = new TStatic ( this , IDC_GENERATE_AMP_LABEL ) ;
	amp = new TEdit ( this , IDC_GENERATE_AMP ) ;
   amp_valid = new TFilterValidator ( "0-9." );
   amp->SetValidator ( amp_valid ) ;

	speriod_label = new TStatic ( this , IDC_GENERATE_SPERIOD_LABEL ) ;
	speriod = new TEdit ( this , IDC_GENERATE_SPERIOD ) ;
   speriod_valid = new TFilterValidator ( "0-9." );
   speriod->SetValidator ( speriod_valid ) ;

	phase_label = new TStatic ( this , IDC_GENERATE_PHASE_LABEL ) ;
	phase = new TEdit ( this , IDC_GENERATE_PHASE ) ;
   phase_valid = new TFilterValidator ( "0-9.-" );
   phase->SetValidator ( phase_valid ) ;

	points = new TEdit ( this , IDC_GENERATE_POINTS ) ;
   points_valid = new TRangeValidator ( 1 , 999999 ) ;
   points->SetValidator ( points_valid ) ;

	dest = new TComboBox ( this , IDC_GENERATE_DEST ) ;
}

GenerateDialog::~GenerateDialog ()
{
}

void GenerateDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   ar1->SetText ( "0.0" ) ;
   ar2->SetText ( "0.0" ) ;
   ar10->SetText ( "0.0" ) ;
   ma1->SetText ( "0.0" ) ;
   ma2->SetText ( "0.0" ) ;
   ma10->SetText ( "0.0" ) ;
   mean->SetText ( "0.0" ) ;
   std->SetText ( "1.0" ) ;
   low->SetText ( "0.0" ) ;
   high->SetText ( "1.0" ) ;
   height->SetText ( "1.0" ) ;
   rperiod->SetText ( "12.0" ) ;
   amp->SetText ( "1.0" ) ;
   speriod->SetText ( "12.0" ) ;
   phase->SetText ( "0.0" ) ;

   points->SetText ( "100" ) ;
   dest->SetText ( "" ) ;

   parm_type = 2 ;
   Normal->Check () ;

   ar1_label->Show ( false ) ;
   ar1->Show ( false ) ;
   ar2_label->Show ( false ) ;
   ar2->Show ( false ) ;
   ar10_label->Show ( false ) ;
   ar10->Show ( false ) ;
   ma1_label->Show ( false ) ;
   ma1->Show ( false ) ;
   ma2_label->Show ( false ) ;
   ma2->Show ( false ) ;
   ma10_label->Show ( false ) ;
   ma10->Show ( false ) ;
   mean_label->Show ( true ) ;
   mean->Show ( true ) ;
   std_label->Show ( true ) ;
   std->Show ( true ) ;
   low_label->Show ( false ) ;
   low->Show ( false ) ;
   high_label->Show ( false ) ;
   high->Show ( false ) ;
   height_label->Show ( false ) ;
   height->Show ( false ) ;
   rperiod_label->Show ( false ) ;
   rperiod->Show ( false ) ;
   amp_label->Show ( false ) ;
   amp->Show ( false ) ;
   speriod_label->Show ( false ) ;
   speriod->Show ( false ) ;
   phase_label->Show ( false ) ;
   phase->Show ( false ) ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      dest->AddString ( sigptr->name ) ;
      }
   dest->SetSelIndex ( -1 ) ;
}

void GenerateDialog::CmOk ()
{
   char msg[84] ;

   ar1->GetText ( msg , 83 ) ;
   parm_ar1 = atof ( msg ) ;
   ar2->GetText ( msg , 83 ) ;
   parm_ar2 = atof ( msg ) ;
   ar10->GetText ( msg , 83 ) ;
   parm_ar10 = atof ( msg ) ;
   ma1->GetText ( msg , 83 ) ;
   parm_ma1 = atof ( msg ) ;
   ma2->GetText ( msg , 83 ) ;
   parm_ma2 = atof ( msg ) ;
   ma10->GetText ( msg , 83 ) ;
   parm_ma10 = atof ( msg ) ;
   mean->GetText ( msg , 83 ) ;
   parm_mean = atof ( msg ) ;
   std->GetText ( msg , 83 ) ;
   parm_std = atof ( msg ) ;
   low->GetText ( msg , 83 ) ;
   parm_low = atof ( msg ) ;
   high->GetText ( msg , 83 ) ;
   parm_high = atof ( msg ) ;
   height->GetText ( msg , 83 ) ;
   parm_height = atof ( msg ) ;
   rperiod->GetText ( msg , 83 ) ;
   parm_rperiod = atof ( msg ) ;
   amp->GetText ( msg , 83 ) ;
   parm_amp = atof ( msg ) ;
   speriod->GetText ( msg , 83 ) ;
   parm_speriod = atof ( msg ) ;
   phase->GetText ( msg , 83 ) ;
   parm_phase = atof ( msg ) ;

   points->GetText ( msg , 83 ) ;
   parm_points = atoi ( msg ) ;

   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;

	TDialog::CmOk () ;
}

void GenerateDialog::EvArma ()
{
   parm_type = 1 ;

   ar1_label->Show ( true ) ;
   ar1->Show ( true ) ;
   ar2_label->Show ( true ) ;
   ar2->Show ( true ) ;
   ar10_label->Show ( true ) ;
   ar10->Show ( true ) ;
   ma1_label->Show ( true ) ;
   ma1->Show ( true ) ;
   ma2_label->Show ( true ) ;
   ma2->Show ( true ) ;
   ma10_label->Show ( true ) ;
   ma10->Show ( true ) ;
   mean_label->Show ( false ) ;
   mean->Show ( false ) ;
   std_label->Show ( false ) ;
   std->Show ( false ) ;
   low_label->Show ( false ) ;
   low->Show ( false ) ;
   high_label->Show ( false ) ;
   high->Show ( false ) ;
   height_label->Show ( false ) ;
   height->Show ( false ) ;
   rperiod_label->Show ( false ) ;
   rperiod->Show ( false ) ;
   amp_label->Show ( false ) ;
   amp->Show ( false ) ;
   speriod_label->Show ( false ) ;
   speriod->Show ( false ) ;
   phase_label->Show ( false ) ;
   phase->Show ( false ) ;
}

void GenerateDialog::EvNormal ()
{
   parm_type = 2 ;

   ar1_label->Show ( false ) ;
   ar1->Show ( false ) ;
   ar2_label->Show ( false ) ;
   ar2->Show ( false ) ;
   ar10_label->Show ( false ) ;
   ar10->Show ( false ) ;
   ma1_label->Show ( false ) ;
   ma1->Show ( false ) ;
   ma2_label->Show ( false ) ;
   ma2->Show ( false ) ;
   ma10_label->Show ( false ) ;
   ma10->Show ( false ) ;
   mean_label->Show ( true ) ;
   mean->Show ( true ) ;
   std_label->Show ( true ) ;
   std->Show ( true ) ;
   low_label->Show ( false ) ;
   low->Show ( false ) ;
   high_label->Show ( false ) ;
   high->Show ( false ) ;
   height_label->Show ( false ) ;
   height->Show ( false ) ;
   rperiod_label->Show ( false ) ;
   rperiod->Show ( false ) ;
   amp_label->Show ( false ) ;
   amp->Show ( false ) ;
   speriod_label->Show ( false ) ;
   speriod->Show ( false ) ;
   phase_label->Show ( false ) ;
   phase->Show ( false ) ;
}

void GenerateDialog::EvUniform ()
{
   parm_type = 3 ;

   ar1_label->Show ( false ) ;
   ar1->Show ( false ) ;
   ar2_label->Show ( false ) ;
   ar2->Show ( false ) ;
   ar10_label->Show ( false ) ;
   ar10->Show ( false ) ;
   ma1_label->Show ( false ) ;
   ma1->Show ( false ) ;
   ma2_label->Show ( false ) ;
   ma2->Show ( false ) ;
   ma10_label->Show ( false ) ;
   ma10->Show ( false ) ;
   mean_label->Show ( false ) ;
   mean->Show ( false ) ;
   std_label->Show ( false ) ;
   std->Show ( false ) ;
   low_label->Show ( true ) ;
   low->Show ( true ) ;
   high_label->Show ( true ) ;
   high->Show ( true ) ;
   height_label->Show ( false ) ;
   height->Show ( false ) ;
   rperiod_label->Show ( false ) ;
   rperiod->Show ( false ) ;
   amp_label->Show ( false ) ;
   amp->Show ( false ) ;
   speriod_label->Show ( false ) ;
   speriod->Show ( false ) ;
   phase_label->Show ( false ) ;
   phase->Show ( false ) ;
}

void GenerateDialog::EvRamp ()
{
   parm_type = 4 ;

   ar1_label->Show ( false ) ;
   ar1->Show ( false ) ;
   ar2_label->Show ( false ) ;
   ar2->Show ( false ) ;
   ar10_label->Show ( false ) ;
   ar10->Show ( false ) ;
   ma1_label->Show ( false ) ;
   ma1->Show ( false ) ;
   ma2_label->Show ( false ) ;
   ma2->Show ( false ) ;
   ma10_label->Show ( false ) ;
   ma10->Show ( false ) ;
   mean_label->Show ( false ) ;
   mean->Show ( false ) ;
   std_label->Show ( false ) ;
   std->Show ( false ) ;
   low_label->Show ( false ) ;
   low->Show ( false ) ;
   high_label->Show ( false ) ;
   high->Show ( false ) ;
   height_label->Show ( true ) ;
   height->Show ( true ) ;
   rperiod_label->Show ( true ) ;
   rperiod->Show ( true ) ;
   amp_label->Show ( false ) ;
   amp->Show ( false ) ;
   speriod_label->Show ( false ) ;
   speriod->Show ( false ) ;
   phase_label->Show ( false ) ;
   phase->Show ( false ) ;
}

void GenerateDialog::EvSine ()
{
   parm_type = 5 ;

   ar1_label->Show ( false ) ;
   ar1->Show ( false ) ;
   ar2_label->Show ( false ) ;
   ar2->Show ( false ) ;
   ar10_label->Show ( false ) ;
   ar10->Show ( false ) ;
   ma1_label->Show ( false ) ;
   ma1->Show ( false ) ;
   ma2_label->Show ( false ) ;
   ma2->Show ( false ) ;
   ma10_label->Show ( false ) ;
   ma10->Show ( false ) ;
   mean_label->Show ( false ) ;
   mean->Show ( false ) ;
   std_label->Show ( false ) ;
   std->Show ( false ) ;
   low_label->Show ( false ) ;
   low->Show ( false ) ;
   high_label->Show ( false ) ;
   high->Show ( false ) ;
   height_label->Show ( false ) ;
   height->Show ( false ) ;
   rperiod_label->Show ( false ) ;
   rperiod->Show ( false ) ;
   amp_label->Show ( true ) ;
   amp->Show ( true ) ;
   speriod_label->Show ( true ) ;
   speriod->Show ( true ) ;
   phase_label->Show ( true ) ;
   phase->Show ( true ) ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void generate ( TWindow *parent )
{
   int nsigs ;
	char error[256], rest[256] ;
	GenerateDialog *gd ;
   Signal **signals ;

   nsigs = get_signals ( &signals ) ;
	gd = new GenerateDialog ( parent , nsigs , signals ) ;

	if (gd->Execute() == IDOK) {
      if (process ( ID_PRED_NAME , gd->dest_name , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete gd ;
         return ;
         }
      switch (gd->parm_type) {
         case 1:
            sprintf ( rest , "%d ARMA %lf %lf %lf %lf %lf %lf", gd->parm_points,
               gd->parm_ar1, gd->parm_ar2, gd->parm_ar10, gd->parm_ma1,
               gd->parm_ma2, gd->parm_ma10 ) ;
            if (process ( ID_PRED_GENERATE , rest , NULL , error , NULL ) < 0)
               parent->MessageBox ( error , "ERROR" ) ;
            break ;
         case 2:
            sprintf ( rest , "%d NORMAL %lf %lf", gd->parm_points,
                      gd->parm_mean, gd->parm_std ) ;
            if (process ( ID_PRED_GENERATE , rest , NULL , error , NULL ) < 0)
               parent->MessageBox ( error , "ERROR" ) ;
            break ;
         case 3:
            sprintf ( rest , "%d UNIFORM %lf %lf", gd->parm_points,
                      gd->parm_low, gd->parm_high ) ;
            if (process ( ID_PRED_GENERATE , rest , NULL , error , NULL ) < 0)
               parent->MessageBox ( error , "ERROR" ) ;
            break ;
         case 4:
            sprintf ( rest , "%d RAMP %lf %lf", gd->parm_points,
                      gd->parm_height, gd->parm_rperiod ) ;
            if (process ( ID_PRED_GENERATE , rest , NULL , error , NULL ) < 0)
               parent->MessageBox ( error , "ERROR" ) ;
            break ;
         case 5:
            sprintf ( rest , "%d SINE %lf %lf %lf", gd->parm_points,
                      gd->parm_amp, gd->parm_speriod,
                      gd->parm_phase ) ;
            if (process ( ID_PRED_GENERATE , rest , NULL , error , NULL ) < 0)
               parent->MessageBox ( error , "ERROR" ) ;
            break ;
         }
      }

	delete gd ;
}


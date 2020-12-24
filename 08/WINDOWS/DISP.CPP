/******************************************************************************/
/*                                                                            */
/*  DISP - Display a signal                                                   */
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

#include "disp.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( DispDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_DISPLAY_OPTIMAL , EvOptimal ) ,
  EV_COMMAND ( IDC_DISPLAY_SYMMETRIC , EvSymmetric ) ,
  EV_COMMAND ( IDC_DISPLAY_FIXED , EvFixed ) ,
  EV_COMMAND ( IDC_DISPLAY_CONFIDENCE , EvConfidence ) ,
END_RESPONSE_TABLE ;

DispDialog::DispDialog ( TWindow *parent , MiscParams *mp , int p_nsignals ,
                         Signal **p_signals )
	: TDialog ( parent , IDD_DISPLAY ), TWindow ( parent )
{
   n_signals = p_nsignals ;
   signals = p_signals ;

   parm_range = mp->display_range ;
   parm_domain0 = mp->display_domain0 ;
   parm_domain1 = mp->display_domain1 ;
   parm_conf = 0 ;
   parm_min = mp->display_min ;
   parm_max = mp->display_max ;
   parm_origin = mp->display_origin ;
   parm_rate = mp->display_rate ;

	range_optimal = new TRadioButton ( this , IDC_DISPLAY_OPTIMAL ) ;
	range_symmetric = new TRadioButton ( this , IDC_DISPLAY_SYMMETRIC ) ;
	range_fixed = new TRadioButton ( this , IDC_DISPLAY_FIXED ) ;
	conf = new TCheckBox ( this , IDC_DISPLAY_CONFIDENCE ) ;

	domain0_label = new TStatic ( this , IDC_DISPLAY_DOMAIN0_LABEL ) ;
	domain0 = new TEdit ( this , IDC_DISPLAY_DOMAIN0 ) ;
   domain0_valid = new TFilterValidator ( "0-9" );
   domain0->SetValidator ( domain0_valid ) ;

	domain1_label = new TStatic ( this , IDC_DISPLAY_DOMAIN1_LABEL ) ;
	domain1 = new TEdit ( this , IDC_DISPLAY_DOMAIN1 ) ;
   domain1_valid = new TFilterValidator ( "0-9" );
   domain1->SetValidator ( domain1_valid ) ;

	sig_min_label = new TStatic ( this , IDC_DISPLAY_SIGMIN_LABEL ) ;
	sig_min = new TEdit ( this , IDC_DISPLAY_SIGMIN ) ;
   sig_min_valid = new TFilterValidator ( "0-9.-" );
   sig_min->SetValidator ( sig_min_valid ) ;

	sig_max_label = new TStatic ( this , IDC_DISPLAY_SIGMAX_LABEL ) ;
	sig_max = new TEdit ( this , IDC_DISPLAY_SIGMAX ) ;
   sig_max_valid = new TFilterValidator ( "0-9.-" );
   sig_max->SetValidator ( sig_max_valid ) ;

	origin_label = new TStatic ( this , IDC_DISPLAY_ORIGIN_LABEL ) ;
	origin = new TEdit ( this , IDC_DISPLAY_ORIGIN ) ;
   origin_valid = new TFilterValidator ( "0-9.-" );
   origin->SetValidator ( origin_valid ) ;

	rate_label = new TStatic ( this , IDC_DISPLAY_RATE_LABEL ) ;
	rate = new TEdit ( this , IDC_DISPLAY_RATE ) ;
   rate_valid = new TFilterValidator ( "0-9.-" );
   rate->SetValidator ( rate_valid ) ;

	source = new TListBox ( this , IDC_DISPLAY_SOURCE ) ;
}

DispDialog::~DispDialog ()
{
}

void DispDialog::SetupWindow ()
{
   int i ;
   char msg[84] ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   if (parm_conf)
      conf->Check () ;
   else 
      conf->SetCheck ( BF_UNCHECKED ) ;

   switch (parm_range) {
      case 0: range_optimal->Check() ;
         sig_min_label->Show ( false ) ;
         sig_min->Show ( false ) ;
         sig_max_label->Show ( false ) ;
         sig_max->Show ( false ) ;
         break ;
      case 1: range_symmetric->Check() ;
         sig_min_label->Show ( false ) ;
         sig_min->Show ( false ) ;
         sig_max_label->Show ( false ) ;
         sig_max->Show ( false ) ;
         break ;
      case 2: range_fixed->Check() ;
         sig_min_label->Show ( true ) ;
         sig_min->Show ( true ) ;
         sig_max_label->Show ( true ) ;
         sig_max->Show ( true ) ;
         break ;
      }

   sprintf ( msg , "%d" , parm_domain0 ) ;
   domain0->SetText ( msg ) ;

   sprintf ( msg , "%d" , parm_domain1 ) ;
   domain1->SetText ( msg ) ;

   sprintf ( msg , "%lf" , parm_min ) ;
   sig_min->SetText ( msg ) ;

   sprintf ( msg , "%lf" , parm_max ) ;
   sig_max->SetText ( msg ) ;

   sprintf ( msg , "%lf" , parm_origin ) ;
   origin->SetText ( msg ) ;

   sprintf ( msg , "%lf" , parm_rate ) ;
   rate->SetText ( msg ) ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      source->AddString ( sigptr->name ) ;
      }
   source->SetSelIndex ( 0 ) ;
}

void DispDialog::CmOk ()
{
   char msg[84] ;

   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
   domain0->GetText ( msg , 83 ) ;
   parm_domain0 = atoi ( msg ) ;
   domain1->GetText ( msg , 83 ) ;
   parm_domain1 = atoi ( msg ) ;
   sig_min->GetText ( msg , 83 ) ;
   parm_min = atof ( msg ) ;
   sig_max->GetText ( msg , 83 ) ;
   parm_max = atof ( msg ) ;
   origin->GetText ( msg , 83 ) ;
   parm_origin = atof ( msg ) ;
   rate->GetText ( msg , 83 ) ;
   parm_rate = atof ( msg ) ;

   TDialog::CmOk () ;
}

void DispDialog::EvOptimal ()
{
   parm_range = 0 ;
   sig_min_label->Show ( false ) ;
   sig_min->Show ( false ) ;
   sig_max_label->Show ( false ) ;
   sig_max->Show ( false ) ;
}

void DispDialog::EvSymmetric ()
{
   parm_range = 1 ;
   sig_min_label->Show ( false ) ;
   sig_min->Show ( false ) ;
   sig_max_label->Show ( false ) ;
   sig_max->Show ( false ) ;
}

void DispDialog::EvFixed ()
{
   parm_range = 2 ;
   sig_min_label->Show ( true ) ;
   sig_min->Show ( true ) ;
   sig_max_label->Show ( true ) ;
   sig_max->Show ( true ) ;
}

void DispDialog::EvConfidence ()
{
   if (conf->GetCheck() == BF_CHECKED)
      parm_conf = 1 ;
   else 
      parm_conf = 0 ;
}

/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void do_display ( TWindow *parent )
{
   int nsigs, type ;
	char error[256], rest[256] ;
   Signal **signals ;
   MiscParams *mp ;
	DispDialog *disp ;

   mp = get_misc_params () ;
   nsigs = get_signals ( &signals ) ;

	disp = new DispDialog ( parent , mp , nsigs , signals ) ;

	if (disp->Execute() == IDOK) {
      switch (disp->parm_range) {
         case 0: strcpy ( rest , "OPTIMAL" ) ;  break ;
         case 1: strcpy ( rest , "SYMMETRIC" ) ;  break ;
         case 2: sprintf ( rest , "%lf %lf", disp->parm_min, disp->parm_max ) ;
                 break ;
         }
      if (process ( ID_PRED_DISPLAY_RANGE , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete disp ;
         return ;
         }
      sprintf ( rest , "%d %d", disp->parm_domain0, disp->parm_domain1 ) ;
      if (process ( ID_PRED_DISPLAY_DOMAIN , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete disp ;
         return ;
         }
      sprintf ( rest , "%lf", disp->parm_origin ) ;
      if (process ( ID_PRED_DISPLAY_ORIGIN , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete disp ;
         return ;
         }
      sprintf ( rest , "%lf", disp->parm_rate ) ;
      if (process ( ID_PRED_DISPLAY_RATE , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete disp ;
         return ;
         }
      if (disp->parm_conf)
         type = ID_PRED_DISPLAY_CONFIDENCE ;
      else
         type = ID_PRED_DISPLAY ;

      if (process ( type , disp->source_name , NULL , error , NULL ) < 0)
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete disp ;
}


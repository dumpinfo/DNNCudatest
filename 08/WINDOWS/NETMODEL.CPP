/******************************************************************************/
/*                                                                            */
/*  NETMODEL - Allow the user to select a neural network model                */
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

#include "netmodel.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( NetModelDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_NETMODEL_PNN_GAUSS , EvPNN_GAUSS ) ,
  EV_COMMAND ( IDC_NETMODEL_PNN_RECIP , EvPNN_RECIP ) ,
  EV_COMMAND ( IDC_NETMODEL_SEPVAR , EvSEPVAR ) ,
  EV_COMMAND ( IDC_NETMODEL_SEPCLASS , EvSEPCLASS ) ,
  EV_COMMAND ( IDC_NETMODEL_MLFN_REAL , EvMLFN_REAL ) ,
  EV_COMMAND ( IDC_NETMODEL_MLFN_CPXIN , EvMLFN_CPXIN ) ,
  EV_COMMAND ( IDC_NETMODEL_MLFN_CPXHID , EvMLFN_CPXHID ) ,
  EV_COMMAND ( IDC_NETMODEL_MLFN_CPX , EvMLFN_CPX ) ,
  EV_COMMAND ( IDC_NETMODEL_LINEAR , EvLINEAR ) ,
END_RESPONSE_TABLE ;

NetModelDialog::NetModelDialog ( TWindow *parent , int p_model ,
                                 int p_nhid1 , int p_nhid2 , int p_domain ,
                                 int p_linear , int p_kernel )
	: TDialog ( parent , IDD_NET_MODEL ), TWindow ( parent )
{
   model = p_model ;
   nhid1 = p_nhid1 ;
   nhid2 = p_nhid2 ;
   domain = p_domain ;
   out_act = p_linear ;
   kernel = p_kernel ;

	pnn_gauss = new TRadioButton ( this , IDC_NETMODEL_PNN_GAUSS ) ;
	pnn_recip = new TRadioButton ( this , IDC_NETMODEL_PNN_RECIP ) ;
	sepvar = new TRadioButton ( this , IDC_NETMODEL_SEPVAR ) ;
	sepclass = new TRadioButton ( this , IDC_NETMODEL_SEPCLASS ) ;
	mlfn_real = new TRadioButton ( this , IDC_NETMODEL_MLFN_REAL ) ;
	mlfn_cpxin = new TRadioButton ( this , IDC_NETMODEL_MLFN_CPXIN ) ;
	mlfn_cpxhid = new TRadioButton ( this , IDC_NETMODEL_MLFN_CPXHID ) ;
	mlfn_cpx = new TRadioButton ( this , IDC_NETMODEL_MLFN_CPX ) ;
	linear = new TCheckBox ( this , IDC_NETMODEL_LINEAR ) ;

   valid1 = new TRangeValidator ( 0 , 9999 ) ;
   valid2 = new TRangeValidator ( 0 , 9999 ) ;

	hid1 = new TEdit ( this , IDC_NETMODEL_HID1 ) ;
   hid1->SetValidator ( valid1 ) ;
	hid1label = new TStatic ( this , IDC_NETMODEL_HID1_LABEL ) ;
	hid2 = new TEdit ( this , IDC_NETMODEL_HID2 ) ;
   hid2->SetValidator ( valid2 ) ;
	hid2label = new TStatic ( this , IDC_NETMODEL_HID2_LABEL ) ;
}

NetModelDialog::~NetModelDialog ()
{
}

void NetModelDialog::SetupWindow ()
{
   char msg[84] ;

	TDialog::SetupWindow () ;

   if (out_act)
      linear->Check () ;
   else 
      linear->SetCheck ( BF_UNCHECKED ) ;

   sprintf ( msg , "%d" , nhid1 ) ;
   hid1->SetText ( msg ) ;

   sprintf ( msg , "%d" , nhid2 ) ;
   hid2->SetText ( msg ) ;

   switch (model) {
      case NETMOD_PNN:
         if (kernel == KERNEL_GAUSS)
            pnn_gauss->Check() ;
         else if (kernel == KERNEL_RECIP)
            pnn_recip->Check() ;
         hid1->Show ( false ) ;
         hid1label->Show ( false ) ;
         hid2->Show ( false ) ;
         hid2label->Show ( false ) ;
         linear->Show ( false ) ;
         break ;
      case NETMOD_SEPVAR:
         sepvar->Check() ;
         hid1->Show ( false ) ;
         hid1label->Show ( false ) ;
         hid2->Show ( false ) ;
         hid2label->Show ( false ) ;
         linear->Show ( false ) ;
         break ;
      case NETMOD_SEPCLASS:
         sepclass->Check() ;
         hid1->Show ( false ) ;
         hid1label->Show ( false ) ;
         hid2->Show ( false ) ;
         hid2label->Show ( false ) ;
         linear->Show ( false ) ;
         break ;
      case NETMOD_MLFN:
         if (domain == DOMAIN_REAL)
            mlfn_real->Check() ;
         else if (domain == DOMAIN_COMPLEX)
            mlfn_cpx->Check() ;
         else if (domain == DOMAIN_COMPLEX_INPUT)
            mlfn_cpxin->Check() ;
         else if (domain == DOMAIN_COMPLEX_HIDDEN)
            mlfn_cpxhid->Check() ;
         hid1->Show ( true ) ;
         hid1label->Show ( true ) ;
         hid2->Show ( true ) ;
         hid2label->Show ( true ) ;
         linear->Show ( true ) ;
         break ;
      }
}

void NetModelDialog::CmOk ()
{
   char msg[84] ;
   hid1->GetText ( msg , 84 ) ;
   nhid1 = atoi ( msg ) ;
   hid2->GetText ( msg , 84 ) ;
   nhid2 = atoi ( msg ) ;
	TDialog::CmOk () ;
}

void NetModelDialog::EvPNN_GAUSS ()
{
   model = NETMOD_PNN ;
   kernel = KERNEL_GAUSS ;
   hid1->Show ( false ) ;
   hid1label->Show ( false ) ;
   hid2->Show ( false ) ;
   hid2label->Show ( false ) ;
   linear->Show ( false ) ;
}

void NetModelDialog::EvPNN_RECIP ()
{
   model = NETMOD_PNN ;
   kernel = KERNEL_RECIP ;
   hid1->Show ( false ) ;
   hid1label->Show ( false ) ;
   hid2->Show ( false ) ;
   hid2label->Show ( false ) ;
   linear->Show ( false ) ;
}

void NetModelDialog::EvSEPVAR ()
{
   model = NETMOD_SEPVAR ;
   kernel = KERNEL_GAUSS ;
   hid1->Show ( false ) ;
   hid1label->Show ( false ) ;
   hid2->Show ( false ) ;
   hid2label->Show ( false ) ;
   linear->Show ( false ) ;
}

void NetModelDialog::EvSEPCLASS ()
{
   model = NETMOD_SEPCLASS ;
   kernel = KERNEL_GAUSS ;
   hid1->Show ( false ) ;
   hid1label->Show ( false ) ;
   hid2->Show ( false ) ;
   hid2label->Show ( false ) ;
   linear->Show ( false ) ;
}

void NetModelDialog::EvMLFN_REAL ()
{
   model = NETMOD_MLFN ;
   domain = DOMAIN_REAL ;
   hid1->Show ( true ) ;
   hid1label->Show ( true ) ;
   hid2->Show ( true ) ;
   hid2label->Show ( true ) ;
   linear->Show ( true ) ;
}

void NetModelDialog::EvMLFN_CPXIN ()
{
   model = NETMOD_MLFN ;
   domain = DOMAIN_COMPLEX_INPUT ;
   nhid2 = 0 ;
   hid2->SetText ( "0" ) ;
   hid1->Show ( true ) ;
   hid1label->Show ( true ) ;
   hid2->Show ( true ) ;
   hid2label->Show ( true ) ;
   linear->Show ( true ) ;
}

void NetModelDialog::EvMLFN_CPXHID ()
{
   model = NETMOD_MLFN ;
   domain = DOMAIN_COMPLEX_HIDDEN ;
   nhid2 = 0 ;
   hid2->SetText ( "0" ) ;
   hid1->Show ( true ) ;
   hid1label->Show ( true ) ;
   hid2->Show ( true ) ;
   hid2label->Show ( true ) ;
   linear->Show ( true ) ;
}

void NetModelDialog::EvMLFN_CPX ()
{
   model = NETMOD_MLFN ;
   domain = DOMAIN_COMPLEX ;
   nhid2 = 0 ;
   hid2->SetText ( "0" ) ;
   hid1->Show ( true ) ;
   hid1label->Show ( true ) ;
   hid2->Show ( true ) ;
   hid2label->Show ( true ) ;
   linear->Show ( true ) ;
}

void NetModelDialog::EvLINEAR ()
{
   if (linear->GetCheck() == BF_CHECKED)
      out_act = 1 ;
   else 
      out_act = 0 ;
}

/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void get_net_model ( TWindow *parent )
{
   int model, nhid1, nhid2, domain, linear, kernel ;
	char error[256], rest[256] ;
	NetModelDialog *nm ;

   get_net_params ( &model , &nhid1 , &nhid2 , &domain ,
                    &linear , &kernel ) ;

	nm = new NetModelDialog ( parent , model , nhid1 , nhid2 , domain ,
                             linear , kernel ) ;

	if (nm->Execute() == IDOK) {
      model = nm->model ;
      nhid1 = nm->nhid1 ;
      nhid2 = nm->nhid2 ;
      domain = nm->domain ;
      linear = nm->out_act ;
      kernel = nm->kernel ;
      switch (model) {
         case NETMOD_PNN:
		      strcpy ( rest , "PNN" ) ;
            break ;
         case NETMOD_SEPVAR:
		      strcpy ( rest , "SEPVAR" ) ;
            break ;
         case NETMOD_SEPCLASS:
		      strcpy ( rest , "SEPCLASS" ) ;
            break ;
         case NETMOD_MLFN:
		      strcpy ( rest , "MLFN" ) ;
            break ;
            }
      if (process ( ID_PRED_NETMOD , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete nm ;
         return ;
         }
      sprintf ( rest , "%d" , nhid1 ) ;
      if (process ( ID_PRED_MLFN_HID1 , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete nm ;
         return ;
         }
      sprintf ( rest , "%d" , nhid2 ) ;
      if (process ( ID_PRED_MLFN_HID2 , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete nm ;
         return ;
         }
      switch (domain) {
         case DOMAIN_REAL:
		      strcpy ( rest , "REAL" ) ;
            break ;
         case DOMAIN_COMPLEX:
		      strcpy ( rest , "COMPLEX" ) ;
            break ;
         case DOMAIN_COMPLEX_INPUT:
		      strcpy ( rest , "COMPLEX INPUT" ) ;
            break ;
         case DOMAIN_COMPLEX_HIDDEN:
		      strcpy ( rest , "COMPLEX HIDDEN" ) ;
            break ;
            }
      if (process ( ID_PRED_MLFN_DOMAIN , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete nm ;
         return ;
         }
      if (linear)
         strcpy ( rest , "LINEAR" ) ;
      else 
         strcpy ( rest , "NONLINEAR" ) ;
      if (process ( ID_PRED_MLFN_OUTPUT_ACT , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete nm ;
         return ;
         }
      if (kernel == KERNEL_GAUSS)
         strcpy ( rest , "GAUSS" ) ;
      if (kernel == KERNEL_RECIP)
         strcpy ( rest , "RECIPROCAL" ) ;
      if (process ( ID_PRED_KERNEL , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete nm ;
         return ;
         }
      }

	delete nm ;
}


/******************************************************************************/
/*                                                                            */
/*  TEST_NET - Test a neural network                                          */
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

#include "test_net.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( TestNetDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_NETWORK_TEST_EXTENDED , EvExtended ) ,
END_RESPONSE_TABLE ;

TestNetDialog::TestNetDialog ( TWindow *parent , double p_thresh , int p_nnets ,
                               Network **p_nets )
	: TDialog ( parent , IDD_NETWORK_TEST ), TWindow ( parent )
{
   parm_thresh = p_thresh ;
   n_nets = p_nnets ;
   nets = p_nets ;

	netnames = new TListBox ( this , IDC_NETWORK_TEST_NAMES ) ;
	type = new TCheckBox ( this , IDC_NETWORK_TEST_EXTENDED ) ;
	thresh = new TEdit ( this , IDC_NETWORK_TEST_THRESH ) ;
   thresh_valid = new TFilterValidator ( "0-9." );
   thresh->SetValidator ( thresh_valid ) ;
}

TestNetDialog::~TestNetDialog ()
{
}

void TestNetDialog::SetupWindow ()
{
   int i ;
   char msg[84] ;
   Network *netptr ;

	TDialog::SetupWindow () ;

   strcpy ( net_name , "" ) ;

   sprintf ( msg , "%.2lf" , parm_thresh ) ;
   thresh->SetText ( msg ) ;

   parm_type = ID_PRED_TEST_NETWORK ;
   type->SetCheck ( BF_UNCHECKED ) ;

   for (i=0 ; i<n_nets ; i++) {
      netptr = nets[i] ;
      netnames->AddString ( netptr->name ) ;
      }
   netnames->SetSelIndex ( 0 ) ;
}

void TestNetDialog::CmOk ()
{
   char msg[84] ;

   netnames->GetSelString ( net_name , 255 ) ;
   strupr ( net_name ) ;

   thresh->GetText ( msg , 83 ) ;
   parm_thresh = atof ( msg ) ;

   if (type->GetCheck() == BF_CHECKED)
      parm_type = ID_PRED_EXTENDED_TEST_NETWORK ;
   else 
      parm_type = ID_PRED_TEST_NETWORK ;

	TDialog::CmOk () ;
}

void TestNetDialog::EvExtended ()
{
   if (type->GetCheck() == BF_CHECKED)
      parm_type = ID_PRED_EXTENDED_TEST_NETWORK ;
   else 
      parm_type = ID_PRED_TEST_NETWORK ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void test_net ( TWindow *parent )
{
   int nnets ;
	char error[256], rest[256], result[256] ;
	TestNetDialog *tn ;
   Network **nets ;
   MiscParams *mp ;

   nnets = get_networks ( &nets ) ;
   mp = get_misc_params () ;
	tn = new TestNetDialog ( parent , mp->threshold , nnets , nets ) ;

	if (tn->Execute() == IDOK) {
      sprintf ( rest , "%lf", tn->parm_thresh ) ;
      if (process ( ID_PRED_THRESHOLD , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete tn ;
         return ;
         }
      if (process ( tn->parm_type , tn->net_name , NULL , error , result ) < 0)
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete tn ;
}

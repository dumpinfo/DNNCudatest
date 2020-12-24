/******************************************************************************/
/*                                                                            */
/*  ORTHOG - Define and apply an orthogonalization model                      */
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

#include "orthog.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( OrthogDefineDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_ORTHOG_DEFINE_PRINCO , EvPrinco ) ,
  EV_COMMAND ( IDC_ORTHOG_DEFINE_CENTROID , EvCentroid ) ,
  EV_COMMAND ( IDC_ORTHOG_DEFINE_DISCRIM , EvDiscrim ) ,
  EV_COMMAND ( IDC_ORTHOG_DEFINE_STD , EvStd ) ,
END_RESPONSE_TABLE ;

OrthogDefineDialog::OrthogDefineDialog ( TWindow *parent ,
                                         int p_northogs , Orthog **p_orthogs )
	: TDialog ( parent , IDD_ORTHOG_DEFINE ), TWindow ( parent )
{
   n_orthogs = p_northogs ;
   orthogs = p_orthogs ;

	princo = new TRadioButton ( this , IDC_ORTHOG_DEFINE_PRINCO ) ;
	centroid = new TRadioButton ( this , IDC_ORTHOG_DEFINE_CENTROID ) ;
	discrim = new TRadioButton ( this , IDC_ORTHOG_DEFINE_DISCRIM ) ;

	std = new TCheckBox ( this , IDC_ORTHOG_DEFINE_STD ) ;

	factors = new TEdit ( this , IDC_ORTHOG_DEFINE_FACTORS ) ;
   factors_valid = new TRangeValidator ( 1 , 999999 ) ;
   factors->SetValidator ( factors_valid ) ;
	limit = new TEdit ( this , IDC_ORTHOG_DEFINE_LIMIT ) ;
   limit_valid = new TFilterValidator ( "0-9." ) ;
   limit->SetValidator ( limit_valid ) ;

	orthog = new TComboBox ( this , IDC_ORTHOG_DEFINE_NAME ) ;
}

OrthogDefineDialog::~OrthogDefineDialog ()
{
}

void OrthogDefineDialog::SetupWindow ()
{
   int i ;
   Orthog *orthptr ;

	TDialog::SetupWindow () ;

   strcpy ( orthog_name , "" ) ;

   factors->SetText ( "9999" ) ;
   limit->SetText ( "100.0" ) ;

   parm_type = 1 ;
   princo->Check () ;

   parm_std = 1 ;
   std->Check () ;
   std->Show ( true ) ;

   for (i=0 ; i<n_orthogs ; i++) {
      orthptr = orthogs[i] ;
      orthog->AddString ( orthptr->name ) ;
      }
   orthog->SetSelIndex ( -1 ) ;
}

void OrthogDefineDialog::CmOk ()
{
   char msg[84] ;

   orthog->GetText ( orthog_name , 255 ) ;
   strupr ( orthog_name ) ;
   factors->GetText ( msg , 83 ) ;
   parm_factors = atoi ( msg ) ;
   limit->GetText ( msg , 83 ) ;
   parm_limit = atof ( msg ) ;
   if (std->GetCheck() == BF_CHECKED)
      parm_std = 1 ;
   else 
      parm_std = 0 ;

	TDialog::CmOk () ;
}

void OrthogDefineDialog::EvPrinco ()
{
   parm_type = 1 ;
   std->Show ( true ) ;
}

void OrthogDefineDialog::EvCentroid ()
{
   parm_type = 2 ;
   std->Show ( true ) ;
}

void OrthogDefineDialog::EvDiscrim ()
{
   parm_type = 3 ;
   std->Show ( false ) ;
}

void OrthogDefineDialog::EvStd ()
{
   if (std->GetCheck() == BF_CHECKED)
      parm_std = 1 ;
   else 
      parm_std = 0 ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void define_orthog ( TWindow *parent )
{
   int northogs ;
	char error[256], rest[256] ;
	OrthogDefineDialog *orth ;
   Orthog **orthogs ;

   northogs = get_orthogs ( &orthogs ) ;
	orth = new OrthogDefineDialog ( parent , northogs , orthogs ) ;

	if (orth->Execute() == IDOK) {
      switch (orth->parm_type) {
         case 1: strcpy ( rest , "PRINCIPAL COMPONENTS" ) ;  break ;
         case 2: strcpy ( rest , "CENTROIDS" ) ;  break ;
         case 3: strcpy ( rest , "DISCRIMINANT" ) ;  break ;
         }
      if (process ( ID_PRED_ORTHOG_TYPE , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete orth ;
         return ;
         }
      if (orth->parm_std)
         strcpy ( rest , "YES" ) ;
      else
         strcpy ( rest , "NO" ) ;
      if (process ( ID_PRED_ORTHOG_STD , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete orth ;
         return ;
         }
      sprintf ( rest , "%d", orth->parm_factors ) ;
      if (process ( ID_PRED_ORTHOG_FACTORS , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete orth ;
         return ;
         }
      sprintf ( rest , "%lf", orth->parm_limit ) ;
      if (process ( ID_PRED_ORTHOG_LIMIT , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete orth ;
         return ;
         }
      if (process ( ID_PRED_DEFINE_ORTHOG , orth->orthog_name , NULL , error , NULL ) < 0)
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete orth ;
}

/*
--------------------------------------------------------------------------------

   Apply

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( OrthogApplyDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
END_RESPONSE_TABLE ;

OrthogApplyDialog::OrthogApplyDialog ( TWindow *parent ,
                                       int p_northogs , Orthog **p_orthogs ,
                                       int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_ORTHOG_APPLY ), TWindow ( parent )
{
   n_orthogs = p_northogs ;
   orthogs = p_orthogs ;

   n_signals = p_nsignals ;
   signals = p_signals ;

	orthog = new TListBox ( this , IDC_ORTHOG_APPLY_NAME ) ;
	dest = new TComboBox ( this , IDC_ORTHOG_APPLY_DEST ) ;
}

OrthogApplyDialog::~OrthogApplyDialog ()
{
}

void OrthogApplyDialog::SetupWindow ()
{
   int i ;
   Orthog *orthptr ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   strcpy ( orthog_name , "" ) ;
   strcpy ( dest_name , "" ) ;

   for (i=0 ; i<n_orthogs ; i++) {
      orthptr = orthogs[i] ;
      orthog->AddString ( orthptr->name ) ;
      }
   orthog->SetSelIndex ( 0 ) ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      dest->AddString ( sigptr->name ) ;
      }
   dest->SetSelIndex ( -1 ) ;
}

void OrthogApplyDialog::CmOk ()
{
   orthog->GetSelString ( orthog_name , 255 ) ;
   strupr ( orthog_name ) ;
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;

	TDialog::CmOk () ;
}

/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void apply_orthog ( TWindow *parent )
{
   int northogs, nsigs ;
	char error[256] ;
	OrthogApplyDialog *orth ;
   Orthog **orthogs ;
   Signal **signals ;

   northogs = get_orthogs ( &orthogs ) ;
   nsigs = get_signals ( &signals ) ;
	orth = new OrthogApplyDialog ( parent , northogs , orthogs , nsigs ,
                                  signals ) ;

	if (orth->Execute() == IDOK) {
      if (process ( ID_PRED_NAME , orth->dest_name , NULL , error , NULL ) < 0){
         parent->MessageBox ( error , "ERROR" ) ;
         delete orth ;
         return ;
         }
      if (process ( ID_PRED_APPLY_ORTHOG , orth->orthog_name , NULL , error , NULL ) < 0)
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete orth ;
}

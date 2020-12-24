/******************************************************************************/
/*                                                                            */
/*  CLEAR - Clear the various objects                                         */
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

#include "clear.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( ClearDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_CLEAR_INPUT , EvInlist ) ,
  EV_COMMAND ( IDC_CLEAR_OUTPUT , EvOutlist ) ,
  EV_COMMAND ( IDC_CLEAR_CLASSES , EvClasses ) ,
  EV_COMMAND ( IDC_CLEAR_TRAINING , EvTrain ) ,
  EV_COMMAND ( IDC_CLEAR_TEST , EvTest ) ,
  EV_COMMAND ( IDC_CLEAR_CONF , EvConf ) ,
  EV_COMMAND ( IDC_CLEAR_NET , EvNet ) ,
  EV_COMMAND ( IDC_CLEAR_NETS , EvNets ) ,
  EV_COMMAND ( IDC_CLEAR_ARMA , EvArma ) ,
  EV_COMMAND ( IDC_CLEAR_ARMAS , EvArmas ) ,
  EV_COMMAND ( IDC_CLEAR_ORTHOG , EvOrthog ) ,
  EV_COMMAND ( IDC_CLEAR_ORTHOGS , EvOrthogs ) ,
  EV_COMMAND ( IDC_CLEAR_ALL , EvAll ) ,
END_RESPONSE_TABLE ;

ClearDialog::ClearDialog ( TWindow *parent , int p_nnets , Network **p_nets ,
        int p_nARMAs , ARMA **p_ARMAs , int p_northogs , Orthog **p_orthogs )
	: TDialog ( parent , IDD_CLEAR ), TWindow ( parent )
{
   n_nets = p_nnets ;
   networks = p_nets ;
   n_ARMAs = p_nARMAs ;
   ARMAs = p_ARMAs ;
   n_orthogs = p_northogs ;
   orths = p_orthogs ;

   inlist = new TRadioButton ( this , IDC_CLEAR_INPUT ) ;
   outlist = new TRadioButton ( this , IDC_CLEAR_OUTPUT ) ;
   classes = new TRadioButton ( this , IDC_CLEAR_CLASSES ) ;
   train = new TRadioButton ( this , IDC_CLEAR_TRAINING ) ;
   test = new TRadioButton ( this , IDC_CLEAR_TEST ) ;
   conf = new TRadioButton ( this , IDC_CLEAR_CONF ) ;
   net = new TRadioButton ( this , IDC_CLEAR_NET ) ;
   nets = new TRadioButton ( this , IDC_CLEAR_NETS ) ;
   arma = new TRadioButton ( this , IDC_CLEAR_ARMA ) ;
   armas = new TRadioButton ( this , IDC_CLEAR_ARMAS ) ;
   orthog = new TRadioButton ( this , IDC_CLEAR_ORTHOG ) ;
   orthogs = new TRadioButton ( this , IDC_CLEAR_ORTHOGS ) ;
   all = new TRadioButton ( this , IDC_CLEAR_ALL ) ;

	net_name = new TComboBox ( this , IDC_CLEAR_NET_NAME ) ;
	arma_name = new TComboBox ( this , IDC_CLEAR_ARMA_NAME ) ;
	orthog_name = new TComboBox ( this , IDC_CLEAR_ORTHOG_NAME ) ;
}

ClearDialog::~ClearDialog ()
{
}

void ClearDialog::SetupWindow ()
{
   int i ;
   Network *netptr ;
   ARMA *armaptr ;
   Orthog *orthptr ;

	TDialog::SetupWindow () ;

   parm_type = ID_PRED_CLEAR_INPUT ;
   inlist->Check () ;

   for (i=0 ; i<n_nets ; i++) {
      netptr = networks[i] ;
      net_name->AddString ( netptr->name ) ;
      }
   net_name->SetSelIndex ( 0 ) ;

   for (i=0 ; i<n_ARMAs ; i++) {
      armaptr = ARMAs[i] ;
      arma_name->AddString ( armaptr->name ) ;
      }
   arma_name->SetSelIndex ( 0 ) ;

   for (i=0 ; i<n_orthogs ; i++) {
      orthptr = orths[i] ;
      orthog_name->AddString ( orthptr->name ) ;
      }
   orthog_name->SetSelIndex ( 0 ) ;

   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::CmOk ()
{
   net_name->GetText ( parm_net , 255 ) ;
   arma_name->GetText ( parm_arma , 255 ) ;
   orthog_name->GetText ( parm_orthog , 255 ) ;

   strupr ( parm_net ) ;
   strupr ( parm_arma ) ;
   strupr ( parm_orthog ) ;

	TDialog::CmOk () ;
}

void ClearDialog::EvInlist ()
{
   parm_type = ID_PRED_CLEAR_INPUT ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvOutlist () 
{
   parm_type = ID_PRED_CLEAR_OUTPUT ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvClasses () 
{
   parm_type = ID_PRED_CLEAR_CLASSES ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvTrain () 
{
   parm_type = ID_PRED_CLEAR_TRAINING ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvTest () 
{
   parm_type = ID_PRED_CLEAR_TEST ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvConf () 
{
   parm_type = ID_PRED_CLEAR_CONFIDENCE ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvNet () 
{
   parm_type = ID_PRED_CLEAR_NETWORK ;
   net_name->Show ( true ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvNets () 
{
   parm_type = ID_PRED_CLEAR_NETWORKS ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvArma () 
{
   parm_type = ID_PRED_CLEAR_ARMA ;
   net_name->Show ( false ) ;
   arma_name->Show ( true ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvArmas () 
{
   parm_type = ID_PRED_CLEAR_ARMAS ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvOrthog () 
{
   parm_type = ID_PRED_CLEAR_ORTHOG ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( true ) ;
}

void ClearDialog::EvOrthogs ()
{
   parm_type = ID_PRED_CLEAR_ORTHOGS ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}

void ClearDialog::EvAll () 
{
   parm_type = ID_PRED_CLEAR_ALL ;
   net_name->Show ( false ) ;
   arma_name->Show ( false ) ;
   orthog_name->Show ( false ) ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void do_clear ( TWindow *parent )
{
   int i, nnets, narmas, northogs ;
	char error[256], rest[256] ;
	ClearDialog *cd ;
   Network **networks ;
   ARMA **armas ;
   Orthog **orthogs ;

   nnets = get_networks ( &networks ) ;
   narmas = get_ARMAs ( &armas , &i ) ;
   northogs = get_orthogs ( &orthogs ) ;

   cd = new ClearDialog ( parent , nnets , networks , narmas , armas ,
                          northogs , orthogs ) ;

	if (cd->Execute() == IDOK) {
      strcpy ( rest , "" ) ;
      if (cd->parm_type == ID_PRED_CLEAR_NETWORK)
         strcpy ( rest , cd->parm_net ) ;
      else if (cd->parm_type == ID_PRED_CLEAR_ARMA)
         strcpy ( rest , cd->parm_arma ) ;
      if (cd->parm_type == ID_PRED_CLEAR_ORTHOG)
         strcpy ( rest , cd->parm_orthog ) ;
      if (process ( cd->parm_type , rest , NULL , error , NULL ) < 0 )
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete cd ;
}


/******************************************************************************/
/*                                                                            */
/*  NET_PRED - Neural network prediction                                      */
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

#include "net_pred.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( NetPredDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_NETWORK_PREDICT_CONF , EvConf ) ,
END_RESPONSE_TABLE ;

NetPredDialog::NetPredDialog ( TWindow *parent , double p_prob ,
                               int p_nnets , Network **p_nets ,
                               int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_NETWORK_PREDICT ), TWindow ( parent )
{
   prob = p_prob ;
   n_nets = p_nnets ;
   nets = p_nets ;
   n_signals = p_nsignals ;
   signals = p_signals ;

   conf = new TCheckBox ( this , IDC_NETWORK_PREDICT_CONF ) ;

	dist_label = new TStatic ( this , IDC_NETWORK_PREDICT_DIST_LABEL ) ;
	dist = new TEdit ( this , IDC_NETWORK_PREDICT_DIST ) ;
   dist_valid = new TRangeValidator ( 1 , 999999 ) ;
   dist->SetValidator ( dist_valid ) ;

	pct_label = new TStatic ( this , IDC_NETWORK_PREDICT_PCT_LABEL ) ;
	pct = new TEdit ( this , IDC_NETWORK_PREDICT_PCT ) ;
   pct_valid = new TFilterValidator ( "0-9." ) ;
   pct->SetValidator ( pct_valid ) ;

	points = new TEdit ( this , IDC_NETWORK_PREDICT_POINTS ) ;
   points_valid = new TRangeValidator ( 1 , 999999 ) ;
   points->SetValidator ( points_valid ) ;

	net_names = new TListBox ( this , IDC_NETWORK_PREDICT_NAME ) ;
	dest = new TComboBox ( this , IDC_NETWORK_PREDICT_DEST ) ;
}

NetPredDialog::~NetPredDialog ()
{
}

void NetPredDialog::SetupWindow ()
{
   int i ;
   char msg[84] ;
   Network *netptr ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   strcpy ( net_name , "" ) ;
   strcpy ( dest_name , "" ) ;

   points->SetText ( "1" ) ;

   dist->SetText ( "1" ) ;
   dist_label->Show ( false ) ;
   dist->Show ( false ) ;

   sprintf ( msg , "%.1lf" , 100.0 * prob ) ;
   pct->SetText ( msg ) ;
   pct_label->Show ( false ) ;
   pct->Show ( false ) ;

   parm_conf = 0 ;
   conf->SetCheck ( BF_UNCHECKED ) ;

   for (i=0 ; i<n_nets ; i++) {
      netptr = nets[i] ;
      net_names->AddString ( netptr->name ) ;
      }
   net_names->SetSelIndex ( 0 ) ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      dest->AddString ( sigptr->name ) ;
      }
   dest->SetSelIndex ( -1 ) ;
}

void NetPredDialog::CmOk ()
{
   char msg[84] ;

   net_names->GetSelString ( net_name , 255 ) ;
   strupr ( net_name ) ;
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;

   points->GetText ( msg , 83 ) ;
   parm_points = atoi ( msg ) ;

   dist->GetText ( msg , 83 ) ;
   parm_dist = atoi ( msg ) ;

   pct->GetText ( msg , 83 ) ;
   parm_pct = atof ( msg ) ;

	TDialog::CmOk () ;
}

void NetPredDialog::EvConf ()
{
   if (conf->GetCheck() == BF_CHECKED) {
      parm_conf = 1 ;
      dist_label->Show ( true ) ;
      dist->Show ( true ) ;
      pct_label->Show ( true ) ;
      pct->Show ( true ) ;
      }
   else {
      parm_conf = 0 ;
      dist_label->Show ( false ) ;
      dist->Show ( false ) ;
      pct_label->Show ( false ) ;
      pct->Show ( false ) ;
      }
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void predict_net ( TWindow *parent )
{
   int nnets, nsigs ;
	char error[256], rest[256] ;
	NetPredDialog *np ;
   Network **nets ;
   Signal **signals ;
   MiscParams *mp ;

   mp = get_misc_params () ;
   nnets = get_networks ( &nets ) ;
   nsigs = get_signals ( &signals ) ;
	np = new NetPredDialog ( parent , mp->conf_prob , nnets , nets ,
                            nsigs , signals ) ;

	if (np->Execute() == IDOK) {
      if (process ( ID_PRED_NAME , np->dest_name , NULL , error , NULL ) < 0 ) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete np ;
         return ;
         }
      if (np->parm_conf  &&  np->parm_dist) {
         sprintf ( rest , "%lf" , np->parm_pct ) ;
         if (process ( ID_PRED_CONFIDENCE_PROB , rest , NULL , error, NULL ) < 0 ) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete np ;
            return ;
            }
         sprintf ( rest , "%d %s" , np->parm_dist, np->net_name ) ;
         if (process ( ID_PRED_NETWORK_CONFIDENCE , rest , NULL , error, NULL ) < 0 ) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete np ;
            return ;
            }
         }
      sprintf ( rest , "%d %s" , np->parm_points, np->net_name ) ;
      if (process ( ID_PRED_NETWORK_PREDICT , rest , NULL , error, NULL ) < 0 )
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete np ;
}

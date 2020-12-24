/******************************************************************************/
/*                                                                            */
/*  ARMA - All routines related to ARMA models                                */
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

#include "arma.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( ARMAtrainDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_ARMA_TRAIN_FIXED , EvFixed ) ,
  EV_COMMAND ( IDC_ARMA_TRAIN_PROGRESS , EvProgress ) ,
END_RESPONSE_TABLE ;

ARMAtrainDialog::ARMAtrainDialog ( TWindow *parent , LearnParams *lp ,
                                   int p_fixed , int p_nARMAs , ARMA **p_ARMAs )
	: TDialog ( parent , IDD_ARMA_TRAIN ), TWindow ( parent )
{
   params = *lp ;
   parm_fixed = p_fixed ;
   n_ARMAs = p_nARMAs ;
   ARMAs = p_ARMAs ;

	fixed = new TCheckBox ( this , IDC_ARMA_TRAIN_FIXED ) ;
	progress = new TCheckBox ( this , IDC_ARMA_TRAIN_PROGRESS ) ;

	error = new TEdit ( this , IDC_ARMA_TRAIN_ERROR ) ;
   error_valid = new TFilterValidator ( "0-9." );
   error->SetValidator ( error_valid ) ;

	acc = new TEdit ( this , IDC_ARMA_TRAIN_ACC ) ;
   acc_valid = new TRangeValidator ( 1 , 15 );
   acc->SetValidator ( acc_valid ) ;

	refine = new TEdit ( this , IDC_ARMA_TRAIN_REFINE ) ;
   refine_valid = new TRangeValidator ( 1 , 15 );
   refine->SetValidator ( refine_valid ) ;

	arma_names = new TComboBox ( this , IDC_ARMA_TRAIN_NAME ) ;
}

ARMAtrainDialog::~ARMAtrainDialog ()
{
}

void ARMAtrainDialog::SetupWindow ()
{
   int i ;
   char msg[84] ;
   ARMA *armaptr ;

	TDialog::SetupWindow () ;

   strcpy ( arma_name , "" ) ;

   if (parm_fixed)
      fixed->Check () ;
   else
      fixed->SetCheck ( BF_UNCHECKED ) ;

   if (params.progress)
      progress->Check () ;
   else
      progress->SetCheck ( BF_UNCHECKED ) ;

   sprintf ( msg , "%.4lf" , params.quit_err ) ;
   error->SetText ( msg ) ;

   sprintf ( msg , "%d" , params.acc ) ;
   acc->SetText ( msg ) ;

   sprintf ( msg , "%d" , params.refine ) ;
   refine->SetText ( msg ) ;

   for (i=0 ; i<n_ARMAs ; i++) {
      armaptr = ARMAs[i] ;
      arma_names->AddString ( armaptr->name ) ;
      }

   arma_names->SetSelIndex ( -1 ) ;
}

void ARMAtrainDialog::CmOk ()
{
   char msg[84] ;

   arma_names->GetText ( arma_name , 255 ) ;
   strupr ( arma_name ) ;

   error->GetText ( msg , 83 ) ;
   params.quit_err = atof ( msg ) ;
   acc->GetText ( msg , 83 ) ;
   params.acc = atoi ( msg ) ;
   refine->GetText ( msg , 83 ) ;
   params.refine = atoi ( msg ) ;

	TDialog::CmOk () ;
}


void ARMAtrainDialog::EvFixed ()
{
   if (fixed->GetCheck() == BF_CHECKED)
      parm_fixed = 1 ;
   else 
      parm_fixed = 0 ;
}

void ARMAtrainDialog::EvProgress ()
{
   if (progress->GetCheck() == BF_CHECKED)
      params.progress = 1 ;
   else 
      params.progress = 0 ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void train_arma ( TWindow *parent )
{
   int narmas, fixed ;
	char error[256], rest[256], result[256] ;
   LearnParams *lp ;
	ARMAtrainDialog *at ;
   ARMA **armas ;

   lp = get_learn_params () ;
   narmas = get_ARMAs ( &armas , &fixed ) ;
	at = new ARMAtrainDialog ( parent , lp , fixed , narmas , armas ) ;

	if (at->Execute() == IDOK) {
      if (at->parm_fixed)
         strcpy ( rest , "YES" ) ;
      else
         strcpy ( rest , "NO" ) ;
      process ( ID_PRED_ARMA_FIXED , rest , NULL , error , NULL ) ;
      if (at->params.progress)
         process ( ID_PRED_PROGRESS_ON , rest , NULL , error , NULL ) ;
      else 
         process ( ID_PRED_PROGRESS_OFF , rest , NULL , error , NULL ) ;
      sprintf ( rest , "%lf" , at->params.quit_err ) ;
      if (process ( ID_PRED_QUITERR , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete at ;
         return ;
         }
      sprintf ( rest , "%d" , at->params.acc ) ;
      if (process ( ID_PRED_ACCURACY , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete at ;
         return ;
         }
      sprintf ( rest , "%d" , at->params.refine ) ;
      if (process ( ID_PRED_REFINE , rest , NULL , error , NULL ) < 0) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete at ;
         return ;
         }
      if (process ( ID_PRED_TRAIN_ARMA , at->arma_name , NULL , error , result ) < 0)
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete at ;
}

/*
--------------------------------------------------------------------------------

   Predict

--------------------------------------------------------------------------------
*/


DEFINE_RESPONSE_TABLE1 ( ARMApredictDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_ARMA_PREDICT_CONF , EvConf ) ,
END_RESPONSE_TABLE ;

ARMApredictDialog::ARMApredictDialog ( TWindow *parent , double p_prob ,
                                       int p_nARMAs , ARMA **p_ARMAs ,
                                       int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_ARMA_PREDICT ), TWindow ( parent )
{
   prob = p_prob ;
   n_ARMAs = p_nARMAs ;
   ARMAs = p_ARMAs ;
   n_signals = p_nsignals ;
   signals = p_signals ;

   conf = new TCheckBox ( this , IDC_ARMA_PREDICT_CONF ) ;

	dist_label = new TStatic ( this , IDC_ARMA_PREDICT_DIST_LABEL ) ;
	dist = new TEdit ( this , IDC_ARMA_PREDICT_DIST ) ;
   dist_valid = new TRangeValidator ( 1 , 999999 ) ;
   dist->SetValidator ( dist_valid ) ;

	pct_label = new TStatic ( this , IDC_ARMA_PREDICT_PCT_LABEL ) ;
	pct = new TEdit ( this , IDC_ARMA_PREDICT_PCT ) ;
   pct_valid = new TFilterValidator ( "0-9." ) ;
   pct->SetValidator ( pct_valid ) ;

	points = new TEdit ( this , IDC_ARMA_PREDICT_POINTS ) ;
   points_valid = new TRangeValidator ( 1 , 999999 ) ;
   points->SetValidator ( points_valid ) ;

	arma_names = new TListBox ( this , IDC_ARMA_PREDICT_NAME ) ;
	dest = new TComboBox ( this , IDC_ARMA_PREDICT_DEST ) ;
}

ARMApredictDialog::~ARMApredictDialog ()
{
}

void ARMApredictDialog::SetupWindow ()
{
   int i ;
   char msg[84] ;
   ARMA *armaptr ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   strcpy ( arma_name , "" ) ;
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

   for (i=0 ; i<n_ARMAs ; i++) {
      armaptr = ARMAs[i] ;
      arma_names->AddString ( armaptr->name ) ;
      }
   arma_names->SetSelIndex ( 0 ) ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      dest->AddString ( sigptr->name ) ;
      }
   dest->SetSelIndex ( -1 ) ;
}

void ARMApredictDialog::CmOk ()
{
   char msg[84] ;

   arma_names->GetSelString ( arma_name , 255 ) ;
   strupr ( arma_name ) ;
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

void ARMApredictDialog::EvConf ()
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

void predict_arma ( TWindow *parent )
{
   int i, narmas, nsigs ;
	char error[256], rest[256], result[256] ;
	ARMApredictDialog *at ;
   ARMA **armas ;
   Signal **signals ;
   MiscParams *mp ;

   mp = get_misc_params () ;
   narmas = get_ARMAs ( &armas , &i ) ;
   nsigs = get_signals ( &signals ) ;
	at = new ARMApredictDialog ( parent , mp->conf_prob , narmas , armas ,
                                nsigs , signals ) ;

	if (at->Execute() == IDOK) {
      if (process ( ID_PRED_NAME , at->dest_name , NULL , error , NULL ) < 0 ) {
         parent->MessageBox ( error , "ERROR" ) ;
         delete at ;
         return ;
         }
      if (at->parm_conf  &&  at->parm_dist) {
         sprintf ( rest , "%lf" , at->parm_pct ) ;
         if (process ( ID_PRED_CONFIDENCE_PROB , rest , NULL , error, NULL ) < 0 ) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete at ;
            return ;
            }
         sprintf ( rest , "%d %s" , at->parm_dist, at->arma_name ) ;
         if (process ( ID_PRED_ARMA_CONFIDENCE , rest , NULL , error, NULL ) < 0 ) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete at ;
            return ;
            }
         }
      sprintf ( rest , "%d %s" , at->parm_points, at->arma_name ) ;
      if (process ( ID_PRED_ARMA_PREDICT , rest , NULL , error , result ) < 0 )
         parent->MessageBox ( error , "ERROR" ) ;
      }

	delete at ;
}

/*
--------------------------------------------------------------------------------

   Predicted shocks

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( ShockDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
END_RESPONSE_TABLE ;

ShockDialog::ShockDialog ( TWindow *parent ,
                           int p_nsignals , Signal **p_signals ,
                           int p_ninsouts , InputOutput **p_insouts )
	: TDialog ( parent , IDD_ARMA_SHOCK ), TWindow ( parent )
{
   int i ;

   n_signals = p_nsignals ;
   signals = p_signals ;
   n_insouts = p_ninsouts ;
   insouts = p_insouts ;

   for (i=0 ; i<MAX_WIN_ARMA_SHOCK ; i++) {
      shock_name[i] = new TComboBox ( this , IDC_ARMA_SHOCK_SHOCKNAME+i ) ;
      out_name[i] = new TStatic ( this , IDC_ARMA_SHOCK_OUTNAME+i ) ;
      }
}

ShockDialog::~ShockDialog ()
{
}

void ShockDialog::SetupWindow ()
{
   int i, j ;
   Signal *sigptr ;
   TComboBox *shock ;
   InputOutput *ioptr ;

	TDialog::SetupWindow () ;

   for (i=0 ; i<MAX_WIN_ARMA_SHOCK ; i++) {
      shock_used[i] = 0 ;
      out_name[i]->SetText ( "" ) ;
      shock = shock_name[i] ;
      for (j=0 ; j<n_signals ; j++) {
         sigptr = signals[j] ;
         shock->AddString ( sigptr->name ) ;
         }
      shock->SetSelIndex ( -1 ) ;
      }

   nout = 0 ;
   for (i=0 ; i<n_insouts ; i++) {
      ioptr = insouts[i] ;
      if (ioptr->is_input)
         continue ;
      if (nout == MAX_WIN_ARMA_SHOCK)
         continue ;
      out_name[nout]->SetText ( signals[ioptr->which]->name ) ;
      out_name[nout]->Show ( true ) ;
      if (ioptr->shock >= 0) {
         shock_name[nout]->SetSelString ( signals[ioptr->shock]->name , 0 ) ;
         shock_used[nout] = 1 ;  // Flag so user cannot remove it
         }
      ++nout ;
      }
}

void ShockDialog::CmOk ()
{
   int i, error_shown ;

   error_shown = 0 ;
   for (i=0 ; i<MAX_WIN_ARMA_SHOCK ; i++) {
      shock_name[i]->GetText ( shock_names[i] , 255 ) ;
      strupr ( shock_names[i] ) ;
      out_name[i]->GetText ( out_names[i] , 255 ) ;
      strupr ( out_names[i] ) ;
      if (shock_used[i]  &&  (! strlen ( shock_names[i]))  &&  ! error_shown) {
         this->MessageBox ( "Cannot remove a shock.  CLEAR OUTPUTS." , "ERROR");
         error_shown = 1 ;
         }
      }

	TDialog::CmOk () ;
}

/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void shock_arma ( TWindow *parent )
{
   int i, nsigs, nio ;
	char error[256], rest[256] ;
	ShockDialog *sd ;
   Signal **signals ;
   InputOutput **io ;

   nsigs = get_signals ( &signals ) ;
   nio = get_ins_outs ( &io ) ;
	sd = new ShockDialog ( parent , nsigs , signals , nio , io ) ;

	if (sd->Execute() == IDOK) {
      for (i=0 ; i<MAX_WIN_ARMA_SHOCK ; i++) {
         if (strlen ( sd->shock_names[i] )) {
            sprintf ( rest , "%s FOR %s", sd->shock_names[i], sd->out_names[i]);
            if (process ( ID_PRED_ARMA_SHOCK , rest , NULL , error , NULL ) < 0) {
               parent->MessageBox ( error , "ERROR" ) ;
               delete sd ;
               return ;
               }
            }
         }
      }

	delete sd ;
}

/******************************************************************************/
/*                                                                            */
/*  NET_TRN - Train a neural network                                          */
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

#include "net_trn.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( PNNtrnDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_TRAIN_PNN_PROGRESS , EvProgress ) ,
  EV_COMMAND ( IDC_TRAIN_PNN_CV , EvCVtrain ) ,
END_RESPONSE_TABLE ;

PNNtrnDialog::PNNtrnDialog ( TWindow *parent , LearnParams *lp , int p_nnets ,
                             Network **p_nets )
	: TDialog ( parent , IDD_TRAIN_PNN ), TWindow ( parent )
{
   params = *lp ;
   anneal = *(lp->ap) ;
   n_nets = p_nnets ;
   nets = p_nets ;

	netnames = new TComboBox ( this , IDC_TRAIN_PNN_NETNAMES ) ;
	progress = new TCheckBox ( this , IDC_TRAIN_PNN_PROGRESS ) ;
	cv_train = new TCheckBox ( this , IDC_TRAIN_PNN_CV ) ;
	error_label = new TStatic ( this , IDC_TRAIN_PNN_ERROR_LABEL ) ;
	error = new TEdit ( this , IDC_TRAIN_PNN_ERROR ) ;
   error_valid = new TFilterValidator ( "0-9." );
   error->SetValidator ( error_valid ) ;
	acc_label = new TStatic ( this , IDC_TRAIN_PNN_ACC_LABEL ) ;
	acc = new TEdit ( this , IDC_TRAIN_PNN_ACC ) ;
   acc_valid = new TRangeValidator ( 1 , 15 );
   acc->SetValidator ( acc_valid ) ;
	refine_label = new TStatic ( this , IDC_TRAIN_PNN_REFINE_LABEL ) ;
	refine = new TEdit ( this , IDC_TRAIN_PNN_REFINE ) ;
   refine_valid = new TRangeValidator ( 1 , 15 );
   refine->SetValidator ( refine_valid ) ;

	siglo_label = new TStatic ( this , IDC_TRAIN_PNN_SIGLO_LABEL ) ;
	siglo = new TEdit ( this , IDC_TRAIN_PNN_SIGLO ) ;
   siglo_valid = new TFilterValidator ( "0-9." );
   siglo->SetValidator ( siglo_valid ) ;

	sighi_label = new TStatic ( this , IDC_TRAIN_PNN_SIGHI_LABEL ) ;
	sighi = new TEdit ( this , IDC_TRAIN_PNN_SIGHI ) ;
   sighi_valid = new TFilterValidator ( "0-9." );
   sighi->SetValidator ( sighi_valid ) ;

	nsigs_label = new TStatic ( this , IDC_TRAIN_PNN_NSIGS_LABEL ) ;
	nsigs = new TEdit ( this , IDC_TRAIN_PNN_NSIGS ) ;
   nsigs_valid = new TRangeValidator ( 3 , 9999 );
   nsigs->SetValidator ( nsigs_valid ) ;
}

PNNtrnDialog::~PNNtrnDialog ()
{
}

void PNNtrnDialog::SetupWindow ()
{
   int i ;
   char msg[84] ;
   Network *netptr ;

	TDialog::SetupWindow () ;

   if (params.progress)
      progress->Check () ;
   else
      progress->SetCheck ( BF_UNCHECKED ) ;

   cv_train->SetCheck ( BF_UNCHECKED ) ;
   parm_type = ID_PRED_TRAIN_NETWORK ;

   sprintf ( msg , "%.4lf" , params.quit_err ) ;
   error->SetText ( msg ) ;

   sprintf ( msg , "%d" , params.acc ) ;
   acc->SetText ( msg ) ;

   sprintf ( msg , "%d" , params.refine ) ;
   refine->SetText ( msg ) ;

   for (i=0 ; i<n_nets ; i++) {
      netptr = nets[i] ;
      netnames->AddString ( netptr->name ) ;
      }
   netnames->SetSelIndex ( 0 ) ;

   sprintf ( msg , "%.4lf" , params.siglo ) ;
   siglo->SetText ( msg ) ;

   sprintf ( msg , "%.4lf" , params.sighi ) ;
   sighi->SetText ( msg ) ;

   sprintf ( msg , "%d" , params.nsigs ) ;
   nsigs->SetText ( msg ) ;
}

void PNNtrnDialog::CmOk ()
{
   char msg[84] ;

   netnames->GetText ( net_name , 255 ) ;
   strupr ( net_name ) ;
   error->GetText ( msg , 83 ) ;
   params.quit_err = atof ( msg ) ;
   acc->GetText ( msg , 83 ) ;
   params.acc = atoi ( msg ) ;
   refine->GetText ( msg , 83 ) ;
   params.refine = atoi ( msg ) ;

   siglo->GetText ( msg , 83 ) ;
   params.siglo = atof ( msg ) ;

   sighi->GetText ( msg , 83 ) ;
   params.sighi = atof ( msg ) ;

   nsigs->GetText ( msg , 83 ) ;
   params.nsigs = atoi ( msg ) ;

	TDialog::CmOk () ;
}

void PNNtrnDialog::EvProgress ()
{
   if (progress->GetCheck() == BF_CHECKED)
      params.progress = 1 ;
   else 
      params.progress = 0 ;
}

void PNNtrnDialog::EvCVtrain ()
{
   if (cv_train->GetCheck() == BF_CHECKED)
      parm_type = ID_PRED_CVTRAIN_NETWORK ;
   else 
      parm_type = ID_PRED_TRAIN_NETWORK ;
}

/*
--------------------------------------------------------------------------------

   MLFN

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( MLFNtrnDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_PROGRESS , EvProgress ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_CV , EvCVtrain ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_ERR_MSE , EvErrMse ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_ERR_ABS , EvErrAbs ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_ERR_KK , EvErrKK ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_ERR_XENT , EvErrXent ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_AN1 , EvAN1 ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_AN1CJ , EvAN1CJ ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_AN1LM , EvAN1LM ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_AN2 , EvAN2 ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_AN2CJ , EvAN2CJ ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_AN2LM , EvAN2LM ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_REGCJ , EvRegCJ ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_REGLM , EvRegLM ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_SS , EvSS ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_METHOD_SSG , EvSSG ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_INIT_GAUSS , EvInitGauss ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_INIT_CAUCHY , EvInitCauchy ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_INIT_EXP , EvInitExp ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_INIT_FAST , EvInitFast ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_INIT_CLIMB , EvInitClimb ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_INIT_GAUSS , EvEscapeGauss ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_ESCAPE_CAUCHY , EvEscapeCauchy ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_ESCAPE_EXP , EvEscapeExp ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_ESCAPE_FAST , EvEscapeFast ) ,
  EV_COMMAND ( IDC_TRAIN_MLFN_ESCAPE_CLIMB , EvEscapeClimb ) ,
END_RESPONSE_TABLE ;

MLFNtrnDialog::MLFNtrnDialog ( TWindow *parent , LearnParams *lp , int p_nnets ,
                               Network **p_nets )
	: TDialog ( parent , IDD_TRAIN_MLFN ), TWindow ( parent )
{
   params = *lp ;
   anneal = *(lp->ap) ;
   n_nets = p_nnets ;
   nets = p_nets ;

	netnames = new TComboBox ( this , IDC_TRAIN_MLFN_NETNAMES ) ;
	progress = new TCheckBox ( this , IDC_TRAIN_MLFN_PROGRESS ) ;
	cv_train = new TCheckBox ( this , IDC_TRAIN_MLFN_CV ) ;
	error_label = new TStatic ( this , IDC_TRAIN_MLFN_ERROR_LABEL ) ;
	error = new TEdit ( this , IDC_TRAIN_MLFN_ERROR ) ;
   error_valid = new TFilterValidator ( "0-9." );
   error->SetValidator ( error_valid ) ;
	acc_label = new TStatic ( this , IDC_TRAIN_MLFN_ACC_LABEL ) ;
	acc = new TEdit ( this , IDC_TRAIN_MLFN_ACC ) ;
   acc_valid = new TRangeValidator ( 1 , 15 );
   acc->SetValidator ( acc_valid ) ;
	refine_label = new TStatic ( this , IDC_TRAIN_MLFN_REFINE_LABEL ) ;
	refine = new TEdit ( this , IDC_TRAIN_MLFN_REFINE ) ;
   refine_valid = new TRangeValidator ( 1 , 15 );
   refine->SetValidator ( refine_valid ) ;

   errtype = new TGroupBox ( this , IDC_TRAIN_MLFN_ERRTYPE ) ;
	err_msq = new TRadioButton ( this , IDC_TRAIN_MLFN_ERR_MSE , errtype ) ;
	err_abs = new TRadioButton ( this , IDC_TRAIN_MLFN_ERR_ABS , errtype ) ;
	err_kk = new TRadioButton ( this , IDC_TRAIN_MLFN_ERR_KK , errtype ) ;
	err_xent = new TRadioButton ( this , IDC_TRAIN_MLFN_ERR_XENT , errtype ) ;

   method = new TGroupBox ( this , IDC_TRAIN_MLFN_METHOD ) ;
	method_an1 = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_AN1 , method ) ;
	method_an1cj = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_AN1CJ , method ) ;
	method_an1lm = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_AN1LM , method ) ;
	method_an2 = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_AN2 , method ) ;
	method_an2cj = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_AN2CJ , method ) ;
	method_an2lm = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_AN2LM , method ) ;
	method_regcj = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_REGCJ , method ) ;
	method_reglm = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_REGLM , method ) ;
	method_ss = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_SS , method ) ;
	method_ssg = new TRadioButton ( this , IDC_TRAIN_MLFN_METHOD_SSG , method ) ;

	pretries_label = new TStatic ( this , IDC_TRAIN_MLFN_PRETRIES_LABEL ) ;
	pretries = new TEdit ( this , IDC_TRAIN_MLFN_PRETRIES ) ;
   pretries_valid = new TRangeValidator ( 1 , 999999 );
   pretries->SetValidator ( pretries_valid ) ;
	restarts_label = new TStatic ( this , IDC_TRAIN_MLFN_RESTARTS_LABEL ) ;
	restarts = new TEdit ( this , IDC_TRAIN_MLFN_RESTARTS ) ;
   restarts_valid = new TRangeValidator ( 0 , 999999 );
   restarts->SetValidator ( restarts_valid ) ;

	init_label = new TStatic ( this , IDC_TRAIN_MLFN_INIT ) ;
	escape_label = new TStatic ( this , IDC_TRAIN_MLFN_ESCAPE ) ;

	ntemps_label = new TStatic ( this , IDC_TRAIN_MLFN_NTEMPS ) ;
	init_ntemps = new TEdit ( this , IDC_TRAIN_MLFN_INIT_NTEMPS ) ;
   init_ntemps_valid = new TRangeValidator ( 1 , 999999 );
   init_ntemps->SetValidator ( init_ntemps_valid ) ;

	niters_label = new TStatic ( this , IDC_TRAIN_MLFN_NITERS ) ;
	init_niters = new TEdit ( this , IDC_TRAIN_MLFN_INIT_NITERS ) ;
   init_niters_valid = new TRangeValidator ( 1 , 999999 );
   init_niters->SetValidator ( init_niters_valid ) ;

	setback_label = new TStatic ( this , IDC_TRAIN_MLFN_SETBACK ) ;
	init_setback = new TEdit ( this , IDC_TRAIN_MLFN_INIT_SETBACK ) ;
   init_setback_valid = new TRangeValidator ( 0 , 999999 );
   init_setback->SetValidator ( init_setback_valid ) ;

	start_label = new TStatic ( this , IDC_TRAIN_MLFN_START ) ;
	init_start = new TEdit ( this , IDC_TRAIN_MLFN_INIT_START ) ;
   init_start_valid = new TFilterValidator ( "0-9." );
   init_start->SetValidator ( init_start_valid ) ;

	stop_label = new TStatic ( this , IDC_TRAIN_MLFN_STOP ) ;
	init_stop = new TEdit ( this , IDC_TRAIN_MLFN_INIT_STOP ) ;
   init_stop_valid = new TFilterValidator ( "0-9." );
   init_stop->SetValidator ( init_stop_valid ) ;

	random_label = new TStatic ( this , IDC_TRAIN_MLFN_RANDOM ) ;
   init_random = new TGroupBox ( this , IDC_TRAIN_MLFN_INIT_RANDOM ) ;
	init_gauss = new TRadioButton ( this , IDC_TRAIN_MLFN_INIT_GAUSS , init_random ) ;
	init_cauchy = new TRadioButton ( this , IDC_TRAIN_MLFN_INIT_CAUCHY , init_random ) ;

	reduc_label = new TStatic ( this , IDC_TRAIN_MLFN_REDUC ) ;
   init_reduction = new TGroupBox ( this , IDC_TRAIN_MLFN_INIT_REDUCTION ) ;
	init_exp = new TRadioButton ( this , IDC_TRAIN_MLFN_INIT_EXP , init_reduction ) ;
	init_fast = new TRadioButton ( this , IDC_TRAIN_MLFN_INIT_FAST , init_reduction ) ;

	climb_label = new TStatic ( this , IDC_TRAIN_MLFN_CLIMB ) ;
	init_climb = new TCheckBox ( this , IDC_TRAIN_MLFN_INIT_CLIMB ) ;

	ratio_label = new TStatic ( this , IDC_TRAIN_MLFN_RATIO ) ;
	init_ratio = new TEdit ( this , IDC_TRAIN_MLFN_INIT_RATIO ) ;
   init_ratio_valid = new TFilterValidator ( "0-9." );
   init_ratio->SetValidator ( init_ratio_valid ) ;

	escape_ntemps = new TEdit ( this , IDC_TRAIN_MLFN_ESCAPE_NTEMPS ) ;
   escape_ntemps_valid = new TRangeValidator ( 1 , 999999 );
   escape_ntemps->SetValidator ( escape_ntemps_valid ) ;

	escape_niters = new TEdit ( this , IDC_TRAIN_MLFN_ESCAPE_NITERS ) ;
   escape_niters_valid = new TRangeValidator ( 1 , 999999 );
   escape_niters->SetValidator ( escape_niters_valid ) ;

	escape_setback = new TEdit ( this , IDC_TRAIN_MLFN_ESCAPE_SETBACK ) ;
   escape_setback_valid = new TRangeValidator ( 0 , 999999 );
   escape_setback->SetValidator ( escape_setback_valid ) ;

	escape_start = new TEdit ( this , IDC_TRAIN_MLFN_ESCAPE_START ) ;
   escape_start_valid = new TFilterValidator ( "0-9." );
   escape_start->SetValidator ( escape_start_valid ) ;

	escape_stop = new TEdit ( this , IDC_TRAIN_MLFN_ESCAPE_STOP ) ;
   escape_stop_valid = new TFilterValidator ( "0-9." );
   escape_stop->SetValidator ( escape_stop_valid ) ;

   escape_random = new TGroupBox ( this , IDC_TRAIN_MLFN_ESCAPE_RANDOM ) ;
	escape_gauss = new TRadioButton ( this , IDC_TRAIN_MLFN_ESCAPE_GAUSS , escape_random ) ;
	escape_cauchy = new TRadioButton ( this , IDC_TRAIN_MLFN_ESCAPE_CAUCHY , escape_random ) ;

   escape_reduction = new TGroupBox ( this , IDC_TRAIN_MLFN_ESCAPE_REDUCTION ) ;
	escape_exp = new TRadioButton ( this , IDC_TRAIN_MLFN_ESCAPE_EXP , escape_reduction ) ;
	escape_fast = new TRadioButton ( this , IDC_TRAIN_MLFN_ESCAPE_FAST , escape_reduction ) ;

	escape_climb = new TCheckBox ( this , IDC_TRAIN_MLFN_ESCAPE_CLIMB ) ;

	escape_ratio = new TEdit ( this , IDC_TRAIN_MLFN_ESCAPE_RATIO ) ;
   escape_ratio_valid = new TFilterValidator ( "0-9." );
   escape_ratio->SetValidator ( escape_ratio_valid ) ;
}

MLFNtrnDialog::~MLFNtrnDialog ()
{
}

void MLFNtrnDialog::SetupWindow ()
{
   int i ;
   char msg[84] ;
   Network *netptr ;

	TDialog::SetupWindow () ;

   if (params.progress)
      progress->Check () ;
   else 
      progress->SetCheck ( BF_UNCHECKED ) ;

   cv_train->SetCheck ( BF_UNCHECKED ) ;
   parm_type = ID_PRED_TRAIN_NETWORK ;

   sprintf ( msg , "%.4lf" , params.quit_err ) ;
   error->SetText ( msg ) ;

   sprintf ( msg , "%d" , params.acc ) ;
   acc->SetText ( msg ) ;

   sprintf ( msg , "%d" , params.refine ) ;
   refine->SetText ( msg ) ;

   for (i=0 ; i<n_nets ; i++) {
      netptr = nets[i] ;
      netnames->AddString ( netptr->name ) ;
      }

   netnames->SetSelIndex ( 0 ) ;

   switch (params.errtype) {
      case ERRTYPE_MSE: err_msq->Check () ;  break ;
      case ERRTYPE_ABS: err_abs->Check () ;  break ;
      case ERRTYPE_KK: err_kk->Check () ;  break ;
      case ERRTYPE_XENT: err_xent->Check () ;  break ;
      }

   switch (params.method) {
      case METHOD_AN1: method_an1->Check () ;  break ;
      case METHOD_AN1_CJ: method_an1cj->Check () ;  break ;
      case METHOD_AN1_LM: method_an1lm->Check () ;  break ;
      case METHOD_AN2: method_an2->Check () ;  break ;
      case METHOD_AN2_CJ: method_an2cj->Check () ;  break ;
      case METHOD_AN2_LM: method_an2lm->Check () ;  break ;
      case METHOD_REGRS_CJ: method_regcj->Check () ;  break ;
      case METHOD_REGRS_LM: method_reglm->Check () ;  break ;
      case METHOD_SS: method_ss->Check () ;  break ;
      case METHOD_SSG: method_ssg->Check () ;  break ;
      }

   sprintf ( msg , "%d" , params.pretries ) ;
   pretries->SetText ( msg ) ;
   sprintf ( msg , "%d" , params.retries ) ;
   restarts->SetText ( msg ) ;

   sprintf ( msg , "%d" , anneal.ntempI ) ;
   init_ntemps->SetText ( msg ) ;
   sprintf ( msg , "%d" , anneal.niterI ) ;
   init_niters->SetText ( msg ) ;
   sprintf ( msg , "%d" , anneal.sbI ) ;
   init_setback->SetText ( msg ) ;
   sprintf ( msg , "%.4lf" , anneal.startI ) ;
   init_start->SetText ( msg ) ;
   sprintf ( msg , "%.4lf" , anneal.endI ) ;
   init_stop->SetText ( msg ) ;
   if (anneal.randomI == NormalDensity)
      init_gauss->Check () ;
   else if (anneal.randomI == CauchyDensity)
      init_cauchy->Check () ;
   if (anneal.reducI == ANNEAL_REDUCE_EXPONENTIAL)
      init_exp->Check () ;
   if (anneal.reducI == ANNEAL_REDUCE_FAST)
      init_fast->Check () ;
   if (anneal.climbI)
      init_climb->Check () ;
   else
      init_climb->SetCheck ( BF_UNCHECKED ) ;
   sprintf ( msg , "%.4lf" , anneal.ratioI ) ;
   init_ratio->SetText ( msg ) ;

   sprintf ( msg , "%d" , anneal.ntempE ) ;
   escape_ntemps->SetText ( msg ) ;
   sprintf ( msg , "%d" , anneal.niterE ) ;
   escape_niters->SetText ( msg ) ;
   sprintf ( msg , "%d" , anneal.sbE ) ;
   escape_setback->SetText ( msg ) ;
   sprintf ( msg , "%.4lf" , anneal.startE ) ;
   escape_start->SetText ( msg ) ;
   sprintf ( msg , "%.4lf" , anneal.endE ) ;
   escape_stop->SetText ( msg ) ;
   if (anneal.randomE == NormalDensity)
      escape_gauss->Check () ;
   else if (anneal.randomE == CauchyDensity)
      escape_cauchy->Check () ;
   if (anneal.reducE == ANNEAL_REDUCE_EXPONENTIAL)
      escape_exp->Check () ;
   if (anneal.reducE == ANNEAL_REDUCE_FAST)
      escape_fast->Check () ;
   if (anneal.climbE)
      escape_climb->Check () ;
   else
      escape_climb->SetCheck ( BF_UNCHECKED ) ;
   sprintf ( msg , "%.4lf" , anneal.ratioE ) ;
   escape_ratio->SetText ( msg ) ;
}

void MLFNtrnDialog::CmOk ()
{
   char msg[84] ;

   netnames->GetText ( net_name , 255 ) ;
   strupr ( net_name ) ;
   error->GetText ( msg , 83 ) ;
   params.quit_err = atof ( msg ) ;
   acc->GetText ( msg , 83 ) ;
   params.acc = atoi ( msg ) ;
   refine->GetText ( msg , 83 ) ;
   params.refine = atoi ( msg ) ;

   pretries->GetText ( msg , 83 ) ;
   params.pretries = atoi ( msg ) ;
   restarts->GetText ( msg , 83 ) ;
   params.retries = atoi ( msg ) ;

   init_ntemps->GetText ( msg , 83 ) ;
   anneal.ntempI = atoi ( msg ) ;
   init_niters->GetText ( msg , 83 ) ;
   anneal.niterI = atoi ( msg ) ;
   init_setback->GetText ( msg , 83 ) ;
   anneal.sbI = atoi ( msg ) ;
   init_start->GetText ( msg , 83 ) ;
   anneal.startI = atof ( msg ) ;
   init_stop->GetText ( msg , 83 ) ;
   anneal.endI = atof ( msg ) ;
   init_ratio->GetText ( msg , 83 ) ;
   anneal.ratioI = atof ( msg ) ;

   escape_ntemps->GetText ( msg , 83 ) ;
   anneal.ntempE = atoi ( msg ) ;
   escape_niters->GetText ( msg , 83 ) ;
   anneal.niterE = atoi ( msg ) ;
   escape_setback->GetText ( msg , 83 ) ;
   anneal.sbE = atoi ( msg ) ;
   escape_start->GetText ( msg , 83 ) ;
   anneal.startE = atof ( msg ) ;
   escape_stop->GetText ( msg , 83 ) ;
   anneal.endE = atof ( msg ) ;
   escape_ratio->GetText ( msg , 83 ) ;
   anneal.ratioE = atof ( msg ) ;

	TDialog::CmOk () ;
}

void MLFNtrnDialog::EvProgress ()
{
   if (progress->GetCheck() == BF_CHECKED)
      params.progress = 1 ;
   else 
      params.progress = 0 ;
}

void MLFNtrnDialog::EvCVtrain ()
{
   if (cv_train->GetCheck() == BF_CHECKED)
      parm_type = ID_PRED_CVTRAIN_NETWORK ;
   else 
      parm_type = ID_PRED_TRAIN_NETWORK ;
}

void MLFNtrnDialog::EvErrMse ()
{
   params.errtype = ERRTYPE_MSE ;
}
void MLFNtrnDialog::EvErrAbs ()
{
   params.errtype = ERRTYPE_ABS ;
}
void MLFNtrnDialog::EvErrKK ()
{
   params.errtype = ERRTYPE_KK ;
}
void MLFNtrnDialog::EvErrXent ()
{
   params.errtype = ERRTYPE_XENT ;
}
void MLFNtrnDialog::EvAN1 ()
{
   params.method = METHOD_AN1 ;
   pretries_label->Show ( false ) ;
   pretries->Show ( false ) ;
   restarts_label->Show ( true ) ;
   restarts->Show ( true ) ;
   ntemps_label->Show ( true ) ;
   niters_label->Show ( true ) ;
   setback_label->Show ( true ) ;
   start_label->Show ( true ) ;
   stop_label->Show ( true ) ;
   random_label->Show ( true ) ;
   reduc_label->Show ( false ) ;
   climb_label->Show ( false ) ;
   ratio_label->Show ( false ) ;
   init_label->Show ( true ) ;
   init_ntemps->Show ( true ) ;
   init_niters->Show ( true ) ;
   init_setback->Show ( true ) ;
   init_start->Show ( true ) ;
   init_stop->Show ( true ) ;
   init_gauss->Show ( true ) ;
   init_cauchy->Show ( true ) ;
   init_random->Show ( true ) ;
   init_exp->Show ( false ) ;
   init_fast->Show ( false ) ;
   init_reduction->Show ( false ) ;
   init_climb->Show ( false ) ;
   init_ratio->Show ( false ) ;
   escape_label->Show ( false ) ;
   escape_ntemps->Show ( false ) ;
   escape_niters->Show ( false ) ;
   escape_setback->Show ( false ) ;
   escape_start->Show ( false ) ;
   escape_stop->Show ( false ) ;
   escape_gauss->Show ( false ) ;
   escape_cauchy->Show ( false ) ;
   escape_random->Show ( false ) ;
   escape_exp->Show ( false ) ;
   escape_fast->Show ( false ) ;
   escape_reduction->Show ( false ) ;
   escape_climb->Show ( false ) ;
   escape_ratio->Show ( false ) ;
}
void MLFNtrnDialog::EvAN1CJ ()
{
   params.method = METHOD_AN1_CJ ;
   pretries_label->Show ( true ) ;
   pretries->Show ( true ) ;
   restarts_label->Show ( true ) ;
   restarts->Show ( true ) ;
   ntemps_label->Show ( true ) ;
   niters_label->Show ( true ) ;
   setback_label->Show ( true ) ;
   start_label->Show ( true ) ;
   stop_label->Show ( true ) ;
   random_label->Show ( true ) ;
   reduc_label->Show ( false ) ;
   climb_label->Show ( false ) ;
   ratio_label->Show ( false ) ;
   init_label->Show ( true ) ;
   init_ntemps->Show ( true ) ;
   init_niters->Show ( true ) ;
   init_setback->Show ( true ) ;
   init_start->Show ( true ) ;
   init_stop->Show ( true ) ;
   init_gauss->Show ( true ) ;
   init_cauchy->Show ( true ) ;
   init_random->Show ( true ) ;
   init_exp->Show ( false ) ;
   init_fast->Show ( false ) ;
   init_reduction->Show ( false ) ;
   init_climb->Show ( false ) ;
   init_ratio->Show ( false ) ;
   escape_label->Show ( true ) ;
   escape_ntemps->Show ( true ) ;
   escape_niters->Show ( true ) ;
   escape_setback->Show ( true ) ;
   escape_start->Show ( true ) ;
   escape_stop->Show ( true ) ;
   escape_gauss->Show ( true ) ;
   escape_cauchy->Show ( true ) ;
   escape_random->Show ( true ) ;
   escape_exp->Show ( false ) ;
   escape_fast->Show ( false ) ;
   escape_reduction->Show ( false ) ;
   escape_climb->Show ( false ) ;
   escape_ratio->Show ( false ) ;
}
void MLFNtrnDialog::EvAN1LM ()
{
   params.method = METHOD_AN1_LM ;
   pretries_label->Show ( true ) ;
   pretries->Show ( true ) ;
   restarts_label->Show ( true ) ;
   restarts->Show ( true ) ;
   ntemps_label->Show ( true ) ;
   niters_label->Show ( true ) ;
   setback_label->Show ( true ) ;
   start_label->Show ( true ) ;
   stop_label->Show ( true ) ;
   random_label->Show ( true ) ;
   reduc_label->Show ( false ) ;
   climb_label->Show ( false ) ;
   ratio_label->Show ( false ) ;
   init_label->Show ( true ) ;
   init_ntemps->Show ( true ) ;
   init_niters->Show ( true ) ;
   init_setback->Show ( true ) ;
   init_start->Show ( true ) ;
   init_stop->Show ( true ) ;
   init_gauss->Show ( true ) ;
   init_cauchy->Show ( true ) ;
   init_random->Show ( true ) ;
   init_exp->Show ( false ) ;
   init_fast->Show ( false ) ;
   init_reduction->Show ( false ) ;
   init_climb->Show ( false ) ;
   init_ratio->Show ( false ) ;
   escape_label->Show ( true ) ;
   escape_ntemps->Show ( true ) ;
   escape_niters->Show ( true ) ;
   escape_setback->Show ( true ) ;
   escape_start->Show ( true ) ;
   escape_stop->Show ( true ) ;
   escape_gauss->Show ( true ) ;
   escape_cauchy->Show ( true ) ;
   escape_random->Show ( true ) ;
   escape_exp->Show ( false ) ;
   escape_fast->Show ( false ) ;
   escape_reduction->Show ( false ) ;
   escape_climb->Show ( false ) ;
   escape_ratio->Show ( false ) ;
}
void MLFNtrnDialog::EvAN2 ()
{
   params.method = METHOD_AN2 ;
   pretries_label->Show ( false ) ;
   pretries->Show ( false ) ;
   restarts_label->Show ( true ) ;
   restarts->Show ( true ) ;
   ntemps_label->Show ( true ) ;
   niters_label->Show ( true ) ;
   setback_label->Show ( true ) ;
   start_label->Show ( true ) ;
   stop_label->Show ( true ) ;
   random_label->Show ( true ) ;
   reduc_label->Show ( true ) ;
   climb_label->Show ( true ) ;
   ratio_label->Show ( true ) ;
   init_label->Show ( true ) ;
   init_ntemps->Show ( true ) ;
   init_niters->Show ( true ) ;
   init_setback->Show ( true ) ;
   init_start->Show ( true ) ;
   init_stop->Show ( true ) ;
   init_gauss->Show ( true ) ;
   init_cauchy->Show ( true ) ;
   init_random->Show ( true ) ;
   init_exp->Show ( true ) ;
   init_fast->Show ( true ) ;
   init_reduction->Show ( true ) ;
   init_climb->Show ( true ) ;
   init_ratio->Show ( true ) ;
   escape_label->Show ( false ) ;
   escape_ntemps->Show ( false ) ;
   escape_niters->Show ( false ) ;
   escape_setback->Show ( false ) ;
   escape_start->Show ( false ) ;
   escape_stop->Show ( false ) ;
   escape_gauss->Show ( false ) ;
   escape_cauchy->Show ( false ) ;
   escape_random->Show ( false ) ;
   escape_exp->Show ( false ) ;
   escape_fast->Show ( false ) ;
   escape_reduction->Show ( false ) ;
   escape_climb->Show ( false ) ;
   escape_ratio->Show ( false ) ;
}
void MLFNtrnDialog::EvAN2CJ ()
{
   params.method = METHOD_AN2_CJ ;
   pretries_label->Show ( true ) ;
   pretries->Show ( true ) ;
   restarts_label->Show ( true ) ;
   restarts->Show ( true ) ;
   ntemps_label->Show ( true ) ;
   niters_label->Show ( true ) ;
   setback_label->Show ( true ) ;
   start_label->Show ( true ) ;
   stop_label->Show ( true ) ;
   random_label->Show ( true ) ;
   reduc_label->Show ( true ) ;
   climb_label->Show ( true ) ;
   ratio_label->Show ( true ) ;
   init_label->Show ( true ) ;
   init_ntemps->Show ( true ) ;
   init_niters->Show ( true ) ;
   init_setback->Show ( true ) ;
   init_start->Show ( true ) ;
   init_stop->Show ( true ) ;
   init_gauss->Show ( true ) ;
   init_cauchy->Show ( true ) ;
   init_random->Show ( true ) ;
   init_exp->Show ( true ) ;
   init_fast->Show ( true ) ;
   init_reduction->Show ( true ) ;
   init_climb->Show ( true ) ;
   init_ratio->Show ( true ) ;
   escape_label->Show ( true ) ;
   escape_ntemps->Show ( true ) ;
   escape_niters->Show ( true ) ;
   escape_setback->Show ( true ) ;
   escape_start->Show ( true ) ;
   escape_stop->Show ( true ) ;
   escape_gauss->Show ( true ) ;
   escape_cauchy->Show ( true ) ;
   escape_random->Show ( true ) ;
   escape_exp->Show ( true ) ;
   escape_fast->Show ( true ) ;
   escape_reduction->Show ( true ) ;
   escape_climb->Show ( true ) ;
   escape_ratio->Show ( true ) ;
}
void MLFNtrnDialog::EvAN2LM ()
{
   params.method = METHOD_AN2_LM ;
   pretries_label->Show ( true ) ;
   pretries->Show ( true ) ;
   restarts_label->Show ( true ) ;
   restarts->Show ( true ) ;
   ntemps_label->Show ( true ) ;
   niters_label->Show ( true ) ;
   setback_label->Show ( true ) ;
   start_label->Show ( true ) ;
   stop_label->Show ( true ) ;
   random_label->Show ( true ) ;
   reduc_label->Show ( true ) ;
   climb_label->Show ( true ) ;
   ratio_label->Show ( true ) ;
   init_label->Show ( true ) ;
   init_ntemps->Show ( true ) ;
   init_niters->Show ( true ) ;
   init_setback->Show ( true ) ;
   init_start->Show ( true ) ;
   init_stop->Show ( true ) ;
   init_gauss->Show ( true ) ;
   init_cauchy->Show ( true ) ;
   init_random->Show ( true ) ;
   init_exp->Show ( true ) ;
   init_fast->Show ( true ) ;
   init_reduction->Show ( true ) ;
   init_climb->Show ( true ) ;
   init_ratio->Show ( true ) ;
   escape_label->Show ( true ) ;
   escape_ntemps->Show ( true ) ;
   escape_niters->Show ( true ) ;
   escape_setback->Show ( true ) ;
   escape_start->Show ( true ) ;
   escape_stop->Show ( true ) ;
   escape_gauss->Show ( true ) ;
   escape_cauchy->Show ( true ) ;
   escape_random->Show ( true ) ;
   escape_exp->Show ( true ) ;
   escape_fast->Show ( true ) ;
   escape_reduction->Show ( true ) ;
   escape_climb->Show ( true ) ;
   escape_ratio->Show ( true ) ;
}
void MLFNtrnDialog::EvRegCJ ()
{
   params.method = METHOD_REGRS_CJ ;
   pretries_label->Show ( false ) ;
   pretries->Show ( false ) ;
   restarts_label->Show ( false ) ;
   restarts->Show ( false ) ;
   ntemps_label->Show ( false ) ;
   niters_label->Show ( false ) ;
   setback_label->Show ( false ) ;
   start_label->Show ( false ) ;
   stop_label->Show ( false ) ;
   random_label->Show ( false ) ;
   reduc_label->Show ( false ) ;
   climb_label->Show ( false ) ;
   ratio_label->Show ( false ) ;
   init_label->Show ( false ) ;
   init_ntemps->Show ( false ) ;
   init_niters->Show ( false ) ;
   init_setback->Show ( false ) ;
   init_start->Show ( false ) ;
   init_stop->Show ( false ) ;
   init_gauss->Show ( false ) ;
   init_cauchy->Show ( false ) ;
   init_random->Show ( false ) ;
   init_exp->Show ( false ) ;
   init_fast->Show ( false ) ;
   init_reduction->Show ( false ) ;
   init_climb->Show ( false ) ;
   init_ratio->Show ( false ) ;
   escape_label->Show ( false ) ;
   escape_ntemps->Show ( false ) ;
   escape_niters->Show ( false ) ;
   escape_setback->Show ( false ) ;
   escape_start->Show ( false ) ;
   escape_stop->Show ( false ) ;
   escape_gauss->Show ( false ) ;
   escape_cauchy->Show ( false ) ;
   escape_random->Show ( false ) ;
   escape_exp->Show ( false ) ;
   escape_fast->Show ( false ) ;
   escape_reduction->Show ( false ) ;
   escape_climb->Show ( false ) ;
   escape_ratio->Show ( false ) ;
}
void MLFNtrnDialog::EvRegLM ()
{
   params.method = METHOD_REGRS_LM ;
   pretries_label->Show ( false ) ;
   pretries->Show ( false ) ;
   restarts_label->Show ( false ) ;
   restarts->Show ( false ) ;
   ntemps_label->Show ( false ) ;
   niters_label->Show ( false ) ;
   setback_label->Show ( false ) ;
   start_label->Show ( false ) ;
   stop_label->Show ( false ) ;
   random_label->Show ( false ) ;
   reduc_label->Show ( false ) ;
   climb_label->Show ( false ) ;
   ratio_label->Show ( false ) ;
   init_label->Show ( false ) ;
   init_ntemps->Show ( false ) ;
   init_niters->Show ( false ) ;
   init_setback->Show ( false ) ;
   init_start->Show ( false ) ;
   init_stop->Show ( false ) ;
   init_gauss->Show ( false ) ;
   init_cauchy->Show ( false ) ;
   init_random->Show ( false ) ;
   init_exp->Show ( false ) ;
   init_fast->Show ( false ) ;
   init_reduction->Show ( false ) ;
   init_climb->Show ( false ) ;
   init_ratio->Show ( false ) ;
   escape_label->Show ( false ) ;
   escape_ntemps->Show ( false ) ;
   escape_niters->Show ( false ) ;
   escape_setback->Show ( false ) ;
   escape_start->Show ( false ) ;
   escape_stop->Show ( false ) ;
   escape_gauss->Show ( false ) ;
   escape_cauchy->Show ( false ) ;
   escape_random->Show ( false ) ;
   escape_exp->Show ( false ) ;
   escape_fast->Show ( false ) ;
   escape_reduction->Show ( false ) ;
   escape_climb->Show ( false ) ;
   escape_ratio->Show ( false ) ;
}
void MLFNtrnDialog::EvSS ()
{
   params.method = METHOD_SS ;
   pretries_label->Show ( false ) ;
   pretries->Show ( false ) ;
   restarts_label->Show ( true ) ;
   restarts->Show ( true ) ;
   ntemps_label->Show ( true ) ;
   niters_label->Show ( true ) ;
   setback_label->Show ( true ) ;
   start_label->Show ( true ) ;
   stop_label->Show ( true ) ;
   random_label->Show ( true ) ;
   reduc_label->Show ( false ) ;
   climb_label->Show ( false ) ;
   ratio_label->Show ( false ) ;
   init_label->Show ( true ) ;
   init_ntemps->Show ( true ) ;
   init_niters->Show ( true ) ;
   init_setback->Show ( true ) ;
   init_start->Show ( true ) ;
   init_stop->Show ( true ) ;
   init_gauss->Show ( true ) ;
   init_cauchy->Show ( true ) ;
   init_random->Show ( true ) ;
   init_exp->Show ( false ) ;
   init_fast->Show ( false ) ;
   init_reduction->Show ( false ) ;
   init_climb->Show ( false ) ;
   init_ratio->Show ( false ) ;
   escape_label->Show ( false ) ;
   escape_ntemps->Show ( false ) ;
   escape_niters->Show ( false ) ;
   escape_setback->Show ( false ) ;
   escape_start->Show ( false ) ;
   escape_stop->Show ( false ) ;
   escape_gauss->Show ( false ) ;
   escape_cauchy->Show ( false ) ;
   escape_random->Show ( false ) ;
   escape_exp->Show ( false ) ;
   escape_fast->Show ( false ) ;
   escape_reduction->Show ( false ) ;
   escape_climb->Show ( false ) ;
   escape_ratio->Show ( false ) ;
}
void MLFNtrnDialog::EvSSG ()
{
   params.method = METHOD_SSG ;
   pretries_label->Show ( false ) ;
   pretries->Show ( false ) ;
   restarts_label->Show ( true ) ;
   restarts->Show ( true ) ;
   ntemps_label->Show ( true ) ;
   niters_label->Show ( true ) ;
   setback_label->Show ( true ) ;
   start_label->Show ( true ) ;
   stop_label->Show ( true ) ;
   random_label->Show ( true ) ;
   reduc_label->Show ( false ) ;
   climb_label->Show ( false ) ;
   ratio_label->Show ( false ) ;
   init_label->Show ( true ) ;
   init_ntemps->Show ( true ) ;
   init_niters->Show ( true ) ;
   init_setback->Show ( true ) ;
   init_start->Show ( true ) ;
   init_stop->Show ( true ) ;
   init_gauss->Show ( true ) ;
   init_cauchy->Show ( true ) ;
   init_random->Show ( true ) ;
   init_exp->Show ( false ) ;
   init_fast->Show ( false ) ;
   init_reduction->Show ( false ) ;
   init_climb->Show ( false ) ;
   init_ratio->Show ( false ) ;
   escape_label->Show ( false ) ;
   escape_ntemps->Show ( false ) ;
   escape_niters->Show ( false ) ;
   escape_setback->Show ( false ) ;
   escape_start->Show ( false ) ;
   escape_stop->Show ( false ) ;
   escape_gauss->Show ( false ) ;
   escape_cauchy->Show ( false ) ;
   escape_random->Show ( false ) ;
   escape_exp->Show ( false ) ;
   escape_fast->Show ( false ) ;
   escape_reduction->Show ( false ) ;
   escape_climb->Show ( false ) ;
   escape_ratio->Show ( false ) ;
}
void MLFNtrnDialog::EvInitGauss ()
{
   anneal.randomI = NormalDensity ;
}
void MLFNtrnDialog::EvInitCauchy ()
{
   anneal.randomI = CauchyDensity ;
}
void MLFNtrnDialog::EvInitExp ()
{
   anneal.reducI = ANNEAL_REDUCE_EXPONENTIAL ;
}
void MLFNtrnDialog::EvInitFast ()
{
   anneal.reducI = ANNEAL_REDUCE_FAST ;
}
void MLFNtrnDialog::EvInitClimb ()
{
   if (init_climb->GetCheck() == BF_CHECKED)
      anneal.climbI = 1 ;
   else 
      anneal.climbI = 0 ;
}
void MLFNtrnDialog::EvEscapeGauss ()
{
   anneal.randomE = NormalDensity ;
}
void MLFNtrnDialog::EvEscapeCauchy ()
{
   anneal.randomE = CauchyDensity ;
}
void MLFNtrnDialog::EvEscapeExp ()
{
   anneal.reducE = ANNEAL_REDUCE_EXPONENTIAL ;
}
void MLFNtrnDialog::EvEscapeFast ()
{
   anneal.reducE = ANNEAL_REDUCE_FAST ;
}
void MLFNtrnDialog::EvEscapeClimb ()
{
   if (escape_climb->GetCheck() == BF_CHECKED)
      anneal.climbE = 1 ;
   else 
      anneal.climbE = 0 ;
}
void MLFNtrnDialog::EvEscapeNoClimb ()
{
   anneal.climbE = 0 ;
}


/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void train_net ( TWindow *parent )
{
   int nnets, netmodel, nhid1, nhid2, domain, linear, kernel ;
   char error[256], rest[256], result[256] ;
   Network **nets ;
   LearnParams *lp ;
   PNNtrnDialog *tpnn ;
   MLFNtrnDialog *tmlfn ;

   lp = get_learn_params () ;
   nnets = get_networks ( &nets ) ;
   get_net_params ( &netmodel , &nhid1 , &nhid2 , &domain ,
                      &linear , &kernel ) ;

   if ((netmodel == NETMOD_PNN) || (netmodel == NETMOD_SEPVAR)
    || (netmodel == NETMOD_SEPCLASS)) {
      tpnn = new PNNtrnDialog ( parent , lp , nnets , nets ) ;
      if (! tpnn)
         return ;
      if (tpnn->Execute() == IDOK) {
         if (tpnn->params.progress)
            process ( ID_PRED_PROGRESS_ON , rest , NULL , error , NULL ) ;
         else 
            process ( ID_PRED_PROGRESS_OFF , rest , NULL , error , NULL ) ;
         sprintf ( rest , "%lf" , tpnn->params.quit_err ) ;
         if (process ( ID_PRED_QUITERR , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tpnn ;
            return ;
            }
         sprintf ( rest , "%d" , tpnn->params.acc ) ;
         if (process ( ID_PRED_ACCURACY , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tpnn ;
            return ;
            }
         sprintf ( rest , "%d" , tpnn->params.refine ) ;
         if (process ( ID_PRED_REFINE , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tpnn ;
            return ;
            }

         sprintf ( rest , "%lf" , tpnn->params.siglo ) ;
         if (process ( ID_PRED_SIGLO , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tpnn ;
            return ;
            }
         sprintf ( rest , "%lf" , tpnn->params.sighi ) ;
         if (process ( ID_PRED_SIGHI , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tpnn ;
            return ;
            }
         sprintf ( rest , "%d" , tpnn->params.nsigs ) ;
         if (process ( ID_PRED_SIGTRIES , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tpnn ;
            return ;
            }
         if (process ( tpnn->parm_type , tpnn->net_name , NULL , error , result ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tpnn ;
            return ;
            }
         }
      delete tpnn ;
      return ;
		}

   else if (netmodel == NETMOD_MLFN) {
      tmlfn = new MLFNtrnDialog ( parent , lp , nnets , nets ) ;
      if (! tmlfn)
         return ;
      if (tmlfn->Execute() == IDOK) {
         if (tmlfn->params.progress)
            process ( ID_PRED_PROGRESS_ON , rest , NULL , error , NULL ) ;
         else 
            process ( ID_PRED_PROGRESS_OFF , rest , NULL , error , NULL ) ;
         sprintf ( rest , "%lf" , tmlfn->params.quit_err ) ;
         if (process ( ID_PRED_QUITERR , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%d" , tmlfn->params.acc ) ;
         if (process ( ID_PRED_ACCURACY , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%d" , tmlfn->params.refine ) ;
         if (process ( ID_PRED_REFINE , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }

         switch (tmlfn->params.errtype) {
            case ERRTYPE_MSE: strcpy ( rest , "MEAN SQUARE" ) ;  break ;
            case ERRTYPE_ABS: strcpy ( rest , "ABSOLUTE" ) ;  break ;
            case ERRTYPE_KK: strcpy ( rest , "KK" ) ;  break ;
            case ERRTYPE_XENT: strcpy ( rest , "CROSS ENTROPY" ) ;  break ;
            }
         if (process( ID_PRED_MLFN_ERROR_TYPE, rest, NULL, error , NULL ) < 0){
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         switch (tmlfn->params.method) {
            case METHOD_AN1: strcpy ( rest , "AN1" ) ;  break ;
            case METHOD_AN1_CJ: strcpy ( rest , "AN1_CJ" ) ;  break ;
            case METHOD_AN1_LM: strcpy ( rest , "AN1_LM" ) ;  break ;
            case METHOD_AN2: strcpy ( rest , "AN2" ) ;  break ;
            case METHOD_AN2_CJ: strcpy ( rest , "AN2_CJ" ) ;  break ;
            case METHOD_AN2_LM: strcpy ( rest , "AN2_LM" ) ;  break ;
            case METHOD_REGRS_CJ: strcpy ( rest , "REGRESS_CJ" ) ;  break ;
            case METHOD_REGRS_LM: strcpy ( rest , "REGRESS_LM" ) ;  break ;
            case METHOD_SS: strcpy ( rest , "SS" ) ;  break ;
            case METHOD_SSG: strcpy ( rest , "SSG" ) ;  break ;
            }
         if (process( ID_PRED_MLFN_LEARNING, rest, NULL, error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%d" , tmlfn->params.pretries ) ;
         if (process ( ID_PRED_MLFN_PRETRIES , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%d" , tmlfn->params.retries ) ;
         if (process ( ID_PRED_MLFN_RESTARTS , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }

         sprintf ( rest , "%d" , tmlfn->anneal.ntempI ) ;
         if (process ( ID_PRED_ANNEAL_INIT_TEMPS , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%d" , tmlfn->anneal.niterI ) ;
         if (process ( ID_PRED_ANNEAL_INIT_ITERS , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%d" , tmlfn->anneal.sbI ) ;
         if (process ( ID_PRED_ANNEAL_INIT_SETBACK , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%lf" , tmlfn->anneal.startI ) ;
         if (process ( ID_PRED_ANNEAL_INIT_START , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%lf" , tmlfn->anneal.endI ) ;
         if (process ( ID_PRED_ANNEAL_INIT_STOP , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         switch (tmlfn->anneal.randomI) {
            case NormalDensity: strcpy ( rest , "GAUSS" ) ;  break ;
            case CauchyDensity: strcpy ( rest , "CAUCHY" ) ;  break ;
            }
         if (process( ID_PRED_ANNEAL_INIT_RANDOM, rest, NULL, error , NULL ) < 0){
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         switch (tmlfn->anneal.reducI) {
            case ANNEAL_REDUCE_EXPONENTIAL: strcpy ( rest , "EXPONENTIAL" ) ;  break ;
            case ANNEAL_REDUCE_FAST: strcpy ( rest , "FAST" ) ;  break ;
            }
         if (process( ID_PRED_ANNEAL_INIT_REDUCTION, rest, NULL, error , NULL ) < 0){
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         if (tmlfn->anneal.climbI)
            strcpy ( rest , "CLIMB" ) ;
         else
            strcpy ( rest , "NOCLIMB" ) ;
         if (process( ID_PRED_ANNEAL_INIT_ACCEPT, rest, NULL, error , NULL ) < 0){
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%lf" , tmlfn->anneal.ratioI ) ;
         if (process ( ID_PRED_ANNEAL_INIT_RATIO , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }

         sprintf ( rest , "%d" , tmlfn->anneal.ntempE ) ;
         if (process ( ID_PRED_ANNEAL_ESCAPE_TEMPS , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%d" , tmlfn->anneal.niterE ) ;
         if (process ( ID_PRED_ANNEAL_ESCAPE_ITERS , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%d" , tmlfn->anneal.sbE ) ;
         if (process ( ID_PRED_ANNEAL_ESCAPE_SETBACK , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%lf" , tmlfn->anneal.startE ) ;
         if (process ( ID_PRED_ANNEAL_ESCAPE_START , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%lf" , tmlfn->anneal.endE ) ;
         if (process ( ID_PRED_ANNEAL_ESCAPE_STOP , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         switch (tmlfn->anneal.randomE) {
            case NormalDensity: strcpy ( rest , "GAUSS" ) ;  break ;
            case CauchyDensity: strcpy ( rest , "CAUCHY" ) ;  break ;
            }
         if (process( ID_PRED_ANNEAL_ESCAPE_RANDOM, rest, NULL, error , NULL ) < 0){
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         switch (tmlfn->anneal.reducE) {
            case ANNEAL_REDUCE_EXPONENTIAL: strcpy ( rest , "EXPONENTIAL" ) ;  break ;
            case ANNEAL_REDUCE_FAST: strcpy ( rest , "FAST" ) ;  break ;
            }
         if (process( ID_PRED_ANNEAL_ESCAPE_REDUCTION, rest, NULL, error , NULL ) < 0){
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         if (tmlfn->anneal.climbE)
            strcpy ( rest , "CLIMB" ) ;
         else
            strcpy ( rest , "NOCLIMB" ) ;
         if (process( ID_PRED_ANNEAL_ESCAPE_ACCEPT, rest, NULL, error , NULL ) < 0){
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         sprintf ( rest , "%lf" , tmlfn->anneal.ratioE ) ;
         if (process ( ID_PRED_ANNEAL_ESCAPE_RATIO , rest , NULL , error , NULL ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         if (process ( tmlfn->parm_type , tmlfn->net_name , NULL , error , result ) < 0) {
            parent->MessageBox ( error , "ERROR" ) ;
            delete tmlfn ;
            return ;
            }
         }
      delete tmlfn ;
      return ;
      }
}


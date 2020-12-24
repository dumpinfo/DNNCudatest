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

class PNNtrnDialog: public TDialog {

public:
   PNNtrnDialog ( TWindow *parent , LearnParams *lp , int p_nnets ,
                  Network **p_nets ) ;
	~PNNtrnDialog () ;

   int parm_type ;
   LearnParams params ;
   AnnealParams anneal ;
   char net_name[256] ;


protected:

   int n_nets ;
   Network **nets ;

   TComboBox *netnames ;
   TCheckBox *progress, *cv_train ;
   TStatic *error_label, *acc_label, *refine_label ;
   TEdit *error, *acc, *refine ;
   TFilterValidator *error_valid ;
   TRangeValidator *acc_valid, *refine_valid ;

   TStatic *siglo_label, *sighi_label, *nsigs_label ;
   TEdit *siglo, *sighi, *nsigs ;
   TFilterValidator *siglo_valid, *sighi_valid ;
   TRangeValidator *nsigs_valid ;

   void SetupWindow () ;
   void CmOk () ;
   void EvProgress () ;
   void EvCVtrain () ;

	DECLARE_RESPONSE_TABLE ( PNNtrnDialog ) ;
} ;

class MLFNtrnDialog: public TDialog {

public:
   MLFNtrnDialog ( TWindow *parent , LearnParams *lp , int p_nnets ,
                   Network **p_nets ) ;
	~MLFNtrnDialog () ;

   int parm_type ;
   LearnParams params ;
   AnnealParams anneal ;
   char net_name[256] ;


protected:

   int n_nets ;
   Network **nets ;

   TComboBox *netnames ;
   TCheckBox *progress, *cv_train ;
   TStatic *error_label, *acc_label, *refine_label ;
   TEdit *error, *acc, *refine ;
   TFilterValidator *error_valid ;
   TRangeValidator *acc_valid, *refine_valid ;

   TRadioButton *err_msq, *err_abs, *err_kk, *err_xent ;
   TGroupBox *errtype ;
   TRadioButton *method_an1, *method_an1cj, *method_an1lm ; 
   TRadioButton *method_an2, *method_an2cj, *method_an2lm ; 
   TRadioButton *method_regcj, *method_reglm ; 
   TRadioButton *method_ss, *method_ssg ;
   TGroupBox *method ;
   TStatic *pretries_label, *restarts_label ;
   TEdit *pretries, *restarts ;
   TRangeValidator *pretries_valid, *restarts_valid ;

   TStatic *init_label, *escape_label ;
   TStatic *ntemps_label, *niters_label, *setback_label ;
   TEdit *init_ntemps, *init_niters, *init_setback ;
   TRangeValidator *init_ntemps_valid, *init_niters_valid, *init_setback_valid ;
   TStatic *start_label, *stop_label ;
   TEdit *init_start, *init_stop ;
   TFilterValidator *init_start_valid, *init_stop_valid ;
   TStatic *random_label, *reduc_label, *climb_label ;
   TRadioButton *init_gauss, *init_cauchy ;
   TGroupBox *init_random ;
   TRadioButton *init_exp, *init_fast ;
   TGroupBox *init_reduction ;
   TCheckBox *init_climb ;
   TStatic *ratio_label ;
   TEdit *init_ratio ;
   TFilterValidator *init_ratio_valid ;

   TEdit *escape_ntemps, *escape_niters, *escape_setback ;
   TRangeValidator *escape_ntemps_valid, *escape_niters_valid, *escape_setback_valid ;
   TEdit *escape_start, *escape_stop ;
   TFilterValidator *escape_start_valid, *escape_stop_valid ;
   TRadioButton *escape_gauss, *escape_cauchy ;
   TGroupBox *escape_random ;
   TRadioButton *escape_exp, *escape_fast ;
   TGroupBox *escape_reduction ;
   TCheckBox *escape_climb ;
   TEdit *escape_ratio ;
   TFilterValidator *escape_ratio_valid ;

   void SetupWindow () ;
   void CmOk () ;
   void EvProgress () ;
   void EvCVtrain () ;
   void EvErrMse () ;
   void EvErrAbs () ;
   void EvErrKK () ;
   void EvErrXent () ;
   void EvAN1 () ;
   void EvAN1CJ () ;
   void EvAN1LM () ;
   void EvAN2 () ;
   void EvAN2CJ () ;
   void EvAN2LM () ;
   void EvRegCJ () ;
   void EvRegLM () ;
   void EvSS () ;
   void EvSSG () ;
   void EvInitGauss () ;
   void EvInitCauchy () ;
   void EvInitExp () ;
   void EvInitFast () ;
   void EvInitClimb () ;
   void EvInitNoClimb () ;
   void EvEscapeGauss () ;
   void EvEscapeCauchy () ;
   void EvEscapeExp () ;
   void EvEscapeFast () ;
   void EvEscapeClimb () ;
   void EvEscapeNoClimb () ;

	DECLARE_RESPONSE_TABLE ( MLFNtrnDialog ) ;
} ;

extern void train_net ( TWindow *parent ) ;

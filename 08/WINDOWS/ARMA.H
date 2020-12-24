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

class ARMAtrainDialog: public TDialog {

public:
   ARMAtrainDialog ( TWindow *parent , LearnParams *lp , int p_fixed ,
                     int p_nARMAs , ARMA **p_ARMAs ) ;
	~ARMAtrainDialog () ;

   LearnParams params ;
   int parm_fixed ;
   char arma_name[256] ;


protected:

   int n_ARMAs ;
   ARMA **ARMAs ;

   TCheckBox *fixed, *progress ;
   TEdit *error, *acc, *refine ;
   TFilterValidator *error_valid ;
   TRangeValidator *acc_valid, *refine_valid ;
   TComboBox *arma_names ;

   void SetupWindow () ;
   void CmOk () ;
   void EvFixed () ;
   void EvProgress () ;

	DECLARE_RESPONSE_TABLE ( ARMAtrainDialog ) ;
} ;

extern void train_arma ( TWindow *parent ) ;

/*
--------------------------------------------------------------------------------

   Predict

--------------------------------------------------------------------------------
*/

class ARMApredictDialog: public TDialog {

public:
   ARMApredictDialog ( TWindow *parent , double p_prob , int p_nARMAs ,
                       ARMA **p_ARMAs , int p_nsignals , Signal **p_signals ) ;
	~ARMApredictDialog () ;

   int parm_points, parm_conf, parm_dist ;
   double parm_pct ;
   char arma_name[256], dest_name[256] ;


protected:

   double prob ;
   int n_ARMAs ;
   ARMA **ARMAs ;
   int n_signals ;
   Signal **signals ;

   TEdit *points ;
   TRangeValidator *points_valid ;
   TCheckBox *conf ;
   TStatic *dist_label ;
   TEdit *dist ;
   TRangeValidator *dist_valid ;
   TStatic *pct_label ;
   TEdit *pct ;
   TFilterValidator *pct_valid ;
   TListBox *arma_names ;
   TComboBox *dest ;

   void SetupWindow () ;
   void CmOk () ;
   void EvConf () ;

	DECLARE_RESPONSE_TABLE ( ARMApredictDialog ) ;
} ;

extern void predict_arma ( TWindow *parent ) ;

/*
--------------------------------------------------------------------------------

   Predicted shocks

--------------------------------------------------------------------------------
*/

#define MAX_WIN_ARMA_SHOCK 10

class ShockDialog: public TDialog {

public:
   ShockDialog ( TWindow *parent , int p_nsignals , Signal **p_signals ,
                 int p_ninsouts , InputOutput **p_insouts ) ;
	~ShockDialog () ;

   int nout ;
   char shock_names[MAX_WIN_ARMA_SHOCK][256], out_names[MAX_WIN_ARMA_SHOCK][256] ;

protected:

   int n_signals ;
   Signal **signals ;
   int n_insouts ;
   InputOutput **insouts ;

   int shock_used[MAX_WIN_ARMA_SHOCK] ;
   TComboBox *shock_name[MAX_WIN_ARMA_SHOCK] ;
   TStatic *out_name[MAX_WIN_ARMA_SHOCK] ;

   void SetupWindow () ;
   void CmOk () ;

	DECLARE_RESPONSE_TABLE ( ShockDialog ) ;
} ;

extern void shock_arma ( TWindow *parent ) ;

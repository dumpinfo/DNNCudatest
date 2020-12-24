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

class DispDialog: public TDialog {

public:
   DispDialog ( TWindow *parent , MiscParams *mp , int p_nsignals ,
                Signal **p_signals ) ;
	~DispDialog () ;

   int parm_range, parm_domain0, parm_domain1, parm_conf ;
   double parm_min, parm_max, parm_rate, parm_origin ;
   char source_name[256] ;

protected:

   int n_signals ;
   Signal **signals ;

   TRadioButton *range_optimal, *range_symmetric, *range_fixed ;
   TStatic *domain0_label, *domain1_label ;
   TEdit *domain0, *domain1 ;
   TFilterValidator *domain0_valid, *domain1_valid ;
   TStatic *sig_min_label, *sig_max_label ;
   TEdit *sig_min, *sig_max ;
   TFilterValidator *sig_min_valid, *sig_max_valid ;
   TStatic *origin_label, *rate_label ;
   TEdit *origin, *rate ;
   TFilterValidator *origin_valid, *rate_valid ;
   TCheckBox *conf ;
   TListBox *source ;

   void SetupWindow () ;
   void CmOk () ;
   void EvOptimal () ;
   void EvSymmetric () ;
   void EvFixed () ;
   void EvConfidence () ;

	DECLARE_RESPONSE_TABLE ( DispDialog ) ;
} ;

extern void do_display ( TWindow *parent ) ;



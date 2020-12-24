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

class NetPredDialog: public TDialog {

public:
   NetPredDialog ( TWindow *parent , double p_prob , int p_nnets ,
                   Network **p_nets , int p_nsignals , Signal **p_signals ) ;
	~NetPredDialog () ;

   int parm_points, parm_conf, parm_dist ;
   double parm_pct ;
   char net_name[256], dest_name[256] ;


protected:

   double prob ;
   int n_nets ;
   Network **nets ;
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
   TListBox *net_names ;
   TComboBox *dest ;

   void SetupWindow () ;
   void CmOk () ;
   void EvConf () ;

	DECLARE_RESPONSE_TABLE ( NetPredDialog ) ;
} ;

extern void predict_net ( TWindow *parent ) ;

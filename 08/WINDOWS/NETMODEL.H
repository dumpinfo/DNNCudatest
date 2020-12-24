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

class NetModelDialog: public TDialog {

public:
   NetModelDialog ( TWindow *parent , int model ,
                    int nhid1 , int nhid2 , int domain ,
                    int linear , int kernel ) ;
	~NetModelDialog () ;

   int model, nhid1, nhid2, domain, out_act, kernel ;
   TEdit *hid1, *hid2 ;

protected:

   TRadioButton *pnn_gauss, *pnn_recip, *sepvar, *sepclass ;
   TRadioButton *mlfn_real, *mlfn_cpxin, *mlfn_cpxhid, *mlfn_cpx ;
   TCheckBox *linear ;
   TRangeValidator *valid1, *valid2 ;
   TStatic *hid1label, *hid2label ;

   void SetupWindow () ;
   void CmOk () ;
   void EvPNN_GAUSS () ;
   void EvPNN_RECIP () ;
   void EvSEPVAR () ;
   void EvSEPCLASS () ;
   void EvMLFN_REAL () ;
   void EvMLFN_CPXIN () ;
   void EvMLFN_CPXHID () ;
   void EvMLFN_CPX () ;
   void EvLINEAR () ;

	DECLARE_RESPONSE_TABLE ( NetModelDialog ) ;
} ;

extern void get_net_model ( TWindow *parent ) ;

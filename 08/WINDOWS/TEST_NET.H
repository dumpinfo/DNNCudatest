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

class TestNetDialog: public TDialog {

public:
   TestNetDialog ( TWindow *parent , double p_thresh ,
                   int p_nnets , Network **p_nets ) ;
	~TestNetDialog () ;

   int parm_type ;
   double parm_thresh ;
   char net_name[256] ;


protected:

   int n_nets ;
   Network **nets ;

   TCheckBox *type ;
   TEdit *thresh ;
   TFilterValidator *thresh_valid ;
   TListBox *netnames ;

   void SetupWindow () ;
   void CmOk () ;
   void EvExtended () ;

	DECLARE_RESPONSE_TABLE ( TestNetDialog ) ;
} ;

extern void test_net ( TWindow *parent ) ;

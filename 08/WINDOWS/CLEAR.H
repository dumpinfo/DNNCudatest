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

class ClearDialog: public TDialog {

public:
   ClearDialog ( TWindow *parent , int p_nnets , Network **p_nets ,
        int p_nARMAs , ARMA **p_ARMAs , int p_northogs , Orthog **p_orthogs ) ;
	~ClearDialog () ;

   int parm_type ;
   char parm_net[256], parm_arma[256], parm_orthog[256] ;

protected:

   int n_nets ;
   Network **networks ;
   int n_ARMAs ;
   ARMA **ARMAs ;
   int n_orthogs ;
   Orthog **orths ;

   TRadioButton *inlist, *outlist, *classes, *train, *test, *conf ;
   TRadioButton *net, *nets, *arma, *armas, *orthog, *orthogs, *all ;
   TComboBox *net_name, *arma_name, *orthog_name ;

   void SetupWindow () ;
   void CmOk () ;
   void EvInlist () ;
   void EvOutlist () ;
   void EvClasses () ;
   void EvTrain () ;
   void EvTest () ;
   void EvConf () ;
   void EvNet () ;
   void EvNets () ;
   void EvArma () ;
   void EvArmas () ;
   void EvOrthog () ;
   void EvOrthogs () ;
   void EvAll () ;

	DECLARE_RESPONSE_TABLE ( ClearDialog ) ;
} ;

extern void do_clear ( TWindow *parent ) ;


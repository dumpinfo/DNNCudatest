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

/*
   ReadSignalDialog
*/

class ReadSignalDialog: public TDialog {

public:
   ReadSignalDialog ( TWindow *parent , int p_nsignals , Signal **p_signals ) ;
	~ReadSignalDialog () ;

   char file_name[256], dest_name[256] ;


protected:

   int n_signals ;
   Signal **signals ;

   TComboBox *dest ;

   void SetupWindow () ;
   void CmOk () ;
   void EvFileName () ;

	DECLARE_RESPONSE_TABLE ( ReadSignalDialog ) ;
} ;

extern void read_signal ( TWindow *parent ) ;


/*
   SaveSignalDialog
*/

class SaveSignalDialog: public TDialog {

public:
   SaveSignalDialog ( TWindow *parent , int p_nsignals , Signal **p_signals ) ;
	~SaveSignalDialog () ;

   char file_name[256], source_name[256] ;


protected:

   int n_signals ;
   Signal **signals ;

   TListBox *source ;

   void SetupWindow () ;
   void CmOk () ;
   void EvFileName () ;

	DECLARE_RESPONSE_TABLE ( SaveSignalDialog ) ;
} ;

extern void save_signal ( TWindow *parent ) ;

/*
   ReadNetworkDialog
*/

class ReadNetworkDialog: public TDialog {

public:
   ReadNetworkDialog ( TWindow *parent , int p_nnets , Network **p_nets ) ;
	~ReadNetworkDialog () ;

   char file_name[256], dest_name[256] ;


protected:

   int n_nets ;
   Network **nets ;

   TComboBox *dest ;

   void SetupWindow () ;
   void CmOk () ;
   void EvFileName () ;

	DECLARE_RESPONSE_TABLE ( ReadNetworkDialog ) ;
} ;

extern void read_network ( TWindow *parent ) ;


/*
   SaveNetworkDialog
*/

class SaveNetworkDialog: public TDialog {

public:
   SaveNetworkDialog ( TWindow *parent , int p_nnets , Network **p_nets ) ;
	~SaveNetworkDialog () ;

   int parm_print ;
   char file_name[256], source_name[256] ;


protected:

   int n_nets ;
   Network **nets ;

   TCheckBox *print ;
   TListBox *source ;

   void SetupWindow () ;
   void CmOk () ;
   void EvFileName () ;
   void EvPrint () ;

	DECLARE_RESPONSE_TABLE ( SaveNetworkDialog ) ;
} ;

extern void save_network ( TWindow *parent ) ;


/*
   ReadARMADialog
*/

class ReadARMADialog: public TDialog {

public:
   ReadARMADialog ( TWindow *parent , int p_nARMAs , ARMA **p_ARMAs ) ;
	~ReadARMADialog () ;

   char file_name[256], dest_name[256] ;


protected:

   int n_ARMAs ;
   ARMA **ARMAs ;

   TComboBox *dest ;

   void SetupWindow () ;
   void CmOk () ;
   void EvFileName () ;

	DECLARE_RESPONSE_TABLE ( ReadARMADialog ) ;
} ;

extern void read_arma ( TWindow *parent ) ;


/*
   SaveARMADialog
*/

class SaveARMADialog: public TDialog {

public:
   SaveARMADialog ( TWindow *parent , int p_nARMAs , ARMA **p_ARMAs ) ;
	~SaveARMADialog () ;

   char file_name[256], source_name[256] ;


protected:

   int n_ARMAs ;
   ARMA **ARMAs ;

   TListBox *source ;

   void SetupWindow () ;
   void CmOk () ;
   void EvFileName () ;

	DECLARE_RESPONSE_TABLE ( SaveARMADialog ) ;
} ;

extern void save_arma ( TWindow *parent ) ;


/*
   ReadOrthogDialog
*/

class ReadOrthogDialog: public TDialog {

public:
   ReadOrthogDialog ( TWindow *parent , int p_northogs , Orthog **p_orthogs ) ;
	~ReadOrthogDialog () ;

   char file_name[256], dest_name[256] ;


protected:

   int n_orthogs ;
   Orthog **orthogs ;

   TComboBox *dest ;

   void SetupWindow () ;
   void CmOk () ;
   void EvFileName () ;

	DECLARE_RESPONSE_TABLE ( ReadOrthogDialog ) ;
} ;

extern void read_orthog ( TWindow *parent ) ;


/*
   SaveOrthogDialog
*/

class SaveOrthogDialog: public TDialog {

public:
   SaveOrthogDialog ( TWindow *parent , int p_northogs , Orthog **p_orthogs ) ;
	~SaveOrthogDialog () ;

   char file_name[256], source_name[256] ;


protected:

   int n_orthogs ;
   Orthog **orthogs ;

   TListBox *source ;

   void SetupWindow () ;
   void CmOk () ;
   void EvFileName () ;

	DECLARE_RESPONSE_TABLE ( SaveOrthogDialog ) ;
} ;

extern void save_orthog ( TWindow *parent ) ;

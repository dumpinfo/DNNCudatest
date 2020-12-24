/******************************************************************************/
/*                                                                            */
/*  READSAVE - Read and save signals, networks, ARMAs, and ORTHOGs            */
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

#include "readsave.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( ReadSignalDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_READSIG_FILENAME , EvFileName ) ,
END_RESPONSE_TABLE ;

ReadSignalDialog::ReadSignalDialog ( TWindow *parent ,
                                     int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_READSIG ), TWindow ( parent )
{
   n_signals = p_nsignals ;
   signals = p_signals ;

	dest = new TComboBox ( this , IDC_READSIG_OBJNAME ) ;
   strcpy ( file_name , "" ) ;
   strcpy ( dest_name , "" ) ;
}

ReadSignalDialog::~ReadSignalDialog ()
{
}

void ReadSignalDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      dest->AddString ( sigptr->name ) ;
      }
   
   dest->SetSelIndex ( -1 ) ;
}

void ReadSignalDialog::CmOk ()
{
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;
	TDialog::CmOk () ;
}

void ReadSignalDialog::EvFileName ()
{
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("Data files (*.DAT)|*.DAT|All Files (*.*)|*.*|") ;
	*FileData.FileName = 0 ;

	if (TFileOpenDialog ( this , FileData ).Execute()  ==  IDOK)
      strcpy ( file_name , FileData.FileName ) ;
}


/*
	This is a global routine called to do the busywork
*/

void read_signal ( TWindow *parent )
{
   int nsigs ;
	char error[256] ;
	ReadSignalDialog *rs ;
   Signal **signals ;

   nsigs = get_signals ( &signals ) ;
	rs = new ReadSignalDialog ( parent , nsigs , signals ) ;

	if (rs->Execute() == IDOK) {
      if (strlen ( rs->file_name )  &&  strlen ( rs->dest_name )) {
         if (process ( ID_PRED_NAME , rs->dest_name , NULL , error , NULL ) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         if (process ( ID_PRED_READSIG , rs->file_name , NULL , error , NULL) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      if (! strlen ( rs->file_name ))
         parent->MessageBox ( "No file name specified" , "ERROR" ) ;
      if (! strlen ( rs->dest_name ))
         parent->MessageBox ( "No signal name specified" , "ERROR" ) ;
      }

	delete rs ;
}


/*
--------------------------------------------------------------------------------

   SaveSignalDialog

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( SaveSignalDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_SAVESIG_FILENAME , EvFileName ) ,
END_RESPONSE_TABLE ;

SaveSignalDialog::SaveSignalDialog ( TWindow *parent ,
                                     int p_nsignals , Signal **p_signals )
	: TDialog ( parent , IDD_SAVESIG ), TWindow ( parent )
{
   n_signals = p_nsignals ;
   signals = p_signals ;

	source = new TListBox ( this , IDC_SAVESIG_OBJNAME ) ;
   strcpy ( file_name , "" ) ;
   strcpy ( source_name , "" ) ;
}

SaveSignalDialog::~SaveSignalDialog ()
{
}

void SaveSignalDialog::SetupWindow ()
{
   int i ;
   Signal *sigptr ;

	TDialog::SetupWindow () ;

   for (i=0 ; i<n_signals ; i++) {
      sigptr = signals[i] ;
      source->AddString ( sigptr->name ) ;
      }
   
   source->SetSelIndex ( 0 ) ;
}

void SaveSignalDialog::CmOk ()
{
   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
	TDialog::CmOk () ;
}

void SaveSignalDialog::EvFileName ()
{
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("Data files (*.DAT)|*.DAT|All Files (*.*)|*.*|") ;
	*FileData.FileName = 0 ;

	if (TFileSaveDialog ( this , FileData ).Execute()  ==  IDOK)
      strcpy ( file_name , FileData.FileName ) ;
}


/*
	This is a global routine called to do the busywork
*/

void save_signal ( TWindow *parent )
{
   int nsigs ;
	char error[256], rest[256] ;
	SaveSignalDialog *ss ;
   Signal **signals ;

   nsigs = get_signals ( &signals ) ;
	ss = new SaveSignalDialog ( parent , nsigs , signals ) ;

	if (ss->Execute() == IDOK) {
      if (strlen ( ss->file_name )  &&  strlen ( ss->source_name )) {
         sprintf ( rest , "%s TO %s", ss->source_name, ss->file_name ) ;
         if (process ( ID_PRED_SAVESIG , rest , NULL , error , NULL) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      if (! strlen ( ss->file_name ))
         parent->MessageBox ( "No file name specified" , "ERROR" ) ;
      if (! strlen ( ss->source_name ))
         parent->MessageBox ( "No signal name specified" , "ERROR" ) ;
      }

	delete ss ;
}

/*
--------------------------------------------------------------------------------

   ReadNetworkDialog

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( ReadNetworkDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_READNET_FILENAME , EvFileName ) ,
END_RESPONSE_TABLE ;

ReadNetworkDialog::ReadNetworkDialog ( TWindow *parent ,
                                       int p_nnets , Network **p_nets )
	: TDialog ( parent , IDD_READNET ), TWindow ( parent )
{
   n_nets = p_nnets ;
   nets = p_nets ;

	dest = new TComboBox ( this , IDC_READNET_OBJNAME ) ;
   strcpy ( file_name , "" ) ;
   strcpy ( dest_name , "" ) ;
}

ReadNetworkDialog::~ReadNetworkDialog ()
{
}

void ReadNetworkDialog::SetupWindow ()
{
   int i ;
   Network *netptr ;

	TDialog::SetupWindow () ;

   for (i=0 ; i<n_nets ; i++) {
      netptr = nets[i] ;
      dest->AddString ( netptr->name ) ;
      }
   
   dest->SetSelIndex ( -1 ) ;
}

void ReadNetworkDialog::CmOk ()
{
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;
	TDialog::CmOk () ;
}

void ReadNetworkDialog::EvFileName ()
{
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("Net files (*.WTS)|*.WTS|All Files (*.*)|*.*|") ;
	*FileData.FileName = 0 ;

	if (TFileOpenDialog ( this , FileData ).Execute()  ==  IDOK)
      strcpy ( file_name , FileData.FileName ) ;
}


/*
	This is a global routine called to do the busywork
*/

void read_network ( TWindow *parent )
{
   int nnets ;
	char error[256], rest[256] ;
	ReadNetworkDialog *rn ;
   Network **networks ;

   nnets = get_networks ( &networks ) ;
	rn = new ReadNetworkDialog ( parent , nnets , networks ) ;

	if (rn->Execute() == IDOK) {
      if (strlen ( rn->file_name )  &&  strlen ( rn->dest_name )) {
         sprintf ( rest , "%s FROM %s", rn->dest_name, rn->file_name ) ;
         if (process ( ID_PRED_RESTORE_NETWORK , rest , NULL , error, NULL) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      if (! strlen ( rn->file_name ))
         parent->MessageBox ( "No file name specified" , "ERROR" ) ;
      if (! strlen ( rn->dest_name ))
         parent->MessageBox ( "No network name specified" , "ERROR" ) ;
      }

	delete rn ;
}


/*
--------------------------------------------------------------------------------

   SaveNetworkDialog

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( SaveNetworkDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_SAVENET_FILENAME , EvFileName ) ,
  EV_COMMAND ( IDC_SAVENET_PRINT , EvPrint ) ,
END_RESPONSE_TABLE ;

SaveNetworkDialog::SaveNetworkDialog ( TWindow *parent ,
                                       int p_nnets , Network **p_nets )
	: TDialog ( parent , IDD_SAVENET ), TWindow ( parent )
{
   n_nets = p_nnets ;
   nets = p_nets ;

   print = new TCheckBox ( this , IDC_SAVENET_PRINT ) ;
	source = new TListBox ( this , IDC_SAVENET_OBJNAME ) ;
   strcpy ( file_name , "" ) ;
   strcpy ( source_name , "" ) ;
}

SaveNetworkDialog::~SaveNetworkDialog ()
{
}

void SaveNetworkDialog::SetupWindow ()
{
   int i ;
   Network *netptr ;

	TDialog::SetupWindow () ;

   parm_print = 0 ;
   print->SetCheck ( BF_UNCHECKED ) ;

   for (i=0 ; i<n_nets ; i++) {
      netptr = nets[i] ;
      source->AddString ( netptr->name ) ;
      }
   
   source->SetSelIndex ( 0 ) ;
}

void SaveNetworkDialog::CmOk ()
{
   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
	TDialog::CmOk () ;
}

void SaveNetworkDialog::EvFileName ()
{
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("Net files (*.WTS)|*.WTS|All Files (*.*)|*.*|") ;
	*FileData.FileName = 0 ;

	if (TFileSaveDialog ( this , FileData ).Execute()  ==  IDOK)
      strcpy ( file_name , FileData.FileName ) ;
}

void SaveNetworkDialog::EvPrint ()
{
   if (print->GetCheck() == BF_CHECKED)
      parm_print = 1 ;
   else 
      parm_print = 0 ;
}


/*
	This is a global routine called to do the busywork
*/

void save_network ( TWindow *parent )
{
   int nnets, type ;
	char error[256], rest[256] ;
	SaveNetworkDialog *sn ;
   Network **networks ;

   nnets = get_networks ( &networks ) ;
	sn = new SaveNetworkDialog ( parent , nnets , networks ) ;

	if (sn->Execute() == IDOK) {
      if (strlen ( sn->file_name )  &&  strlen ( sn->source_name )) {
         type = sn->parm_print  ?  ID_PRED_PRINT_NETWORK : ID_PRED_SAVE_NETWORK;
         sprintf ( rest , "%s TO %s", sn->source_name, sn->file_name ) ;
         if (process ( type , rest , NULL , error , NULL) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      if (! strlen ( sn->file_name ))
         parent->MessageBox ( "No file name specified" , "ERROR" ) ;
      if (! strlen ( sn->source_name ))
         parent->MessageBox ( "No signal name specified" , "ERROR" ) ;
      }

	delete sn ;
}

/*
--------------------------------------------------------------------------------

   ReadARMADialog

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( ReadARMADialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_READARMA_FILENAME , EvFileName ) ,
END_RESPONSE_TABLE ;

ReadARMADialog::ReadARMADialog ( TWindow *parent ,
                                 int p_nARMAs , ARMA **p_ARMAs )
	: TDialog ( parent , IDD_READARMA ), TWindow ( parent )
{
   n_ARMAs = p_nARMAs ;
   ARMAs = p_ARMAs ;

	dest = new TComboBox ( this , IDC_READARMA_OBJNAME ) ;
   strcpy ( file_name , "" ) ;
   strcpy ( dest_name , "" ) ;
}

ReadARMADialog::~ReadARMADialog ()
{
}

void ReadARMADialog::SetupWindow ()
{
   int i ;
   ARMA *armaptr ;

	TDialog::SetupWindow () ;

   for (i=0 ; i<n_ARMAs ; i++) {
      armaptr = ARMAs[i] ;
      dest->AddString ( armaptr->name ) ;
      }
   
   dest->SetSelIndex ( -1 ) ;
}

void ReadARMADialog::CmOk ()
{
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;
	TDialog::CmOk () ;
}

void ReadARMADialog::EvFileName ()
{
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("ARMA files (*.ARM)|*.ARM|All Files (*.*)|*.*|") ;
	*FileData.FileName = 0 ;

	if (TFileOpenDialog ( this , FileData ).Execute()  ==  IDOK)
      strcpy ( file_name , FileData.FileName ) ;
}


/*
	This is a global routine called to do the busywork
*/

void read_arma ( TWindow *parent )
{
   int i, narmas ;
	char error[256], rest[256] ;
	ReadARMADialog *ra ;
   ARMA **armas ;

   narmas = get_ARMAs ( &armas , &i ) ;
	ra = new ReadARMADialog ( parent , narmas , armas ) ;

	if (ra->Execute() == IDOK) {
      if (strlen ( ra->file_name )  &&  strlen ( ra->dest_name )) {
         sprintf ( rest , "%s FROM %s", ra->dest_name, ra->file_name ) ;
         if (process ( ID_PRED_RESTORE_ARMA , rest , NULL , error , NULL) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      if (! strlen ( ra->file_name ))
         parent->MessageBox ( "No file name specified" , "ERROR" ) ;
      if (! strlen ( ra->dest_name ))
         parent->MessageBox ( "No ARMA name specified" , "ERROR" ) ;
      }

	delete ra ;
}


/*
--------------------------------------------------------------------------------

   SaveARMADialog

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( SaveARMADialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_SAVEARMA_FILENAME , EvFileName ) ,
END_RESPONSE_TABLE ;

SaveARMADialog::SaveARMADialog ( TWindow *parent ,
                                 int p_nARMAs , ARMA **p_ARMAs )
	: TDialog ( parent , IDD_SAVEARMA ), TWindow ( parent )
{
   n_ARMAs = p_nARMAs ;
   ARMAs = p_ARMAs ;

	source = new TListBox ( this , IDC_SAVEARMA_OBJNAME ) ;
   strcpy ( file_name , "" ) ;
   strcpy ( source_name , "" ) ;
}

SaveARMADialog::~SaveARMADialog ()
{
}

void SaveARMADialog::SetupWindow ()
{
   int i ;
   ARMA *armaptr ;

	TDialog::SetupWindow () ;

   for (i=0 ; i<n_ARMAs ; i++) {
      armaptr = ARMAs[i] ;
      source->AddString ( armaptr->name ) ;
      }
   
   source->SetSelIndex ( 0 ) ;
}

void SaveARMADialog::CmOk ()
{
   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
	TDialog::CmOk () ;
}

void SaveARMADialog::EvFileName ()
{
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("ARMA files (*.ARM)|*.ARM|All Files (*.*)|*.*|") ;
	*FileData.FileName = 0 ;

	if (TFileSaveDialog ( this , FileData ).Execute()  ==  IDOK)
      strcpy ( file_name , FileData.FileName ) ;
}


/*
	This is a global routine called to do the busywork
*/

void save_arma ( TWindow *parent )
{
   int narmas, i ;
	char error[256], rest[256] ;
	SaveARMADialog *sa ;
   ARMA **armas ;

   narmas = get_ARMAs ( &armas , &i ) ;
	sa = new SaveARMADialog ( parent , narmas , armas ) ;

	if (sa->Execute() == IDOK) {
      if (strlen ( sa->file_name )  &&  strlen ( sa->source_name )) {
         sprintf ( rest , "%s TO %s", sa->source_name, sa->file_name ) ;
         if (process ( ID_PRED_SAVE_ARMA , rest , NULL , error , NULL) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      if (! strlen ( sa->file_name ))
         parent->MessageBox ( "No file name specified" , "ERROR" ) ;
      if (! strlen ( sa->source_name ))
         parent->MessageBox ( "No signal name specified" , "ERROR" ) ;
      }

	delete sa ;
}

/*
--------------------------------------------------------------------------------

   ReadOrthogDialog

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( ReadOrthogDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_READORTHOG_FILENAME , EvFileName ) ,
END_RESPONSE_TABLE ;

ReadOrthogDialog::ReadOrthogDialog ( TWindow *parent ,
                                     int p_northogs , Orthog **p_orthogs )
	: TDialog ( parent , IDD_READORTHOG ), TWindow ( parent )
{
   n_orthogs = p_northogs ;
   orthogs = p_orthogs ;

	dest = new TComboBox ( this , IDC_READORTHOG_OBJNAME ) ;
   strcpy ( file_name , "" ) ;
   strcpy ( dest_name , "" ) ;
}

ReadOrthogDialog::~ReadOrthogDialog ()
{
}

void ReadOrthogDialog::SetupWindow ()
{
   int i ;
   Orthog *orthptr ;

	TDialog::SetupWindow () ;

   for (i=0 ; i<n_orthogs ; i++) {
      orthptr = orthogs[i] ;
      dest->AddString ( orthptr->name ) ;
      }
   
   dest->SetSelIndex ( -1 ) ;
}

void ReadOrthogDialog::CmOk ()
{
   dest->GetText ( dest_name , 255 ) ;
   strupr ( dest_name ) ;
	TDialog::CmOk () ;
}

void ReadOrthogDialog::EvFileName ()
{
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("ORTHOG files (*.ORT)|*.ORT|All Files (*.*)|*.*|") ;
	*FileData.FileName = 0 ;

	if (TFileOpenDialog ( this , FileData ).Execute()  ==  IDOK)
      strcpy ( file_name , FileData.FileName ) ;
}


/*
	This is a global routine called to do the busywork
*/

void read_orthog ( TWindow *parent )
{
   int northogs ;
	char error[256], rest[256] ;
	ReadOrthogDialog *ro ;
   Orthog **orthogs ;

   northogs = get_orthogs ( &orthogs ) ;
	ro = new ReadOrthogDialog ( parent , northogs , orthogs ) ;

	if (ro->Execute() == IDOK) {
      if (strlen ( ro->file_name )  &&  strlen ( ro->dest_name )) {
         sprintf ( rest , "%s FROM %s", ro->dest_name, ro->file_name ) ;
         if (process ( ID_PRED_RESTORE_ORTHOG , rest , NULL , error , NULL) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      if (! strlen ( ro->file_name ))
         parent->MessageBox ( "No file name specified" , "ERROR" ) ;
      if (! strlen ( ro->dest_name ))
         parent->MessageBox ( "No orthogonalization name specified" , "ERROR" ) ;
      }

	delete ro ;
}


/*
--------------------------------------------------------------------------------

   SaveOrthogDialog

--------------------------------------------------------------------------------
*/

DEFINE_RESPONSE_TABLE1 ( SaveOrthogDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
  EV_COMMAND ( IDC_SAVEORTHOG_FILENAME , EvFileName ) ,
END_RESPONSE_TABLE ;

SaveOrthogDialog::SaveOrthogDialog ( TWindow *parent ,
                                     int p_northogs , Orthog **p_orthogs )
	: TDialog ( parent , IDD_SAVEORTHOG ), TWindow ( parent )
{
   n_orthogs = p_northogs ;
   orthogs = p_orthogs ;

	source = new TListBox ( this , IDC_SAVEORTHOG_OBJNAME ) ;
   strcpy ( file_name , "" ) ;
   strcpy ( source_name , "" ) ;
}

SaveOrthogDialog::~SaveOrthogDialog ()
{
}

void SaveOrthogDialog::SetupWindow ()
{
   int i ;
   Orthog *orthptr ;

	TDialog::SetupWindow () ;

   for (i=0 ; i<n_orthogs ; i++) {
      orthptr = orthogs[i] ;
      source->AddString ( orthptr->name ) ;
      }
   
   source->SetSelIndex ( 0 ) ;
}

void SaveOrthogDialog::CmOk ()
{
   source->GetSelString ( source_name , 255 ) ;
   strupr ( source_name ) ;
	TDialog::CmOk () ;
}

void SaveOrthogDialog::EvFileName ()
{
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("ORTHOG files (*.ORT)|*.ORT|All Files (*.*)|*.*|") ;
	*FileData.FileName = 0 ;

	if (TFileSaveDialog ( this , FileData ).Execute()  ==  IDOK)
      strcpy ( file_name , FileData.FileName ) ;
}


/*
	This is a global routine called to do the busywork
*/

void save_orthog ( TWindow *parent )
{
   int northogs ;
	char error[256], rest[256] ;
	SaveOrthogDialog *so ;
   Orthog **orthogs ;

   northogs = get_orthogs ( &orthogs ) ;
	so = new SaveOrthogDialog ( parent , northogs , orthogs ) ;

	if (so->Execute() == IDOK) {
      if (strlen ( so->file_name )  &&  strlen ( so->source_name )) {
         sprintf ( rest , "%s TO %s", so->source_name, so->file_name ) ;
         if (process ( ID_PRED_SAVE_ORTHOG , rest , NULL , error , NULL) < 0)
            parent->MessageBox ( error , "ERROR" ) ;
         }
      if (! strlen ( so->file_name ))
         parent->MessageBox ( "No file name specified" , "ERROR" ) ;
      if (! strlen ( so->source_name ))
         parent->MessageBox ( "No orthogonalization name specified" , "ERROR" );
      }

	delete so ;
}

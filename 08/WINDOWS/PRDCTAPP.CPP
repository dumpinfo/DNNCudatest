/******************************************************************************/
/*                                                                            */
/*  PRDCTAPP - Main for NPREDICT                                              */
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

#include <malloc.h>

#include "..\common\const.h"  // System, limitation constants, typedefs, structs
#include "..\common\classes.h"  // Includes all class headers
#include "..\common\funcdefs.h" // Function prototypes

#include "com_file.h"   // Things in COM_FILE.CPP (Read command file)
#include "prog_win.h"   // Things in PROG_WIN.CPP (ProgressWindow)
#include "readsave.h"   // Things in READSAVE.CPP (Read and save things)
#include "clear.h"      // Things in CLEAR.CPP (Clear objects)
#include "disp.h"       // Things in DISP.CPP (Display a signal)
#include "generate.h"   // Things in GENERATE.CPP (Generate a signal)
#include "copy.h"       // Things in COPY.CPP (Copy a signal)
#include "modify.h"     // Things in MODIFY.CPP (Modify a signal)
#include "filter.h"     // Things in FILTER.CPP (Filter a signal)
#include "lag_corr.h"   // Things in LAG_CORR.CPP (Lagged correlations)
#include "spectrum.h"   // Things in SPECTRUM.CPP (Power spectrum)
#include "arith.h"      // Things in ARITH.CPP (Arithmetic combination)
#include "cumulate.h"   // Things in CUMULATE.CPP (Cumulate training/test sets)
#include "confcomp.h"   // Things in CONFCOMP.CPP (Confidence compnsation)
#include "orthog.h"     // Things in ORTHOG.CPP (Orthogonalization)
#include "in_out.h"     // Things in IN_OUT.CPP (Input and Output lists)
#include "netmodel.h"   // Things in NETMODEL.CPP (Network model)
#include "net_trn.h"    // Things in TRN_NET.CPP (Network training)
#include "test_net.h"   // Things in TEST_NET.CPP (Network testing)
#include "net_pred.h"   // Things in NET_PRED.CPP (Network prediction)
#include "arma.h"       // Things in ARMA.CPP (All ARMA stuff)

#include "prdctapp.h"
#include "prdcmdic.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1(npredictApp, TApplication)
    EV_COMMAND(CM_FILE_READ_SIGNAL, CmReadSignal),
    EV_COMMAND(CM_FILE_SAVE_SIGNAL, CmSaveSignal),
    EV_COMMAND(CM_FILE_READ_NET, CmReadNetwork),
    EV_COMMAND(CM_FILE_SAVE_NET, CmSaveNetwork),
    EV_COMMAND(CM_FILE_READ_ARMA, CmReadARMA),
    EV_COMMAND(CM_FILE_SAVE_ARMA, CmSaveARMA),
    EV_COMMAND(CM_FILE_READ_ORTHOG, CmReadOrthog),
    EV_COMMAND(CM_FILE_SAVE_ORTHOG, CmSaveOrthog),
    EV_COMMAND(CM_FILE_COMMAND_FILE, CmCommandFile),
    EV_COMMAND(CM_FILE_CLEAR, CmClear),
    EV_COMMAND(CM_FILE_AUDIT_LOG, CmAuditLog),
    EV_COMMAND(CM_SIGOPS_DISPLAY_SIGNAL, CmDisplaySignal),
    EV_COMMAND(CM_SIGOPS_GENERATE, CmGenerate),
    EV_COMMAND(CM_SIGOPS_COPY, CmCopy),
    EV_COMMAND(CM_SIGOPS_MODIFY, CmModify),
    EV_COMMAND(CM_SIGOPS_FILTER, CmFilter),
    EV_COMMAND(CM_SIGOPS_LAGCORR, CmLagCorr),
    EV_COMMAND(CM_SIGOPS_SPECTRUM, CmSpectrum),
    EV_COMMAND(CM_SIGOPS_ARITH, CmArith),
    EV_COMMAND(CM_SIGOPS_INSOUTS, CmInputOutput),
    EV_COMMAND(CM_SIGOPS_CUMULATE, CmCumulate),
    EV_COMMAND(CM_SIGOPS_CONFCOMP, CmConfcomp),
    EV_COMMAND(CM_ORTHOG_DEFINE, CmDefineOrthog),
    EV_COMMAND(CM_ORTHOG_APPLY, CmApplyOrthog),
    EV_COMMAND(CM_NETWORK_CLASS, CmClass),
    EV_COMMAND(CM_NETWORK_PRIOR, CmPrior),
    EV_COMMAND(CM_NETWORK_MODEL, CmNetworkModel),
    EV_COMMAND(CM_NETWORK_TRAIN_NETWORK, CmNetworkTrain),
    EV_COMMAND(CM_NETWORK_TEST_NETWORK, CmNetworkTest),
    EV_COMMAND(CM_NETWORK_PREDICT, CmNetworkPredict),
    EV_COMMAND(CM_ARMA_TRAIN, CmARMAtrain),
    EV_COMMAND(CM_ARMA_SHOCK, CmARMAshock),
    EV_COMMAND(CM_ARMA_PREDICT, CmARMApredict),
    EV_COMMAND(CM_HELPABOUT, CmHelpAbout),
    EV_COMMAND(CM_HELPSIZE, CmHelpSize),
END_RESPONSE_TABLE;


/*
   These are defined in MEM.CPP
*/

extern int mem_keep_log ;       // Keep a log file?
extern char mem_file_name[] ;   // Log file name
extern int mem_max_used ;       // Maximum memory ever in use

/*
   These are made global.  The are declared external in EXTERN.H.
*/

TDecoratedMDIFrame *frame ;
npredictMDIClient  *mdiClient ;

/*
--------------------------------------------------------------------------------

   Constructor

--------------------------------------------------------------------------------
*/

npredictApp::npredictApp () : TApplication("npredict")
{
}


/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

npredictApp::~npredictApp ()
{
}


/*
--------------------------------------------------------------------------------

	InitMainWindow

--------------------------------------------------------------------------------
*/


void npredictApp::InitMainWindow ()
{
	MEMTEXT ( "---> Initializing main window" ) ;

	mdiClient = new npredictMDIClient ;
	frame = new TDecoratedMDIFrame ( Name , MDI_MENU , *mdiClient , TRUE ) ;

	init_progress ( frame ) ; // Ditto for PROG_WIN.CPP

	nCmdShow = (nCmdShow != SW_SHOWMINNOACTIVE) ? SW_SHOWNORMAL : nCmdShow;

	frame->SetIcon(this, IDI_MDIAPPLICATION);

	frame->AssignMenu(MDI_MENU);
	 
	TStatusBar *sb = new TStatusBar(frame, TGadget::Recessed,
											  TStatusBar::CapsLock        |
											  TStatusBar::NumLock         |
											  TStatusBar::ScrollLock      |
											  TStatusBar::Overtype);

	frame->Insert(*sb, TDecoratedFrame::Bottom);
	SetMainWindow(frame);
}

/*
--------------------------------------------------------------------------------

	OwlMain

--------------------------------------------------------------------------------
*/

int OwlMain (int , char* [])
{
	npredictApp App ;
	int result ;
	char results[256] ;
	FILE *fp ;
/*
	Initialize memory allocation debugging
*/

#if 1
	mem_keep_log = 1 ;
	_fullpath ( mem_file_name , "mem.log" , 256 ) ;
	fp = fopen ( mem_file_name , "wt" ) ;
	if (fp == NULL) {
		mem_file_name[0] = 0 ;
		mem_keep_log = 0 ;
		}
	else 
		fclose ( fp ) ;
#else
	mem_file_name[0] = 0 ;
	mem_keep_log = 0 ;
#endif

	mem_max_used = 0 ;

	process ( ID_PRED_INITIALIZE , "" , (ControlData *) NULL , "" , results ) ;

	result = App.Run();

	process ( ID_PRED_BYE , "" , (ControlData *) NULL , "" , results ) ;

	MEMCLOSE () ;


   if (heapcheck () == _HEAPCORRUPT)
		frame->MessageBox ( "Heap is corrupted!" , "ERROR" ) ;
   else
      _heapmin () ;
	return result;
}

void npredictApp::CmReadSignal ()
{
   read_signal ( frame ) ;
}

void npredictApp::CmSaveSignal ()
{
   save_signal ( frame ) ;
}

void npredictApp::CmReadNetwork ()
{
   read_network ( frame ) ;
}

void npredictApp::CmSaveNetwork ()
{
   save_network ( frame ) ;
}

void npredictApp::CmReadARMA ()
{
   read_arma ( frame ) ;
}

void npredictApp::CmSaveARMA ()
{
   save_arma ( frame ) ;
}

void npredictApp::CmReadOrthog ()
{
   read_orthog ( frame ) ;
}

void npredictApp::CmSaveOrthog ()
{
   save_orthog ( frame ) ;
}

void npredictApp::CmCommandFile ()
{
   read_command_file ( frame ) ;
}

void npredictApp::CmClear ()
{
   do_clear ( frame ) ;
}

void npredictApp::CmAuditLog ()
{
	int id ;
	char error[256], results[256] ;
	TOpenSaveDialog::TData FileData ;

	FileData.Flags = OFN_PATHMUSTEXIST | OFN_HIDEREADONLY ;
	FileData.SetFilter("Audit log files (*.LOG)|*.LOG|All Files (*.*)|*.*|") ;
	strcpy ( FileData.FileName , "AUDIT.LOG" ) ;

	id = TFileSaveDialog ( frame , FileData ).Execute() ;
	if (id != IDOK)
		return ;

	if (process ( ID_PRED_AUDIT_LOG , FileData.FileName ,
	              NULL , error , results ) < 0)
		frame->MessageBox ( error , "ERROR" ) ;
}

void npredictApp::CmDisplaySignal ()
{
	do_display ( frame ) ;
}

void npredictApp::CmGenerate ()
{
	generate ( frame ) ;
}

void npredictApp::CmCopy ()
{
	do_copy ( frame ) ;
}

void npredictApp::CmModify ()
{
	do_modify ( frame ) ;
}

void npredictApp::CmFilter ()
{
	do_filter ( frame ) ;
}

void npredictApp::CmLagCorr ()
{
	lag_corr ( frame ) ;
}

void npredictApp::CmSpectrum ()
{
	do_spectrum ( frame ) ;
}

void npredictApp::CmArith ()
{
	arith ( frame ) ;
}

void npredictApp::CmInputOutput ()
{
   ins_outs ( frame ) ;
}

void npredictApp::CmCumulate ()
{
   cumulate ( frame ) ;
}

void npredictApp::CmConfcomp ()
{
   set_conf_comp ( frame ) ;
}

void npredictApp::CmDefineOrthog ()
{
	define_orthog ( frame ) ;
}

void npredictApp::CmApplyOrthog ()
{
	apply_orthog ( frame ) ;
}

void npredictApp::CmClass ()
{
	int id ;
	char error[256], name[256], results[256] ;

	name[0] = 0 ;
	id = TInputDialog ( frame , "Class name" ,
		  "Enter class name for subsequent training" , name , 255 ).Execute() ;
	if (id != IDOK)
		return ;

   strupr ( name ) ;
	if (process ( ID_PRED_CLASS , name , NULL , error , results ) < 0)
		frame->MessageBox ( error , "ERROR" ) ;
}

void npredictApp::CmPrior ()
{
	int id ;
	char error[256], text[256], results[256] ;

	text[0] = 0 ;
	id = TInputDialog ( frame , "Prior probability" ,
		  "Enter class prior for subsequent training" , text , 255 ).Execute() ;
	if (id != IDOK)
		return ;

	if (process ( ID_PRED_PRIOR , text , NULL , error , results ) < 0)
		frame->MessageBox ( error , "ERROR" ) ;
}

void npredictApp::CmNetworkModel ()
{
	get_net_model ( frame ) ;
}

void npredictApp::CmNetworkTrain ()
{
   train_net ( frame ) ;
}

void npredictApp::CmNetworkTest ()
{
   test_net ( frame ) ;
}

void npredictApp::CmNetworkPredict ()
{
   predict_net ( frame ) ;
}

void npredictApp::CmARMAtrain ()
{
   train_arma ( frame ) ;
}

void npredictApp::CmARMAshock ()
{
   shock_arma ( frame ) ;
}

void npredictApp::CmARMApredict ()
{
   predict_arma ( frame ) ;
}

/*
--------------------------------------------------------------------------------

	CmHelpAbout

--------------------------------------------------------------------------------
*/

void npredictApp::CmHelpAbout ()
{
   frame->MessageBox ( "NPREDICT ver. 1.0  Copyright (c) 1995 Timothy Masters" ,
                       "About NPREDICT" ) ;
}

/*
--------------------------------------------------------------------------------

	CmHelpSize

--------------------------------------------------------------------------------
*/

void npredictApp::CmHelpSize ()
{
   int nr, nc ;
   char msg[84] ;
	TRect rect ;
   TMDIChild *child ;
	// Show the size of the active window (to help screen grabbing)
   child = mdiClient->GetActiveMDIChild () ;
   if (child == NULL)
      return ;
	child->GetClientRect ( rect ) ;
   nr = rect.bottom ;
   nc = rect.right ;
   sprintf ( msg , "%d rows by %d cols (ratio=%.3lf)",
             nr, nc, (double) nr / (double) nc ) ;
   frame->MessageBox ( msg , "Window size" ) ;
}

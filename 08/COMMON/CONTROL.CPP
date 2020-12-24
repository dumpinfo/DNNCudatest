/******************************************************************************/
/*                                                                            */
/*  CONTROL - Routines for reading control file and interpreting commands     */
/*                                                                            */
/*    get_control_line                                                        */
/*    interpret_control_line (returns -1 if unknown command)                  */
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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"    // System, limitation constants, typedefs, structs
#include "classes.h"  // Includes all class headers
#include "funcdefs.h" // Function prototypes

static void parse_control ( char *control , char **command ,
                     int *command_length , char **param , int *param_length ) ;
static void chop_blanks ( char *str ) ;
void str_to_upr ( char *str ) ;


/*
--------------------------------------------------------------------------------

   get_control_line

--------------------------------------------------------------------------------
*/

void get_control_line ( ControlData *cbuf , char *line )
{
   int n ;

   strcpy ( line , cbuf->buf ) ;
   n = strlen ( line ) + 1 ;   // Include 0 at end of string
   cbuf->n -= n ;
   if (cbuf->n)
      memcpy ( cbuf->buf , cbuf->buf+n , cbuf->n ) ;
   else {
      MEMTEXT ( "CONTROL: control buffer" ) ;
		FREE ( cbuf->buf ) ;
		cbuf->buf = NULL ;   // Insurance
      }
}


/*
--------------------------------------------------------------------------------

   interpret_control_line

--------------------------------------------------------------------------------
*/

int interpret_control_line ( char *line , char **rest )
{
   char *command ;               // Pointer to its command part
   int command_length, n_rest ;  // Lengths of command part and rest

   parse_control ( line , &command , &command_length , rest , &n_rest ) ;

   if (! command_length)
      return 0 ;

/*
   Interpret it and return the command id code.
*/

   if (! strcmp ( command , "AUDIT LOG" ))
      return ID_PRED_AUDIT_LOG ;

   if (! strcmp ( command , "BYE" ))
      return ID_PRED_BYE ;

   if (! strcmp ( command , "CLEAR ALL" ))
      return ID_PRED_CLEAR_ALL ;

   if (! strcmp ( command , "PROGRESS ON" ))
      return ID_PRED_PROGRESS_ON ;

   if (! strcmp ( command , "PROGRESS OFF" ))
      return ID_PRED_PROGRESS_OFF ;

   if (! strcmp ( command , "COMMAND FILE" ))
      return ID_PRED_COMMANDFILE ;

   if (! strcmp ( command , "NAME" ))
      return ID_PRED_NAME ;

   if (! strcmp ( command , "READ SIGNAL FILE" ))
      return ID_PRED_READSIG ;

   if (! strcmp ( command , "SAVE SIGNAL" ))
      return ID_PRED_SAVESIG ;

   if (! strcmp ( command , "DISPLAY" ))
      return ID_PRED_DISPLAY ;

   if (! strcmp ( command , "DISPLAY RANGE" ))
      return ID_PRED_DISPLAY_RANGE ;

   if (! strcmp ( command , "DISPLAY DOMAIN" ))
      return ID_PRED_DISPLAY_DOMAIN ;

   if (! strcmp ( command , "DISPLAY RATE" ))
      return ID_PRED_DISPLAY_RATE ;

   if (! strcmp ( command , "DISPLAY ORIGIN" ))
      return ID_PRED_DISPLAY_ORIGIN ;

   if (! strcmp ( command , "DISPLAY CONFIDENCE" ))
      return ID_PRED_DISPLAY_CONFIDENCE ;

   if (! strcmp ( command , "CLEAR INPUT LIST" ))
      return ID_PRED_CLEAR_INPUT ;

   if (! strcmp ( command , "CLEAR OUTPUT LIST" ))
      return ID_PRED_CLEAR_OUTPUT ;

   if (! strcmp ( command , "INPUT" ))
      return ID_PRED_INPUT ;

   if (! strcmp ( command , "OUTPUT" ))
      return ID_PRED_OUTPUT ;

   if (! strcmp ( command , "CLEAR CLASSES" ))
      return ID_PRED_CLEAR_CLASSES ;

   if (! strcmp ( command , "CLASS" ))
      return ID_PRED_CLASS ;

   if (! strcmp ( command , "PRIOR" ))
      return ID_PRED_PRIOR ;

   if (! strcmp ( command , "CLEAR TRAINING SET" ))
      return ID_PRED_CLEAR_TRAINING ;

   if (! strcmp ( command , "CUMULATE TRAINING SET" ))
      return ID_PRED_CUMULATE_TRAINING ;

   if (! strcmp ( command , "CLEAR TEST SET" ))
      return ID_PRED_CLEAR_TEST ;

   if (! strcmp ( command , "CUMULATE TEST SET" ))
      return ID_PRED_CUMULATE_TEST ;

   if (! strcmp ( command , "CUMULATE INCLUDE" ))
      return ID_PRED_CUMULATE_INCLUDE ;

   if (! strcmp ( command , "CUMULATE EXCLUDE" ))
      return ID_PRED_CUMULATE_EXCLUDE ;

   if (! strcmp ( command , "NETWORK MODEL" ))
      return ID_PRED_NETMOD ;

   if (! strcmp ( command , "KERNEL" ))
      return ID_PRED_KERNEL ;

   if (! strcmp ( command , "MLFN HID 1" ))
      return ID_PRED_MLFN_HID1 ;

   if (! strcmp ( command , "MLFN HID 2" ))
      return ID_PRED_MLFN_HID2 ;

   if (! strcmp ( command , "MLFN OUTPUT ACTIVATION" ))
      return ID_PRED_MLFN_OUTPUT_ACT ;

   if (! strcmp ( command , "MLFN DOMAIN" ))
      return ID_PRED_MLFN_DOMAIN ;

   if (! strcmp ( command , "ALLOWABLE ERROR" ))
      return ID_PRED_QUITERR ;

   if (! strcmp ( command , "SIGMA LOW" ))
      return ID_PRED_SIGLO ;

   if (! strcmp ( command , "SIGMA HIGH" ))
      return ID_PRED_SIGHI ;

   if (! strcmp ( command , "SIGMA TRIES" ))
      return ID_PRED_SIGTRIES ;

   if (! strcmp ( command , "MLFN ERROR TYPE" ))
      return ID_PRED_MLFN_ERROR_TYPE ;

   if (! strcmp ( command , "MLFN LEARNING METHOD" ))
      return ID_PRED_MLFN_LEARNING ;

   if (! strcmp ( command , "MLFN RESTARTS" ))
      return ID_PRED_MLFN_RESTARTS ;

   if (! strcmp ( command , "MLFN PRETRIES" ))
      return ID_PRED_MLFN_PRETRIES ;

   if (! strcmp ( command , "ACCURACY" ))
      return ID_PRED_ACCURACY ;

   if (! strcmp ( command , "REFINE" ))
      return ID_PRED_REFINE ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION TEMPERATURES" ))
      return ID_PRED_ANNEAL_INIT_TEMPS ;

   if (! strcmp ( command , "ANNEALING ESCAPE TEMPERATURES" ))
      return ID_PRED_ANNEAL_ESCAPE_TEMPS ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION ITERATIONS" ))
      return ID_PRED_ANNEAL_INIT_ITERS ;

   if (! strcmp ( command , "ANNEALING ESCAPE ITERATIONS" ))
      return ID_PRED_ANNEAL_ESCAPE_ITERS ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION SETBACK" ))
      return ID_PRED_ANNEAL_INIT_SETBACK ;

   if (! strcmp ( command , "ANNEALING ESCAPE SETBACK" ))
      return ID_PRED_ANNEAL_ESCAPE_SETBACK ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION START" ))
      return ID_PRED_ANNEAL_INIT_START ;

   if (! strcmp ( command , "ANNEALING ESCAPE START" ))
      return ID_PRED_ANNEAL_ESCAPE_START ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION STOP" ))
      return ID_PRED_ANNEAL_INIT_STOP ;

   if (! strcmp ( command , "ANNEALING ESCAPE STOP" ))
      return ID_PRED_ANNEAL_ESCAPE_STOP ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION RATIO" ))
      return ID_PRED_ANNEAL_INIT_RATIO ;

   if (! strcmp ( command , "ANNEALING ESCAPE RATIO" ))
      return ID_PRED_ANNEAL_ESCAPE_RATIO ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION REDUCTION" ))
      return ID_PRED_ANNEAL_INIT_REDUCTION ;

   if (! strcmp ( command , "ANNEALING ESCAPE REDUCTION" ))
      return ID_PRED_ANNEAL_ESCAPE_REDUCTION ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION RANDOM" ))
      return ID_PRED_ANNEAL_INIT_RANDOM ;

   if (! strcmp ( command , "ANNEALING ESCAPE RANDOM" ))
      return ID_PRED_ANNEAL_ESCAPE_RANDOM ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION ACCEPT" ))
      return ID_PRED_ANNEAL_INIT_ACCEPT ;

   if (! strcmp ( command , "ANNEALING ESCAPE ACCEPT" ))
      return ID_PRED_ANNEAL_ESCAPE_ACCEPT ;

   if (! strcmp ( command , "CLEAR NETWORKS" ))
      return ID_PRED_CLEAR_NETWORKS ;

   if (! strcmp ( command , "CLEAR NETWORK" ))
      return ID_PRED_CLEAR_NETWORK ;

   if (! strcmp ( command , "TRAIN NETWORK" ))
      return ID_PRED_TRAIN_NETWORK ;

   if (! strcmp ( command , "CV TRAIN NETWORK" ))
      return ID_PRED_CVTRAIN_NETWORK ;

   if (! strcmp ( command , "SAVE NETWORK" ))
      return ID_PRED_SAVE_NETWORK ;

   if (! strcmp ( command , "RESTORE NETWORK" ))
      return ID_PRED_RESTORE_NETWORK ;

   if (! strcmp ( command , "PRINT NETWORK" ))
      return ID_PRED_PRINT_NETWORK ;

   if (! strcmp ( command , "THRESHOLD" ))
      return ID_PRED_THRESHOLD ;

   if (! strcmp ( command , "TEST NETWORK" ))
      return ID_PRED_TEST_NETWORK ;

   if (! strcmp ( command , "EXTENDED TEST NETWORK" ))
      return ID_PRED_EXTENDED_TEST_NETWORK ;

   if (! strcmp ( command , "GENERATE" ))
      return ID_PRED_GENERATE ;

   if (! strcmp ( command , "NETWORK PREDICT" ))
      return ID_PRED_NETWORK_PREDICT ;

   if (! strcmp ( command , "NETWORK CONFIDENCE" ))
      return ID_PRED_NETWORK_CONFIDENCE ;

   if (! strcmp ( command , "CLEAR ARMAS" ))
      return ID_PRED_CLEAR_ARMAS ;

   if (! strcmp ( command , "CLEAR ARMA" ))
      return ID_PRED_CLEAR_ARMA ;

   if (! strcmp ( command , "TRAIN ARMA" ))
      return ID_PRED_TRAIN_ARMA ;

   if (! strcmp ( command , "ARMA PREDICT" ))
      return ID_PRED_ARMA_PREDICT ;

   if (! strcmp ( command , "ARMA CONFIDENCE" ))
      return ID_PRED_ARMA_CONFIDENCE ;

   if (! strcmp ( command , "ARMA SHOCK" ))
      return ID_PRED_ARMA_SHOCK ;

   if (! strcmp ( command , "ARMA FIXED" ))
      return ID_PRED_ARMA_FIXED ;

   if (! strcmp ( command , "SAVE ARMA" ))
      return ID_PRED_SAVE_ARMA ;

   if (! strcmp ( command , "RESTORE ARMA" ))
      return ID_PRED_RESTORE_ARMA ;

   if (! strcmp ( command , "CENTER" ))
      return ID_PRED_CENTER ;

   if (! strcmp ( command , "MEDIAN CENTER" ))
      return ID_PRED_MEDIAN_CENTER ;

   if (! strcmp ( command , "UNDO CENTER" ))
      return ID_PRED_UNDO_CENTER ;

   if (! strcmp ( command , "CONFIDENCE CENTER" ))
      return ID_PRED_CONFIDENCE_CENTER ;

   if (! strcmp ( command , "DETREND" ))
      return ID_PRED_DETREND ;

   if (! strcmp ( command , "UNDO DETREND" ))
      return ID_PRED_UNDO_DETREND ;

   if (! strcmp ( command , "CONFIDENCE DETREND" ))
      return ID_PRED_CONFIDENCE_DETREND ;

   if (! strcmp ( command , "OFFSET" ))
      return ID_PRED_OFFSET ;

   if (! strcmp ( command , "UNDO OFFSET" ))
      return ID_PRED_UNDO_OFFSET ;

   if (! strcmp ( command , "CONFIDENCE OFFSET" ))
      return ID_PRED_CONFIDENCE_OFFSET ;

   if (! strcmp ( command , "SCALE" ))
      return ID_PRED_SCALE ;

   if (! strcmp ( command , "UNDO SCALE" ))
      return ID_PRED_UNDO_SCALE ;

   if (! strcmp ( command , "CONFIDENCE SCALE" ))
      return ID_PRED_CONFIDENCE_SCALE ;

   if (! strcmp ( command , "STANDARDIZE" ))
      return ID_PRED_STANDARDIZE ;

   if (! strcmp ( command , "UNDO STANDARDIZE" ))
      return ID_PRED_UNDO_STANDARDIZE ;

   if (! strcmp ( command , "CONFIDENCE STANDARDIZE" ))
      return ID_PRED_CONFIDENCE_STANDARDIZE ;

   if (! strcmp ( command , "DIFFERENCE" ))
      return ID_PRED_DIFFERENCE ;

   if (! strcmp ( command , "UNDO DIFFERENCE" ))
      return ID_PRED_UNDO_DIFFERENCE ;

   if (! strcmp ( command , "CONFIDENCE DIFFERENCE" ))
      return ID_PRED_CONFIDENCE_DIFFERENCE ;

   if (! strcmp ( command , "SEASONAL DIFFERENCE" ))
      return ID_PRED_SEASONAL_DIFFERENCE ;

   if (! strcmp ( command , "UNDO SEASONAL DIFFERENCE" ))
      return ID_PRED_UNDO_SEASONAL_DIFFERENCE ;

   if (! strcmp ( command , "CONFIDENCE SEASONAL DIFFERENCE" ))
      return ID_PRED_CONFIDENCE_SEASONAL ;

   if (! strcmp ( command , "LOG" ))
      return ID_PRED_LOG ;

   if (! strcmp ( command , "EXP" ))
      return ID_PRED_EXP ;

   if (! strcmp ( command , "CONFIDENCE LOG" ))
      return ID_PRED_CONFIDENCE_LOG ;

   if (! strcmp ( command , "CLEAR CONFIDENCE COMPENSATION" ))
      return ID_PRED_CLEAR_CONFIDENCE ;

   if (! strcmp ( command , "CONFIDENCE PROBABILITY" ))
      return ID_PRED_CONFIDENCE_PROB ;

   if (! strcmp ( command , "ADD" ))
      return ID_PRED_ADD ;

   if (! strcmp ( command , "SUBTRACT" ))
      return ID_PRED_SUBTRACT ;

   if (! strcmp ( command , "MULTIPLY" ))
      return ID_PRED_MULTIPLY ;

   if (! strcmp ( command , "DIVIDE" ))
      return ID_PRED_DIVIDE ;

   if (! strcmp ( command , "ORTHOGONALIZATION TYPE" ))
      return ID_PRED_ORTHOG_TYPE ;

   if (! strcmp ( command , "ORTHOGONALIZATION FACTORS" ))
      return ID_PRED_ORTHOG_FACTORS ;

   if (! strcmp ( command , "ORTHOGONALIZATION LIMIT" ))
      return ID_PRED_ORTHOG_LIMIT ;

   if (! strcmp ( command , "ORTHOGONALIZATION STANDARDIZE" ))
      return ID_PRED_ORTHOG_STD ;

   if (! strcmp ( command , "DEFINE ORTHOGONALIZATION" ))
      return ID_PRED_DEFINE_ORTHOG ;

   if (! strcmp ( command , "APPLY ORTHOGONALIZATION" ))
      return ID_PRED_APPLY_ORTHOG ;

   if (! strcmp ( command , "SAVE ORTHOGONALIZATION" ))
      return ID_PRED_SAVE_ORTHOG ;

   if (! strcmp ( command , "RESTORE ORTHOGONALIZATION" ))
      return ID_PRED_RESTORE_ORTHOG ;

   if (! strcmp ( command , "CLEAR ORTHOGONALIZATION" ))
      return ID_PRED_CLEAR_ORTHOG ;

   if (! strcmp ( command , "CLEAR ORTHOGONALIZATIONS" ))
      return ID_PRED_CLEAR_ORTHOGS ;

   if (! strcmp ( command , "AUTOCORRELATION" ))
      return ID_PRED_AUTOCORR ;

   if (! strcmp ( command , "CROSSCORRELATION" ))
      return ID_PRED_CROSSCORR ;

   if (! strcmp ( command , "PARTIAL AUTOCORRELATION" ))
      return ID_PRED_P_AUTOCORR ;

   if (! strcmp ( command , "PARTIAL CROSSCORRELATION" ))
      return ID_PRED_P_CROSSCORR ;

   if (! strcmp ( command , "COPY" ))
      return ID_PRED_COPY ;

   if (! strcmp ( command , "INTEGRATE" ))
      return ID_PRED_INTEGRATE ;

   if (! strcmp ( command , "SPECTRUM" ))
      return ID_PRED_SPECTRUM ;

   if (! strcmp ( command , "SPECTRUM WINDOW" ))
      return ID_PRED_SPECTRUM_WINDOW ;

   if (! strcmp ( command , "SAVGOL" ))
      return ID_PRED_SAVGOL ;

   if (! strcmp ( command , "MAXENT" ))
      return ID_PRED_MAXENT ;

   if (! strcmp ( command , "LOWPASS" ))
      return ID_PRED_LOWPASS ;

   if (! strcmp ( command , "HIGHPASS" ))
      return ID_PRED_HIGHPASS ;

   if (! strcmp ( command , "BANDPASS" ))
      return ID_PRED_BANDPASS ;

   if (! strcmp ( command , "QMF" ))
      return ID_PRED_QMF ;

   if (! strcmp ( command , "MORLET" ))
      return ID_PRED_MORLET ;

   if (! strcmp ( command , "PADDING" ))
      return ID_PRED_PADDING ;

   if (! strcmp ( command , "MOVING AVERAGE" ))
      return ID_PRED_MOV_AVG ;

   return -1 ;

}

/*
--------------------------------------------------------------------------------

   parse_control - The command line consists of two parts: the command
      and optional parameters.  This separates these parts and returns
      pointers to each.

--------------------------------------------------------------------------------
*/

void parse_control (
   char *control ,        // Control line input here
   char **command ,       // Returned pointer to command part
   int *command_length ,  // Length of command part
   char **param ,         // Returned pointer to parameter part
   int *param_length      // Length of parameter part
   )
{
   int i, n ;

/*
   Remove any comments
*/

   n = strlen ( control ) ;
   for (i=0 ; i<n ; i++) {
      if (control[i] == ';') {
         control[i] = 0 ;
         break ;
         }
      }

/*
   Remove leading and trailing blanks, convert to upper case
*/

   while (*control == ' ')
      ++control ;
   chop_blanks ( control ) ;
   str_to_upr ( control ) ;                // Return everything in upper case

/*
   Parse the line
*/

   n = strlen ( control ) ;                // This is after cleaning
   for (i=0 ; i<n ; i++) {                 // Span the entire line in this loop
      if (control[i] == '=') {             // End of command when we get '='
         control[i] = 0 ;                  // So end it
         chop_blanks ( control ) ;         // Must return just command
         *command = control ;              // Pointer to command
         *command_length = strlen ( control ) ; // and its length
         *param = control + i + 1 ;        // This is the parameter part
         while (**param == ' ')            // There may be blanks after = sign
            ++*param ;                     // So remove them
         *param_length = strlen ( *param ) ; // User also wants length
         return ;                          // Return here if =parameter version
         }
      }

/*
   It is also legal to have no = sign or parameter.  Handle that here.
*/

   *command = control ;
   *command_length = strlen ( control ) ;
   *param = 0 ;
   *param_length = 0 ;
}

static void chop_blanks ( char *string )
{
   int n = strlen ( string ) ;
   while (n--) {
      if (string[n] == ' ')
         string[n] = 0 ;
      else
         break ;
      }
}

void str_to_upr ( char *string )
{
   char *str = string-1;
   while (*++str)
      *str = (char) toupper ( *str ) ;
}



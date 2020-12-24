// Copyright (c) 1994 John Wiley & Sons, Inc.  All rights reserved.
// Reproduction or translation of this work beyond that permitted in
// section 117 of the 1976 United States Copyright Act without the
// express written permission of the copyright owner is unlawful.
// Requests for further information should be addressed to the
// Permission Department, John Wiley & Sons, Inc.  The purchaser may
// make backup copies for his/her own use only and not for distribution
// or resale.  The publisher assumes no responsibility for errors,
// omissions, or damages, caused by the use of these programs or from
// the use of the information contained herein.

/******************************************************************************/
/*                                                                            */
/*  MLFN - Main program for implementing all multiple-layer feedforward nets  */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

/*
   These are used in MEM.CPP for diagnostic memory allocation
*/

extern int mem_log ;       // Keep a log on disk?
extern long mem_used ;     // Keeps track of memory usage
extern char mem_name[] ;   // Full path name of log file

/*
   Forward declarations for static routines
*/

static int check_anneal ( char *command , char *rest ,
                   struct AnnealParams *anneal_params ) ;
static int check_learn_params ( char *command , char *rest ,
                         struct LearnParams *learn_params ) ;
static int get_yn ( char *msg ) ;
static int ok_to_clear_tset ( TrainingSet **tset ) ;
static int ok_to_clear_weights ( LayerNet **network ) ;

/*
--------------------------------------------------------------------------------

   Main entry point

--------------------------------------------------------------------------------
*/

int main (
   int argc ,    // Number of command line arguments (includes prog name)
   char *argv[]  // Arguments (prog name is argv[0])
   )

{

/*
   Declarations of local variables
*/

/*
   User's command control line related variables are here.
   Control_file_number and control_files permit nesting of 'CONTROL' commands.
   If control_file_number equals -1, control commands are read from stdin.
   Otherwise they are read from that file in FILE *control_files.
   Up to MAX_CONTROL_FILES can be stacked.
*/

   int control_file_number = -1 ;           // Stack pointer for control files
   FILE *control_files[MAX_CONTROL_FILES] ; // This is the stack

   char *control_line ;    // User's commands here
   char *command, *rest ;  // Pointers to its command and parameter parts
   int n_command, n_rest ; // Lengths of those parts

/*
   These are network parameters which may be set by the user via commands.
   Most are initialized to defaults which indicate that the user has not
   yet set them.  As they are set, their current values are placed here.
   When learning is done for a network, their values are copied from here
   into the network object.  When a network is read, the object's values
   are copied from it to here.  Otherwise, these variables are not used;
   the values in the network object itself are used.  The only purpose of
   these variables is to keep track of current values.
*/

   int net_model = -1 ;     // LayerNet model (see NETMOD_? in CONST.H)
   int out_model = -1 ;     // Output model (see OUTMOD_? in CONST.H)
   int out_linear = 0 ;     // Output activations linear?  Default is no.
   int n_inputs = -1 ;      // Number of input neurons
   int n_outputs = -1 ;     // Number of output neurons
   int n_hidden1 = -1 ;     // Number of hidden layer one neurons
   int n_hidden2 = -1 ;     // Ditto layer 2 (0 if just one hidden layer)


   TrainingSet *tset = NULL ;            // Training set here
   LayerNet *network = NULL ;            // LayerNet here
   struct LearnParams learn_params ;     // General learning parameters
   struct AnnealParams anneal_params ;   // Simulated annealing parameters

   int classif_output = -1 ;  // Current class (0=reject) for classif training
   char out_file[80] = "" ;   // File for EXECUTE output
   double threshold ;         // CLASSIFY confusion reject cutoff

/*
   Miscellaneous variables
*/

   int i, n, m ;
   double p ;
   char *msg ;
   FILE *fp ;

/*
--------------------------------------------------------------------------------

   Program starts here.

   Verify that a careless user didn't fail to set the integer size
   correctly when compiling.

--------------------------------------------------------------------------------
*/

#if VERSION_16_BIT
   if (sizeof(int) > 2) {
      printf ( "\nRecompile with VERSION_16_BIT set to 0 in CONST.H" ) ;
      exit ( 1 ) ;
      }
#else
   if (sizeof(int) < 4) {
      printf ( "\nRecompile with VERSION_16_BIT set to 1 in CONST.H" ) ;
      exit ( 1 ) ;
      }
#endif

   printf ( "\n\nMLFN - Train and test multiple-layer feedforward networks" ) ;
   printf("\n\nCopyright (c) 1994 John Wiley & Sons, Inc.  All rights reserved.");
   printf("\nReproduction or translation of this work beyond that permitted in");
   printf("\nsection 117 of the 1976 United States Copyright Act without the") ;
   printf("\nexpress written permission of the copyright owner is unlawful.") ;
   printf("\nRequests for further information should be addressed to the") ;
   printf("\nPermission Department, John Wiley & Sons, Inc.  The purchaser may") ;
   printf("\nmake backup copies for his/her own use only and not for distribution");
   printf("\nor resale.  The publisher assumes no responsibility for errors,") ;
   printf("\nomissions, or damages, caused by the use of these programs or from") ;
   printf("\nthe use of the information contained herein.\n\n") ;


/*
   Process command line parameters
*/

   mem_name[0] = 0 ;  // Default is no memory allocation file

   for (i=1 ; i<argc ; i++) {  // Process all command line args
      str_to_upr ( argv[i] ) ; // Easier if all upper case

      if (! strcmp ( argv[i] , "/DEBUG" )) {
         sscanf ( argv[++i] , "%s" , mem_name ) ;
         if ((strlen ( mem_name ) > 1)  ||  ! isalpha ( mem_name[0] )) {
            printf ( "\nIllegal DEBUG drive (%s); must be 1 letter." ) ;
            exit ( 1 ) ;
            }
         continue ;
         }

      printf ( "\nIllegal command line parameter (%s)", argv[i] ) ;
      exit ( 1 ) ;
      }

/*
   Initialize memory allocation debugging
*/

   if (strlen ( mem_name )) {
      strcat ( mem_name , ":mem.log" ) ;
      fp = fopen ( mem_name , "wt" ) ;
      if (fp == NULL) {
         printf ( "\nCannot open debugging file %s", mem_name ) ;
         exit ( 1 ) ;
         }
      fclose ( fp ) ;
      mem_log = 1 ;
      }
   else 
      mem_log = 0 ;

   mem_used = 0 ;

/*
   Initialize defaults
*/

   learn_params.method = METHOD_AN1_CJ ;
   learn_params.errtype = ERRTYPE_MSE ;
   learn_params.quit_err = 0.0 ;
   learn_params.retries = 32767 ;
   learn_params.cj_progress = 0 ;
   learn_params.cj_acc = 7 ;
   learn_params.cj_refine = 2 ;
   learn_params.cj_pretries = 5 ;

   anneal_params.temps0 = 3 ;
   anneal_params.temps = 4 ;
   anneal_params.iters0 = 50 ;
   anneal_params.iters = 20 ;
   anneal_params.setback0 = 50 ;
   anneal_params.setback = 20 ;
   anneal_params.start0 = 1.0 ;
   anneal_params.start = 4.0 ;
   anneal_params.stop0 = 0.2 ;
   anneal_params.stop = 0.02 ;

   learn_params.ap = &anneal_params ;

   threshold = 0.0 ;

   MEMTEXT ( "MLFN: control_line, msg" ) ;
   if (((control_line = (char *) MALLOC ( CONTROL_LINE_LENGTH+1 )) == NULL)
    || ((msg = (char *) MALLOC ( CONTROL_LINE_LENGTH+1 )) == NULL)) {
      printf ( "\nInsufficient memory" ) ;
      exit ( 1 ) ;
      }

/*
   Main loop processes all commands
*/

   for (;;) {

      get_control_line ( control_line , &control_file_number, control_files ) ;

      split_control_line ( control_line , &command , &n_command ,
                           &rest , &n_rest ) ;

      if (! n_command)
         continue ;

      if (n_rest)
         sprintf ( msg , "%s = %s", command, rest ) ;
      else 
         strcpy ( msg , command ) ;
      normal_message ( msg ) ;

/*
   Act on the command
*/

      if (! strcmp ( command , "BYE" ))
         break ;

      if (! strcmp ( command , "COMMAND FILE" )) {
         stack_control_file ( rest , &control_file_number , control_files ) ;
         continue ;
         }

      if (! strcmp ( command , "DOMAIN" )) {
         if (! strcmp ( rest , "REAL" ))
            n = NETMOD_REAL ;
         else if (! strcmp ( rest , "COMPLEX" ))
            n = NETMOD_COMPLEX ;
         else if (! strcmp ( rest , "COMPLEX-INPUT" ))
            n = NETMOD_COMPLEX_INPUT ;
         else if (! strcmp ( rest , "COMPLEX-HIDDEN" ))
            n = NETMOD_COMPLEX_HIDDEN ;
         else {
            sprintf ( msg , "Illegal DOMAIN = %s", rest ) ;
            error_message ( msg ) ;
            continue ;
            }
         if (net_model == n)
            continue ;
         if (ok_to_clear_weights( &network ))
            net_model = n ;
         else
            warning_message ( "Command aborted" ) ;
         continue ;
         }

      if (! strcmp ( command , "MODE" )) {
         if (! strcmp ( rest , "CLASSIFICATION" ))
            n = OUTMOD_CLASSIFY ;
         else if (! strcmp ( rest , "AUTOASSOCIATION" ))
            n = OUTMOD_AUTO ;
         else if (! strcmp ( rest , "MAPPING" ))
            n = OUTMOD_GENERAL ;
         else {
            sprintf ( msg , "Illegal MODE = %s", rest ) ;
            error_message ( msg ) ;
            continue ;
            }
         if (out_model == n)
            continue ;
         if ((ok_to_clear_tset( &tset )) && (ok_to_clear_weights( &network)))
            out_model = n ;
         else
            warning_message ( "Command aborted" ) ;
         continue ;
         }

      if (! strcmp ( command , "OUTPUT ACTIVATION" )) {
         if (! strcmp ( rest , "LINEAR" ))
            n = 1 ;
         else if (! strcmp ( rest , "NONLINEAR" ))
            n = 0 ;
         else {
            sprintf ( msg , "Illegal OUTPUT ACTIVATION: %s", rest ) ;
            error_message ( msg ) ;
            continue ;
            }
         out_linear = n ;
         continue ;
         }

      if (! strcmp ( command , "INPUTS" )) {
         m = sscanf ( rest , "%d" , &n ) ;
         if ((m <= 0)  ||  (n <= 0)) {
            sprintf ( msg , "Illegal INPUTS = %s", rest ) ;
            error_message ( msg ) ;
            continue ;
            }
         if (n_inputs == n)
            continue ;
         if ((ok_to_clear_tset( &tset)) && (ok_to_clear_weights(&network)))
            n_inputs = n ;
         else
            warning_message ( "Command aborted" ) ;
         continue ;
         }

      if (! strcmp ( command , "OUTPUTS" )) {
         m = sscanf ( rest , "%d" , &n ) ;
         if ((m <= 0)  ||  (n <= 0)) {
            sprintf ( msg , "Illegal OUTPUTS = %s", rest ) ;
            error_message ( msg ) ;
            continue ;
            }
         if (n_outputs == n)
            continue ;
         if ((ok_to_clear_tset( &tset)) && (ok_to_clear_weights(&network)))
            n_outputs = n ;
         else
            warning_message ( "Command aborted" ) ;
         continue ;
         }

      if (! strcmp ( command , "FIRST HIDDEN" )) {
         m = sscanf ( rest , "%d" , &n ) ;
         if ((m <= 0)  ||  (n < 0)) {
            sprintf ( msg , "Illegal FIRST HIDDEN = %s", rest ) ;
            error_message ( msg ) ;
            continue ;
            }
         if (n_hidden1 == n)
            continue ;
         if (ok_to_clear_weights( &network ))
            n_hidden1 = n ;
         else
            warning_message ( "Command aborted" ) ;
         continue ;
         }

      if (! strcmp ( command , "SECOND HIDDEN" )) {
         m = sscanf ( rest , "%d" , &n ) ;
         if ((m <= 0)  ||  (n < 0)) {
            sprintf ( msg , "Illegal SECOND HIDDEN = %s", rest ) ;
            error_message ( msg ) ;
            continue ;
            }
         if (n  &&  ! n_hidden1) {
            error_message ( "SECOND HIDDEN must be 0 if FIRST HIDDEN IS 0." ) ;
            continue ;
            }
         if (n_hidden2 == n)
            continue ;
         if (ok_to_clear_weights( &network ))
            n_hidden2 = n ;
         else
            warning_message ( "Command aborted" ) ;
         continue ;
         }

      if (! strcmp ( command , "CUMULATE TRAINING SET" )) {
         if ((out_model == OUTMOD_AUTO)  &&  (n_outputs != n_inputs)) {
            error_message (
              "AUTOASSOCIATIVE output mode but OUTPUTS not equal to INPUTS" ) ;
            continue ;
            }
         if (net_model <= 0)
            error_message ( "CUMULATE TRAINING SET used before DOMAIN set." ) ;
         else if (out_model <= 0)
            error_message ( "CUMULATE TRAINING SET used before MODE set." ) ;
         else if ((out_model == OUTMOD_AUTO)  &&
                  (net_model == NETMOD_COMPLEX_INPUT))
           error_message("AUTOASSOCIATIVE valid only for REAL or COMPLEX net.");
         else if ((out_model == OUTMOD_AUTO)  &&
                  (net_model == NETMOD_COMPLEX_HIDDEN))
           error_message("AUTOASSOCIATIVE valid only for REAL or COMPLEX net.");
         else if ((out_model == OUTMOD_CLASSIFY)  &&
                  (net_model == NETMOD_COMPLEX))
            error_message ( "CLASSIFICATION not valid for COMPLEX net." ) ;
         else if (n_inputs <= 0)
            error_message ( "CUMULATE TRAINING SET used before INPUTS set." ) ;
         else if (n_outputs <= 0)
            error_message ( "CUMULATE TRAINING SET used before OUTPUTS set." ) ;
         else {
            if (tset == NULL) {
               MEMTEXT ( "MLFN: new tset" ) ;
               tset = new TrainingSet ( net_model , out_model ,
                                        n_inputs , n_outputs ) ;
               }
            tset->train ( rest , classif_output ) ;
            }
         continue ;
         }

      if (check_anneal ( command , rest , &anneal_params ))
         continue ;

      if (check_learn_params ( command , rest , &learn_params ))
         continue ;

      if (! strcmp ( command , "COMPUTE WEIGHTS" )) {
         if ((tset == NULL)  ||  (tset->ntrain == 0)) {
            error_message ( "Cannot COMPUTE WEIGHTS; No training set exists." ) ;
            continue ;
            }
         if (network == NULL) {
            if (n_hidden1 < 0) {
               error_message ( "COMPUTE WEIGHTS used before FIRST HIDDEN set.");
               continue ;
               }
            else if (n_hidden2 < 0) {
               error_message( "COMPUTE WEIGHTS used before SECOND HIDDEN set.");
               continue ;
               }
            else {
               MEMTEXT ( "MLFN: new LayerNet" ) ;
               network = new LayerNet ( net_model , out_model , out_linear ,
                       n_inputs , n_hidden1 , n_hidden2 , n_outputs , 1 , 1 ) ;
               }
            }
         if ((network == NULL)  ||  (! network->ok)) {  // Malloc failure?
            memory_message ( "to create network." ) ;
            if (network != NULL) {
               delete network ;
               network = NULL ;
               }
            continue ;
            }
         if ((net_model != NETMOD_REAL)  &&  n_hidden2) {
            error_message ( "Cannot have two hidden layers in COMPLEX models.");
            delete network ;
            network = NULL ;
            continue ;
            }
         if ((net_model == NETMOD_COMPLEX_HIDDEN)  &&  ! n_hidden1) {
            error_message ( "Must have hidden layer in COMPLEX-HIDDEN model.");
            delete network ;
            network = NULL ;
            continue ;
            }
         if ((learn_params.method == METHOD_REGRS_CJ)  &&  n_hidden1) {
            error_message( "Cannot have hidden layer for REGRESS_CJ learning.");
            if (! network->errtype) {  // Indicates untrained network
               delete network ;
               network = NULL ;
               }
            continue ;
            }
         if ((learn_params.errtype == ERRTYPE_XENT)  &&
             (out_model != OUTMOD_CLASSIFY)) {
            error_message( "CROSS ENTROPY valid only for CLASSIFY output." ) ;
            if (! network->errtype) {  // Indicates untrained network
               delete network ;
               network = NULL ;
               }
            continue ;
            }
         network->learn ( tset , &learn_params ) ;
         if (! network->errtype) {  // Indicates massive failure
            MEMTEXT ( "MLFN: learn failure delete network" ) ;
            delete network ;
            network = NULL ;
            }
         else {
            sprintf ( msg , "Final error = %.6lf", network->neterr ) ;
            normal_message ( msg ) ;
            }
         continue ;
         }

      if (! strcmp ( command , "PRINT WEIGHTS" )) {
         if (network == NULL)
            error_message ( "There are no learned weights to print." ) ;
         else
            network->wt_print ( rest ) ;
         continue ;
         }

      if (! strcmp ( command , "SAVE NETWORK" )) {
         if (network == NULL)
            error_message ( "There are no learned weights to save." ) ;
         else
            wt_save ( network , 0 , rest ) ;
         continue ;
         }

      if (! strcmp ( command , "RESTORE NETWORK" )) {
         if (network != NULL) {
            MEMTEXT ( "MLFN: delete network for restore" ) ;
            delete network ;
            network = NULL ;
            }
         network = wt_restore ( rest ) ;
         if (network == NULL)
            continue ;
         if (tset != NULL) {
            if ((n_inputs != network->nin)
             || (n_outputs != network->nout)
             || (net_model != network->model)
             || (out_model != network->outmod)) {
               error_message ( "Net conflicts with existing training set." ) ;
               MEMTEXT ( "MLFN: delete network" ) ;
               delete network ;
               network = NULL ;
               continue ;
               }
            }
         net_model = network->model ;
         out_model = network->outmod ;
         out_linear = network->outlin ;
         n_inputs = network->nin ;
         n_outputs = network->nout ;
         n_hidden1 = network->nhid1 ;
         n_hidden2 = network->nhid2 ;
         continue ;
         }

      if (! strcmp ( command , "ERASE TRAINING SET" )) {
         if (tset != NULL) {
            MEMTEXT ( "MLFN: delete tset" ) ;
            delete tset ;
            tset = NULL ;
            }
         continue ;
         }

      if (! strcmp ( command , "ERASE NETWORK" )) {
         if (network != NULL) {
            MEMTEXT ( "MLFN: delete network" ) ;
            delete network ;
            network = NULL ;
            }
         continue ;
         }

      if (! strcmp ( command , "CLASS" )) {
         if (n_outputs < 0) {
            error_message ( "CLASS used before OUTPUTS set." ) ;
            continue ;
            }
         if (out_model != OUTMOD_CLASSIFY) {
            error_message
                  ( "CLASS only valid when MODE = CLASSIFICATION" ) ;
            continue ;
            }
         m = sscanf ( rest , "%d" , &n ) ;
         if ((m <= 0)  ||  (n < 0)) {
            sprintf ( msg , "Illegal CLASS = %s", rest ) ;
            error_message ( msg ) ;
            }
         else if (n > n_outputs) {
            sprintf ( msg , "CLASS=%d exceeds OUTPUTS=%d",
                      n, n_outputs ) ;
            error_message ( msg ) ;
            }
         else
            classif_output = n ;
         continue ;
         }


      if (! strcmp ( command , "RUN OUTPUT" )) {
         strcpy ( out_file , rest ) ;
         continue ;
         }

      if (! strcmp ( command , "RUN NETWORK WITH INPUT" )) {
         if (network == NULL)
            error_message ( "There is no trained network" ) ;
         else
            network->execute_from_file ( rest , out_file ) ;
         continue ;
         }

      if (! strcmp ( command , "TEST NETWORK WITH INPUT" )) {
         if (network == NULL)
            error_message ( "There is no trained network" ) ;
         else if (out_model == OUTMOD_CLASSIFY)
            error_message ( "TEST not valid in CLASSIFICATION output mode" ) ;
         else
            network->test_from_file ( rest ) ;
         continue ;
         }

      if (! strcmp ( command , "CLASSIFY WITH INPUT" )) {
         if (network == NULL)
            error_message ( "There is no trained network" ) ;
         else if (out_model != OUTMOD_CLASSIFY)
           error_message ( "CLASSIFY valid only in CLASSIFICATION output mode");
         else
            network->classify_from_file ( rest , threshold ) ;
         continue ;
         }

      if (! strcmp ( command , "ZERO CONFUSION ROW" )) {
         if (network == NULL)
            error_message ( "There is no trained network" ) ;
         else if (out_model != OUTMOD_CLASSIFY)
           error_message ( "ZERO CONFUSION valid only in CLASSIFICATION mode") ;
         else
            network->reset_confusion () ;
         continue ;
         }

      if (! strcmp ( command , "ACTIVATION THRESHOLD FOR CONFUSION" )) {
         p = atof ( rest ) ;
         if ((p < 0.0)  ||  (p > 100.0)) {
            sprintf ( msg , "Illegal ACTIVATION THRESHOLD: %s", rest ) ;
            error_message ( msg ) ;
            }
         else
            threshold = p / 50.0 - 1.0 ;
         continue ;
         }

      if (! strcmp ( command , "LIST CONFUSION ROW" )) {
         if (network == NULL)
            error_message ( "There is no trained network" ) ;
         else if (out_model != OUTMOD_CLASSIFY)
            error_message ( "CONFUSION valid only in CLASSIFICATION output mode" ) ;
         else
            network->show_confusion () ;
         continue ;
         }

      if (! strcmp ( command , "WRITE CONFUSION ROW TO FILE" )) {
         if (network == NULL)
            error_message ( "There is no trained network" ) ;
         else if (out_model != OUTMOD_CLASSIFY)
            error_message ( "CONFUSION valid only in CLASSIFICATION output mode" ) ;
         else
            network->save_confusion ( rest ) ;
         continue ;
         }

      sprintf ( msg , "Unknown command: %s", command ) ;
      error_message ( msg ) ;

      } // Endless command loop

   MEMTEXT ( "MLFN: control_line, msg" ) ;
   FREE ( control_line ) ;
   FREE ( msg ) ;
   MEMCLOSE () ;
   return ( 0 ) ;
}

/*
--------------------------------------------------------------------------------

   Static routines

--------------------------------------------------------------------------------
*/

static int ok_to_clear_tset( TrainingSet **tset )
{
   if (*tset == NULL)
      return 1 ;

   if (get_yn ( "Existing training set must be erased.  OK" )) {
      MEMTEXT ( "MLFN: delete tset" ) ;
      delete *tset ;
      *tset = NULL ;
      return 1 ;
      }
   else
      return 0 ;
}

static int ok_to_clear_weights( LayerNet **network )
{
   if (*network == NULL)
      return 1 ;

   if (get_yn ( "Existing learned weights must be erased.  OK" )) {
      MEMTEXT ( "MLFN: delete network" ) ;
      delete *network ;
      *network = NULL ;
      return 1 ;
      }
   else
      return 0 ;
}

static int get_yn ( char *msg )
{
   int key ;
   printf ( "\n%s? (Y/N):", msg ) ;
   for (;;) {
      key = getch () ;
      if ((key == 'y')  ||  (key == 'Y'))
         return 1 ;
      if ((key == 'n')  ||  (key == 'N'))
         return 0 ;
      printf ( " Y or N:" ) ;
      }
}

static int check_anneal ( char *command , char *rest ,
                          struct AnnealParams *anneal_params )
{
   int m, n ;
   double p ;
   char msg[80] ;

   if (! strcmp ( command , "ANNEALING INITIALIZATION TEMPERATURES" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n <= 0)) {
         sprintf ( msg , "Illegal ANNEALING TEMPERATURES: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->temps0 = n ;
      return 1 ;
      }

   if (! strcmp ( command , "ANNEALING ESCAPE TEMPERATURES" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n <= 0)) {
         sprintf ( msg , "Illegal ANNEALING TEMPERATURES: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->temps = n ;
      return 1 ;
      }

   if (! strcmp ( command , "ANNEALING INITIALIZATION ITERATIONS" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n <= 0)) {
         sprintf ( msg , "Illegal ANNEALING ITERATIONS: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->iters0 = n ;
      return 1 ;
      }

   if (! strcmp ( command , "ANNEALING ESCAPE ITERATIONS" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n <= 0)) {
         sprintf ( msg , "Illegal ANNEALING ITERATIONS: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->iters = n ;
      return 1 ;
      }
    if (! strcmp ( command , "ANNEALING INITIALIZATION SETBACK" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n < 0)) {
         sprintf ( msg , "Illegal ANNEALING SETBACK: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->setback0 = n ;
      return 1 ;
      }

   if (! strcmp ( command , "ANNEALING ESCAPE SETBACK" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n < 0)) {
         sprintf ( msg , "Illegal ANNEALING SETBACK: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->setback = n ;
      return 1 ;
      }

   if (! strcmp ( command , "ANNEALING INITIALIZATION START" )) {
      p = atof ( rest ) ;
      if (p <= 0.0) {
         sprintf ( msg , "Illegal ANNEALING START: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->start0 = p ;
      return 1 ;
      }

   if (! strcmp ( command , "ANNEALING ESCAPE START" )) {
      p = atof ( rest ) ;
      if (p <= 0.0) {
         sprintf ( msg , "Illegal ANNEALING START: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->start = p ;
      return 1 ;
      }

   if (! strcmp ( command , "ANNEALING INITIALIZATION STOP" )) {
      p = atof ( rest ) ;
      if (p <= 0.0) {
         sprintf ( msg , "Illegal ANNEALING STOP: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->stop0 = p ;
      return 1 ;
      }


   if (! strcmp ( command , "ANNEALING ESCAPE STOP" )) {
      p = atof ( rest ) ;
      if (p <= 0.0) {
         sprintf ( msg , "Illegal ANNEALING STOP: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         anneal_params->stop = p ;
      return 1 ;
      }
   return 0 ;
}


static int check_learn_params ( char *command , char *rest ,
                                struct LearnParams *learn_params )
{
   int m, n ;
   double p ;
   char msg[80] ;

   if (! strcmp ( command , "LEARNING ALGORITHM" )) {
      if (! strcmp ( rest , "SIMULATED ANNEALING" ))
         learn_params->method = METHOD_AN1 ;
      else if (! strcmp ( rest , "ANNEALING_CJ" ))
         learn_params->method = METHOD_AN1_CJ ;
      else if (! strcmp ( rest , "REGRESS_CJ" ))
         learn_params->method = METHOD_REGRS_CJ ;
      else {
         sprintf ( msg , "Illegal LEARNING ALGORITHM = %s", rest ) ;
         error_message ( msg ) ;
         }
      return 1 ;
      }

   if (! strcmp ( command , "ERROR TYPE" )) {
      if (! strcmp ( rest , "MEAN SQUARE" ))
         learn_params->errtype = ERRTYPE_MSE ;
      else if (! strcmp ( rest , "ABSOLUTE" ))
         learn_params->errtype = ERRTYPE_ABS ;
      else if (! strcmp ( rest , "KK" ))
         learn_params->errtype = ERRTYPE_KK ;
      else if (! strcmp ( rest , "CROSS ENTROPY" ))
         learn_params->errtype = ERRTYPE_XENT ;
      else if (! strcmp ( rest , "MAX" ))
         learn_params->errtype = ERRTYPE_MAX ;
      else if (! strcmp ( rest , "SIXTEENTH" ))
         learn_params->errtype = ERRTYPE_SIXTEENTH ;
      else if (! strcmp ( rest , "MEAN LOG" ))
         learn_params->errtype = ERRTYPE_MEANLOG ;
      else if (! strcmp ( rest , "LOG MEAN" ))
         learn_params->errtype = ERRTYPE_LOGMEAN ;
       else {
         sprintf ( msg , "Illegal ERROR TYPE: %s", rest ) ;
         error_message ( msg ) ;
         }
      return 1 ;
      }

   if (! strcmp ( command , "ALLOWABLE ERROR" )) {
      p = atof ( rest ) ;
      if (p < 0.0) {
         sprintf ( msg , "Illegal ALLOWABLE ERROR = %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->quit_err = p ;
      return 1 ;
      }

   if (! strcmp ( command , "MAXIMUM RESTARTS" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n < 0)) {
         sprintf ( msg , "Illegal MAXIMUM RESTARTS: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->retries = n ;
      return 1 ;
      }

   if (! strcmp ( command , "CJ ACCURACY" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n < 1)) {
         sprintf ( msg , "Illegal CJ ACCURACY: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->cj_acc = n ;
      return 1 ;
      }

   if (! strcmp ( command , "CJ REFINE" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n < 0)) {
         sprintf ( msg , "Illegal CJ REFINE: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->cj_refine = n ;
      return 1 ;
      }

   if (! strcmp ( command , "CJ PRETRIES" )) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n < 1)) {
         sprintf ( msg , "Illegal CJ PRETRIES: %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->cj_pretries = n ;
      return 1 ;
      }

   if (! strcmp ( command , "CJ PROGRESS" )) {
      learn_params->cj_progress = 1 ;
      return 1 ;
      }

   if (! strcmp ( command , "CJ NO PROGRESS" )) {
      learn_params->cj_progress = 0 ;
      return 1 ;
      }

   return 0 ;
}

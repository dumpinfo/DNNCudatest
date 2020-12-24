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
/*  PNN - Main program for implementing all probabilistic nets                */
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
   This is used in prog_win to print relatively massive progress info
*/

int show_progress = 0 ;

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
   The actual training set and trained network
*/

   TrainingSet *tset = NULL ;  // Training set here
   PNNet *network = NULL ;     // PNNet here

/*
   This structure is for the control text files (if any).
   If there is control data in the control buffer, control text is gotten
   from there.  Otherwise the user is prompted for input.
   A CONTROL command causes the following sequence to occur:
     1) A new memory block is allocated (as needed) and the file is read
     2) If there was still control data in the old buffer, the file's
        buffer is extended and the old data is appended
     3) The address of the control buffer is updated after freeing the old one
*/

   ControlData cbuf ;

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

   struct NetParams net_params ;         // Network model parameters
   struct LearnParams learn_params ;     // General learning parameters
   struct MiscParams misc_params ;       // Other parameters

/*
--------------------------------------------------------------------------------

   Program starts here.

   Verify that a careless user didn't fail to set the integer size
   correctly when compiling.

--------------------------------------------------------------------------------
*/

   int i, id ;
   char *rest ;
   FILE *fp ;

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

   printf ( "\n\nPNN - Train and test probabilistic neural networks" ) ;
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

      if (! strcmp ( argv[i] , "/PROGRESS" )) {
         show_progress = 1 ;
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
   Initialize defaults and "undefined" values
*/

   defaults ( &net_params , &learn_params , &misc_params ) ;

   cbuf.n = 0 ;
   cbuf.next = 0 ;
   cbuf.buf = NULL ;

/*
   Main loop processes all commands
*/

   for (;;) {
      id = get_control_line ( &cbuf , &rest ) ;
      if (process ( id , rest , &net_params , &learn_params , &misc_params ,
                    &cbuf , &network , &tset ))
         break ;
      } // Endless command loop

   if (cbuf.n) {  // Only happens if QUIT in control file before end
      MEMTEXT ( "PNN: control buffer" ) ;
      FREE ( cbuf.buf ) ;
      }

   MEMCLOSE () ;
   return EXIT_SUCCESS ;
}


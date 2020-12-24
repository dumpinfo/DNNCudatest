/******************************************************************************/
/*                                                                            */
/*  PREDICT - Main program                                                    */
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
#include "..\common\const.h"  // System, limitation constants, typedefs, structs
#include "..\common\classes.h"  // Includes all class headers
#include "..\common\funcdefs.h" // Function prototypes

/*
   These are defined in MEM.CPP
*/

extern int mem_keep_log ;       // Keep a log file?
extern char mem_file_name[] ;   // Log file name
extern int mem_max_used ;       // Maximum memory ever in use

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
   This structure is for the control text files (if any).
   If there is control data in the control buffer, control text is gotten
   from there.  Otherwise the user is prompted for input.
   A COMMAND FILE command causes the following sequence to occur:
     1) A new memory block is allocated (as needed) and the file is read
     2) If there was still control data in the old buffer, the file's
        buffer is extended and the old data is appended
     3) The address of the control buffer is updated after freeing the old one
*/

   ControlData cbuf ;

/*
--------------------------------------------------------------------------------

   Program starts here.

--------------------------------------------------------------------------------
*/

   int i, id ;
   char *rest, line[256], error[256], results[256] ;
   FILE *fp ;

   if (sizeof(int) < 4) {
      printf ( "\nThis program requires 4-byte integers." ) ;
      exit ( 1 ) ;
      }

   printf ( "\n\nNPREDICT - Time series prediction" ) ;
   printf (   "\n           Version 1.0 Copyright 1995 Tim Masters.\n" ) ;

/*
   Process command line parameters
*/

   mem_keep_log = 0 ;       // Default is no memory allocation file
   mem_file_name[0] = 0 ;
   mem_max_used = 0 ;

   for (i=1 ; i<argc ; i++) {  // First command line argument is program name
      str_to_upr ( argv[i] ) ; // Keep in upper case for uniformity

      if (! strcmp ( argv[i] , "/DEBUG" )) {
         sscanf ( argv[++i] , "%s" , mem_file_name ) ;
         if ((strlen ( mem_file_name ) > 1) || ! isalpha ( mem_file_name[0] )) {
            printf ( "\nDEBUG drive must be 1 letter, not %s." ) ;
            exit ( 1 ) ;
            }
         strcat ( mem_file_name , ":MEM.LOG" ) ;
         fp = fopen ( mem_file_name , "wt" ) ;
         if (fp == NULL) {
            printf ( "\nCannot open /DEBUG file (%s)", mem_file_name ) ;
            exit ( 1 ) ;
            }
         fclose ( fp ) ;
         mem_keep_log = 1 ;
         continue ;
         }

      printf ( "\nUndefined command line parameter (%s)", argv[i] ) ;
      exit ( 1 ) ;
      }

/*
   Initialize the graphics adapter
*/

   if (init_graphics ()) {  // This is for graphics mode (in GRAPHICS.CPP)
      printf ( "\nWarning... The video graphics adapter is not supported." ) ;
      printf ( "\n           DISPLAY commands will not be honored." ) ;
      }
   init_textmode () ;       // This is for text mode (in GRAPHICS.CPP)

/*
--------------------------------------------------------------------------------

   Initialize command buffer and user defaults

--------------------------------------------------------------------------------
*/

   cbuf.n = 0 ;
   cbuf.next = 0 ;
   cbuf.buf = NULL ;

   process ( ID_PRED_INITIALIZE , rest , &cbuf , line , results ) ;


/*
   Main loop processes all commands
*/

   for (;;) {
      if (cbuf.n) {
         get_control_line ( &cbuf , line ) ;
         printf ( "\n%s", line ) ;
         }
      else {
         printf ( "\nInput (or BYE) " ) ;  // No control data
         gets ( line ) ;                   // so read from stdin
         }
      id = interpret_control_line ( line , &rest ) ;
      if (! id)      // No command (blank or comment)
         continue ;
      if (id < 0) {
         printf ( "\nUnknown command: %s", line ) ;
         if (cbuf.n)  // If reading control file
            break ;   // Then abort reading
         }

      id = process ( id , rest , &cbuf , error , results ) ;

      if (strlen ( results ))
         printf ( "\n%s", results ) ;

      if (id > 0)     // >0 means BYE, <0 means error
         break ;

      if (id < 0) {
         printf ( "\nERROR: %s", error ) ;
         if (cbuf.n)     // If reading control file
            break ;      // Then abort
         }

      } // Endless command loop

   if (cbuf.n) {  // Only happens if BYE in control file before end
      MEMTEXT ( "PREDICT: control buffer" ) ;
      FREE ( cbuf.buf ) ;
      }

   close_textmode () ;       // This is for text mode (in GRAPHICS.CPP)
   MEMCLOSE () ;
   return EXIT_SUCCESS ;
}


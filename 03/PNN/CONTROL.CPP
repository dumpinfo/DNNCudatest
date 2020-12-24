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
/*  CONTROL - Gets the next control line in the control buffer, or if it      */
/*            is empty gets user's input.                                     */
/*            It then interprets the command and returns its id.              */
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
   Forward declarations for static routines
*/

static void remove_comments ( char *str ) ;
static void remove_leading_blanks ( char **str ) ;
static void remove_trailing_blanks ( char *str ) ;
static void split_control_line ( char *line , char **command , int *n_command ,
                                 char **rest , int *n_rest ) ;
void str_to_upr ( char *str ) ;


int get_control_line ( ControlData *cbuf , char **rest )
{
   int n ;
   char msg[256] ;
   static char line[256] ;  // Line copied to here
   char *command ;          // Pointer to its command part
   int n_command, n_rest ;  // Lengths of command part and rest

   if (! cbuf->n) {
      printf ( "\nInput (or BYE) " ) ;  // No control data
      gets ( line ) ;                     // so read from stdin
      }

   else {
      strcpy ( line , cbuf->buf ) ;
      n = strlen ( line ) + 1 ;   // Include 0 at end of string
      cbuf->n -= n ;
      if (cbuf->n)
         memcpy ( cbuf->buf , cbuf->buf+n , cbuf->n ) ;
      else {
         MEMTEXT ( "CONTROL: control buffer" ) ;
         FREE ( cbuf->buf ) ;
         }
      }

/*
   At this point, the next command string is in 'line' and the command buffer
   is appropriately crunched.
   The first step is to split the line into its command part and the
   rest of the line, which contains any data needed by the command.
*/

   split_control_line ( line , &command , &n_command , rest , &n_rest ) ;

   if (! n_command)
      return 0 ;

   if (n_rest)
      sprintf ( msg , "%s = %s", command, *rest ) ;
   else 
      strcpy ( msg , command ) ;

   normal_message ( msg ) ;

/*
   Interpret it and return the command id code.
*/

   if (! strcmp ( command , "BYE" ))
      return ID_PNNCOM_BYE ;

   if (! strcmp ( command , "COMMAND FILE" ))
      return ID_PNNCOM_COMMANDFILE ;

   if (! strcmp ( command , "NETWORK MODEL" ))
      return ID_PNNCOM_NETMOD ;

   if (! strcmp ( command , "KERNEL" ))
      return ID_PNNCOM_KERNEL ;

   if (! strcmp ( command , "MODE" ))
      return ID_PNNCOM_OUTMOD ;

   if (! strcmp ( command , "INPUTS" ))
      return ID_PNNCOM_NIN ;

   if (! strcmp ( command , "OUTPUTS" ))
      return ID_PNNCOM_NOUT ;

   if (! strcmp ( command , "CUMULATE TRAINING SET" ))
      return ID_PNNCOM_TRAIN ;

   if (! strcmp ( command , "ALLOWABLE ERROR" ))
      return ID_PNNCOM_QUITERR ;

   if (! strcmp ( command , "TOLERANCE" ))
      return ID_PNNCOM_QUITTOL ;

   if (! strcmp ( command , "SIGMA LOW" ))
      return ID_PNNCOM_SIGLO ;

   if (! strcmp ( command , "SIGMA HIGH" ))
      return ID_PNNCOM_SIGHI ;

   if (! strcmp ( command , "SIGMA TRIES" ))
      return ID_PNNCOM_SIGTRIES ;

   if (! strcmp ( command , "COMPUTE WEIGHTS" ))
      return ID_PNNCOM_LEARN ;

   if (! strcmp ( command , "SAVE NETWORK" ))
      return ID_PNNCOM_SAVENET ;

   if (! strcmp ( command , "RESTORE NETWORK" ))
      return ID_PNNCOM_RESTNET ;

   if (! strcmp ( command , "ERASE TRAINING SET" ))
      return ID_PNNCOM_CLEARTRAIN ;

   if (! strcmp ( command , "ERASE NETWORK" ))
      return ID_PNNCOM_CLEARNET ;

   if (! strcmp ( command , "CLASS" ))
      return ID_PNNCOM_CLASSOUT ;

   if (! strcmp ( command , "PRIOR" ))
      return ID_PNNCOM_PRIOR ;

   if (! strcmp ( command , "RUN OUTPUT" ))
      return ID_PNNCOM_OUTFILE ;

   if (! strcmp ( command , "RUN NETWORK WITH INPUT" ))
      return ID_PNNCOM_EXECUTE ;

   if (! strcmp ( command , "TEST NETWORK WITH INPUT" ))
      return ID_PNNCOM_TEST ;

   if (! strcmp ( command , "CLASSIFY WITH INPUT" ))
      return ID_PNNCOM_CLASSIFY ;

   if (! strcmp ( command , "ZERO CONFUSION ROW" ))
      return ID_PNNCOM_RESET ;

   if (! strcmp ( command , "ACTIVATION THRESHOLD FOR CONFUSION" ))
      return ID_PNNCOM_THRESH ;

   if (! strcmp ( command , "LIST CONFUSION ROW" ))
      return ID_PNNCOM_SHOWCON ;

   if (! strcmp ( command , "WRITE CONFUSION ROW TO FILE" ))
      return ID_PNNCOM_SAVECON ;

   if (! strcmp ( command , "MAX MOMENT" ))
      return ID_PNNCOM_MAXMOM ;

   sprintf ( msg , "Unknown command: %s", command ) ;
   error_message ( msg ) ;
   return 0 ;

}

/*
--------------------------------------------------------------------------------

   split_control_line - This splits the control line into the command part
      and the parameter part.  It replaces the delimeter (which terminates the
      command) with a 0, and returns a pointer to the rest of the line.
      It also returns a pointer to the command part of the line, which will
      not be the input line if it started with blanks.  The command part is
      converted to upper case.
      Note that no actual copies are made.  The returned string pointers
      simply point to parts of the input line, which is modified as needed.

--------------------------------------------------------------------------------
*/

void split_control_line (
   char *line ,     // Control line input here
   char **command , // Returned pointer to command part
   int *n_command , // Length of command part
   char **rest ,    // Returned pointer to parameter part
   int *n_rest      // Length of parameter part
   )
{
   int i, n ;

   remove_leading_blanks ( &line ) ;
   remove_comments ( line ) ;

   n = strlen ( line ) ;
   for (i=0 ; i<n ; i++) {
      if (line[i] == '=') {                 // Terminates command part
         line[i] = 0 ;                      // so terminate this substring
         remove_trailing_blanks ( line ) ;  // Blanks may preceed colon
         *command = line ;                  // Return pointer to command part
         *n_command = strlen ( *command ) ; // and its length
         str_to_upr ( *command ) ;          // Return it upper case
         *rest = line + i + 1 ;             // This is the parameter part
         remove_leading_blanks ( rest ) ;   // which may have leading blanks
         str_to_upr ( *rest ) ;             // Return it upper case
         *n_rest = strlen ( *rest ) ;       // Return its length
         return ;                           // Normal return here
         }
      }

   remove_trailing_blanks ( line ) ;
   *command = line ;
   str_to_upr ( *command ) ;          // Return it upper case
   *n_command = strlen ( line ) ;
   *rest = 0 ;
   *n_rest = 0 ;
}

/*
--------------------------------------------------------------------------------

   Other local utility routines

--------------------------------------------------------------------------------
*/

static void remove_leading_blanks ( char **str )
{
   while (**str == ' ')
      ++*str ;
}

static void remove_trailing_blanks ( char *str )
{
   int n = strlen ( str ) ;
   while (n--) {
      if (str[n] == ' ')
         str[n] = 0 ;
      else
         break ;
      }
}

static void remove_comments ( char *str )
{
   int i, n ;
   n = strlen ( str ) ;
   for (i=0 ; i<n ; i++) {
      if (str[i] == ';') {
         str[i] = 0 ;
         while (i--) {
            if (str[i] == ' ')
               str[i] = 0 ;
            else
               return ;
            }
         }
      }
}

void str_to_upr ( char *str )
{
   char *ptr = str-1;
   while (*++ptr)
      *ptr = toupper ( *ptr ) ;
}

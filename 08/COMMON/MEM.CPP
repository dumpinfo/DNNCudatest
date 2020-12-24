/******************************************************************************/
/*                                                                            */
/*  MEM.CPP                                                                   */
/*                                                                            */
/*  This set of routines is used to intercept all calls to malloc, realloc,   */
/*  and free, checking the legality of their actions.                         */
/*  The routine memtext is also here to allow runtime commenting of the       */
/*  log file.                                                                 */
/*  Finally, memclose should be called at program completion to verify that   */
/*  no memory is still dangling.                                              */
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
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

#define MAX_ALLOCS 4096

/*
   These three globals must be initialized in the main program
*/

int mem_keep_log ;        // Keep a log file?
char mem_file_name[256] ; // Log file name
int mem_max_used=0 ;      // Maximum memory ever in use

static int nallocs=0 ;                   // Number of allocations
static int allocs[MAX_ALLOCS] ;          // Ptrs to them are here
static unsigned alloc_size[MAX_ALLOCS] ; // Size of those allocs
static int total_use=0 ;                 // Total bytes allocated
static FILE *fp_rec ;                    // File pointer for recording actions

void *memalloc ( unsigned n )
{       
   void *ptr ;

   if (n == 0) {
      if (mem_keep_log) {
         fp_rec = fopen ( mem_file_name , "at" ) ;
         fprintf ( fp_rec , "\nMEM.CPP: memalloc called with length=0" ) ;
         fclose ( fp_rec ) ;
         }
      return NULL ;
      }

   if (nallocs == MAX_ALLOCS) {
      if (mem_keep_log) {
         fp_rec = fopen ( mem_file_name , "at" ) ;
         fprintf ( fp_rec , "\nMEM.CPP: memalloc MAX_ALLOCS exceeded" ) ;
         fclose ( fp_rec ) ;
         }
      return NULL ;
      }

   ptr = (void *) malloc ( n ) ;

   if (ptr != NULL) {
      allocs[nallocs] = (int) ptr ;
      alloc_size[nallocs++] = n ;
      total_use += n ;
      if (total_use > mem_max_used)
         mem_max_used = total_use ;
      }

   if (mem_keep_log) {
      fp_rec = fopen ( mem_file_name , "at" ) ;
      fprintf ( fp_rec , "\nAlloc=%d  %u bytes  %d allocs  total memory=%d" ,
                         (int) ptr , n, nallocs, total_use ) ;
      fclose ( fp_rec ) ;
      }

   return ( ptr ) ;
}

void memfree ( void *ptr )
{
   int i ;

   i = nallocs ;
   while (i--) {
      if (allocs[i] == (int) ptr)
         break ;
      }

   if (i < 0) {
      if (mem_keep_log) {
         fp_rec = fopen ( mem_file_name , "at" ) ;
         fprintf ( fp_rec , "\nMEM.CPP: illegal FREE" ) ;
         fclose ( fp_rec ) ;
         }
      exit ( 1 ) ;
      }

   --nallocs ;
   total_use -= alloc_size[i] ;

   if (mem_keep_log) {
      fp_rec = fopen ( mem_file_name , "at" ) ;
      fprintf ( fp_rec , "\nFree=%d  %d allocs  total memory=%d",
                (int) ptr, nallocs, total_use );
      fclose ( fp_rec ) ;
      }

   while (i<nallocs) {   // Remove old ptr
      alloc_size[i] = alloc_size[i+1] ;
      allocs[i] = allocs[i+1] ;
      ++i ;
      }

   free ( ptr ) ;
}

void *memrealloc ( void *ptr , unsigned n )
{
   int i ;
   void *newptr ;

   i = nallocs ;
   while (i--) {
      if (allocs[i] == (int) ptr)
         break ;
      }

   if (i < 0) {
      if (mem_keep_log) {
         fp_rec = fopen ( mem_file_name , "at" ) ;
         fprintf ( fp_rec , "\nMEM.CPP: Illegal REALLOC." ) ;
         fclose ( fp_rec ) ;
         }
      return NULL ;
      }

   newptr = (void *) realloc ( ptr , n ) ;

   if (mem_keep_log) {
      fp_rec = fopen ( mem_file_name , "at" ) ;
      fprintf ( fp_rec , "\nRealloc=%d  %u bytes",
         (int) ptr, n ) ;
      fprintf( fp_rec, " New=%d  total memory=%d",
         (int) newptr, total_use+n-alloc_size[i]);
      fclose ( fp_rec ) ;
      }

   if (newptr != NULL) {
      total_use = total_use + n - alloc_size[i] ;
      alloc_size[i] = n ;
      allocs[i] = (int) newptr ;
      if (total_use > mem_max_used)
         mem_max_used = total_use ;
      }

   return newptr ;
}

void memtext ( char *text )
{
   if (mem_keep_log) {
      fp_rec = fopen ( mem_file_name , "at" ) ;
      fprintf ( fp_rec , "\n%s", text ) ;
      fclose ( fp_rec ) ;
      }
}

void notext ( char * )
{
   return ;
}

void memclose ()
{
   int i ;

   if (mem_keep_log) {
      fp_rec = fopen ( mem_file_name , "at" ) ;
      fprintf( fp_rec , "\nMax memory use=%d  Dangling allocs=%d",
               mem_max_used , nallocs ) ;
      i = nallocs ;
      while (i--)
         fprintf ( fp_rec , "\n%d", (int) allocs[i] ) ;
      fclose (fp_rec ) ;
      }
}

void nomemclose ()
{
   return ;
}

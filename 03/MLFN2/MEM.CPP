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
/*  MEM - Supervised memory allocation                                        */
/*                                                                            */
/*  memalloc - Allocate memory                                                */
/*  memrealloc - Reallocate memory                                            */
/*  memfree - Free allocated memory                                           */
/*  memtext - Write comments to the memory allocation file                    */
/*  memclose - Do final memory status check                                   */
/*                                                                            */
/*  This routine opens, then closes the record file for each transaction,     */
/*  ensuring that a record will always exist despite catastrophic failure.    */
/*  Flushing the file theoretically does the same thing, but is definitely    */
/*  less reliable than closing on some systems.                               */
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

#define MAX_BLOCKS 1024

int mem_log ;        // Keep a log on disk?
long mem_used=0 ;    // Keeps track of max memory usage
char mem_name[80] ;  // Full path name of log file

static long blocks[MAX_BLOCKS] ;    // Ptrs to allocated blocks
static unsigned bsize[MAX_BLOCKS] ; // Size of those blocks
static int nblocks=0 ;              // Number of blocks
static long memtot=0 ;              // Total bytes allocated
static FILE *memfp ;                // Recording file

void *memalloc ( unsigned n )
{       
   void *ptr ;

   if (n == 0) {
      printf ( "\nMEM ERROR: zero length" ) ;
      exit ( 1 ) ;
      }

   if (nblocks == MAX_BLOCKS) {
      printf ( "\nMEM ERROR: limit exceeded" ) ;
      exit ( 1 ) ;
      }

   ptr = (void *) malloc ( n ) ;

   if (ptr != NULL) {
      bsize[nblocks] = n ;
      blocks[nblocks++] = (long) ptr ;
      memtot += n ;
      if (memtot > mem_used)
         mem_used = memtot ;
      }

   if (mem_log) {
      memfp = fopen ( mem_name , "at" ) ;
      fprintf ( memfp , "\nA:%ld  (%d)  %u bytes  tot=%ld" ,
                         (long) ptr , nblocks, n, memtot ) ;
      fclose ( memfp ) ;
      }

   return ( ptr ) ;
}

void memfree ( void *ptr )
{
   int i ;

   for (i=nblocks-1 ; i>=0 ; i--) {  // Find this block
      if (blocks[i] == (long) ptr)
         break ;
      }

   if (mem_log) {
      memfp = fopen ( mem_name , "at" ) ;
      fprintf ( memfp , "\nF:%ld  (%d)", (long) ptr, nblocks ) ;
      if (i < 0)
         fprintf ( memfp , " ILLEGAL!" ) ;
      fclose ( memfp ) ;
      }

   if (i < 0) {
      printf ( "\nMEM ERROR: ILLEGAL FREE." ) ;
      exit ( 1 ) ;
      }

   memtot -= bsize[i] ;
   --nblocks ;

   while (i<nblocks) {   // Compress out because no longer used
      blocks[i] = blocks[i+1] ;
      bsize[i] = bsize[i+1] ;
      ++i ;
      }

   free ( ptr ) ;
}

void *memrealloc ( void *ptr , unsigned n )
{
   int i ;
   void *newptr ;

   for (i=nblocks-1 ; i>=0 ; i--) {  // Find this block
      if (blocks[i] == (long) ptr)
         break ;
      }

   if (mem_log) {
      memfp = fopen ( mem_name , "at" ) ;
      fprintf ( memfp , "\nR:%ld  (%d)  %u bytes",
         (long) ptr, nblocks, n ) ;
      if (i >= 0)
         fprintf ( memfp , "  tot=%ld", memtot+n-bsize[i] ) ;
      else 
         fprintf ( memfp , " ILLEGAL!" ) ;
      fclose ( memfp ) ;
      }

   if (i < 0) {
      printf ( "\nMEM ERROR: ILLEGAL REALLOC." ) ;
      exit ( 1 ) ;
      }

   newptr = (void *) realloc ( ptr , n ) ;

   if (newptr != NULL) {
      memtot = memtot + n - bsize[i] ;
      bsize[i] = n ;
      blocks[i] = (long) newptr ;
      if (memtot > mem_used)
         mem_used = memtot ;
      }

   return newptr ;
}


void memtext ( char *text )
{
   if (mem_log) {
      memfp = fopen ( mem_name , "at" ) ;
      fprintf ( memfp , "\n%s", text ) ;
      fclose ( memfp ) ;
      }
}

void notext ( char * )
{
   return ;
}

void memclose ()
{
   int i ;

   if (mem_log) {
      memfp = fopen ( mem_name , "at" ) ;
      fprintf( memfp, "\nFinal blocks=%d  max memory=%ld", nblocks , mem_used );
      for (i=0 ; i<nblocks ; i++)
         fprintf ( memfp , "\n%ld", (long) blocks[i] ) ;
      fclose (memfp ) ;
      }
}

void nomemclose ()
{
   return ;
}

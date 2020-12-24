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
/*  FLRAND - Generate full 32 bit period-free random numbers                  */
/*    This routine is nonportable in that it assumes 32-bit longs!            */
/*                                                                            */
/*    void sflrand ( long iseed ) - Set the random seed                       */
/*    long flrand () - Return a full 32 bit random integer                    */
/*    double unifrand () - Return uniform random in [0,1)                     */
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
--------------------------------------------------------------------------------

   We start out with a pair of shuffled uniform random number generators.
   Each works by using the linear congruential method to generate random
   numbers with a few more bits range.

--------------------------------------------------------------------------------
*/

#define TABLE_LENGTH_1 103
#define IM1 714025L      // These constants are from Press et. al.
#define IA1 1366L        // "Numerical Recipes in C"
#define IC1 150889L      // Do not tamper with them unless you are an expert

static long seed1 = 797L ;            // Keep the current seed here
static long table1[TABLE_LENGTH_1] ;  // Keep shuffle table here
static int table_initialized1 = 0 ;   // Has it been initialized?

static void srand1s ( long iseed )
{
   seed1 = iseed ;
   table_initialized1 = 0 ;    // Must also rebuild table!
}

static long randout1 ;
static long rand1s ()
{
   int i ;

   if (! table_initialized1) {  // Initialize shuffle table before use
      table_initialized1 = 1 ;  // Flag to avoid more inits
      for (i=0 ; i<TABLE_LENGTH_1 ; i++) {  // Fill entire table
         seed1 = (IA1 * seed1 + IC1) % IM1 ;
         table1[i] = seed1 ;
         }
      seed1 = (IA1 * seed1 + IC1) % IM1 ;
      randout1 = seed1 ; // One more for first use
      }

   i = (int) ((double) TABLE_LENGTH_1 * (double) randout1 / (double) IM1) ;
   randout1 = table1[i] ;              // This output comes from table
   seed1 = (IA1 * seed1 + IC1) % IM1 ; // Generate new random number
   table1[i] = seed1 ;                 // to replace used entry
   return randout1 ;                   // then return old entry
}

#define TABLE_LENGTH_2 97
#define IM2 312500L      // These constants are from Press et. al.
#define IA2 741L         // "Numerical Recipes in C"
#define IC2 66037L       // Do not tamper with them unless you are an expert

static long seed2 = 32667L ;          // Keep the current seed here
static long table2[TABLE_LENGTH_2] ;  // Keep shuffle table here
static int table_initialized2 = 0 ;   // Has it been initialized?

static void srand2s ( long iseed )
{
   seed2 = iseed ;
   table_initialized2 = 0 ;    // Must also rebuild table!
}

static long randout2 ;
static long rand2s ()
{
   int i ;

   if (! table_initialized2) {  // Initialize shuffle table before use
      table_initialized2 = 1 ;  // Flag to avoid more inits
      for (i=0 ; i<TABLE_LENGTH_2 ; i++) {  // Fill entire table
         seed2 = (IA2 * seed2 + IC2) % IM2 ;
         table2[i] = seed2 ;
         }
      seed2 = (IA2 * seed2 + IC2) % IM2 ;
      randout2 = seed2 ; // One more for first use
      }

   i = (int) ((double) TABLE_LENGTH_2 * (double) randout2 / (double) IM2) ;
   randout2 = table2[i] ;              // This output comes from table
   seed2 = (IA2 * seed2 + IC2) % IM2 ; // Generate new random number
   table2[i] = seed2 ;                 // to replace used entry
   return randout2 ;                   // then return old entry
}


/*
--------------------------------------------------------------------------------

   These are 16-bit subgenerators

--------------------------------------------------------------------------------
*/

static long rand16_1 ()
{
   long k ;
   long mult = IM1 / 65536L ;
   long max = mult * 65536L ;

   for (;;) {
      k = rand1s () ;
      if (k < max )
         return k / mult ;
      }
}

static long rand16_2 ()
{
   long k ;
   long mult = IM2 / 65536L ;
   long max = mult * 65536L ;

   for (;;) {
      k = rand2s () ;
      if (k < max )
         return k / mult ;
      }
}

/*
--------------------------------------------------------------------------------

   Here is the actual routine.  It combines the two randoms generated above,
   then shuffles the results.

--------------------------------------------------------------------------------
*/

#define TABLE_LENGTH 113

static long table[TABLE_LENGTH] ;  // Keep shuffle table here
static int table_initialized = 0 ; // Has it been initialized?

/*
   Set the random seed
*/

void sflrand ( long iseed )
{
   srand1s ( (iseed >> 16 ) & 65535L) ;
   srand2s ( iseed & 65535L ) ; // Set the individual seeds
   table_initialized = 0 ;      // Must also rebuild table!
}

/*
   This is the actual random number generator
*/

static long randout ;
long flrand ()
{
   int i ;

   if (! table_initialized) {  // Initialize shuffle table before use
      table_initialized = 1 ;  // Flag to avoid more inits
      for (i=0 ; i<TABLE_LENGTH ; i++)  // Fill entire table
         table[i] = (rand16_1() << 16)  |  rand16_2() ; // Put it in the table
      randout = (rand16_1() << 16)  |  rand16_2() ; // One more for first use
      }

   i = (int) ((double) TABLE_LENGTH * (double) (randout & 0x7FFFFFFFL) /
       ((double) 0x7FFFFFFFL + 1.0)) ; // Avoid cast problems with hi bit
   randout = table[i] ;            // This output comes from table
   table[i] = (rand16_1() << 16)  |  rand16_2() ; // Replace used entry
   return randout ;                // then return old entry
}

/*
--------------------------------------------------------------------------------

   Generate a uniform in [0, 1).

--------------------------------------------------------------------------------
*/

double unifrand ()
{
   double r1, r2, r ;
   double denom = 0x7FFFFFFFL + 1.0 ;

   for (;;) {
      r1 = flrand () & 0x7FFFFFFFL ;
      r2 = flrand () & 0x7FFFFFFFL ;
      r = (r1 + r2 / denom) / denom ;
      if (r < 1.0)
         return r ;
      }
}

/******************************************************************************/
/*                                                                            */
/*  CONJGRAD - Use conjugate gradients to find a local minimum of a function  */
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

#define DEBUG 0

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

static double gamma ( int n , double *g , double *grad ) ;
static void find_new_dir ( int n , double gam , double *g ,
                           double *h , double *grad ) ;

/*
   This routine uses the general univariate minimizers 'glob_min' and
   'brentmin' to minimize along the gradient line.  So, just like we do in
   the various class's 'learn' routines, we must have a local function for
   them to call, and it must have access to the relevant data.
   These statics handle that.
*/

static double univar_crit ( double t ) ; // Local univariate criterion
static double *local_x, *local_base, *local_direc ;   // It uses these
static int local_n ;
static double (*local_criter) (double * , int , double * ) ;

double conjgrad (
   int maxits ,           // Iteration limit
   double critlim ,       // Quit if crit drops this low
   double tol ,           // Convergence tolerance
   double (*criter) (double * , int , double * ) , // Criterion func
   int n ,                // Number of variables
   double *x ,            // In/out of independent variable
   double ystart ,        // Input of starting function value
   double *base ,         // Work vector n long
   double *direc ,        // Work vector n long
   double *g ,            // Work vector n long
   double *h ,            // Work vector n long
   int progress           // Print progress?
   )
{
   int i, iter, user_quit, convergence_counter, poor_cj_counter ;
   double fval, fbest, t1, t2, t3, y1, y2, y3 ;
   double prev_best, toler, gam, improvement, dlen, scale ;
   char msg[800] ;

/*
   Initialize for the local univariate criterion which may be called by
   'glob_min' and 'brentmin' to minimize along the search direction.
*/


   local_x = x ;
   local_base = base ;
   local_direc = direc ;
   local_n = n ;
   local_criter = criter ;


/*
   Initialize that the user has not pressed ESCape.
   Evaluate the function and, more importantly, its gradient, at the
   starting point.  This call to criter puts the negative gradient into direc.
   Also initialize the CJ algorithm by putting that vector in g and h.
*/

   user_quit = 0 ;
   fbest = criter ( x , 1 , direc ) ;

   prev_best = 1.e30 ;
   memcpy ( g , direc , n * sizeof(double) ) ;
   memcpy ( h , direc , n * sizeof(double) ) ;

#if DEBUG
   printf ( "\nCJ starting error = %lf", fbest ) ;
#endif

   if (fbest < 0.0) {   // If user pressed ESCape during criter call
      fbest = ystart ;
      user_quit = 1 ;
      goto FINISH ;
      }

/*
   Main loop.  For safety we impose a limit on iterations.
   There are two counters that have somewhat similar purposes.
   The first, convergence_counter, counts how many times an iteration
   failed to reduce the function value to the user's tolerance level.
   We require failure several times in a row before termination.

   The second, poor_cj_counter, has a (generally) higher threshold.
   It keeps track of poor improvement, and imposes successively smaller
   limits on gamma, thus forcing the algorithm back to steepest
   descent if CJ is doing poorly.
*/

   convergence_counter = 0 ;
   poor_cj_counter = 0 ;
   scale = 0.01 ;

   iter = 0 ;
   for (;;) {

      if ((maxits > 0)  &&  (iter++ >= maxits))
         break ;

      if (fbest < critlim)     // Do we satisfy user yet?
         break ;

/*
   Convergence check
*/

      if (prev_best <= 1.0)                  // If the function is small
         toler = tol ;                       // Work on absolutes
      else                                   // But if it is large
         toler = tol * prev_best ;           // Keep things relative

      if ((prev_best - fbest)  <=  toler) {  // If little improvement
         if (++convergence_counter >= 3)     // Then count how many
            break ;                          // And quit if too many
         }
      else                                   // But a good iteration
         convergence_counter = 0 ;           // Resets this counter

/*
   Does the user want to quit?
*/

      if ((user_quit = user_pressed_escape ()) != 0)
         break ;

/*
   Here we do a few quick things for housekeeping.
   We save the base for the linear search in 'base', which lets us
   parameterize from t=0.
   We also cumulate the gradient length.
*/

      dlen = 0.0 ;
      for (i=0 ; i<n ; i++) {
         base[i] = x[i] ;               // We step out from here
         dlen += direc[i] * direc[i] ;  // Length of search vector
         }

      dlen = sqrt ( dlen ) ;            // Actual length

      y2 = prev_best = fbest ;

#if DEBUG
      printf ( "\nStarting GLOBAL   Scale=%lf", scale ) ;
#endif


      user_quit = glob_min ( 0.0 , 2.0 * scale , -3 , 0 , critlim ,
                  univar_crit , &t1 , &y1 , &t2 , &y2 , &t3 , &y3 , progress) ;

      if (user_quit  ||  (y2 < critlim)) { // ESCape or good enough already?
         if (y2 < fbest) {                 // If global caused improvement
            for (i=0 ; i<n ; i++)          // Implement that improvement
               x[i] = base[i] + t2 * direc[i] ;
            fbest = y2 ;
            }
         else {                            // Else revert to starting point
            for (i=0 ; i<n ; i++)
               x[i] = base[i] ;
            }
         break ;
         }

/*
   We just used a crude global strategy to find three points that
   bracket the minimum.  Refine using Brent's method.
   If we are possibly near the end, as indicated by the convergence_counter
   being nonzero, then try extra hard.
*/


      if (convergence_counter)
         fbest = brentmin ( 25 , critlim , tol , 1.e-7 ,
                            univar_crit , &t1 , &t2 , &t3 , y2 , progress ) ;
      else 
         fbest = brentmin ( 15 , critlim , 10.0 * tol , 1.e-5 ,
                            univar_crit , &t1 , &t2 , &t3 , y2 , progress ) ;


#if DEBUG
         printf ( "\nBRENT t=%lf  f=%lf", t2 , fbest ) ;
#endif

      scale = t2 ;
      if (scale < 0.01 * dlen)
         scale = 0.01 * dlen ;
      if (scale > dlen)
         scale = dlen ;

/*
   We just completed the global and refined search.
   Update the current point to reflect the minimum obtained.
   Then evaluate the error and its derivatives there.  (The linear optimizers
   only evaluated the error, not its derivatives.)
   If the user pressed ESCape during dermin, fbest will be returned
   negative.
*/

      for (i=0 ; i<n ; i++)
         x[i] = base[i] + t2 * direc[i] ;

      if (fbest < 0.0) {              // If user pressed ESCape
         fbest = -fbest ;
         user_quit = 1 ;
         break ;
         }

      improvement = (prev_best - fbest) / prev_best ;

      sprintf ( msg , "scale=%lf f=%le dlen=%le improvement=%lf%%",
                scale , fbest, dlen, 100.0 * improvement ) ;
      if (progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;

      if (fbest < critlim)     // Do we satisfy user yet?
         break ;


      fval = criter ( x , 1 , direc ) ; // Need (negative) derivs now

      if (fval < 0.0  ||  user_pressed_escape()) { // If user pressed ESCape
         user_quit = 1 ;
         break ;
         }

#if DEBUG
      printf ( "\nf=%lf at (", fval ) ;
#endif
#if DEBUG > 1
      for (i=0 ; i<n ; i++)
         printf ( " %lf", x[i] ) ;
      printf ( ")...\a" ) ;
      getch () ;
#endif

      gam = gamma ( n , g , direc ) ;

#if DEBUG
      dlen = 0.0 ;
      for (i=0 ; i<n ; i++)
         dlen += direc[i] * direc[i] ;
      printf ( "\nGamma = %lf  with grad len = %lf", gam, sqrt(dlen) ) ;
#endif
      if (gam < 0.0)
         gam = 0.0 ;

      if (gam > 10.0)             // limit gamma
         gam = 10.0 ;

      if (improvement < 0.001)    // Count how many times we
         ++poor_cj_counter ;      // got poor improvement
      else                        // in a row
         poor_cj_counter = 0 ;

      if (poor_cj_counter >= 2) { // If several times
         if (gam > 1.0)           // limit gamma
            gam = 1.0 ;
         }

      if (poor_cj_counter >= 6) { // If too many times
         poor_cj_counter = 0 ;    // set gamma to 0
         gam = 0.0 ;              // to use steepest descent (gradient)
#if DEBUG
         printf ( "\nSetting Gamma=0" ) ;
#endif
         }

      find_new_dir ( n , gam , g , h , direc ) ; // Compute search direction

      } // Main loop

FINISH:
   if (user_quit)
      return -fbest ;
   else 
      return fbest ;
}


/*
--------------------------------------------------------------------------------

   Local routine to find gamma

--------------------------------------------------------------------------------
*/

static double gamma ( int n , double *g , double *grad )
{
   int i ;
   double denom, numer ;

   numer = denom = 0.0 ;

   for (i=0 ; i<n ; i++) {
      numer += (grad[i] - g[i]) * grad[i] ;  // Grad is negative gradient
      denom += g[i] * g[i] ;
      }

   if (denom == 0.0)   // Only happens if gradient is zero!
      return 0.0 ;
   else
      return numer / denom ;
}

/*
--------------------------------------------------------------------------------

   Local routine to find correction for next iteration

--------------------------------------------------------------------------------
*/

static void find_new_dir ( int n , double gam ,
                           double *g , double *h , double *grad )
{
   int i ;

   for (i=0 ; i<n ; i++) {
      g[i] = grad[i] ;
      h[i] = grad[i] = g[i] + gam * h[i] ;
      }
}


static double univar_crit ( double t )
{
   int i ;
   for (i=0 ; i<local_n ; i++)
      local_x[i] = local_base[i] + t * local_direc[i] ;
   return local_criter ( local_x , 0 , (double *) NULL ) ;
}



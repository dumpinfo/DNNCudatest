/******************************************************************************/
/*                                                                            */
/*  POWELL - Use Powell's method to find a local minimum of a function        */
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
static double (*local_criter) ( double * ) ;

double powell (
   int maxits ,           // Iteration limit
   double critlim ,       // Quit if crit drops this low
   double tol ,           // Convergence tolerance
   double (*criter) ( double * ) , // Criterion func
   int n ,                // Number of variables
   double *x ,            // In/out of independent variable
   double ystart ,        // Input of starting function value
   double *base ,         // Work vector n long
   double *p0 ,           // Work vector n long
   double *direc ,        // Work vector n*n long
   int progress           // Print progress?
   )
{
   int i, j, idir, iter, user_quit, convergence_counter, idelta, replaced ;
   double fval, fbest, f0, test, t1, t2, t3, y1, y2, y3 ;
   double prev_best, toler, delta, scale, len ;
   char msg[84] ;

/*
   Initialize for the local univariate criterion which may be called by
   'glob_min' and 'brentmin' to minimize along the search direction.
*/


   local_x = x ;
   local_base = base ;
   local_n = n ;
   local_criter = criter ;

/*
   Initialize the direction matrix to be a reflected identity.
   This reflection is in deference to the fact that this routine will
   be especially used for ARMA weight optimization.  Input terms, which
   preceed the MA terms in each weight vector, will on the first pass
   already have been optimized with regression.  Thus, reflection saves
   a few useless iterations.  For other problems it has no meaning.
*/

   for (i=0 ; i<n ; i++) {
      for (j=0 ; j<n ; j++)
         direc[i*n+j] = (j == (n-i-1)) ? 1.0 : 0.0 ;
      }

/*
   Main loop.  For safety we impose a limit on iterations.
*/

   replaced = -1 ;          // Assume no replacement will be done
   prev_best = 1.e30 ;
   fbest = ystart ;
   scale = 0.2 ;
   iter = convergence_counter = user_quit = 0 ;

   for (;;) {

      if ((iter++ >= maxits)  &&  (maxits > 0))
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
         if (++convergence_counter >= 2)     // Then count how many
            break ;                          // And quit if too many
         }
      else                                   // But a good iteration
         convergence_counter = 0 ;           // Resets this counter

      prev_best = fbest ;

/*
   Does the user want to quit?
*/

      if ((user_quit = user_pressed_escape ()) != 0)
         break ;

/*
   Loop over all search directions, minimizing in each.
   Keep track of the direction that gave the most improvement.
   For efficiency, we keep 'replaced' as the vector that just got replaced
   (or -1 if no replacement was done).  Skip the first direction if that
   was the one just replaced!
*/

#if 0
      printf ( "\nStarting new iter at " ) ;
      for (i=0 ; i<n ; i++)
         printf ( " %lf", x[i] ) ;
      printf ( " = %lf", criter ( x ) ) ;
      getch () ;
#endif

      for (i=0 ; i<n ; i++)            // Preserve the starting point
         p0[i] = x[i] ;
      f0 = fbest ;                     // And function value at that point
      delta = -1.0 ;                   // Will keep best improvement here

      for (idir=0 ; idir<n ; idir++) { // For each direction
         if ((n>1)  &&  ! idir  &&  ! replaced) // If we just replaced the
            continue ;                 // first vector, avoid waste
         for (i=0 ; i<n ; i++)         // Local criter steps out from here
            base[i] = x[i] ;           // So it must be current point
         y2 = fbest ;                  // Glob_min can use first f value
         local_direc = direc + idir * n ; // This is the idir direction
         user_quit = glob_min ( 0.0 , 0.1 * scale , -2 , 0 , critlim ,
                                univar_crit , &t1 , &y1 , &t2 ,
                                &y2 , &t3 , &y3 , -1 ) ;
         if (user_quit  ||  (y2 < critlim)) { // ESCape or good enough already?
            if (y2 < fbest) {                 // If global caused improvement
               for (i=0 ; i<n ; i++)          // Implement that improvement
                  x[i] = base[i] + t2 * local_direc[i] ;
               fbest = y2 ;
               }
            else {                            // Else revert to starting point
               for (i=0 ; i<n ; i++)
                  x[i] = base[i] ;
               }
            goto FINISH ;
            }

#if DEBUG
         printf ( "\nGLOBAL %d: t=%lf  scale=%lf  f=%lf",
                  idir, t2 / scale , scale, y2 ) ;
#endif

         if (convergence_counter)  // If failing, try extra hard
            fval = brentmin ( 40 , critlim , tol , 1.e-7 ,
                              univar_crit , &t1 , &t2 , &t3 , y2 , -1 ) ;
         else                      // But normally refine only moderately
            fval = brentmin ( 20 , critlim , 10.0 * tol , 1.e-5 ,
                              univar_crit , &t1 , &t2 , &t3 , y2 , -1 ) ;
         scale = fabs(t2) / n  +  (1.0 - 1.0/n) * scale ; // Keep reasonable

#if DEBUG
         printf ( "\nBRENT %d: t=%lf  scale=%lf  f=%lf",
                  idir, t2 / scale , scale, fval ) ;
#endif

         for (i=0 ; i<n ; i++)          // Get current point from parametric
            x[i] = base[i] + t2 * local_direc[i] ;
         if (fval < 0.0) {              // If user pressed ESCape
            fbest = -fval ;
            user_quit = 1 ;
            goto FINISH ;
            }
         if (fbest - fval > delta) {    // Keep track of best direction
            delta = fbest - fval ;
            idelta = idir ;
            }
         fbest = fval ;                  // This is always the best so far
         } // For all directions

/*
   Before looping through all n directions, we stood at point p0 with f=f0.
   We now stand at point x with f=fbest.
   It is quite possible that the average direction of motion points right
   along a ravine.  Thus, it behooves us to step out in that direction.
   Try it.  We might luck out.
*/

      for (i=0 ; i<n ; i++) {
         p0[i] = x[i] - p0[i] ;    // Preserve average direction here
         base[i] = x[i] + p0[i] ;  // Step out to this point (borrow base)
         }
      fval = criter ( base ) ;     // Evaluate function at this test point

/*
   If this step improved, and if a more sophisticated second derivative
   test is passed, we replace the old direction of max improvement with this
   average direction (after first minimizing in this great direction).
   Also, set 'replaced' to the vector that was replaced if that is done.
   Otherwise set it to -1.  We do that so on the next iteration we skip
   minimization along the first direction if that happened to be the one
   that was just replaced!
*/

      replaced = -1 ;                  // Assume no replacement will be done
      if (fval < fbest) {              // If this trial step improved

         fbest = fval ;                // Might as well keep this better point
         for (i=0 ; i<n ; i++)         // Keep x current
            x[i] = base[i] ;
#if DEBUG
         printf ( "\nTest step improved" ) ;
#endif
         test = f0 - fbest - delta ;
         test = 2.0 * (f0 - 2.0 * fbest + fval) * test * test ;
         if (test < delta * (f0 - fval) * (f0 - fval)) { // Use this direction
            replaced = idelta ;        // Record this upcoming replacement
            len = 0.0 ;                // Will cumulate length of new dir
            for (i=0 ; i<n ; i++)
               len += p0[i] * p0[i] ;
            len = sqrt ( len ) ;
            for (i=0 ; i<n ; i++)
               p0[i] /= len ;          // Keep direction unit length
            local_direc = p0 ;         // We put the average direction here 
            y2 = fbest ;               // Glob_min can use first f value
            user_quit = glob_min ( 0.0 , 0.1 * scale , -2 , 0 , critlim ,
                                   univar_crit , &t1 , &y1 , &t2 ,
                                   &y2 , &t3 , &y3 , -1 ) ;
            if (user_quit  ||  (y2 < critlim)) { // ESCape or good enough already?
               if (y2 < fbest) {                 // If global caused improvement
                  for (i=0 ; i<n ; i++)          // Implement that improvement
                     x[i] = base[i] + t2 * local_direc[i] ;
                  fbest = y2 ;
                  }
               else {                            // Else revert to starting point
                  for (i=0 ; i<n ; i++)
                     x[i] = base[i] ;
                  }
               break ;
               }
#if DEBUG
            printf ( "\nAVG GLOBAL t=%lf  scale=%lf  f=%lf",
                     t2 / scale , scale, y2 ) ;
#endif
            if (convergence_counter)  // If failing, try extra hard
               fval = brentmin ( 40 , critlim , tol , 1.e-7 ,
                                 univar_crit , &t1 , &t2 , &t3 , y2 , -1 ) ;
            else                      // But normally refine only moderately
               fval = brentmin ( 20 , critlim , 10.0 * tol , 1.e-5 ,
                                 univar_crit , &t1 , &t2 , &t3 , y2 , -1 ) ;
            scale = fabs(t2) / n  +  (1.0 - 1.0/n) * scale ; // Scale reasonable
#if DEBUG
            printf ( "\nAVG BRENT t=%lf  scale=%lf  f=%lf",
                     t2 / scale , scale, fval ) ;
#endif
            for (i=0 ; i<n ; i++)          // Get current point from parametric
               x[i] = base[i] + t2 * local_direc[i] ;
            if (fval < 0.0) {              // If user pressed ESCape
               fbest = -fval ;
               user_quit = 1 ;
               goto FINISH ;
               }
            fbest = fval ;                 // This is always the best so far

            for (i=0 ; i<n ; i++)
               direc[idelta*n+i] = p0[i] ; // Replace best direction with this
            sprintf ( msg , "Replacing direction %d.  Length=%lf", idelta, len);
            if (progress)
               write_progress ( msg ) ;
            else 
               write_non_progress ( msg ) ;
            } // If second derivative test passed
         } // If test step improved
      sprintf ( msg , "Iteration %d  f=%lf  (old=%lf)", iter, fbest, prev_best);
      if (progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;
      } // Main loop

FINISH:
   if (user_quit)
      return -fbest ;
   else 
      return fbest ;
}


/*
--------------------------------------------------------------------------------

   local criterion for univariate minimization

--------------------------------------------------------------------------------
*/

static double univar_crit ( double t )
{
   int i ;

   for (i=0 ; i<local_n ; i++)
      local_x[i] = local_base[i] + t * local_direc[i] ;
   return local_criter ( local_x ) ;
}




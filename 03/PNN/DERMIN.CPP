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
/*  DERMIN - Use conjugate gradients to find a local minimum of a function    */
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
static double (*local_criter) (double * , double * , double * , int ) ;

double dermin (
   int itmax ,            // Iteration limit
   double critlim ,       // Quit if crit drops this low
   double eps ,           // Small, but greater than machine precision
   double tol ,           // Convergence tolerance
   double (*criter) (double * , double * , double * , int ) , // Criterion func
   int n ,                // Number of variables
   double *x ,            // In/out of independent variable
   double ystart ,        // Input of starting function value
   double *base ,         // Work vector n long
   double *direc ,        // Work vector n long
   double *g ,            // Work vector n long
   double *h ,            // Work vector n long
   double *deriv2         // Work vector n long
   )
{
   int i, iter, user_quit, convergence_counter, poor_cj_counter ;
   double fval, fbest, high, scale, t1, t2, t3, y1, y2, y3, dlen, dot1, dot2 ;
   double prev_best, toler, gam, improvement ;
   char msg[84] ;

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
   Evaluate the function and, more importantly, its derivatives, at the
   starting point.  This call to criter puts the gradient into direc, but
   we flip its sign to get the downhill search direction.
   Also initialize the CJ algorithm by putting that vector in g and h.
*/

   user_quit = 0 ;
   fbest = criter ( x , direc , deriv2 , 1 ) ;
   prev_best = 1.e30 ;
   for (i=0 ; i<n ; i++)
      direc[i] = -direc[i] ;
   memcpy ( g , direc , n * sizeof(double) ) ;
   memcpy ( h , direc , n * sizeof(double) ) ;


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
   It keeps track of poor improvement, and imposes successively small
   limits on gamma, thus forcing the algorithm back to steepest
   descent if CJ is doing poorly.
*/

   convergence_counter = 0 ;
   poor_cj_counter = 0 ;

   for (iter=0 ; iter<itmax ; iter++) {

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
   We find the greatest second derivative.  This makes an excellent
   scaling factor for the search direction so that the initial global
   search for a trio containing the minimum is fast.  Because this is so
   stable, we use it to bound the generally better but unstable Newton scale.
   We also compute the length of the search vector and its dot product with
   the gradient vector, as well as the directional second derivative.
   That lets us use a sort of Newton's method to help  us scale the
   initial global search to be as fast as possible.  In the ideal case,
   the 't' parameter will be exactly equal to 'scale', the center point
   of the call to glob_min.
*/

      dot1 = dot2 = dlen = 0.0 ;        // For finding directional derivs
      high = 1.e-4 ;                    // For scaling glob_min
      for (i=0 ; i<n ; i++) {
         base[i] = x[i] ;               // We step out from here
         if (deriv2[i] > high)          // Keep track of second derivatives
            high = deriv2[i] ;          // For linear search via glob_min
         dot1 += direc[i] * g[i] ;      // Directional first derivative
         dot2 += direc[i] * direc[i] * deriv2[i] ; // and second
         dlen += direc[i] * direc[i] ;  // Length of search vector
         }

      dlen = sqrt ( dlen ) ;            // Actual length

/*
   The search direction is in 'direc' and the maximum second derivative
   is in 'high'.  That stable value makes a good approximate scaling factor.
   The ideal Newton scaling factor is numerically unstable.
   So compute the Newton ideal, then bound it to be near the less ideal
   but far more stable maximum second derivative.
   Pass the first function value, corresponding to t=0, to the routine
   in *y2 and flag this by using a negative npts.
*/

      scale = dot1 / dot2 ;          // Newton's ideal but unstable scale
      high = 1.5 / high ;            // Less ideal but more stable heuristic
      if (high < 1.e-4)              // Subjectively keep it realistic
         high = 1.e-4 ;

      if (scale < 0.0)               // This is truly pathological
         scale = high ;              // So stick with old reliable
      else if (scale < 0.1 * high)   // Bound the Newton scale
         scale = 0.1 * high ;        // To be close to the stable scale
      else if (scale > 10.0 * high)  // Bound it both above and below
         scale = 10.0 * high ;


      y2 = prev_best = fbest ;

      user_quit = glob_min ( 0.0 , 2.0 * scale , -3 , 0 , critlim ,
                  univar_crit , &t1 , &y1 , &t2 , &y2 , &t3 , &y3 ) ;

      if (user_quit  ||  (y2 < critlim)) { // ESCape or good enough already?
         if (y2 < fbest) {                 // If global caused improvement
            for (i=0 ; i<n ; i++) {        // Implement that improvement
               x[i] = base[i] + t2 * direc[i] ;
               if (x[i] < 1.e-10)           // Limit it away from zero
                  x[i] = 1.e-10 ;           // Fairly arbitrary constant
               }
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
         fbest = brentmin ( 20 , critlim , eps , 1.e-7 ,
                            univar_crit , &t1 , &t2 , &t3 , y2 ) ;
      else 
         fbest = brentmin ( 10 , critlim , 1.e-6 , 1.e-5 ,
                            univar_crit , &t1 , &t2 , &t3 , y2 ) ;

/*
   We just completed the global and refined search.
   Update the current point to reflect the minimum obtained.
   Then evaluate the error and its derivatives there.  (The linear optimizers
   only evaluated the error, not its derivatives.)
   If the user pressed ESCape during dermin, fbest will be returned
   negative.
*/

      for (i=0 ; i<n ; i++) {
         x[i] = base[i] + t2 * direc[i] ;
         if (x[i] < 1.e-10)           // Limit it away from zero
            x[i] = 1.e-10 ;           // Fairly arbitrary constant
         }

      if (fbest < 0.0) {              // If user pressed ESCape
         fbest = -fbest ;
         user_quit = 1 ;
         break ;
         }

      improvement = (prev_best - fbest) / prev_best ;

      if (fbest < critlim)     // Do we satisfy user yet?
         break ;

      fval = criter ( x , direc , deriv2 , 1 ) ; // Need derivs now
      for (i=0 ; i<n ; i++)                      // Flip sign to get
         direc[i] = -direc[i] ;                  // negative gradient

      if (fval < 0.0) {                          // If user pressed ESCape
         user_quit = 1 ;
         break ;
         }

      sprintf ( msg , "t=%lf f=%le improvement=%lf%%",
                t2 / scale , fval, 100.0 * improvement ) ;
      write_progress ( msg ) ;

      gam = gamma ( n , g , direc ) ;

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

double gamma ( int n , double *g , double *grad )
{
   int i ;
   double denom, numer ;

   numer = denom = 0.0 ;

   for (i=0 ; i<n ; i++) {
      denom += g[i] * g[i] ;
      numer += (grad[i] - g[i]) * grad[i] ;  // Grad is neg gradient
      }

   if (denom == 0.0)   // Should never happen (means gradient is zero!)
      return 0.0 ;
   else
      return numer / denom ;
}

/*
--------------------------------------------------------------------------------

   Local routine to find correction for next iteration

--------------------------------------------------------------------------------
*/

void find_new_dir ( int n , double gam , double *g , double *h , double *grad )
{
   int i ;

   for (i=0 ; i<n ; i++) {
      g[i] = grad[i] ;
      grad[i] = h[i] = g[i] + gam * h[i] ;
      }
}


static double univar_crit ( double t )
{
   int i ;
   for (i=0 ; i<local_n ; i++) {
      local_x[i] = local_base[i] + t * local_direc[i] ;
      if (local_x[i] < 1.e-10)
         local_x[i] = 1.e-10 ;
      }
   return local_criter ( local_x , (double *) NULL , (double *) NULL , 0 ) ;
}


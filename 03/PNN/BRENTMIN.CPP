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
/*  BRENTMIN - Use Brent's method to find a local minimum of a                */
/*             univariate function.                                           */
/*                                                                            */
/*  This is given three points such that the center has lesser function       */
/*  value than its neighbors.  It iteratively refines the interval.           */
/*  These subintervals will be spaced arithmetically or logarithmically       */
/*  according to log_space.                                                   */
/*  If the criterion function drops to critlim or smaller, execution will     */
/*  terminate.                                                                */
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

double brentmin (
   int itmax ,            // Iteration limit
   double critlim ,       // Quit if crit drops this low
   double eps ,           // Small, but greater than machine precision
   double tol ,           // Brent's tolerance (>= sqrt machine precision)
   double (*criter) (double) , // Criterion function
   double *x1 ,           // Lower X value, input and output
   double *x2 ,           // Middle (best), input and output
   double *x3 ,           // And upper, input and output
   double y               // Function value at x2
   )
{
   int iter, user_quit ;
   double prevdist, step, xlow, xmid, xhigh, tol1, tol2 ;
   double xbest, xsecbest, xthirdbest, fbest, fsecbest, fthirdbest ;
   double numer, denom, testdist, xrecent, frecent, t1, t2 ;
   char msg[160] ;

   user_quit = 0 ;

/*
  Initialize prevdist, the distance moved on the previous step, to 0 so that
  the 'if (fabs ( prevdist )  >  tol1)' encountered on the first iteration
  below will fail, forcing a golden section the first time.  Also initialize
  step to 0 to avoid a zealous compiler from pointing out that it was
  referenced before being set.
*/

   prevdist = step = 0.0 ;

/*
   We always keep the minimum bracketed between xlow and xhigh.
   xbest has the min function ordinate so far (or latest if tie).
   xsecbest and xthirdbest are the second and third best.
*/

   xbest = xsecbest = xthirdbest = *x2 ;
   xlow = *x1 ;
   xhigh = *x3 ;

   fbest = fsecbest = fthirdbest = y ;


/*
   Main loop.  For safety we impose a limit on iterations.
*/

   for (iter=0 ; iter<itmax ; iter++) {

      sprintf ( msg , "%.9lf %.9lf %.9lf :  %.6lf %.6lf %.6lf",
                      xlow, xbest, xhigh, fbest, fsecbest, fthirdbest ) ;
      write_progress ( msg ) ;

      if (fbest < critlim)    // Do we satisfy user yet?
         break ;

      xmid = 0.5 * (xlow + xhigh) ;
      tol1 = tol * (fabs ( xbest ) + eps) ;
      tol2 = 2. * tol1 ;

      if ((user_quit = user_pressed_escape ()) != 0)
         break ;

/*
   The following convergence test simultaneously makes sure xhigh and
   xlow are close relative to tol2, and that xbest is near the midpoint.
*/
      if (fabs ( xbest - xmid )  <=  (tol2 - 0.5 * (xhigh - xlow)))
         break ;

/*
   Avoid refining function to limits of precision
*/
      if ((iter >= 2)  &&  ((fthirdbest - fbest) < eps))
         break ;

      if (fabs ( prevdist )  >  tol1) {  // If we moved far enough try parabolic fit
         t1 = (xbest - xsecbest) * (fbest - fthirdbest) ; // Temps for the
         t2 = (xbest - xthirdbest) * (fbest - fsecbest) ; // parabolic estimate
         numer = (xbest - xthirdbest) * t2  -  (xbest - xsecbest) * t1 ;
         denom = 2. * (t1 - t2) ;  // Estimate will be numer / denom
         testdist = prevdist ;     // Will soon verify interval is shrinking
         prevdist = step ;         // Save for next iteration
         if (denom != 0.0)         // Avoid dividing by zero
            step = numer / denom ; // This is the parabolic estimate to min
         else 
            step = 1.e30 ;         // Assures failure of next test

         if ((fabs ( step ) < fabs ( 0.5 * testdist ))// If shrinking
          && (step + xbest > xlow)             // and within known bounds
          && (step + xbest < xhigh)) {         // then we can use the
            xrecent = xbest + step ;           // parabolic estimate
            if ((xrecent - xlow  <  tol2)  ||  // If we are very close
                (xhigh - xrecent <  tol2)) {   // to known bounds
               if (xbest < xmid)               // then stabilize
                  step = tol1 ;
               else
                  step = -tol1 ;
               }
            }
         else {  // Parabolic estimate poor, so use golden section
            prevdist = (xbest >= xmid)  ?  xlow - xbest  :  xhigh - xbest ;          // Poor so use    
            step = .3819660 * prevdist ;
            }
         }
      else { // prevdist did not exceed tol1: we did not move far enough
             // to justify a parabolic fit.  Use golden section.
         prevdist = (xbest >= xmid)  ?  xlow - xbest  :  xhigh - xbest ;
         step = .3819660 * prevdist ;
         }

      if (fabs (step)  >=  tol1)     // In order to numerically justify
         xrecent = xbest + step ;    // another trial we must move a
      else {                         // decent distance.
         if (step > 0.)
            xrecent = xbest + tol1 ;
         else
            xrecent = xbest - tol1 ;
         }

/*
   At long last we have a trial point 'xrecent'.  Evaluate the function.
*/

      frecent = criter ( xrecent ) ;

      if (frecent < 0.0) {
         user_quit = 1 ;
         break ;
         }

      if (frecent <= fbest) {    // If we improved...
         if (xrecent >= xbest)   // Shrink the (xlow,xhigh) interval by
            xlow = xbest ;       // replacing the appropriate endpoint
         else
            xhigh = xbest ;
         xthirdbest = xsecbest ; // Update x and f values for best,
         xsecbest = xbest ;      // second and third best
         xbest = xrecent ;
         fthirdbest = fsecbest ;
         fsecbest = fbest ;
         fbest = frecent ;
         }

      else {                  // We did not improve
         if (xrecent < xbest) // Shrink the (xlow,xhigh) interval by
            xlow = xrecent ;  // replacing the appropriate endpoint
         else
            xhigh = xrecent ;

         if ((frecent <= fsecbest)   // If we at least beat the second best
          || (xsecbest == xbest)) {  // or we had a duplication
            xthirdbest = xsecbest ;  // we can update the second and third
            xsecbest = xrecent ;     // best, though not the best.
            fthirdbest = fsecbest ;  // Recall that we started iters with
            fsecbest = frecent ;     // best, sec and third all equal.
            }
         else if ((frecent <= fthirdbest) // Oh well.  Maybe at least we can
           || (xthirdbest == xbest)       // beat the third best or rid
           || (xthirdbest == xsecbest)) { // ourselves of a duplication
            xthirdbest = xrecent ;        // (which is how we start the
            fthirdbest = frecent ;        // iterations)
            }
         }
      }
 
   *x1 = xlow ;
   *x2 = xbest ;
   *x3 = xhigh ;

   if (user_quit)
      return -fbest ;
   else 
      return fbest ;
}



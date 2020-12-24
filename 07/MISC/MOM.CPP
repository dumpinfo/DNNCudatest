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

#include <math.h>

#define PI 3.141592653589793

void rotobj ( int n , double theta , double *xpts , double *ypts )
{
   int i ;
   double x, y ;
   double c = cos ( theta ) ;
   double s = sin ( theta ) ;

   for (i=0 ; i<n ; i++) {
      x = xpts[i] ;
      y = ypts[i] ;
      xpts[i] = c * x  -  s * y ;  // CCW
      ypts[i] = c * y  +  s * x ;
      }
}

double zpow ( double base , int power )
{
   double result = 1.0 ;

   while (power--)
      result *= base ;
   return result ;
}

int binom ( int n , int m )
{
   int prod = 1 ;
   int den = 1 ;

   while (m) {
      prod *= n-- ;
      den *= m-- ;
      }

   return prod / den ;
}

double rotmom1 ( int p , int q , double m[4][4] , double theta )
{
   int r, s, sgn ;
   double sum = 0.0 ;
   double ct = cos ( theta ) ;
   double st = sin ( theta ) ;

   for (r=0 ; r<=p ; r++) {
      for (s=0 ; s<=q ; s++) {
         sgn = ((q - s) & 1)  ?  -1.0  :  1.0  ;
         sum += sgn * binom ( p , r ) * binom ( q , s ) *
                zpow ( ct , p-r+s ) * zpow ( st , q-s+r ) * m[p+q-r-s][r+s] ;
         }
      }
   return sum ;
}

void rotmom (
   double *moments ,  // In/Out 7 moments: 20, 11, 02, 30, 21, 12, 03
   double theta       // Clockwise rotation
   )
{
   double m[4][4] ;

   m[0][0] = 1.0 ;
   m[1][0] = m[0][1] = 0.0 ;
   m[2][0] = moments[0] ;
   m[1][1] = moments[1] ;
   m[0][2] = moments[2] ;
   m[3][0] = moments[3] ;
   m[2][1] = moments[4] ;
   m[1][2] = moments[5] ;
   m[0][3] = moments[6] ;

   moments[0] = rotmom1 ( 2 , 0 , m , theta ) ;
   moments[1] = rotmom1 ( 1 , 1 , m , theta ) ;
   moments[2] = rotmom1 ( 0 , 2 , m , theta ) ;
   moments[3] = rotmom1 ( 3 , 0 , m , theta ) ;
   moments[4] = rotmom1 ( 2 , 1 , m , theta ) ;
   moments[5] = rotmom1 ( 1 , 2 , m , theta ) ;
   moments[6] = rotmom1 ( 0 , 3 , m , theta ) ;
}

int make_el ( double h , double v , double ccw , double egg , int n ,
              double *xpts , double *ypts )
{
   int ix, iy, nout ;
   double x, y, xr, yr, inc, c, s, d ;

   inc = 2.0 / (double) (n-1) ;
   nout = 0 ;
   c = cos ( ccw ) ;
   s = sin ( ccw ) ;

   for (ix=0 ; ix<n ; ix++) {
      x = -1.0 + ix * inc ;
      for (iy=0 ; iy<n ; iy++) {
         y = -1.0 + iy * inc ;
         xr = c * x  +  s * y ;
         yr = c * y  -  s * x ;  // Rotate CW
         d = (xr / h) * (xr / h) ;
         d += (yr / v) * (yr / v) ;
         d += egg * xr * yr * yr ;
         if (d <= 1.0) {
            xpts[nout] = x / inc ;
            ypts[nout++] = y / inc ;
            }
         }
      }
   return nout ;
}


void moments (
   int n ,        // Number of points
   double *xpts , // X coordinates
   double *ypts , // Y coordinates
   double *mom    // Moments: 20, 11, 02, 30, 21, 12, 03
   )
{
   int i ;
   double xm, ym, x, y, xsq, ysq, factor ;

   xm = ym = 0.0 ;
   for (i=0 ; i<n ; i++) {
      xm += xpts[i] ;
      ym += ypts[i] ;
      }
   xm /= (double) n ;
   ym /= (double) n ;

   for (i=0 ; i<7 ; i++)
      mom[i] = 0.0 ;

   for (i=0 ; i<n ; i++) {
      x = xpts[i] - xm ;
      y = ypts[i] - ym ;
      mom[0] += (xsq = x * x) ;
      mom[1] += x * y ;
      mom[2] += (ysq = y * y) ;
      mom[3] += xsq * x ;
      mom[4] += xsq * y ;
      mom[5] += x * ysq ;
      mom[6] += y * ysq ;
      }

   factor = 1.0 / ((double) n * (double) n) ;

   mom[0] *= factor ;
   mom[1] *= factor ;
   mom[2] *= factor ;

   factor /= sqrt ( (double) n ) ;

   mom[3] *= factor ;
   mom[4] *= factor ;
   mom[5] *= factor ;
   mom[6] *= factor ;
}

double root3 ( double x )
{
   if (x > 0.0)
      return pow ( x , 0.33333333333 ) ;
   else if (x < 0.0)
      return -pow ( -x , 0.33333333333 ) ;
   else
      return 0.0 ;
}

void eigen (
   double *moments ,  // Moments: 20, 11, 02, ...
   double *valmax ,   // Larger eignevalue
   double *valmin ,   // And smaller
   double vectmax[] , // (x, y) eigenvector corresponding to valmax
   double vectmin[]   // And valmin
   )
{
   double a, b, c, r, t1, t2 ;

   a = moments[0] ;   // m20
   b = moments[1] ;   // m11
   c = moments[2] ;   // m02

   r = sqrt ( (a - c) * (a - c)  +  4.0 * b * b ) ;

   *valmax = 0.5 * (a + c + r) ;
   *valmin = 0.5 * (a + c - r) ;

   t1 = a - c + r ;  // If the axes are nearly vertical / horizontal
   t2 = a - c - r ;  // one of these will be tiny, along with b

   if (fabs(t1) >= fabs(t2) ) {  // Avoid using the tiny one
      vectmax[0] = t1 ;          // Use the larger
      vectmax[1] = 2.0 * b ;     // And get the other
      vectmin[0] = 2.0 * b ;     // by orthogonality
      vectmin[1] = -t1 ;
      }
   else {
      vectmax[0] = 2.0 * b ;
      vectmax[1] = -t2 ;
      vectmin[0] = t2 ;
      vectmin[1] = 2.0 * b ;
      }
}


void vars ( double *m , double *momvars , double *huvars , double *othervars )
{
   double valmax, valmin, vectmax[2], vectmin[2], theta ;

   momvars[0] = 7.21 * log ( (m[0] + 0.02) / 0.0995775 ) ;  // m20
   momvars[1] = 20.0 * root3 ( m[1] ) ;                     // m11
   momvars[2] = 7.21 * log ( (m[2] + 0.02) / 0.0995775 ) ;  // m02
   momvars[3] = 20.0 * root3 ( m[3] ) ;                     // m30
   momvars[4] = 20.0 * root3 ( m[4] ) ;                     // m21
   momvars[5] = 20.0 * root3 ( m[5] ) ;                     // m12
   momvars[6] = 20.0 * root3 ( m[6] ) ;                     // m03

   huvars[0] = m[0] + m[2] ;
   huvars[1] = (m[0] - m[2]) * (m[0] - m[2])  + 4.0 * m[1] * m[1] ;
   huvars[2]= (m[3] - 3.0 * m[5]) * (m[3] - 3.0 * m[5]) +
              (m[6] - 3.0 * m[4]) * (m[6] - 3.0 * m[4]) ;
   huvars[3]= (m[3] + m[5]) * (m[3] + m[5])  +  (m[6] + m[4]) * (m[6] + m[4]) ;
   huvars[4]= (m[3] - 3.0 * m[5]) * (m[3] + m[5]) *
      ((m[3] + m[5]) * (m[3] + m[5])  -  3.0 * (m[6] + m[4]) * (m[6] + m[4]))
      + (3.0 * m[4] - m[6]) * (m[6] + m[4]) *
      (3.0 * (m[3] + m[5]) * (m[3] + m[5])  -  (m[6] + m[4]) * (m[6] + m[4])) ;
   huvars[5] = (m[0] - m[2]) * ((m[3] + m[5]) *
      (m[3] + m[5]) - (m[6] + m[4]) * (m[6] + m[4]))
      + 4.0 * m[1] * (m[3] + m[5]) * (m[6] + m[4]) ;
   huvars[6] = (3 * m[4] - m[6]) * (m[3] + m[5]) *
      ((m[3] + m[5]) * (m[3] + m[5])  -  3.0 * (m[6] + m[4]) * (m[6] + m[4]))
      + (3.0 * m[5] - m[3]) * (m[6] + m[4]) *
      (3.0 * (m[3] + m[5]) * (m[3] + m[5])  -  (m[6] + m[4]) * (m[6] + m[4])) ;

   huvars[0] = 10.0 * log ( 10.0 * (huvars[0] - 0.12) ) ;
   huvars[1] = 3.0 * log ( (huvars[1] + 0.0001) / .0101 ) ;
   huvars[2] = 8.0 * log ( (huvars[2] + 0.0005) / .0025 ) ;
   huvars[3] = 2.0 * log ( (huvars[3] + 0.0000001) / .0000101 ) ;
   huvars[4] = 1000.0 * root3 ( huvars[4] ) ;
   huvars[5] = 200.0 * root3 ( huvars[5] ) ;
   huvars[6] = 500.0 * root3 ( huvars[6] ) ;

   eigen ( m , &valmax , &valmin , vectmax , vectmin ) ;

   if (fabs (vectmax[0]) > 0.0)
      theta = atan ( vectmax[1] / vectmax[0] ) ;
   else 
      theta = PI / 2.0 ;

   othervars[0] = 6.3662 * theta ;                           // Theta
   othervars[1] = 10.0 * cos ( 2.0 * theta ) ;               // Cos
   othervars[2] = 10.0 * sin ( 2.0 * theta ) ;               // Sin
   othervars[3] = 2.0 * log ( valmax / valmin - 0.99999 ) ;  // Aspect
   othervars[4] = 2.0 * log ( m[0] / m[2] ) ;                // Fixed aspect
   othervars[5] = 10.0 * log ( (m[0] + m[2] - 0.158155) / 0.091845) ; // Spread

}

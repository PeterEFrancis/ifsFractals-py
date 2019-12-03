# Load External packages
import numpy as np                                      # for wicked fast arrays                                  MORE INFORMATION: https://docs.scipy.org/doc/
from numpy import *                                     # for ease of math function use
from numpy.linalg import *                              # for ease of linalg function use
from numba import njit                                  # for compiling functions into C, so they run faster.     MORE INFORMATION: http://numba.pydata.org/
import os                                               # for deep file manipulation
import math                                             # for math
from scipy.stats import linregress                      # for linear regressions





# internal dependencies
from ifsFractals.basic import *






## Word Fractals

def word_fractal(string):
    n = len(string)
    gap = 1/(n**2)
    width = (1 - (n-1)*gap)/n
    macros = [eval('_'+letter)(n) for letter in string.upper()]
    transformations = []
    for i in range(n):
        letter = macros[i]
        for t in range(len(letter)):
            transformations = transformations + [Translate(i*(width + gap),0) @ ScaleXY(width, 1/n) @ letter[t]]
    return transformations

def _(n):
    return []
def _a(n):
    theta = pi/2.3
    phi = pi/2 - theta
    h = (n-cos(theta))/(n*sin(theta))
    ell = (h*(n+1)*cos(theta))/(2*n-2*cos(theta))
    b = sin(theta)/n + ell
    X1 = h*cos(theta) + sin(theta)/n - cos(theta)/n
    Y1 = 1 - sin(theta)/n - cos(theta)/n
    c = (1 - 1/n - cos(theta)/(2*n) -X1)/sin(theta)
    return [Translate(sin(theta)/n,0) @ Rotate(theta) @ ScaleX(h),
            Translate(b,1/2-1/(2*n)) @ ScaleX(1-b-1/n),
            Translate(X1,Y1) @ Rotate(-phi) @ ScaleX(c),
            Translate(1-1/n,Y1-c*cos(theta)+sin(theta)/(2*n)) @ Rotate(-pi/2) @ ScaleX(Y1-c*cos(theta)+sin(theta)/(2*n))]
def _A(n):
    return [ShearY((1-1/n)*2) @ ScaleX(1/2),
            Translate(1/2, 1-1/n) @ ShearY((1/n-1)*2) @ ScaleX(1/2),
            Translate(n/(4*(n-1)),1/2-1/n) @ ScaleX(1 - n/(2*(n-1)))]
def _B(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n, 1-1/(2*n)) @ ScaleXY(1/2 - 1/(2*n),1/2),
            Translate(1/2 + 1/(2*n), 1/2+1/(4*n)) @ Rotate(pi/2) @ ScaleXY(1/2 - 3/(4*n),1/2),
            Translate(1/n,1/2 - 1/(4*n)) @ ScaleXY(1-2/n, 1/2),
            Translate(1-1/n, 1/2+1/(4*n)) @ Rotate(-pi/2) @ ScaleX(1/2 - 3/(4*n)),
            Translate(1/n,0) @ ScaleX(1 - 1/n)]
def _C(n):
    return [Translate(1/n,0) @ ScaleX(1-1/n),
            Translate(1/n,0) @ Rotate(pi/2),
            Translate((1/n),1-1/n) @ ScaleX(1-1/n)]
def _D(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n,1-1/n) @ ShearY(-1/2) @ ScaleX(1-2/n),
            Translate(1-1/n, 1/2+1/n) @ Rotate(-pi/2) @ ScaleX(1/2+1/n),
            Translate(1/n,0) @ ScaleX(1-2/n)]
def _E(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n,1-1/n) @ ScaleX(1-1/n),
            Translate(1/n,1/2-1/(4*n)) @ ScaleXY(1-2/n,1/2),
            Translate(1/n,0) @ ScaleX(1-1/n)]
def _F(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n,1-1/n) @ ScaleX(1-1/n),
            Translate(1/n,1/2-1/(4*n)) @ ScaleXY(1-2/n,1/2)]
def _G(n):
    return [Translate(1/n,0) @ ScaleX(1-1/n),
            Translate(1/n,0) @ Rotate(pi/2),
            Translate((1/n),1-1/n) @ ScaleX(1-1/n),
            Translate(1-1/n,1/2) @ Rotate(-pi/2) @ ScaleX(1/2 - 1/n),
            Translate(1-1/n-1/4,1/2 - 1/(2*n)) @ ScaleXY(1/4,1/2)]
def _H(n):
    return [Translate(1/n,0) @ Rotate(pi/2),
            Translate(1,0) @ Rotate(pi/2),
            Translate(1/n,1/2-1/(2*n)) @ ScaleX(1-2/n)]
def _I(n):
    return [Translate(0,1-1/n),
            Translate(1/2 - 1/(2*n),1-1/n) @ Rotate(-pi/2) @ ScaleX(1-2/n),
            Scale(1)]
def _J(n):
    return [Translate(0,1-1/n),
            Translate(1/2 - 1/(2*n),1-1/n) @ Rotate(-pi/2) @ ScaleX(1-2/n),
            ScaleX(1/2 + 1/(2*n))]
def _K(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n,1/2) @ ShearY((n/2 - 1)/(n-1)) @ ScaleX(1-1/n),
            Translate(1/n,1/2-1/n) @ ShearY((1 - n/2)/(n-1)) @ ScaleX(1-1/n)]
def _L(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n,0) @ ScaleX(1-1/n)]
def _M(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n,1-1/n) @ ShearY(-1) @ ScaleX(1/2 - 1/n),
            Translate(1/2,1/2) @ ShearY(1) @ ScaleX(1/2 - 1/n),
            Translate(1-1/n,1) @ Rotate(-pi/2)]
def _N(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n,1-1/n) @ ShearY((1/n-1)/(1-2/n)) @ ScaleX(1 - 2/n),
            Translate(1-1/n,1) @ Rotate(-pi/2)]
def _O(n):
    return [Translate(1/n,0) @ ScaleX(1-2/n),
            Translate(1,0) @ Rotate(pi/2),
            Translate(1/n,1-1/n) @ ScaleX(1-2/n),
            Translate(1/n,0) @ Rotate(pi/2)]
def _P(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n, 1-3/(4*n)) @ ScaleXY(1-2/n,3/4),
            Translate(1-1/n, 1) @ Rotate(-pi/2) @ ScaleX(1/2),
            Translate(1/n, 1/2) @ ScaleXY(1-2/n,3/4)]
def _Q(n):
    theta = arctan(1 - 2/n)
    z = (1/2 - 1/n - sin(theta)/n)*cos(theta)
    x = 1/2 - z*sin(theta) - cos(theta)/n
    y = x*tan(theta)
    return [Translate(0,1/2) @ ShearY(1-2/n) @ ScaleX(1/2),
            Translate(0,1/2-1/n) @ ShearY(2/n-1) @ ScaleX(1/2),
            Translate(1/2,0) @ ShearY(1-2/n) @ ScaleX(1/2),
            Translate(1/2,1-1/n) @ ShearY(2/n-1) @ ScaleX(1/2),
            Translate(1/2 + x,y) @ Rotate(theta-pi/2) @ ScaleXY(z,1/2)]
def _R(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n, 1-3/(4*n)) @ ScaleXY(1-2/n,3/4),
            Translate(1-1/n, 1) @ Rotate(-pi/2) @ ScaleX(1/2),
            Translate(1/n, 1/2) @ ScaleXY(1-2/n,3/4),
            Translate(1/n,1/2-1/n) @ ShearY((1/n-1/2)/(1-1/n)) @ ScaleX(1-1/n)]
def _S(n):
    return [ScaleX(1-1/n),
            Translate(1,0) @ Rotate(pi/2) @ ScaleX(1/2-1/(2*n)),
            Translate(1/n,1/2-1/(2*n)) @ ScaleX(1-1/n),
            Translate(1/n,1/2-1/(2*n)) @ ScaleY(1/2-1/(2*n)) @ Rotate(pi/2),
            Translate(0,1-1/n)]
def _T(n):
    return [Translate(0,1-1/n),
            Translate(1/2 - 1/(2*n),1-1/n) @ Rotate(-pi/2) @ ScaleX(1-1/n)]
def _U(n):
    return [Scale(1),
            Translate(0,1) @ Rotate(-pi/2) @ ScaleX(1-1/n),
            Translate(1-1/n,1) @ Rotate(-pi/2) @ ScaleX(1-1/n)]
def _V(n):
    return [Translate(0,1-1/n) @ ShearY(2/n-2) @ ScaleX(1/2),
            Translate(1/2,0) @ ShearY(2-2/n) @ ScaleX(1/2)]
def _W(n):
    return [Translate(0,1) @ Rotate(-pi/2),
            Translate(1/n,0) @ ShearY(1) @ ScaleX(1/2 - 1/n),
            Translate(1/2,1/2-1/n) @ ShearY(-1) @ ScaleX(1/2 - 1/n),
            Translate(1-1/n,1) @ Rotate(-pi/2)]
def _X(n):
    return [Translate(cos(pi/4)/n,0) @ Rotate(pi/4) @ ScaleX(sqrt(2)-1/n),
            Translate(0,1-cos(pi/4)/n) @ Rotate(-pi/4) @ ScaleX(cos(pi/4) - 1/n),
            Translate(1/2,1/2-cos(pi/4)/n) @ Rotate(-pi/4) @ ScaleX(cos(pi/4) - 1/n)]
def _Y(n):
    return [Translate(1/2-1/(2*n),1/2) @ Rotate(-pi/2) @ ScaleX(1/2),
            Translate(0,1-1/n) @ ShearY((1/2)/(1/(2*n) - 1/2)) @ ScaleX(1/2 - 1/(2*n)),
            Translate(1/2 + 1/(2*n),1/2 - 1/n) @ ShearY((1/2)/(1/2 - 1/(2*n))) @ ScaleX(1/2 - 1/(2*n))]
def _Z(n):
    return [Translate(0,1-1/n),
            Translate(1/n,1/n) @ Rotate(pi/2) @ ShearY(1/(1/n-1)) @ ScaleX(1-2/n),
            Scale(1)]

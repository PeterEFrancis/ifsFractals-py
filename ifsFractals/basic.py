# Load External packages
import numpy as np                                      # for wicked fast arrays                                  MORE INFORMATION: https://docs.scipy.org/doc/
from numpy import *                                     # for ease of math function use
from numpy.linalg import *                              # for ease of linalg function use
from numba import njit                                  # for compiling functions into C, so they run faster.     MORE INFORMATION: http://numba.pydata.org/
import random as rd                                     # for random numbers
from termcolor import colored                           # for colored print statements
import os                                               # for deep file manipulation
from typing import List                                 # for specification of types
import math                                             # for math
from scipy.stats import linregress                      # for linear regressions










## Built-In Transformations
def Scale(s):
    return np.array([[s, 0, 0],[0, s, 0],[0, 0, 1]], dtype=np.float64)
def Translate(a, b):
    return np.array([[1, 0, a],[0, 1, b],[0, 0, 1]], dtype=np.float64)
def Rotate(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]], dtype=np.float64)
def ShearX(t):
    return np.array([[1, t, 0],[0,1, 0],[0, 0, 1]], dtype=np.float64)
def ShearY(t):
    return np.array([[1, 0, 0],[t,1, 0],[0, 0, 1]], dtype=np.float64)
def ScaleX(s):
    return np.array([[s, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=np.float64)
def ScaleY(s):
    return np.array([[1, 0, 0],[0, s, 0],[0, 0, 1]], dtype=np.float64)
def ScaleXY(s, t):
    return np.array([[s, 0, 0],[0,t, 0],[0, 0, 1]], dtype=np.float64)




## Built-In Figures
Box = np.array([ [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.], [0., 0., 1.], [1/8, 1/8, 1.], [1/8-1/16, 1/8+1/16, 1.] ]).T
def rect(n):
    return ScaleY(1/n) @ Box
Line = np.array([ [0., 0., 1.], [1., 0., 1.] ]).T

XBox = np.array([ [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.], [0., 0., 1.], [0.5, 0., 1.], [0.5, 1., 1.], [1., 1., 1.], [1., 0.5, 1.], [0., 0.5, 1.], [0., 0., 1.], [1/8, 1/8, 1.], [1/8-1/16, 1/8+1/16, 1.]]).T









## Math Ops
@njit
def opNorm(A):
    G = A[:2].T[:2].T
    return np.sqrt(np.max(np.linalg.eig(G @ G.T)[0]))

def check_transformations(transformations, mode=''):
    if transformations is None:
        raise ValueError('ifsFractals: transformations cannot be NoneType.')
    failed = []
    for i in np.arange(len(transformations)):
        if opNorm(transformations[i]) >= 1:
            failed = failed + [i+1]
    if len(failed) == 0:
        if mode == 'pretty':
            print(colored('The opNorm of every transformation is less than 1 so all of the transformations are contraction mappings.','green'))
        elif mode == 'quiet':
            return True
        else:
            return 'The opNorm of every transformation is less than 1 so all of the transformations are contraction mappings.'
    elif len(failed) == 1:
        if mode == 'pretty':
            print(colored(f'The opNorm of transformation {failed[0]} is greater than or equal to 1 so is not a contraction mapping.','red'))
        elif mode == 'quiet':
            return False
        else:
            return f'The opNorm of transformation {failed[0]} is greater than or equal to 1 so is not a contraction mapping.'
    elif len(failed) > 1:
        if mode == 'pretty':
            print(colored(f'The opNorm of transformations {failed} are greater than or equal to 1 so are not contraction mappings.','red'))
        elif mode == 'quiet':
            return False
        else:
            return f'The opNorm of transformations {failed} are greater than or equal to 1 so are not contraction mappings.'

@njit
def choose_random_index(weights):
    r = rd.uniform(0, 1)
    t = 0
    sum = weights[0]
    while r > sum:
        t += 1
        sum = sum + weights[t]
    return min(t, len(weights)-1)

@njit
def make_eq_weights(n):
    return np.array([1/n]*n)

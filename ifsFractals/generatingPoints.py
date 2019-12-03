# Load External packages
import numpy as np                                      # for wicked fast arrays                                  MORE INFORMATION: https://docs.scipy.org/doc/
from numpy import *                                     # for ease of function use
from numpy.linalg import *                              # for ease of linalg function use
from numba import njit                                  # for compiling functions into C, so they run faster.     MORE INFORMATION: http://numba.pydata.org/
from numba import jit
import random as rd                                     # for random numbers




# internal dependencies
# from .basic import *






## for iterating points (and saving the array of points)
def generate_points(n, transformations, weights=np.array([0.]), startingPoint=np.array([0.,0.,1.]), frame=np.array([0.])):
    start = time.time()
    print(f'Generating {n} points...', end='')
    if all(frame == np.array([0.])):
        G = _generate_points_full(n, transformations, weights, startingPoint)
    else:
        startingPoint = np.array([(frame[1]-frame[0])/2,(frame[3]-frame[2])/2,1.])
        G = _generate_points_zoom(n, transformations, weights, startingPoint, frame)
    print(f' Finished in {time.time()-start} seconds.')
    return G

@njit
def _generate_points_full(n, transformations, weights=np.array([0.]), startingPoint=np.array([0.,0.,1.])):
    if all(weights == np.array([0.])):
        return _generate_points_full_simple(n, transformations, startingPoint)
    output = np.array([[startingPoint[0],startingPoint[1],1.]]*n)
    for i in range(100):
        output[0] = transformations[choose_random_index(weights)] @ output[0]
    for i in range(1,n):
        output[i] = transformations[choose_random_index(weights)] @ output[i-1]
    return (output[:,0],output[:,1])

@njit
def _generate_points_full_simple(n, transformations, startingPoint=np.array([0.,0.,1.])):
    output = np.array([[startingPoint[0],startingPoint[1],1.]]*n)
    for _ in np.arange(100):
        output[0] = transformations[rd.randint(0,len(transformations)-1)] @ output[0]
    for i in np.arange(1,n):
        output[i] = transformations[rd.randint(0,len(transformations)-1)] @ output[i-1]
    return (output[:,0], output[:,1])

@njit
def _generate_points_zoom(n, transformations, weights=np.array([0.]), startingPoint=np.array([0.,0.,1.]),  frame=np.array([0.])):
    if all(weights == np.array([0.])):
        return generate_points_zoom_simple(n, transformations, startingPoint, frame)
    if all(frame == np.array([0.])): # this is to keep the order of function inputs consistant
        raise ValueError('ifsFractals: _generate_points_zoom was not given a zoom frame.')
    output = np.array([[startingPoint[0],startingPoint[1],1]]*n, dtype=np.float64)
    for _ in range(20):
        potentialPoint = transformations[choose_random_index(weights)] @ outputFigures[i-1]
        while (frame[0]>potentialPoint[0]) or (frame[1]<potentialPoint[0]) or (frame[2]>potentialPoint[1]) or (frame[3]<potentialPoint[1]):
            potentialPoint = transformations[choose_random_index(weights)] @ potentialPoint
        output[0] = potentialPoint
    for i in np.arange(1,n):
        potentialPoint = transformations[choose_random_index(weights)] @ outputFigures[i-1]
        while (frame[0]>potentialPoint[0]) or (frame[1]<potentialPoint[0]) or (frame[2]>potentialPoint[1]) or (frame[3]<potentialPoint[1]):
            potentialPoint = transformations[choose_random_index(weights)] @ potentialPoint
        output[i] = potentialPoint
    return (output[:,0], output[:,1])

@njit
def _generate_points_zoom_simple(n, transformations, startingPoint, frame):
    outputFigures = np.array([[startingPoint[0],startingPoint[1],1.]]*n)
    for _ in range(20):
        potentialPoint = transformations[rd.randint(0,len(transformations)-1)] @ potentialPoint
        while (frame[0]>potentialPoint[0]) or (frame[1]<potentialPoint[0]) or (frame[2]>potentialPoint[1]) or (frame[3]<potentialPoint[1]):
            potentialPoint = transformations[rd.randint(0,len(transformations)-1)] @ potentialPoint
        output[0] = potentialPoint
    for i in np.arange(1,n):
        potentialPoint = transformations[rd.randint(0,len(transformations)-1)] @ potentialPoint
        while (frame[0]>potentialPoint[0]) or (frame[1]<potentialPoint[0]) or (frame[2]>potentialPoint[1]) or (frame[3]<potentialPoint[1]):
            potentialPoint = transformations[rd.randint(0,len(transformations)-1)] @ potentialPoint
        output[i] = potentialPoint
    return (output[:,0], output[:,1])


@njit
def find_bounds(transformations, weights=np.array([0.])):
    G = _generate_points_full(100000, transformations=transformations, weights=weights)
    xmin, xmax, ymin, ymax = np.min(G[0]), np.max(G[0]), np.min(G[1]), np.max(G[1])
    errX, errY = (xmax-xmin)/10, (ymax-ymin)/10
    return np.array([xmin-errX, xmax+errX, ymin-errY, ymax+errY], dtype=np.float64)

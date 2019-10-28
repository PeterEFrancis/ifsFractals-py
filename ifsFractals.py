 #  _  __     ______              _        _
 # (_)/ _|   |  ____|            | |      | |
 #  _| |_ ___| |__ _ __ __ _  ___| |_ __ _| |___
 # | |  _/ __|  __| '__/ _` |/ __| __/ _` | / __|
 # | | | \__ \ |  | | | (_| | (__| || (_| | \__ \
 # |_|_| |___/_|  |_|  \__,_|\___|\__\__,_|_|___/
 #



# Load External packages
import numpy as np                                      # for wicked fast arrays                                  MORE INFORMATION: https://docs.scipy.org/doc/
from numpy import *                                     # for ease of math function use
from numpy.linalg import *                              # for ease of linalg function use
from numba import njit                                  # for compiling functions into C, so they run faster.     MORE INFORMATION: http://numba.pydata.org/
from numba import jit
import matplotlib.pyplot as plt                         # for simple plotting
from matplotlib import collections as mc                # for simple plotting
import random                                           # for random numbers
import time                                             # for timing
from termcolor import colored                           # for colored print statements
import os                                               # for deep file manipulation
import imageio                                          # for gif making
from matplotlib.ticker import FormatStrFormatter        # for FormatStrFormatter
from typing import List                                 # for specification of types
from PIL import Image                                   # for faster images
import math                                             # for math
from scipy.stats import linregress                      # for linear regressions


# this should only be temporary
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


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
    r = random.uniform(0, 1)
    t = 0
    sum = weights[0]
    while r > sum:
        t += 1
        sum = sum + weights[t]
    return min(t, len(weights)-1)

@njit
def make_eq_weights(n):
    return np.array([1/n]*n)









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








## for Iterating Figures
def transform(figures, transformations):                                 # takes a list of 3xW numpy arrays and a list of transformations (2D 3x3 numpy arrays)
    newFigures = [np.array([[1.,0.,1.],[0.,1.,1.]]).T]
    for M in figures:
        for T in transformations:
            newFigures = newFigures + [T @ M]
    return newFigures[1:]                                                # returns the list containing each original 3xW numpy array multiplied by each transformation
def generate_figures(n, figures, transformations):                       # takes a natural number n, a list of 3xW numpy arrays and a list of transformations (2D 3x3 numpy arrays)
    if n == 0:
        return transform(figures, [np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])])
    else:
        outputFigures = transform(figures, transformations)
        for i in range(1,n):
            outputFigures = transform(outputFigures, transformations)
        return outputFigures                                             # returns the nth iteration of the transform(), a liste








## the Fractal class
class Fractal(object):
    def __init__(self, transformations, weights=np.array([0.]), size=10, color=(0,0,255)):
        if transformations is None:
            raise ValueError('ifsFractals: transformations cannot be NoneType.')
        self.transformations = transformations
        self.color = color
        if all(weights == np.array([0.])):
            self.weights = make_eq_weights(len(transformations))
        else:
            if len(weights) != len(transformations):
                raise ValueError('ifsFractals: Weights do not match the transformations.')
            if sum(weights) - 1 > .00001:
                raise ValueError('ifsFractals: Weights do not sum to 1.')
            self.weights = weights

        if size==0:
            raise ValueError('ifsFractals: size cannot be 0.')
        self.size = size
        self.xmin, self.xmax, self.ymin, self.ymax = find_bounds(self.transformations,self.weights)
        self.bounds = (self.xmin, self.xmax, self.ymin, self.ymax)
                               #   V    this or should be an and, but then I have to figure out how to deal with "flat" fractals
        if (self.xmax-self.xmin==0 or self.ymax-self.ymin==0) or not check_transformations(self.transformations, mode="quiet"):
            raise ValueError('ifsFractals: Fractal converges to insignificance or absurdity.')
        self.width = math.ceil((self.xmax-self.xmin)*36*self.size)
        self.height = math.ceil((self.ymax-self.ymin)*36*self.size)
        self.isZoomed = False

        self.point = np.array([0,0,1])
        self.developement = 0
        self.pic = Image.new('RGB', (self.width, self.height), (255, 255, 255))
        self.pixels = self.pic.load()

    def _scale(self, point):
        h = self.width * (point[0]-self.xmin)/(self.xmax-self.xmin)                 # (x + 2.182)*(self.width - 1)/4.8378                         Why take an input?
        k = self.height * (self.ymax-point[1])/(self.ymax-self.ymin)                # (9.9983 - y)*(self.height - 1)/9.9983
        return h, k

    def _iterate(self):
        r = random.uniform(0, 1)
        t = 0
        sum = self.weights[0]
        while r > sum:
            t += 1
            sum += self.weights[t]
        self.point = self.transformations[min(t, len(self.weights)-1)] @ self.point

    def add_points(self, n):
        for _ in range(100):
            self._iterate()
        for _ in range(n):
            self._iterate()
            self.pixels[self._scale(self.point)] = self.color
        self.developement += n

    def set_zoom(self, isZoomed=True, xmin=None, xmax=None, ymin=None, ymax=None):
        self.isZoomed = isZoomed
        if xmin is not None:
            self.xmin = xmin
        if xmax is not None:
            self.xmax = xmax
        if ymin is not None:
            self.ymin = ymin
        if ymax is not None:
            self.ymax = ymax

    def add_points_zoom(self, n, x, y):
        self.point = np.array([x,y,1])
        for _ in range(100):
            self._iterate()
            while (self.xmin>self.point[0]) or (self.xmax<self.point[0]) or (self.ymin>self.point[1]) or (self.ymax<self.point[1]):
                self._iterate()
        for _ in range(n):
            self._iterate()
            while (self.xmin>self.point[0]) or (self.xmax<self.point[0]) or (self.ymin>self.point[1]) or (self.ymax<self.point[1]):
                self._iterate()
            self.pixels[self._scale(self.point)] = self.color

    def make_gif(self, name='GIF', n=100_000, zoom=2, frames=7, zoomPoint=None):
        if not os.path.exists('Saved Fractals/For Zooms'):
            os.makedirs('Saved Fractals/For Zooms')
        start = time.time()
        print(f'Generating {frames} Zoomed images: ', end='')
        if zoomPoint is None:
            x, y = (self.xmax + self.xmin)/2, (self.ymax + self.ymin)/2
        else:
            x, y = zoomPoint
        for frame in range(frames+1):
            step = Fractal(self.transformations, self.weights, self.size, self.color)
            scale = 1 - ((zoom-1)/zoom)*sqrt(frame/frames)
            print(f'{frame}({round(1/scale, 2)})  ', end='')
            xminZ = x - (self.xmax - self.xmin) * scale / 2
            xmaxZ = x + (self.xmax - self.xmin) * scale / 2
            yminZ = y - (self.ymax - self.ymin) * scale / 2
            ymaxZ = y + (self.ymax - self.ymin) * scale / 2
            step.set_zoom(True, xminZ, xmaxZ, yminZ, ymaxZ)
            step.add_points_zoom(n, x, y)
            step.save_pic(f'Saved Fractals/For Zooms/{name}_{frame}.png')
        images = [imageio.imread(f'Saved Fractals/For Zooms/{name}_0.png')]*frames
        for frame in range(frames+1):
            images.append(imageio.imread(f'Saved Fractals/For Zooms/{name}_{frame}.png'))
            os.remove(f'Saved Fractals/For Zooms/{name}_{frame}.png')
        imageio.mimsave(f'Saved Fractals/{name}.gif', images)
        end = time.time()-start
        print(f'\nCompleted in {int(end//60)} minutes and {round(end%60)} seconds.')

    def load_in_points(self, externalTup, frame=None):
        n = len(externalTup[0])
        externalArray = np.array([*externalTup,[1.]*n]).T
        if frame is None:
            for row in externalArray:
                self.pixels[self._scale(row)] = self.color
            self.developement += n
        else:
            for row in externalArray:
                if (frame[0]<=row[0]) and (frame[1]>=row[0]) and (frame[2]<=row[1]) and (frame[3]>=row[1]):
                    self.pixels[self._scale(row)] = self.color
            self.developement += n

    def save_pic(self, path):
        self.pic.save(path)

    def display_pic(self):
        # https://stackoverflow.com/a/26649884
        plt.imshow(np.asarray(self.pic))

    def dark_pixels(self):
        return np.count_nonzero(255 - np.asarray(self.pic)) // 3

    def dimension(self, n=3_000_000, startSize=2, endSize=100, samples=None):
        start = time.time()
        test = Fractal(self.transformations, weights=self.weights, size = endSize, color=(0,0,0))
        G = _generate_points_full(n, self.transformations, weights=self.weights)
        test.load_in_points(G)
        tag = str(time.time())
        tag = tag[:10] + '_' + tag[11:]
        test.pic.save(f'DIMENSION_CALCULATION_MAX_SIZE_CHECK_{tag}.png')
        print(f'Verify that the image of the fractal with {n} points is \'filled in\' at size {endSize}. (Image file is in parent directory).')
        elapsed = time.time() - start
        go = input("Press Enter to continue or type stop.")
        middle = time.time()
        if (go == 'stop') or (go == 'STOP') or (go == 'Stop'):
            print('Dimension calculation terminated.')
            os.remove(f'DIMENSION_CALCULATION_MAX_SIZE_CHECK_{tag}.png')
            return None
        os.remove(f'DIMENSION_CALCULATION_MAX_SIZE_CHECK_{tag}.png')
        print('Computing fractal dimension ', end='')
        # find samples
        if samples is None:
            samples = endSize - startSize
            for j in range(endSize - startSize):
                samples -= j
                potentialArray = np.floor(np.geomspace(startSize, endSize, num=samples))
                if potentialArray[0] != potentialArray[1]:
                    break
        print(f'with {samples} samples...')
        numberOfPixelsDark = np.array([0.]*samples)
        numberOfPixelsDark[samples - 1] = test.dark_pixels()
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.geomspace.html#numpy.geomspace
        scalingFactor = np.floor(np.geomspace(startSize, endSize, num=samples))
        for i in range(samples-1):
            if scalingFactor[i] == scalingFactor[i-1]:
                numberOfPixelsDark[i] = numberOfPixelsDark[i-1]
                continue
            test = Fractal(self.transformations, weights=self.weights, size=scalingFactor[i], colour=(0,0,0))
            test.load_in_points(G)
            numberOfPixelsDark[i] = test.dark_pixels()
            if i == 0:
                print(f'Finished samples with sizes: {int(np.floor(startSize))}', end='')
            elif i == samples - 2:
                print(f', {int(scalingFactor[i])}, {int(scalingFactor[i+1])}.')
            else:
                print(f', {int(scalingFactor[i])}', end='')
        print(f'Completed in {np.round((elapsed + time.time() - middle)/60, decimals = 1)} minutes.')
        return np.round(linregress(np.log(scalingFactor), np.log(numberOfPixelsDark))[0], decimals = 2)




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
        output[0] = transformations[random.randint(0,len(transformations)-1)] @ output[0]
    for i in np.arange(1,n):
        output[i] = transformations[random.randint(0,len(transformations)-1)] @ output[i-1]
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
        potentialPoint = transformations[random.randint(0,len(transformations)-1)] @ potentialPoint
        while (frame[0]>potentialPoint[0]) or (frame[1]<potentialPoint[0]) or (frame[2]>potentialPoint[1]) or (frame[3]<potentialPoint[1]):
            potentialPoint = transformations[random.randint(0,len(transformations)-1)] @ potentialPoint
        output[0] = potentialPoint
    for i in np.arange(1,n):
        potentialPoint = transformations[random.randint(0,len(transformations)-1)] @ potentialPoint
        while (frame[0]>potentialPoint[0]) or (frame[1]<potentialPoint[0]) or (frame[2]>potentialPoint[1]) or (frame[3]<potentialPoint[1]):
            potentialPoint = transformations[random.randint(0,len(transformations)-1)] @ potentialPoint
        output[i] = potentialPoint
    return (output[:,0], output[:,1])


@njit
def find_bounds(transformations, weights=np.array([0.])):
    G = _generate_points_full(100000, transformations=transformations, weights=weights)
    xmin, xmax, ymin, ymax = np.min(G[0]), np.max(G[0]), np.min(G[1]), np.max(G[1])
    errX, errY = (xmax-xmin)/10, (ymax-ymin)/10
    return np.array([xmin-errX, xmax+errX, ymin-errY, ymax+errY], dtype=np.float64)


## Simple plots and saves
def plot_figures(figuresToPlot:List[np.ndarray], size:float=5, width:float=1 , colour:str='blue', path:str="TRASH.png"):
    lines = [ [(1.1, 1.1), (1.1, 1.1)] ]
    for figure in figuresToPlot:
        lines = lines + [[ (figure[0][i], figure[1][i]), (figure[0][i+1], figure[1][i+1]) ] for i in range(len(figure.T)-1)]
    lc = mc.LineCollection(lines[1:], linewidths=width, color=colour)
    fig, ax = plt.subplots(figsize=(size,size))
    ax.add_collection(lc)
    ax.set_aspect('equal')
    ax.autoscale()
    plt.savefig(path, bbox_inches='tight', dpi=144, format='png')
    plt.show()
def plot_points(points, size=5, colour="blue", path='TRASH.png'):
    fig, ax = plt.subplots(figsize=(size,size))
    ax.set_aspect('equal')
    ax.plot(*points, ',', color=colour)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=144, format='png')
    plt.axis('on')
#     num, KCnum, (X, Y) = pixel_data(path)
    plt.show()
#     print(f'{num} pixels are dark ({num*100/KCnum} percent)')
def save_points(points, size=5, colour="blue", path='TRASH.png'):
    if path == 'TRASH.png':
        print(colored('No Path Specified!','red'))
        return None
    fig, ax = plt.subplots(figsize=(size,size))
    ax.set_aspect('equal')
    ax.plot(*points, ',', color=colour)
    plt.axis('off')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # https://stackoverflow.com/a/29188910/6504760
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.savefig(path, bbox_inches='tight', dpi=144, format='png');
    fig.clear()
    plt.close(fig)




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

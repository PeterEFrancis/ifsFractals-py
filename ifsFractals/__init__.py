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
import matplotlib.pyplot as plt                         # for simple plotting
from matplotlib import collections as mc                # for simple plotting
import random as rd                                     # for random numbers
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






# import supporting files
from .figures import *
from .generatingPoints import *
from .wordFractals import *
from .basic import *





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
        r = rd.uniform(0, 1)
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

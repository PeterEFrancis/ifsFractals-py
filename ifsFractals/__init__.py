
 #  _  __     ______              _        _
 # (_)/ _|   |  ____|            | |      | |
 #  _| |_ ___| |__ _ __ __ _  ___| |_ __ _| |___
 # | |  _/ __|  __| '__/ _` |/ __| __/ _` | / __|
 # | | | \__ \ |  | | | (_| | (__| || (_| | \__ \
 # |_|_| |___/_|  |_|  \__,_|\___|\__\__,_|_|___/
 #

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import random as rd

from typing import List
from PIL import Image


from IPython.core.display import display, HTML
from IPython.display import IFrame



## Built-In Transformations
def Scale(s:float):
    return np.array([[s, 0, 0],[0, s, 0],[0, 0, 1]], dtype=np.float64)
def Translate(a:float, b:float):
    return np.array([[1, 0, a],[0, 1, b],[0, 0, 1]], dtype=np.float64)
def Rotate(theta:float):
    return np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]], dtype=np.float64)
def ShearX(t:float):
    return np.array([[1, t, 0],[0,1, 0],[0, 0, 1]], dtype=np.float64)
def ShearY(t:float):
    return np.array([[1, 0, 0],[t,1, 0],[0, 0, 1]], dtype=np.float64)
def ScaleX(s:float):
    return np.array([[s, 0, 0],[0, 1, 0],[0, 0, 1]], dtype=np.float64)
def ScaleY(s:float):
    return np.array([[1, 0, 0],[0, s, 0],[0, 0, 1]], dtype=np.float64)
def ScaleXY(s:float, t:float):
    return np.array([[s, 0, 0],[0,t, 0],[0, 0, 1]], dtype=np.float64)







## Built-In Figures
Box = np.array([ [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.], [0., 0., 1.], [1/8, 1/8, 1.], [1/8-1/16, 1/8+1/16, 1.] ]).T
def rect(n):
    return ScaleY(1/n) @ Box
Line = np.array([ [0., 0., 1.], [1., 0., 1.] ]).T

XBox = np.array([ [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.], [0., 0., 1.], [0.5, 0., 1.], [0.5, 1., 1.], [1., 1., 1.], [1., 0.5, 1.], [0., 0.5, 1.], [0., 0., 1.], [1/8, 1/8, 1.], [1/8-1/16, 1/8+1/16, 1.]]).T







def mprint(mat, leftspace=1, fmt="g"):
    # To display matrices. Slightly adapted from
    # https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75.
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        print(''.ljust(leftspace), end="|  ")#("                         | ", end="")
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("|")







def choose_random_index(weights):
    r = rd.uniform(0, 1)
    t = 0
    s = weights[0]
    while r > s:
        t += 1
        s = s + weights[t]
    return min(t, len(weights) - 1)





def opNorm(A):
    G = array(A[:2,:2], dtype=float)
    return sqrt(max(linalg.eig(G @ G.T)[0]))

def check_transformations(transformations, verbose=False):
    assert transformations is not None
    failed = []
    for i in np.arange(len(transformations)):
        if opNorm(transformations[i]) >= 1:
            failed = failed + [i+1]
    if len(failed) == 0:
        if verbose:
            print('The opNorm of every transformation is less than 1 so all of the transformations are contraction mappings.')
        else:
            return True
    elif len(failed) == 1:
        if verbose:
            print(f'The opNorm of transformation {failed[0]} is greater than or equal to 1 so is not a contraction mapping.')
        else:
            return False
    elif len(failed) > 1:
        if verbose:
            print(f'The opNorm of transformations {failed} are greater than or equal to 1 so are not contraction mappings.')
        else:
            return False




## figures

def transform(figures, transformations):
    new_figures = [np.array([[1,0],[0,1],[1,1]], dtype=np.float64)]
    for M in figures:
        for T in transformations:
            new_figures = new_figures + [T @ M]
    return new_figures[1:]

def generate_figures(n, figures, transformations):
    output_figures = transform(figures, transformations)
    for i in np.arange(1,n):
        output_figures = transform(output_figures, transformations)
    return output_figures

def plot_figures(figures:List[np.ndarray], size:int=4, width:float=1.5, color:str='blue'):
    lines = [ [(1.1, 1.1), (1.1, 1.1)] ]
    for M in figures:
        lines = lines + [[ (M[0][i], M[1][i]), (M[0][i+1], M[1][i+1]) ] for i in np.arange(len(np.transpose(M))-1)]
    lc = mc.LineCollection(lines[1:], linewidths=width, color=color)
    fig, ax = plt.subplots(figsize=(size,size))
    ax.add_collection(lc)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.plot()
    return







## points

def generate_points(n, transformations, weights=None, head_start=100):
    if weights is None:
        weights = np.array([1] * len(transformations))
    weights = weights / sum(weights)
    start = np.array([0, 0, 1.0])
    output = np.array([[start[0], start[1], 1.0]]*n)
    for i in range(head_start): # do `head_start` iterations to start converging
        output[0] = transformations[choose_random_index(weights)] @ output[0]
    for i in range(1,n):
        output[i] = transformations[choose_random_index(weights)] @ output[i-1]
    return (output[:,0], output[:,1])

def plot_points(points, size=6):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_aspect('equal')
    ax.plot(*points, '.', ms=0.75)





def find_bounds(transformations, weights=None):
    G = generate_points(1000, transformations=transformations, weights=weights)
    xmin, xmax, ymin, ymax = np.min(G[0]), np.max(G[0]), np.min(G[1]), np.max(G[1])
    errX, errY = (xmax-xmin)/10, (ymax-ymin)/10
    return np.array([xmin-errX, xmax+errX, ymin-errY, ymax+errY], dtype=np.float64)







## the Fractal class
class Fractal(object):
    def __init__(self, transformations, weights=None, size=10, color=(0,0,255)):

        assert transformations is not None
        self.transformations = transformations

        self.color = color

        self.set_weights(weights)

        assert size != 0
        self.size = size

        self.xmin, self.xmax, self.ymin, self.ymax = find_bounds(self.transformations, self.weights)
        self.bounds = (self.xmin, self.xmax, self.ymin, self.ymax)
                               #   V    this or should be an and, but then I have to figure out how to deal with "flat" fractals
        if (self.xmax-self.xmin==0 or self.ymax-self.ymin==0) or not check_transformations(self.transformations):
            raise ValueError('Fractal converges to insignificance or absurdity.')
        self.width = math.ceil((self.xmax-self.xmin)*36*self.size)
        self.height = math.ceil((self.ymax-self.ymin)*36*self.size)

        self.point = np.array([0,0,1])
        self.developement = 0
        self.pic = Image.new('RGB', (self.width, self.height), (255, 255, 255))
        self.pixels = self.pic.load()

    def set_weights(self, weights=None):
        if weights is None:
            weights = np.array([1] * len(self.transformations))
        assert len(weights) == len(self.transformations)
        assert all([i >= 0 for i in weights])
        assert any([i > 0 for i in weights])
        self.weights = weights / sum(weights)

    def _scale(self, point):
        h = self.width * (point[0]-self.xmin)/(self.xmax-self.xmin)                 # (x + 2.182)*(self.width - 1)/4.8378                         Why take an input?
        k = self.height * (self.ymax-point[1])/(self.ymax-self.ymin)                # (9.9983 - y)*(self.height - 1)/9.9983
        return h, k

    def plot_figures(self, depth=1, initial=Box, size:int=4, width:float=1.5, color:str='blue'):
        figures = generate_figures(depth, [initial], self.transformations)
        plot_figures(figures, size=size, width=width, color=color)

    def check_transformations(self, verbose=False):
        return check_transformations(self.transformations, verbose)

    def iterate(self):
        self.point = self.transformations[choose_random_index(self.weights)] @ self.point
        return

    def add_points(self, n):
        for _ in range(n):
            self.iterate()
            self.pixels[self._scale(self.point)] = self.color
        self.developement += n

    def load_in_points(self, externalTup):
        n = len(externalTup[0])
        externalArray = np.array([*externalTup,[1.]*n]).T
        for row in externalArray:
            self.pixels[self._scale(row)] = self.color
        self.developement += n

    def save(self, path):
        self.pic.save(path)

    def display(self):
        # https://stackoverflow.com/a/26649884
        plt.imshow(np.asarray(self.pic))

    def export(self):
        for T in self.transformations:
            return {
                'transformations': T.tolist(),
                'weights': T.weights
            }

    def link(self):
        matrices = [(np.round(T * 1000) / 1000).tolist() for T in self.transformations]
        link = 'https://ifs-fractals.peterefrancis.com/playground/t='
        for m in matrices:
            link += f'M({m[0][0]},{m[0][1]},{m[1][0]},{m[1][1]},{m[0][2]},{m[1][2]})&'
        return link[:-1] + "/w=" + ",".join([str(el) for el in self.weights])

    def link_web(self):
        display(HTML("<a href='" + self.link() + "' target='_blank '>Click me to open in IFS Fractals</a>"))

    def embed_web(self):
        display(IFrame(self.link(), width='100%', height='1000px'))

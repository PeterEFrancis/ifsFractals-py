# Load External packages
import numpy as np                                      # for wicked fast arrays                                  MORE INFORMATION: https://docs.scipy.org/doc/
from numpy import *                                     # for ease of math function use
from numpy.linalg import *                              # for ease of linalg function use
from numba import njit                                  # for compiling functions into C, so they run faster.     MORE INFORMATION: http://numba.pydata.org/
import matplotlib.pyplot as plt                         # for simple plotting
from matplotlib import collections as mc                # for simple plotting
import random as rd                                     # for random numbers
import time                                             # for timing
from matplotlib.ticker import FormatStrFormatter        # for FormatStrFormatter
from typing import List                                 # for specification of types










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





def plot_figures(figuresToPlot:List[np.ndarray], size:float=5, width:float=1 , colour:str='blue', saveAs:str="_"):
    lines = [ [(1.1, 1.1), (1.1, 1.1)] ]
    for figure in figuresToPlot:
        lines = lines + [[ (figure[0][i], figure[1][i]), (figure[0][i+1], figure[1][i+1]) ] for i in range(len(figure.T)-1)]
    lc = mc.LineCollection(lines[1:], linewidths=width, color=colour)
    fig, ax = plt.subplots(figsize=(size,size))
    ax.add_collection(lc)
    ax.set_aspect('equal')
    ax.autoscale()
    if saveAs != '_':
        plt.savefig(saveAs, bbox_inches='tight', dpi=144, format='png')
    plt.show()



















## these are fossils

# def plot_points(points, size=5, colour="blue", path='TRASH.png'):
#     fig, ax = plt.subplots(figsize=(size,size))
#     ax.set_aspect('equal')
#     ax.plot(*points, ',', color=colour)
#     plt.axis('off')
#     plt.savefig(path, bbox_inches='tight', dpi=144, format='png')
#     plt.axis('on')
##     num, KCnum, (X, Y) = pixel_data(path)
    # plt.show()
##     print(f'{num} pixels are dark ({num*100/KCnum} percent)')



# def save_points(points, size=5, colour="blue", path='TRASH.png'):
#     if path == 'TRASH.png':
#         print(colored('No Path Specified!','red'))
#         return None
#     fig, ax = plt.subplots(figsize=(size,size))
#     ax.set_aspect('equal')
#     ax.plot(*points, ',', color=colour)
#     plt.axis('off')
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # https://stackoverflow.com/a/29188910/6504760
#     ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
#     plt.savefig(path, bbox_inches='tight', dpi=144, format='png');
#     fig.clear()
#     plt.close(fig)

# Iterated Function System Fractal Generator

The fractal approximations that can be generated here are fixed points of <u>contraction mappings</u>, more specifically, sets of affine linear transformations $T:\mathbb{R}^2\to\mathbb{R}^2$, each of the form $T(\vec{x})=A\vec{x}+\vec{b}$, where $A$ is a $2\times2$ matrix and $\vec{x}$ and $\vec{b}$ are vectors in $\mathbb{R}^2$. Each transformation from $\mathbb{R}^2(\cong\mathbb{R}^2\times\{1\})$ to itself can be represented as a block $3\times 3$ matrix, in the form $$\begin{bmatrix}A& \vec{b}\\0 & 1\end{bmatrix}.$$ The composition of transformations corresponds to the multiplication of these block matrices.

**Method of Iterating Figures**: Given a set of transformations and a set containing one initial figure in $\mathbb{R}^2$, a fractal approximation can be obtained by repeatedly replacing the figures in the set with the result of applying each of the transformations to the figures of the set. Through an infinite number of iterations, the plot obtained by plotting the contents of the set of figures will approach the fractal. *Numerically, if there are $t$ transformations, after the $n$th iteration, there are $t^n$ figures in the figure set.*

**Method of Iterating Points**: Given a set of transformations and an initial point in $\mathbb{R}^2$, select a random transformation from the set, apply the transformation to the point, and plot the point. An approximation of a fractal can be obtained by repeating this process with the point that was previously plotted since successive points will be closer to be in the fractal. We can also introduce an array of weights that correspond to the probability of selecting of each transformation in the set. Different weights will producing differently shaded approximate images of the fractal. *Numerically, regardless of the number of transformations, after the $n$th iteration, there will be $n+1$ points plotted.*

While both methods will produce the fractal after infinite iterations, the method of iterating points is generally much faster.

***

## Using this IFS Generator

Import the generator library `ifsFractals`
> `from ifsFractals import *`

Define some contraction transformations as $3\times 3$ matrices (as numpy arrays) or compose some of the built-in transformations using numpy array multiplication `@`. The following are built-in transformations:
* `Scale(s)` : $(x,y)\to(sx,sy)$
* `Translate(h,k)`: $(x,y)\to(x+h,y+k)$
* `Rotate(theta)`: $(x,y)\to(x\cos\theta-y\sin\theta, x\sin\theta+y\cos\theta)$
* `ShearX(t)` : $(x,y)\to(x+ty,y)$
* `ShearY(t)` : $(x,y)\to(x,xt+y)$
* `ScaleX(s)` : $(x,y)\to(sx,y)$
* `ScaleY(s)` : $(x,y)\to(x,sy)$
* `ScaleXY(s,t)` : $(x,y)\to(sx,ty)$

Remember, the order that when composed, the transformations will be applied from right to left

> `T1 = np.array([[0.7, 0., 0.15], [0., 0.7, 0.3],[0., 0., 1.]])`<br>
> `T2 = Translate(0.35,0) @ Rotate(np.pi/4) @ Scale(0.35*np.sqrt(2))`<br>
> `T3 = Translate(0.3,0.35) @ Rotate(-np.pi/4) @ Scale(0.35*np.sqrt(2))`<br>
> `T4 = Translate(0.45,0.2) @ ScaleXY(1/10,3/10)`<br>

Create a list of the transformations.
> `T = [T1, T2, T3, T4]`

Use `check_transformations(transformations)` with `transformations` the list of transformations to verify that each of the transformations in the list is in fact a contraction mapping.

> `check_transformations(T, mode)`

If `mode` is set to `'pretty'`, will print the response in colors and if set to `'quiet'`, will return a boolean value, `True` or `False`

### I. Creating a Fractal Image

#### A. Iterating Figures

Define a $3$ by $w$ numpy array whose columns are points of the form $(x,y,1)$, where $(x,y)$ is in $\mathbb{R}^2$, and whose consecutive columns define line segments in an initial figure. The following figure is built-in.
> `Box = np.array([ [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [1/8, 1/8, 1], [1/8-1/16, 1/8+1/16, 1] ]).T`

`rect(n)` will return `ScaleY(1/n) @ Box`.

Use the function `generate_figures(n, figures, transformations)` with `n` the number of iterations, `figures` a list initial figures, and `transformations` the list of transformations, to generate a list of figures (numpy arrays).
> `figuresToPlot = generate_figures(5,[Box],T)`

Use the function `plot_figures(figuresToPlot, size, width , colour, path)` with `figuresToPlot` the figures to plot, and optionally the `size`, `width` of the line, `color` and `path`, to display and save the plot. The default value for `size` is 5, `width` is 1, `color` is `'blue'`, and `path` is `'TRASH.png'`.
> `plot_figures(figuresToPlot, size=10, width=.8, colour='red', path='Saved Fractals/leafFractal.png')`

#### B. Iterating Points

Optionally, define a 'weights' numpy array with the elements corresponding to the probability of selecting of each transformation in the list of transformations. Be sure that the elements of the weights array sum to one and is the same length as the list of transformations.
> `WT = np.array([0.64,0.12,0.12,0.12])`

If a function has a weights array as an input, and none is given, the default is an evenly distributed array with the same number of elements as the length of `transformations`. The function to create such an array is `make_eq_weights(n)`. For example,
> `make_eq_weights(4)`

returns `array([0.25, 0.25, 0.25, 0.25])`.

##### 1. Using the `Fractal` class

Create an instance of the `Fractal` class for the list of transformations, and optionally the `weights`, `size`, and `color`. The default is for evenly distributed weights, `size = 10` and `color='blue'`.
> `leafFractal = Fractal(T, weights=WT, size=10, color='green')`

To add points to the fractal image, implement `add_points(n)` with the `n` the number of points to be added. This does not save point coordinates once they have been plotted onto the image. (See **2. Pre-Plotting Calculation** for a method to save point coordinates).
> `leafFractal.add_points(500000)`

Optionally, add points only in a specific frame by implementing `add_points(n,frame)` with the `n` the number of points to be added and `frame=np.array([Xmin, Xmax, Ymin, Ymax])` a numpy array with the bounds of the frame.
> `leafFractal.add_points(500000, np.array([.3, .7, .2, .4]))`

To save the fractal image, implement `save_pic(path)` with `path` the path of the image to be saved.
> `leafFractal.save_pic(../leafFractal.png)`

To display a small scale of the fractal image, implement `display_pic()`.
> `leafFractal.display_pic()`

To access the current full-scale fractal image, call the `pic` attribute.
> `leafFractal.pic`

To see the developement of the fractal (how many points have been plotted) call the `developement` attribute.
> `leafFractal.developement`

To calculate the fractal dimension, implement `dimension(n, startSize, endSize, samples)` with `n` the number of points added, `startSize` the initial size and `endSize` the final size. The default values are `3_000_000`, `2`, and `100` respectively. Samples defaults to (the maximum number of) exponentially-spaced integers between `startSize` and `endSize` including `startSize + 1`. For more information on fractal dimension see 3b1b video on fractals [here](https://www.youtube.com/watch?v=gB9n2gHsHN4).

To return the number of dark pixels in the fractal image, implement `dark_pixels()`.
> `leafFractal.dark_pixels()`

##### 2. Pre-Plotting Calculation

If needed it is possible to calculate and save the coordinates of points before plotting them. This might be helpful if a very large number of points is being computed but an optimal size to plot them is unknown. However, this requires more memory to be used. To do this, use `generate_points(n, transformations, startingPoint, weights, frame)`, where `startingPoint`, `weights`, and `frame` are optional. `startingPoint` must be a numpy array of the form `np.array([x,y,1])` where $(x,y)$ is in $\mathbb{R}^2$. This function returns a 2-tuple with the x and y coordinates of the `n` points generated.
> `points = generate_points(100000, T, startingPoint=np.array([1.,1.2,1.]), weights=WT, frame=np.array([.3, .7, .2, .4]))`

To load the points into an instance of the `Fractal` class, use `load_in_points(externalArray, frame)`, where `externalArray` is the array of points previously generated and `frame` is optional.
> `leafFractal.load_in_points(points, frame=np.array([.3, .7, .2, .4]))`

To display the points generated without using a `Fractal`, use `plot_points(points, size, colour, path)` similar to `plot_figures`. This will display and save the image.
> `plot_points(points, size = 15, colour='green', path='Saved Fractals/leafFractal.png')`

To save the plot of points without displaying it, use `save_points(points, size, colour, path)` similar to `plot_points`.
> `save_points(points, size = 15, colour='green', path='Saved Fractals/leafFractal.png')`

### II. Application Projects

#### A. Word Fractals
Use `word_fractal(string)` to create a list of transformations for a word fractal of `string`. Note that `string` will default to uppercase.
> `T = word_fractal('NAME')`<br>
> `Name = Fractal(T)`

#### B. Creating a Zoom GIF
Create an instance of the `Fractal` class. Use `make_gif(name, n, zoom, frames, zoomPoint)` with (optionally) `name` the filename of the GIF file that will be saved, `n` the number of points in each frame of the GIF, `zoom` the maximum zoom level, `frames` the number of frames in the GIF (note that the GIF will have 10 fps), and `zoomPoint` the point on the fractal that will be zoomed into. The default value of `name` is `'GIF'`, `n` is `100_000`, `zoom` is `2`, and `frames` is `7`. `zoomPoint` will default to the center of the fractal image. Make sure there is a folder 'Saved Fractals' with a subfolder 'For Zoom' in the same directory as 'IFSFGL.py'
> `leafFractal.make_gif(name='GIF', n=100_000, zoom=2, frames=7, zoomPoint=None)`


<!-- ### III. Image Analysis


### IV. Random Fractal Generation -->

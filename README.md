# ifsFractal

A python module for fast Iterated Function System Fractal generation

Also, you can use some `ifsFractlas` functionality in your browser!

To install, use pip:

>	`pip install ifsFractals --upgrade`

If you are trying to use `ifsFractals` in CoCalc or get an error about a read-only file system, use

> `python3 -m pip install --user ifsFractals --upgrade`

See the [usage docs](https://share.cocalc.com/share/f6aee960-eb32-4259-a1ff-cddafc5864a0/Personal/ifsFractal%20Docs/usage.md?viewer=embed) for included functions and classes.

Example usage:

>    `from ifsFractals import *`
>
>    `C1 = Translate(0.5,0.15) @ Rotate(np.pi/4) @ Scale(1/4)`
>    `C2 = Scale(1/2)`
>    `C3 = Translate(-.5,0.15) @ Rotate(-np.pi/4) @ Scale(1/4)`
>    `C4 = Rotate(-np.pi/6) @ Scale(1/2)`
>    `C5 = Rotate(np.pi/6) @ Scale(1/2)`
>
>    `CT = [C1, C2, C3, C4, C5]`
>
>    `crab = Fractal(CT)`
>
>    `crab.add_points(1_000_000)`
>
>    `crab.save_pic('/Path/to/desired/location/image_name.png')`

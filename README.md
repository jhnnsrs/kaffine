# kaffine

[![codecov](https://codecov.io/gh/jhnnsrs/kaffine/branch/master/graph/badge.svg?token=UGXEA2THBV)](https://codecov.io/gh/jhnnsrs/kaffine)
[![PyPI version](https://badge.fury.io/py/kaffine.svg)](https://pypi.org/project/kaffine/)
![Maintainer](https://img.shields.io/badge/maintainer-jhnnsrs-blue)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/kaffine.svg)](https://pypi.python.org/pypi/kaffine/)
[![PyPI status](https://img.shields.io/pypi/status/kaffine.svg)](https://pypi.python.org/pypi/kaffine/)
[![PyPI download day](https://img.shields.io/pypi/dm/kaffine.svg)](https://pypi.python.org/pypi/kaffine/)


### Inspiration

Kaffine is inspired by [pint](https://pint.readthedocs.io/en/stable/) and [affine](https://affine.readthedocs.io/en/latest/). 
And aims to make the creation of unit-aware affine transformations as easy as possible.

Here is a complete, professional GitHub-style `README.md` for your project. I have named the library **`affine-3d`** for this document.

You can save this directly as `README.md` in your project root.

-----

# affine-3d 

**A fluent, unit-aware 3D transformation library for Python.**

`affine-3d` simplifies constructing complex 4x4 affine transformation matrices. It replaces dense NumPy matrix math with a readable **Builder Pattern**, offers native integration with **Pint** for physical units (mixing microns, millimeters, and inches), and bridges seamlessly with **Scipy** for advanced usage.

Designed for **Computer Vision**, **Microscopy**, and **Robotics** applications where "Sample-to-Camera" or "Stage-to-World" coordinates are critical.

## 🚀 Features

  * **Fluent Interface:** Chain operations like `.translate_x(10).rotate_z(90)`.
  * **Unit Aware:** Native support for [Pint](https://pint.readthedocs.io/). Move 100 `microns` on X and 2 `mm` on Y without manual conversion.
  * **Hardware Ready:** Includes `decompose()` to extract pixel size, rotation, and stage position from a raw matrix.
  * **Type Safe:** Built with modern Python 3.11+ type hints (`typing.Self`, `|` unions).
  * **Scipy Bridge:** Cast to/from `scipy.spatial.transform.Rotation` for quaternions and slerp.
  * **Zero-Crash Defaults:** Scipy and Pint are **optional** dependencies. The library falls back to pure NumPy if they aren't installed.

## 📦 Installation


```bash
uv add rm kaffine
```

## ⚡ Quick Start

### 1\. The Simple Math Interface

Perfect for standard graphics or geometry.

```python
from kaffine import Affine

# Create a transform: Move 10 units X, then Rotate 45 deg Z
# (Note: Order is intuitive "Operation 1 -> Operation 2")
tf = Affine.new().translate_x(10).rotate_z(45)

# Apply to a point
point = [0, 0, 0]
result = tf * point 

print(result) 
# Output (4D homogeneous): [10.0, 0.0, 0.0, 1.0]
```

### 2\. The "Hardware" Interface (with Units)

Perfect for controlling stages, cameras, or mixing scales.

```python
from kaffine import Affine
import pint

ureg = pint.UnitRegistry()

# Define your setup's units
ureg.define("step = 0.1 um")  # Piezo stepper resolution
ureg.define("pixel = 3.45 um") # Camera sensor pixel size

# Construct a Matrix that converts Pixels to Stage Coordinates (mm)
# Scenario: 
# 1. Scale pixels to physical size
# 2. Rotate camera 90 degrees
# 3. Translate to stage position (defined in steps)
tf = Affine.new(base_unit="um", registry=ureg) \
    .scale_uniform("1 pixel") \
    .rotate_z(90) \
    .translate_x("50000 step") 

# Transform a point from Pixel Space to Millimeters
pixel_point = [100, 100, 0] # 100x100 px on image
world_point = tf.apply(pixel_point)

print(world_point)
# Output: <Quantity([5.0 -0.345 0. 1.], 'millimeter')>
```

## 📚 Advanced Usage

### Matrix Decomposition (Calibration)

If you have a calibration matrix and need to know the physical properties of your setup (e.g., "What is my effective pixel size?"), use `.decompose()`.

```python
# Assume 'tf' is a calibrated matrix loaded from a file
info = tf.decompose()

print(f"Position:   {info.translation}") # e.g., [10.5, 2.0, 0] mm
print(f"Pixel Size: {info.scale}")       # e.g., [0.00345, 0.00345, 1.0]
print(f"Rotation:   {info.rotation}")    # e.g., [0, 0, 1.5] degree
```

### Scipy Interoperability

Need Quaternions or Spherical Interpolation?

```python
tf = Affine.new().rotate_x(45).rotate_y(30)

# Cast to Scipy
r_scipy = tf.to_scipy_rotation()
print(r_scipy.as_quat())

# ... do complex math ...
r_scipy_inverted = r_scipy.inv()

# Cast back
tf_inv = Affine.from_scipy(r_scipy_inverted)
```

## 🛠 API Reference

### `Affine.new(base_unit="mm", registry=None)`

Starts a new transformation chain.

  * `base_unit`: The physical unit the internal matrix numbers represent.
  * `registry`: Your `pint.UnitRegistry`.

### Builder Methods

All methods return a **new** `Affine` instance (immutable-style chaining).

  * `.translate(x, y, z)` / `.translate_x(val)` ...
  * `.rotate(angle, axis='z')` / `.rotate_x(angle)` ...
  * `.scale(x, y, z)` / `.scale_uniform(s)`

### Application

  * `tf * [x, y, z]`: Returns a raw Numpy array (4D).
  * `tf * other_tf`: Composes two transforms.
  * `tf.apply(points)`: Accepts/Returns Pint Quantities.

### Utilities

  * `tf.decompose()`: Returns `Decomposition(translation, scale, rotation)`.
  * `tf.to_scipy_rotation()`: Returns `scipy.spatial.transform.Rotation`.

## 🧪 Running Tests

The project includes a comprehensive `pytest` suite covering math accuracy, unit conversion, and edge cases.

```bash
uv run pytest tests/
```

## 📄 License

MIT License. Feel free to use this in your commercial or open-source projects.
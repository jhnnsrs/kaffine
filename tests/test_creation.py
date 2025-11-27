import pytest
import numpy as np
import pint
from kaffine import Affine, HAS_SCIPY

@pytest.fixture
def ureg():
    return pint.UnitRegistry()

def test_return_shape_is_4d():
    """Ensure output is always [x, y, z, w]."""
    a = Affine.new().translate_x(10)
    
    # Input 3D
    point = [0, 0, 0]
    res = a * point
    
    # Check Shape
    assert res.shape == (4,)
    # Check Value: (10, 0, 0, 1)
    np.testing.assert_array_equal(res, [10.0, 0.0, 0.0, 1.0])

def test_batch_processing_4d():
    """Ensure N points return (N, 4)."""
    a = Affine.new().translate_z(5)
    points = [
        [0, 0, 0],
        [1, 1, 1]
    ]
    res = a * points
    
    assert res.shape == (2, 4)
    expected = [
        [0.0, 0.0, 5.0, 1.0],
        [1.0, 1.0, 6.0, 1.0]
    ]
    np.testing.assert_array_equal(res, expected)

def test_input_can_be_4d():
    """Ensure we can pass 4D points in manually."""
    a = Affine.new().translate_x(10)
    # Pass in x=0, y=0, z=0, w=1
    point_4d = [0, 0, 0, 1]
    
    res = a * point_4d
    np.testing.assert_array_equal(res, [10.0, 0.0, 0.0, 1.0])

def test_pint_returns_4d_quantity(ureg):
    """Pint .apply() should return a Quantity with shape (4,)."""
    a = Affine.new(base_unit="mm", registry=ureg).translate_x("10 mm")
    
    # Input 3D list of Quantities
    pt = [0*ureg.mm, 0*ureg.mm, 0*ureg.mm]
    
    res = a.apply(pt)
    
    assert isinstance(res, ureg.Quantity)
    assert res.shape == (4,)
    assert res.units == ureg.mm
    
    # Note: The 4th element will be 1.0 * mm.
    # While physically odd, this is the standard representation for 
    # homogeneous arrays in Pint.
    expected_mag = [10.0, 0.0, 0.0, 1.0]
    np.testing.assert_array_equal(res.magnitude, expected_mag)

@pytest.mark.skipif(not HAS_SCIPY, reason="Scipy not installed")
def test_scipy_integration_preserves_4d_logic():
    a = Affine.new().rotate_z(90)
    point = [1, 0, 0]
    
    res = a * point
    
    # Rotated 90 deg Z: (0, 1, 0, 1)
    np.testing.assert_allclose(res, [0, 1, 0, 1], atol=1e-7)
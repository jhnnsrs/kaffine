import pytest
import numpy as np
import pint
from kaffine import Affine, HAS_SCIPY

@pytest.fixture
def ureg():
    return pint.UnitRegistry()

def test_decompose_pixel_size(ureg):
    """Test extracting pixel size (scale) from a matrix."""
    # Pixel size = 0.5 um per pixel
    # World units = um
    a = Affine.new(base_unit="um", registry=ureg).scale_uniform(0.5)
    
    decomp = a.decompose()
    
    # Check scale
    np.testing.assert_allclose(decomp.scale, [0.5, 0.5, 0.5])

def test_decompose_full_transform(ureg):
    """Test extracting translation, rotation, and scale simultaneously."""
    # 1. Translate 10mm
    # 2. Rotate 90 deg Z
    # 3. Scale 2.0
    a = Affine.new(base_unit="mm", registry=ureg) \
        .scale(2.0) \
        .rotate_z(90) \
        .translate_x("10 mm")
    
    decomp = a.decompose()
    
    # Check Translation
    expected_trans = [10.0, 0.0, 0.0]
    np.testing.assert_allclose(decomp.translation.magnitude, expected_trans)
    assert decomp.translation.units == ureg.mm
    
    # Check Scale
    np.testing.assert_allclose(decomp.scale, [2.0, 2.0, 2.0])
    
    # Check Rotation
    expected_rot = [0, 0, 90]
    np.testing.assert_allclose(decomp.rotation.magnitude, expected_rot, atol=1e-5)
    assert decomp.rotation.units == ureg.degree
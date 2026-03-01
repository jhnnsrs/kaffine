from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Any, Literal, NamedTuple, overload, TYPE_CHECKING

# Use Self from typing if available, otherwise fallback for older Python versions
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:
    from typing_extensions import TypeAlias

# --- Dependencies ---
HAS_SCIPY = False
try:
    from scipy.spatial.transform import Rotation as R

    HAS_SCIPY = True
except ImportError:
    R = None  # type: ignore

HAS_PINT = False
try:
    import pint

    HAS_PINT = True
except ImportError:
    pint = None  # type: ignore


# --- Type Definitions ---
if TYPE_CHECKING:
    # These imports only happen during static type checking
    from scipy.spatial.transform import Rotation
    from pint import UnitRegistry

    RotationType: TypeAlias = Rotation
    RegistryType: TypeAlias = UnitRegistry[Any]
    QuantityType: TypeAlias = int | float
else:
    # Runtime fallbacks to prevent errors if the libraries are missing
    RotationType = Any
    RegistryType = Any
    QuantityType = Any

MetricValue: TypeAlias = float | int | str
Vec3or4: TypeAlias = npt.NDArray[np.float64] | list[float] | tuple[float, ...]
Matrix4x4: TypeAlias = npt.NDArray[np.float64]


# --- Output Data Structure ---
class Decomposition(NamedTuple):
    translation: QuantityType  # Vector [x, y, z] in base_units
    scale: list[float]  # Vector [sx, sy, sz] (Pixel Size)
    rotation: QuantityType  # Vector [rx, ry, rz] in degrees

    def __repr__(self) -> str:
        # Custom pretty print
        return (
            f"Decomposition(\n"
            f"  translation={self.translation},\n"
            f"  scale={self.scale} (pixel size),\n"
            f"  rotation={self.rotation}\n"
            f")"
        )


class Affine:
    _default_registry: RegistryType | None = None

    def __init__(
        self,
        matrix: Matrix4x4 | list[list[float]] | None = None,
        base_unit: str = "µm",
        registry: RegistryType | None = None,
    ) -> None:
        if matrix is None:
            self.mat: Matrix4x4 = np.eye(4, dtype=np.float64)
        else:
            self.mat = np.array(matrix, dtype=np.float64)

        self.base_unit = base_unit

        # Registry Logic
        self.ureg: RegistryType | None
        if registry is not None:
            self.ureg = registry
        else:
            if not HAS_PINT:
                self.ureg = None
            else:
                if Affine._default_registry is None:
                    # pylint: disable=import-outside-toplevel
                    import pint

                    Affine._default_registry = pint.UnitRegistry()
                self.ureg = Affine._default_registry

    def __repr__(self) -> str:
        return (
            f"Affine(base_unit='{self.base_unit}')\n"
            f"{np.array2string(self.mat, precision=3, suppress_small=True)}"
        )

    # ==========================================
    # Decompose (The Reverse Operation)
    # ==========================================
    def decompose(self) -> Decomposition:
        """
        Extracts Translation, Scale (Pixel Size), and Rotation from the matrix.
        Returns Pint Quantities.
        """
        if self.ureg is None:
            raise ImportError("Pint required to return Quantities.")

        # 1. Extract Translation (Last Column)
        trans_vec = self.mat[:3, 3]
        trans_quant = trans_vec * self.ureg(self.base_unit)

        # 2. Extract Scale (Norm of the column vectors of the 3x3 block)
        sx = np.linalg.norm(self.mat[:3, 0])
        sy = np.linalg.norm(self.mat[:3, 1])
        sz = np.linalg.norm(self.mat[:3, 2])

        # Detect negative scale (flipping) via determinant
        det = float(np.linalg.det(self.mat[:3, :3]))
        if det < 0:
            sx = -sx

        # 3. Extract Rotation
        rot_mat = self.mat[:3, :3].copy()

        # Avoid divide by zero
        if not np.isclose(sx, 0):
            rot_mat[:, 0] /= abs(sx)
        if not np.isclose(sy, 0):
            rot_mat[:, 1] /= abs(sy)
        if not np.isclose(sz, 0):
            rot_mat[:, 2] /= abs(sz)

        # Convert to Euler Angles
        if HAS_SCIPY and R is not None:
            r = R.from_matrix(rot_mat)
            euler = r.as_euler("xyz", degrees=True)
        else:
            sy_val = np.sqrt(
                rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0]
            )
            singular = sy_val < 1e-6
            if not singular:
                x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
                y = np.arctan2(-rot_mat[2, 0], sy_val)
                z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
            else:
                x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
                y = np.arctan2(-rot_mat[2, 0], sy_val)
                z = 0.0
            euler = np.degrees([x, y, z])

        rot_quant = euler * self.ureg.degree  # type: ignore

        return Decomposition(trans_quant, [float(sx), float(sy), float(sz)], rot_quant)

    # ==========================================
    # Builder Pattern
    # ==========================================
    @classmethod
    def new(cls, base_unit: str = "µm", registry: RegistryType | None = None) -> Self:
        return cls(np.eye(4, dtype=np.float64), base_unit=base_unit, registry=registry)

    def translate(self, x: MetricValue, y: MetricValue, z: MetricValue) -> Self:
        t_mat = np.eye(4, dtype=np.float64)
        t_mat[:3, 3] = [self._to_base(x), self._to_base(y), self._to_base(z)]
        return self._chain(t_mat)

    def translate_x(self, val: MetricValue) -> Self:
        return self.translate(val, 0, 0)

    def translate_y(self, val: MetricValue) -> Self:
        return self.translate(0, val, 0)

    def translate_z(self, val: MetricValue) -> Self:
        return self.translate(0, 0, val)

    def rotate(self, angle: MetricValue, axis: Literal["x", "y", "z"] = "z") -> Self:
        deg = self._to_degrees(angle)
        rot_mat = np.eye(4, dtype=np.float64)
        if HAS_SCIPY and R is not None:
            rot_mat[:3, :3] = R.from_euler(axis, deg, degrees=True).as_matrix()
        else:
            rot_mat = self._numpy_rotation(deg, axis)
        return self._chain(rot_mat)

    def rotate_x(self, val: MetricValue) -> Self:
        return self.rotate(val, axis="x")

    def rotate_y(self, val: MetricValue) -> Self:
        return self.rotate(val, axis="y")

    def rotate_z(self, val: MetricValue) -> Self:
        return self.rotate(val, axis="z")

    def scale(self, x: float, y: float | None = None, z: float | None = None) -> Self:
        sy = x if y is None else y
        sz = x if z is None else z
        s_mat = np.eye(4, dtype=np.float64)
        s_mat[0, 0], s_mat[1, 1], s_mat[2, 2] = x, sy, sz
        return self._chain(s_mat)

    def scale_uniform(self, s: float) -> Self:
        return self.scale(s, s, s)

    # ==========================================
    # Operator Overloading
    # ==========================================
    @overload
    def __mul__(self, other: Affine) -> Self: ...
    @overload
    def __mul__(self, other: Vec3or4) -> npt.NDArray[np.float64]: ...

    def __mul__(self, other: Affine | Vec3or4) -> Self | npt.NDArray[np.float64]:
        if isinstance(other, Affine):
            # Use __class__ to properly return Self type dynamically
            return self.__class__(self.mat @ other.mat, self.base_unit, self.ureg)
        elif isinstance(other, (np.ndarray, list, tuple)):  # type: ignore
            return self._apply_raw(other)
        else:
            raise TypeError(f"Cannot multiply Affine by {type(other)}")

    def apply(self, points: Vec3or4 | QuantityType) -> QuantityType:
        if self.ureg is None:
            raise ImportError("Pint required to apply to quantities.")

        pts_raw: npt.NDArray[np.float64]
        if hasattr(points, "magnitude"):
            pts_raw = points.to(self.base_unit).magnitude  # type: ignore
        elif isinstance(points, (list, tuple, np.ndarray)):
            pts_raw = np.array([self._to_base(p) for p in points], dtype=np.float64)
        else:
            raise TypeError("Invalid input type")

        res_raw = self._apply_raw(pts_raw)  # type: ignore
        return res_raw * self.ureg(self.base_unit)

    # ==========================================
    # Internals
    # ==========================================
    def _chain(self, next_mat: Matrix4x4) -> Self:
        return self.__class__(next_mat @ self.mat, self.base_unit, self.ureg)

    def _apply_raw(self, pts: Vec3or4) -> npt.NDArray[np.float64]:
        arr_pts = np.array(pts, dtype=np.float64)
        orig_shape = arr_pts.shape
        if arr_pts.ndim == 1:
            arr_pts = arr_pts.reshape(1, -1)

        if arr_pts.shape[1] == 3:
            pts_input = np.hstack([arr_pts, np.ones((arr_pts.shape[0], 1))])
        elif arr_pts.shape[1] == 4:
            pts_input = arr_pts
        else:
            raise ValueError(f"Input points must be 3D or 4D, got shape {orig_shape}")

        res = pts_input @ self.mat.T
        if len(orig_shape) == 1:
            return res.flatten()
        return res

    def _to_base(self, val: MetricValue) -> float:
        if val is None:
            return 0.0
        if isinstance(val, (int, float, np.number)):
            return float(val)
        if isinstance(val, str) and self.ureg:
            val_quant = self.ureg(val)
            return float(val_quant.to(self.base_unit).magnitude)  # type: ignore
        if hasattr(val, "to"):
            return float(val.to(self.base_unit).magnitude)  # type: ignore
        raise TypeError(f"Unsupported type: {type(val)}")

    def _to_degrees(self, val: MetricValue) -> float:
        if isinstance(val, (int, float, np.number)):
            return float(val)
        if isinstance(val, str) and self.ureg:
            val_quant = self.ureg(val)
            return float(val_quant.to("degree").magnitude)  # type: ignore
        if hasattr(val, "to"):
            return float(val.to("degree").magnitude)  # type: ignore
        raise TypeError("Unsupported angle type")

    def _numpy_rotation(self, deg: float, axis: str) -> Matrix4x4:
        rad = np.radians(deg)
        c, s = np.cos(rad), np.sin(rad)
        m = np.eye(4, dtype=np.float64)
        if axis == "x":
            m[1:3, 1:3] = [[c, -s], [s, c]]
        elif axis == "y":
            m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
        elif axis == "z":
            m[0:2, 0:2] = [[c, -s], [s, c]]
        return m

    def to_scipy_rotation(self) -> RotationType:
        if not HAS_SCIPY or R is None:
            raise ImportError("scipy required")
        return R.from_matrix(self.mat[:3, :3])

    def to_four_by_four_list(self) -> list[list[float]]:
        return self.mat.tolist()

    @classmethod
    def from_scipy(
        cls,
        rotation_obj: RotationType,
        translation: Vec3or4 = (0.0, 0.0, 0.0),
        base_unit: str = "µm",
        registry: RegistryType | None = None,
    ) -> Self:
        if not HAS_SCIPY:
            raise ImportError("scipy required")
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rotation_obj.as_matrix()
        mat[:3, 3] = translation
        return cls(mat, base_unit=base_unit, registry=registry)

"""Uniform two-dimensional staggered grid geometry."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    h: float
    origin: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0 for n in (self.nx, self.ny)):
            raise ValueError("grid dimensions must be positive integers")
        if isinstance(self.h, (bool, np.bool_)) or not isinstance(self.h, Real) or not np.isfinite(self.h) or self.h <= 0:
            raise ValueError("spacing must be finite and positive")
        try:
            origin = np.asarray(self.origin, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("origin must contain two finite coordinates") from exc
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin must contain two finite coordinates")
        object.__setattr__(self, "origin", tuple(origin))

    @property
    def cell_count(self):
        return self.nx * self.ny

    @property
    def face_count(self):
        return self.ny * (self.nx + 1) + (self.ny + 1) * self.nx

    def _centers(self, width, height, offset):
        x, y = np.meshgrid(np.arange(width) + offset[0], np.arange(height) + offset[1])
        return self.h * np.column_stack((x.ravel(), y.ravel())) + self.origin

    @property
    def cell_centers(self):
        return self._centers(self.nx, self.ny, (.5, .5))

    @property
    def face_centers(self):
        return np.concatenate((self._centers(self.nx + 1, self.ny, (0, .5)),
                               self._centers(self.nx, self.ny + 1, (.5, 0))))

    @property
    def incidence(self):
        rows, columns, values = [], [], []
        y_start = self.ny * (self.nx + 1)
        for j in range(self.ny):
            for i in range(self.nx):
                row = j * self.nx + i
                faces = (j * (self.nx + 1) + i, j * (self.nx + 1) + i + 1,
                         y_start + j * self.nx + i, y_start + (j + 1) * self.nx + i)
                rows.extend([row] * 4)
                columns.extend(faces)
                values.extend((-1, 1, -1, 1))
        return csr_matrix((values, (rows, columns)), shape=(self.cell_count, self.face_count))

    @property
    def face_areas(self):
        return np.full(self.face_count, self.h, dtype=float)

    @property
    def face_distances(self):
        x = np.full((self.ny, self.nx + 1), self.h, dtype=float)
        y = np.full((self.ny + 1, self.nx), self.h, dtype=float)
        x[:, (0, -1)] = self.h / 2
        y[(0, -1), :] = self.h / 2
        return np.concatenate((x.ravel(), y.ravel()))

    @property
    def cell_volumes(self):
        return np.full(self.cell_count, self.h ** 2, dtype=float)

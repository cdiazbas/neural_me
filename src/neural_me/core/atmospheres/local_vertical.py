"""
Local solar vertical frame support for the stereoscopic atmospheres.

Both :class:`~neural_me.core.atmospheres.stereoscopic.StereoscopicAtmosphere`
and :class:`~neural_me.core.atmospheres.nf_stereoscopic.StereoscopicNFAtmosphere`
store a single global Cartesian magnetic vector in the reference view's
line-of-sight frame.  :class:`LocalVerticalMixin` adds the final rotation into
the ``(B_r, B_tN, B_tW)`` frame to both, without duplicating logic: each host
class only has to implement :meth:`LocalVerticalMixin._global_B`.

The per-pixel rotation is fixed geometry, so it is precomputed once and stored
as a buffer — it follows ``.to(device)`` and is saved in ``state_dict``.  The
rotation stays differentiable, so a regularizer may be placed directly on the
radial field.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from ..geometry import apply_local_vertical, local_vertical_matrix_from_map


class LocalVerticalMixin:
    """Adds the observer → local-solar-vertical rotation to an atmosphere.

    Geometry must be installed once via :meth:`set_local_vertical_geometry` or
    :meth:`set_local_vertical_geometry_from_map` before the field can be
    queried.  See :mod:`neural_me.core.geometry` for the conventions.
    """

    # ------------------------------------------------------------------ #
    #  Hook implemented by the host atmosphere
    # ------------------------------------------------------------------ #

    def _global_B(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the global ``(B_los, B_n, B_w)`` in physical units (Gauss)."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _global_B() to use "
            "LocalVerticalMixin"
        )

    # ------------------------------------------------------------------ #
    #  Geometry installation
    # ------------------------------------------------------------------ #

    def set_local_vertical_geometry(
        self,
        rotation: np.ndarray,
        on_disk: Optional[np.ndarray] = None,
    ):
        """Install the precomputed per-pixel rotation.

        Args:
            rotation: Shape ``grid_shape + (3, 3)``, mapping
                ``(B_los, B_n, B_w)`` to ``(B_r, B_tN, B_tW)``.
            on_disk: Optional boolean mask of shape ``grid_shape`` flagging
                pixels where the geometry is defined.  Defaults to all-True.
        """
        rot = torch.as_tensor(np.asarray(rotation), dtype=torch.float32)
        expected = tuple(self.grid_shape) + (3, 3)
        if tuple(rot.shape) != expected:
            raise ValueError(
                f"rotation must have shape {expected}, got {tuple(rot.shape)}"
            )

        if on_disk is None:
            mask = torch.ones(tuple(self.grid_shape), dtype=torch.bool)
        else:
            mask = torch.as_tensor(np.asarray(on_disk), dtype=torch.bool)
            if tuple(mask.shape) != tuple(self.grid_shape):
                raise ValueError(
                    f"on_disk must have shape {tuple(self.grid_shape)}, "
                    f"got {tuple(mask.shape)}"
                )

        device = getattr(self, "device", None) or rot.device
        rot = rot.to(device)
        mask = mask.to(device)

        if "lv_rotation" in self._buffers:
            self.lv_rotation = rot
            self.lv_on_disk = mask
        else:
            self.register_buffer("lv_rotation", rot)
            self.register_buffer("lv_on_disk", mask)

    def set_local_vertical_geometry_from_map(
        self, smap, heliographic: bool = True, roll: float = 0.0
    ):
        """Derive and install the rotation from a map's WCS.

        Args:
            smap: ``sunpy.map.GenericMap`` describing the grid the global
                magnetic vector lives on, i.e. the reference view's. See
                :mod:`neural_me.core.geometry` for how to build one when
                the FITS headers have been stripped.
            heliographic: ``True`` for a true meridional/zonal tangential
                split (uses the solar rotation axis); ``False`` for the
                image-aligned split.  ``B_r`` is identical either way.
            roll: Roll of the transverse components away from solar north, in
                radians.  See :func:`neural_me.core.geometry.local_vertical_matrix`.
        """
        if tuple(smap.data.shape) != tuple(self.grid_shape):
            raise ValueError(
                f"map shape {tuple(smap.data.shape)} does not match "
                f"grid_shape {tuple(self.grid_shape)}"
            )
        rotation, on_disk = local_vertical_matrix_from_map(
            smap, heliographic=heliographic, roll=roll
        )
        self.set_local_vertical_geometry(rotation, on_disk)

    @property
    def has_local_vertical_geometry(self) -> bool:
        return "lv_rotation" in self._buffers

    # ------------------------------------------------------------------ #
    #  Field in the local vertical frame
    # ------------------------------------------------------------------ #

    def local_vertical_field(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(B_r, B_tN, B_tW)`` in Gauss.

        Radial (normal to the photosphere), tangential North (meridional) and
        tangential West (zonal).  Off-disk pixels carry an identity rotation;
        use :attr:`lv_on_disk` to mask them.
        """
        if not self.has_local_vertical_geometry:
            raise RuntimeError(
                "No local vertical geometry installed. Call "
                "set_local_vertical_geometry_from_map(smap) with the reference "
                "view's sunpy Map first."
            )
        B_los, B_n, B_w = self._global_B()
        return apply_local_vertical(B_los, B_n, B_w, self.lv_rotation)

    def to_local_vertical_stack(self) -> torch.Tensor:
        """Stack of ``[B_r, B_tN, B_tW]`` with shape ``grid_shape + (3,)``."""
        return torch.stack(self.local_vertical_field(), dim=-1)

    def to_local_vertical_numpy(self) -> np.ndarray:
        """:meth:`to_local_vertical_stack` detached on the CPU as a numpy array."""
        return self.to_local_vertical_stack().detach().cpu().numpy()

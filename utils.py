import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import torch


def _cell_bounds(points, latitude=False):
    """Infer cell edges from one-dimensional cell centres."""
    points = np.asarray(points, dtype=np.float64)
    bounds = np.concatenate(
        (
            [1.5 * points[0] - 0.5 * points[1]],
            0.5 * (points[:-1] + points[1:]),
            [1.5 * points[-1] - 0.5 * points[-2]],
        )
    )
    return np.clip(bounds, -90.0, 90.0) if latitude else bounds


def _coordinate_name(data, candidates):
    try:
        return next(
            name
            for name in candidates
            if name in data.coords and data[name].ndim == 1
        )
    except StopIteration as error:
        raise ValueError(f"Expected one of the coordinates {candidates}") from error


def get_remap_matrix(source, target):
    """Return conservative-remapping weights and target latitude/longitude."""
    src_lat_name = _coordinate_name(source, ("lat", "latitude"))
    src_lon_name = _coordinate_name(source, ("lon", "longitude"))
    dst_lat_name = _coordinate_name(target, ("lat", "latitude"))
    dst_lon_name = _coordinate_name(target, ("lon", "longitude"))

    src_lat = np.asarray(source[src_lat_name])
    src_lon = (np.asarray(source[src_lon_name]) + 180.0) % 360.0 - 180.0
    dst_lat = np.asarray(target[dst_lat_name])
    dst_lon = (np.asarray(target[dst_lon_name]) + 180.0) % 360.0 - 180.0

    src_lat_order, src_lon_order = np.argsort(src_lat), np.argsort(src_lon)
    dst_lat_order, dst_lon_order = np.argsort(dst_lat), np.argsort(dst_lon)
    src_lat, src_lon = src_lat[src_lat_order], src_lon[src_lon_order]
    dst_lat, dst_lon = dst_lat[dst_lat_order], dst_lon[dst_lon_order]

    # Equal latitude widths do not have equal areas on a sphere.
    src_lat_b = np.sin(np.deg2rad(_cell_bounds(src_lat, latitude=True)))
    dst_lat_b = np.sin(np.deg2rad(_cell_bounds(dst_lat, latitude=True)))
    lat_overlap = np.maximum(
        0.0,
        np.minimum(dst_lat_b[1:, None], src_lat_b[None, 1:])
        - np.maximum(dst_lat_b[:-1, None], src_lat_b[None, :-1]),
    )
    weights_lat = sparse.csr_matrix(lat_overlap / np.diff(dst_lat_b)[:, None])

    src_lon_b, dst_lon_b = _cell_bounds(src_lon), _cell_bounds(dst_lon)
    lon_overlap = np.zeros((dst_lon.size, src_lon.size))
    # Shift copies of the source grid to include overlap across the dateline.
    for shift in (-360.0, 0.0, 360.0):
        lon_overlap += np.maximum(
            0.0,
            np.minimum(dst_lon_b[1:, None], src_lon_b[None, 1:] + shift)
            - np.maximum(dst_lon_b[:-1, None], src_lon_b[None, :-1] + shift),
        )
    weights_lon = sparse.csr_matrix(lon_overlap / np.diff(dst_lon_b)[:, None])

    # The two-dimensional cell overlap is separable on a rectilinear grid.
    weights = sparse.kron(weights_lat, weights_lon, format="csr")

    # Return rows and columns in the grids' original coordinate order.
    src_order = (src_lat_order[:, None] * source.sizes[src_lon_name] + src_lon_order[None, :]).ravel()
    dst_order = (dst_lat_order[:, None] * target.sizes[dst_lon_name] + dst_lon_order[None, :]).ravel()
    remap_matrix = weights[np.argsort(dst_order)][:, np.argsort(src_order)].tocsr()

    return remap_matrix, (np.asarray(target[dst_lat_name]), np.asarray(target[dst_lon_name]))


def conservative_remap(values, remap_matrix, target_shape):
    """Apply remapping weights, or validate and return an already matching grid."""
    
    if remap_matrix is None:
        return values

    values = np.asarray(values)
    target_shape = tuple(target_shape)

    if np.prod(values.shape[-2:]) != remap_matrix.shape[1]:
        raise ValueError("The input field shape does not match the remapping matrix")

    flat = values.reshape((-1, remap_matrix.shape[1]))
    mapped = (remap_matrix @ flat.T).T
    return mapped.reshape(values.shape[:-2] + target_shape).astype(values.dtype, copy=False)


def plot_tensor(tensor, ax, name="", latlon=None, cmap="viridis", vmin=None, vmax=None):
    values = (
        tensor.detach().cpu().numpy()
        if torch.is_tensor(tensor)
        else np.asarray(tensor)
    )
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D tensor after squeezing, got shape {values.shape}")

    if latlon is None:
        image = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        lat, lon = latlon
        image = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.coastlines(linestyle="-")
        ax.add_feature(cfeature.BORDERS, linestyle="--")
    ax.set_title(name)
    ax.figure.colorbar(image, ax=ax, shrink=0.5)
    return image


def plot_sample(tensors, labels, latlon=None, cmap="viridis", vmin=None, vmax=None):
    """Plot variables as columns and, for batched tensors, samples as rows."""

    n_axes = len(tensors)
    assert n_axes == len(labels)
    batched = tensors[0].ndim > 3
    n_samples = len(tensors[0]) if batched else 1
    subplot_kw = {"projection": ccrs.PlateCarree()} if latlon is not None else {}
    fig, ax = plt.subplots(n_samples, n_axes, figsize=(4 * n_axes, 2.5 * n_samples), subplot_kw=subplot_kw)
    ax = np.asarray(ax).reshape(n_samples, n_axes)
    for row in range(n_samples):
        for i in range(n_axes):
            plot_tensor(tensors[i][row] if batched else tensors[i], ax[row, i], labels[i], latlon, cmap, vmin, vmax)
    fig.tight_layout(h_pad=0.2)
    return fig

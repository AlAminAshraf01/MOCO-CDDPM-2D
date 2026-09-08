"""Adapted from https://github.com/SongweiGe/TATS -- 2D port.

Only the helpers that make sense for 2D data are kept.  ``save_video_grid``
from the 3D code is replaced by ``save_image_grid``.
"""

import math

import imageio
import numpy as np
import torch


def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    """Shift ``src_dim`` to ``dest_dim``.

    e.g. shift_dim(x, 1, -1) turns (b, c, h, w) into (b, h, w, c).
    Rank agnostic, so it is shared unchanged with the 3D code.
    """
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def adopt_weight(global_step, threshold=0, value=0.):
    weight = 1
    if global_step < threshold:
        weight = value
    return weight


def save_image_grid(images, fname, nrow=None, padding=1):
    """``images``: (b, c, h, w) float tensor already scaled to [0, 1]."""
    b, c, h, w = images.shape
    images = images.permute(0, 2, 3, 1)
    images = (images.detach().cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)

    grid = np.zeros(((padding + h) * nrow + padding,
                     (padding + w) * ncol + padding, c), dtype='uint8')
    for i in range(b):
        r = i // ncol
        col = i % ncol
        start_r = (padding + h) * r + padding
        start_c = (padding + w) * col + padding
        grid[start_r:start_r + h, start_c:start_c + w] = images[i]
    if c == 1:
        grid = grid[..., 0]
    imageio.imwrite(fname, grid)
    return grid


def comp_getattr(args, attr_name, default=None):
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    return default


def visualize_tensors(t, name=None, nest=0):
    if name is not None:
        print(name, "current nest: ", nest)
    print("type: ", type(t))
    if isinstance(t, dict):
        print(t.keys())
        for k in t.keys():
            if t[k] is None:
                print(k, "None")
            elif isinstance(t[k], torch.Tensor):
                print(k, t[k].shape)
            elif isinstance(t[k], (dict, list)):
                visualize_tensors(t[k], name, nest + 1)
    elif isinstance(t, list):
        print("list length: ", len(t))
        for t2 in t:
            visualize_tensors(t2, name, nest + 1)
    elif isinstance(t, torch.Tensor):
        print(t.shape)
    else:
        print(t)
    return ""

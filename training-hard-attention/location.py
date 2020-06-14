import numpy as np
import torch

# Configuration
min_coord = 0
max_coord = 208
grid_dim = 50

# Data structures for transforming between representations
int2coord = torch.LongTensor([int(c) for c in np.linspace(min_coord, max_coord-1, grid_dim)])
coord2int = torch.LongTensor([-1]*max_coord)
for i, c in enumerate(int2coord):
    coord2int[c] = i
int2pixel = torch.LongTensor([(x, y)
                              for x in int2coord
                              for y in int2coord])

def stack(locs, dim):
    indices = [l.index for l in locs]
    return Location(torch.stack(indices, dim=dim), "index")

class Location():
    """
    supports formats ['index', 'normalized', 'pixel']
    each describes a tensor of shape:
      - if 'int': LongTensor of arbitrary shape
      - if 'normalized', FloatTensor with last dim of size 2 (and all elements in [-1, 1])
      - if 'pixel', LongTensor with last dim of size 2
    """
    def __init__(self, tensor, format_str, inject_vals=None):
        """
        Can be initialized by either 'index', 'pixel'.
        """
        if inject_vals is not None:
            self._pixel, self._index, self._normalized = inject_vals
        else:
            assert format_str in ['index', 'pixel'], "Unsupported initialization format."
            assert tensor.type() == 'torch.LongTensor'
            if format_str == 'pixel':
                assert tensor.shape[-1] == 2, f"Location tensor should have last dimension \
                                                of size 2, but shape is: {tensor.shape}"
            self._pixel = tensor if format_str == 'pixel' else None
            self._index = tensor if format_str == 'index' else None
            self._normalized = None

    @property
    def pixel(self):
        if self._pixel is None:
            # then we know from __init__ that self._index must be set
            self._pixel = int2pixel[self._index]
        return self._pixel

    @property
    def index(self):
        if self._index is None:
            # then we know from __init__ that self._pixel must be set
            row_col = coord2int[self._pixel]
            assert (row_col < 0).sum() == 0, "Invalid pixel coordinate."
            row_index = row_col[..., 0]
            col_index = row_col[..., 1]
            self._index = row_index * grid_dim + col_index
        return self._index

    @property
    def normalized(self):
        if self._normalized is None:
            # calculate using the property self.pixel
            unit_pixels = self.pixel.type(torch.FloatTensor) / max_coord
            self._normalized = unit_pixels*2 - 1
        return self._normalized

    def __getitem__(self, *args):
        def slice_if_not_none(tensor):
            return tensor.__getitem__(*args) if tensor is not None else None
        new_pixel = slice_if_not_none(self._pixel)
        new_index = slice_if_not_none(self._index)
        new_normed = slice_if_not_none(self._normalized)
        return Location(None, None, inject_vals=(new_pixel, new_index, new_normed))

    def concatenate(self, other, dim):
        """
        Concatenate self with another location in specified `dim`. Returns
        the resulting tensor (while leaving self unchanged).
        """
        def concat_if_not_none(a, b):
            if a is None or b is None:
                return None
            return torch.cat([a, b], dim=dim)
        new_pixel = concat_if_not_none(self.pixel, other.pixel)  # call property for pixel to ensure that not every new attribute is None
        new_index = concat_if_not_none(self._index, other._index)
        new_normed = concat_if_not_none(self._normalized, other._normalized)
        return Location(None, None, inject_vals=(new_pixel, new_index, new_normed))

    def __len__(self):
        return len(self._index) if self._index is not None else len(self._pixel)

    def __iter__(self):
        # based on https://github.com/pytorch/pytorch/blob/master/torch/tensor.py
        return iter(map(lambda i: self[i], range(len(self))))

    def save(self, path):
        # save by saving indices to some file
        torch.save(self.index, path)

    @staticmethod
    def load(path):
        indices = torch.load(path)
        return Location(indices, 'index')

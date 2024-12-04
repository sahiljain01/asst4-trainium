import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)

    # The P dimension size of a tile in both SBUF and PSUM must never 
    # exceed nki.tile_size.pmax == 128.
    c_in_pmax = nl.tile_size.pmax

    # free dimension max
    f_dim_max = nl.tile_size.psum_fmax

    # number of tiles == number of input channels divided by max number of channels
    # partition dimension for any given tile
    n_tiles_c_in = in_channels // c_in_pmax
    weight_matrix = nl.load(W)
    # Process the images in batches
    for b in nl.affine_range(batch_size):
        image = nl.load(X[b])
        intermediate_output = None
        res_psum = nl.zeros((in_channels, out_height, out_width), nl.float32, buffer=nl.psum)

        for filter_height_index in nl.affine_range(filter_height):
            for filter_width_index in nl.affine_range(filter_width):
                filter_weights = weight_matrix[: , :, filter_height_index, filter_width_index]
                image_filtered = nl.copy(image[:, filter_height_index:input_height-filter_height+filter_height_index + 1, filter_width_index:input_width-filter_width+filter_width_index+1])
                image_filtered = image_filtered.reshape((in_channels, out_height * out_width))
                output = nl.matmul(filter_weights, image_filtered)
                res_psum += output.reshape((in_channels, out_height, out_width))

        nl.store(X_out[b], res_psum)

    return X_out


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

    # # Initialize output array
    # X_out_no_pooling = nl.ndarray(
    #     shape=(batch_size, out_channels, out_height, out_width),
    #     dtype=X.dtype,
    #     buffer=nl.sbuf,
    # )

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    print(f"\n X out shape: {X_out.shape} \n")

    # Various tiling dimensions (You may want to define more of them)

    # The P dimension size of a tile in both SBUF and PSUM must never 
    # exceed nki.tile_size.pmax == 128.
    c_in_pmax = nl.tile_size.pmax
    c_out_pmax = nl.tile_size.pmax

    # free dimension max
    f_dim_max = nl.tile_size.psum_fmax

    # number of tiles == number of input channels divided by max number of channels
    # partition dimension for any given tile
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_in_pmax # TODO: these should both be 128?

    # load in the weights into an SBUF array of shape 
    # (n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, 128, kernel_height, kernel_width)
    weight_matrix_orig = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)
    weight_matrix = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
    
    bias_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), 1), dtype=W.dtype, buffer=nl.sbuf)
    for bias_tile_out in nl.affine_range(n_tiles_c_out):
        bias_sbuf[bias_tile_out, :, 0] = nl.load(bias[128 * bias_tile_out: 128 * (bias_tile_out + 1)])

    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in, c_in_pmax, filter_height, filter_width))

    weight_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype = W.dtype, buffer = nl.sbuf)
    weight_copy = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
    weight_matrix = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax), dtype=W.dtype, buffer=nl.sbuf)

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        weight_sbuf[c_out_tile] = nl.load(W[c_out_tile])

    for c_out_tile in nl.affine_range(n_tiles_c_out):
        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    weight_copy[i, j, c_out_tile, c_in_tile, :, :] = nl.copy(weight_sbuf[c_out_tile, :, c_in_tile, :, i, j], dtype = W.dtype)
                    weight_matrix[i, j, c_out_tile, c_in_tile] = nisa.nc_transpose(weight_copy[i, j, c_out_tile, c_in_tile])


    out_chunks = 2
    n_chunks = (out_height + (out_chunks - 1)) // out_chunks
    in_rows = (out_chunks + filter_height - 1)

    # loop over batch
    for b in nl.affine_range(batch_size):
        for n_chunk in nl.sequential_range(n_chunks):
            image = nl.ndarray(
                (n_tiles_c_in, nl.par_dim(c_in_pmax), in_rows, input_width), 
                dtype=W.dtype, 
                buffer=nl.sbuf
            )

            for n_tile_in in nl.affine_range(n_tiles_c_in):
                # load corresponding part of input image
                image[n_tile_in, :, :, :] = nl.load(X[b, 128 * n_tile_in: 128 * (n_tile_in + 1), (n_chunk * out_chunks): (n_chunk * out_chunks) + in_rows, :])

            output_image = nl.ndarray(
                (nl.par_dim(c_out_pmax), out_chunks, out_width), 
                dtype=W.dtype,
                buffer=nl.sbuf
            )

            output_image_pooled = nl.ndarray(
                (nl.par_dim(c_out_pmax), out_chunks // pool_size, out_width // pool_size), 
                dtype=W.dtype,
                buffer=nl.sbuf
            )

            for n_tile_out_index in nl.sequential_range(n_tiles_c_out):
                # assign space in SBUF to store output
                # loop over output_rows:
                for row in nl.sequential_range(out_chunks):
                    # assign space in PSUM to store output row
                    res_psum = nl.zeros((c_in_pmax, out_width), nl.float32, buffer=nl.psum)
                    # loop over kernel_height
                    for filter_height_index in nl.affine_range(filter_height):
                        # loop over kernel_width
                        for filter_width_index in nl.affine_range(filter_width):
                            # loop over n_tiles_c_in
                            for n_tile_in_index in nl.affine_range(n_tiles_c_in):
                                # (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_in_pmax), c_out_pmax)
                                result = nl.matmul(
                                    weight_matrix[filter_height_index, filter_width_index, n_tile_out_index, n_tile_in_index, :, :],
                                    image[n_tile_in_index, :, row + filter_height_index, filter_width_index: filter_width_index + out_width],
                                    transpose_x=True
                                )
                                res_psum += result

                    output_image[:, row, :] = res_psum

                output_image[:, :, :] = nisa.tensor_scalar(output_image, np.add, bias_sbuf[n_tile_out_index, :, 0])
                i_0 = nl.arange(c_out_pmax)[:, None, None, None, None] #
                i_1 = nl.arange(out_chunks // pool_size)[None, :, None, None, None] # y_outer
                i_2 = nl.arange(pool_size)[None, None, :, None, None] # y_inner
                i_3 = nl.arange(out_width // pool_size)[None, None, None, :, None] # x_outer
                i_4 = nl.arange(pool_size)[None, None, None, None, :] # x_inner

                output_image_pooled[:, :, :] = nl.max(output_image[i_0, pool_size*i_1+i_2, pool_size*i_3+i_4], axis=[2,4])
                nl.store(X_out[b, 128 * n_tile_out_index: 128 * (n_tile_out_index + 1), n_chunk * (out_chunks // pool_size): (n_chunk * (out_chunks // pool_size)) + (out_chunks // pool_size), :], output_image_pooled[:, :, :])

    return X_out

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
    c_out_pmax = nl.tile_size.pmax

    # free dimension max
    f_dim_max = nl.tile_size.psum_fmax

    # number of tiles == number of input channels divided by max number of channels
    # partition dimension for any given tile
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_in_pmax # TODO: these should both be 128?

    # load in the weights into an SBUF array of shape (n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, 128, kernel_height, kernel_width)
    weight_matrix = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype=nl.float32, buffer=nl.sbuf)
    weight_matrix_pt2 = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax), dtype=nl.float32, buffer=nl.sbuf)

    for n_tile_in in nl.affine_range(n_tiles_c_in):
        for n_tile_out in nl.affine_range(n_tiles_c_out):
            weight_no_transpose = nl.load(W[128 * n_tile_out: 128 * (n_tile_out + 1), 128 * n_tile_in: 128 * (n_tile_in + 1), :, :])
            # weight_transpose = weight_no_transpose.reshape((filter_height, filter_width, 1, 1, c_in_pmax, c_out_pmax))
            weight_matrix[n_tile_out, :, n_tile_in, :, :, :] = weight_no_transpose
            # weight_matrix_pt2[:, :, n_tile_out, n_tile_in, :, :] = weight_transpose

    # transpose that to get an array of shape (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_in_pmax), c_out_pmax), call this w
    weight_matrix_pt2 = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax), dtype=nl.float32, buffer=nl.sbuf)
    i_p0 = nl.arange(n_tiles_c_out)[:, None, None, None, None, None]
    i_p1 = nl.arange(c_out_pmax)[ None, :, None, None, None, None]
    i_p2 = nl.arange(n_tiles_c_in)[ None, None, :, None, None, None]
    i_p3 = nl.arange(c_in_pmax)[ None, None, None, :, None, None]
    i_p4 = nl.arange(filter_height)[ None, None, None, None, :, None]
    i_p5 = nl.arange(filter_width)[ None, None, None, None, None, :]

    weight_matrix_pt2[i_p4, i_p5, i_p0, i_p2, i_p3, i_p1] = nl.copy(weight_matrix[i_p0, i_p1, i_p2, i_p3, i_p4, i_p5])    

    # nl.device_print("weight matrix orig", weight_matrix[1, 1, 1, 1, :, :])
    weight_matrix_value = weight_matrix_pt2[0, 0, 0, 0, 0, 0]
    # nl.device_print("weight matrix orig", W[1, 1, :, :])

    # loop over batch:
    for b in nl.affine_range(batch_size):
        image = nl.ndarray(
            (n_tiles_c_in, nl.par_dim(c_in_pmax), input_height, input_width), 
            dtype=nl.float32, 
            buffer=nl.sbuf
        )

        for n_tile_in in nl.affine_range(n_tiles_c_in):
            # load corresponding part of input image
            image[n_tile_in, :, :, :] = nl.load(X[b, 128 * n_tile_in: 128 * (n_tile_in + 1), :, :])

        # loop over n_tiles_c_out:
        for n_tile_out_index in nl.affine_range(n_tiles_c_out):
            # assign space in SBUF to store output
            output_image = nl.ndarray(
                (nl.par_dim(c_out_pmax), out_height, out_width), 
                dtype=nl.float32, 
                buffer=nl.sbuf
            )
            # loop over output_rows:
            for row in nl.affine_range(out_height):
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
                                weight_matrix_pt2[filter_height_index, filter_width_index, n_tile_out_index, n_tile_in_index, :, :],
                                image[n_tile_in_index, :, row + filter_height_index, filter_width_index: filter_width_index + out_width]
                            )
                            res_psum += result

                nl.store(X_out[b, n_tile_out_index, :, row, :], res_psum)


    # - assign space in SBUF to store entire image, call it x
    # # shape : (n_tiles_c_in, nl.par_dim(c_in_pmax), image_height, image_width)
    # loop over n_tiles_c_in:
    #     - load corresponding part of input image

    # loop over n_tiles_c_out:
    #     - assign space in SBUF to store output
    #     # shape : (nl.par_dim(c_out_pmax), out_height, out_width)
    #     loop over output_rows:
    #         - assign space in PSUM to store output row
    #         loop over kernel_height:
    #             loop over kernel_width:
    #                 loop over n_tiles_c_in:
    #                     - matmul w[kernel_height, kernel_width, n_tile_c_out, n_tile_cin, :, :].T with
    #                     x[n_tiles_c_in, :, out_row + kernel_height, kernel_width:kernel_width + out_width]
    #         - copy stuff from PSUM back to SBUF
    #     - copy stuff from SBUF back to HBM




    # weight_matrix = nl.load(W)
    # # Process the images in batches
    # for b in nl.affine_range(batch_size):
    #     image = nl.load(X[b])
    #     intermediate_output = None
    #     res_psum = nl.zeros((in_channels, out_height, out_width), nl.float32, buffer=nl.psum)

        # for filter_height_index in nl.affine_range(filter_height):
        #     for filter_width_index in nl.affine_range(filter_width):
    #             filter_weights = weight_matrix[: , :, filter_height_index, filter_width_index]
    #             image_filtered = nl.copy(image[:, filter_height_index:input_height-filter_height+filter_height_index + 1, filter_width_index:input_width-filter_width+filter_width_index+1])
    #             image_filtered = image_filtered.reshape((in_channels, out_height * out_width))
    #             output = nl.matmul(filter_weights, image_filtered)
    #             res_psum += output.reshape((in_channels, out_height, out_width))

    #     nl.store(X_out[b], res_psum)

    return X_out


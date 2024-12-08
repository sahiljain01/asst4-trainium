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
    weight_matrix_orig = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, c_in_pmax, filter_height, filter_width), dtype=nl.float32, buffer=nl.sbuf)
    weight_matrix = nl.ndarray((filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax), dtype=nl.float32, buffer=nl.sbuf)

    for n_tile_in in nl.affine_range(n_tiles_c_in):
        for n_tile_out in nl.affine_range(n_tiles_c_out):
            weight_no_transpose = nl.load(W[128 * n_tile_out: 128 * (n_tile_out + 1), 128 * n_tile_in: 128 * (n_tile_in + 1), :, :])
            weight_matrix_orig[n_tile_out, :, n_tile_in, :, :, :] = weight_no_transpose

    x_sbuf = nl.ndarray((2, nl.par_dim(100), 100, 100), buffer=nl.sbuf, dtype=nl.float32) # specify P dimension to be the second dimension

    # move data around using nl.copy to get an array of shape 
    # (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_out_pmax), c_in_pmax)
    for n_tile_in in nl.affine_range(n_tiles_c_in):
        for n_tile_out in nl.affine_range(n_tiles_c_out):
            for n_tile_in_channel in nl.affine_range(128):
                for n_tile_out_channel in nl.affine_range(128):
                    for k_h in nl.affine_range(filter_height):
                        for k_w in nl.affine_range(filter_width):
                            # (n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, 128, kernel_height, kernel_width)
                            # i_a, i_b, i_c, i_d, i_e, i_f = nl.mgrid[n_tile_out, n_tile_out_channel, n_tile_in, n_tile_in_channel, 0:filter_height, 0:filter_width]
                            # i_a, i_b, i_c, i_d, i_e, i_f = nl.mgrid[0:1, 0:1, 0:1, 0:1, 0:filter_height, 0:filter_width]

                            # i_a = i_a + n_tile_out
                            # i_b = i_b + n_tile_out_channel
                            # i_c = i_c + n_tile_in
                            # i_d = i_d + n_tile_in_channel

                            # i_g, i_l = nl.mgrid[ 0:filter_height, 0:filter_width]

                            # p = nl.copy(
                            #     weight_matrix_orig[0, 0, :, :, :, :]
                            # )
                            weight_matrix[k_h, k_w, n_tile_out, n_tile_in, :, n_tile_in_channel] = nl.copy(
                                weight_matrix_orig[n_tile_out, :, n_tile_in, n_tile_in_channel, k_h, k_w]
                            )
                            # weight_matrix[i_g, i_l, n_tile_out, n_tile_in, n_tile_out_channel, n_tile_in_channel] = nl.copy(
                            #     weight_matrix_orig[i_a, i_b, i_c, i_d, i_e, i_f]
                            # )
                            # p = nl.copy(
                            #     weight_matrix_orig[i_a, i_b, i_c, i_d, i_e, i_f]
                            # )


    # transpose that to get an array of shape (kernel_height, kernel_width, n_tiles_out_channels, n_tiles_in_channels, nl.par_dim(c_in_pmax), c_out_pmax), call this w


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

        # # loop over n_tiles_c_out:
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
                                weight_matrix[filter_height_index, filter_width_index, n_tile_out_index, n_tile_in_index, :, :],
                                image[n_tile_in_index, :, row + filter_height_index, filter_width_index: filter_width_index + out_width]
                            )
                            # # res_psum[:, :] = nl.copy(nl.add(result, res_psum))
                            res_psum += result

                # nl.store(X_out[b, 128 * n_tile_out_index: 128 * (n_tile_out_index + 1), row, :], res_psum)
                output_image[:, row, :] = res_psum

            
            # zeros_matrix = nl.zeros((nl.par_dim(c_out_pmax), out_height, out_width), nl.float32, buffer=nl.psum)
            nl.store(X_out[b, 128 * n_tile_out_index: 128 * (n_tile_out_index + 1), :, :], output_image)
            # nl.store(X_out[b, 128 * n_tile_out_index: 128 * (n_tile_out_index + 1), :, :], zeros_matrix)

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

    # weight_matrix_pt2[i_p4, i_p5, i_p0, i_p2, i_p3, i_p1] = nl.copy(weight_matrix[i_p0, i_p1, i_p2, i_p3, i_p4, i_p5])    
    # print(weight_matrix_pt2.shape)
    # nl.device_print("Weight Matrix", weight_matrix_pt2[0,0,0,0,:, :])



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

if __name__ == "__main__":
    input_channels = 128
    output_channels = 128
    kernel_size = 3
    batch_size = 4
    image_dims = (32, 16)

    X = np.random.rand(
        batch_size, input_channels, image_dims[0], image_dims[1]
    ).astype(np.float32)
    W = np.random.rand(
        output_channels, input_channels, kernel_size, kernel_size
    ).astype(np.float32)
    bias = (
        np.zeros(output_channels).astype(np.float32)
    )

    fused_conv2d_maxpool(X, W, bias)

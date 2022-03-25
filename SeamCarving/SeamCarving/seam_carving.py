from typing import Dict, Any
import utils
import numpy as np

NDArray = Any


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """
    :param image: ِnparray which represents an image
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respectively).
    """
    gradients = utils.get_gradients(image)
    original_height = image.shape[0]
    original_width = image.shape[1]
    height_diff = original_height - out_height
    width_diff = original_width - out_width

    grayscale_original_image = utils.to_grayscale(image)
    if width_diff > 0:  # need to scale down the width
        resized_width_image, vertical_seams = scale_down(image, grayscale_original_image, gradients, width_diff,
                                                         forward_implementation)
    else:  # need to scale up the width
        resized_width_image, vertical_seams = scale_up(image, grayscale_original_image, gradients, abs(width_diff),
                                                       forward_implementation)

    grayscale_resized_image = utils.to_grayscale(resized_width_image)
    np.rot90(image, k=1, axes=(0, 1))  # rotating CCW
    if height_diff > 0:  # need to scale down the height
        resized_image, horizontal_seams = scale_down(resized_width_image, grayscale_resized_image, gradients,
                                                     height_diff, forward_implementation)
    else:  # need to scale up the height
        resized_image, horizontal_seams = scale_up(resized_width_image, grayscale_resized_image, gradients,
                                                   abs(height_diff), forward_implementation)
    np.rot90(image, k=-1, axes=(0, 1))  # rotating CW

    out_images_dict = {'resized': resized_image, 'vertical_seams': vertical_seams, 'horizontal_seams': horizontal_seams}

    return out_images_dict


def scale_down(image: NDArray, grayscale_image: NDArray, gradients: NDArray, dim_diff: int, is_forward: bool):
    """
        :param image: ِnparray which represents an image
        :param grayscale_image: ِnparray which represents a grayscale image
        :param gradients: ِnparray which represents a gradients' matrix
        :param dim_diff: the desired difference in width / height
        :param is_forward: a boolean flag that indicates whether forward or basic implementation is used
        :return: resized_image scaled up width by dim_diff and seams image
    """
    indices_matrix, seams_matrix = calculate_seams(grayscale_image, gradients, dim_diff, image.shape, is_forward)
    resized_image = create_original_without_seams(image, indices_matrix)
    rows_range = np.range(image.shape[0])
    for seam_number in range(dim_diff):
        image[rows_range, seams_matrix[:, seam_number]] = [255, 0, 0]
    return resized_image, image


def scale_up(image: NDArray, grayscale_image: NDArray, gradients: NDArray, dim_diff: int, is_forward: bool):
    """
        :param image: ِnparray which represents an image
        :param grayscale_image: ِnparray which represents a grayscale image
        :param gradients: ِnparray which represents a gradients' matrix
        :param dim_diff: the desired difference in width / height
        :param is_forward: a boolean flag that indicates whether forward or basic implementation is used
        :return: resized_image scaled up width by dim_diff and seams image.
    """
    indices_matrix, seams_matrix = calculate_seams(grayscale_image, gradients, dim_diff, image.shape, is_forward)
    resized_image = create_original_with_dup_seams(image, seams_matrix, dim_diff)
    rows_range = np.range(image.shape[0])
    for seam_number in range(dim_diff):
        image[rows_range, seams_matrix[:, seam_number]] = [0, 0, 0]
    return resized_image, image


def calculate_seams(grayscale_image: NDArray, gradients: NDArray, dim_diff: int, original_shape: (int, int, int),
                    is_forward: bool):
    indices_matrix = np.indices((original_shape[0], original_shape[1]))[1]
    seams_matrix = np.zeros((grayscale_image.shape[0], dim_diff), dtype=np.float32)

    for seam_number in range(dim_diff):
        if is_forward:
            cost_matrix = calculate_cost_matrix_forward(grayscale_image, gradients)
            best_seam = find_best_seam_forward(cost_matrix, indices_matrix)
        else:
            cost_matrix = calculate_cost_matrix_basic(grayscale_image, gradients)
            best_seam = find_best_seam_basic(cost_matrix, indices_matrix)

        seams_matrix[:, seam_number] = best_seam
        remove_seam(grayscale_image, indices_matrix, gradients, best_seam)

    return indices_matrix, seams_matrix


def calculate_cost_matrix_basic(grayscale_image: NDArray, E: NDArray):
    M = np.zeros_like(grayscale_image)
    M[0, :] = E[0, :]
    # column_256 = np.broadcast_to([256.], [grayscale_image.shape[1], 1])
    for row in M[1:, :]:
        shift_right_row = np.concatenate((np.array([256.]), M[row - 1, 1:]))
        shift_left_row = np.concatenate((M[row - 1, 0:-1], np.array([256.]))
        M[row] = E[row] + np.fmin(M[row - 1, :], shift_left_row, shift_right_row)
    return M


def find_best_seam_basic(cost_matrix: NDArray, indices_matrix: NDArray):
    best_orig_seam = np.zeros((cost_matrix.shape[0], 1), dtype=np.float32)
    best_orig_seam[-1] = indices_matrix[-1, np.argmin(cost_matrix[-1, :])]
    for row_index in range(cost_matrix.shape[0] - 2, -1, -1):
        best_prev_index = best_orig_seam[row_index + 1]
        is_right_edge = best_prev_index == cost_matrix.shape[1] - 1
        is_left_edge = best_prev_index == 0

        if is_left_edge and is_right_edge:  # one column
            min_column_index = cost_matrix[row_index, best_prev_index]
        elif is_left_edge:
            min_column_index = np.argmin(cost_matrix[row_index, best_prev_index],
                                         cost_matrix[row_index, best_prev_index + 1])
        elif is_right_edge:
            min_column_index = np.argmin(cost_matrix[row_index, best_prev_index],
                                         cost_matrix[row_index, best_prev_index - 1])
        else:
            min_column_index = np.argmin(cost_matrix[row_index, best_prev_index],
                                         cost_matrix[row_index, best_prev_index - 1],
                                         cost_matrix[row_index, best_prev_index + 1])

        best_orig_seam[row_index] = indices_matrix[row_index, min_column_index]
    return best_orig_seam


def remove_seam(grayscale_image: NDArray, indices_matrix: NDArray, gradients: NDArray, seam: NDArray):
    for row_index in range(grayscale_image.shape[0]):
        grayscale_image[row_index, :] = np.concatenate(grayscale_image[row_index, :seam[row_index]],
                                                       grayscale_image[row_index, seam[row_index] + 1:])
        indices_matrix[row_index, :] = np.concatenate(indices_matrix[row_index, :seam[row_index]],
                                                      indices_matrix[row_index, seam[row_index] + 1:])
        gradients[row_index, :] = np.concatenate(gradients[row_index, :seam[row_index]],
                                                 gradients[row_index, seam[row_index] + 1:])


def create_original_without_seams(image: NDArray, indices_matrix: NDArray):
    # resized_image = np.zeros_like(original_image)
    return np.take(image, indices_matrix)


def create_original_with_dup_seams(image: NDArray, seams_matrix: NDArray, dim_diff: int):
    resized_image = np.empty(image.shape[0], image.shape[1] + dim_diff)
    for row_idx in range(image.shape[0]):
        duplicated_values = np.take(image[row_idx, :], seams_matrix[row_idx, :])
        resized_image[row_idx, :] = np.insert(image[row_idx, :], seams_matrix[row_idx, :], duplicated_values)
    return resized_image


def calculate_cost_matrix_forward(grayscale_image: NDArray, E: NDArray):
    # TODO continue
    pass


def find_best_seam_forward(cost_matrix: NDArray, indices_matrix: NDArray):
    # TODO continue
    pass

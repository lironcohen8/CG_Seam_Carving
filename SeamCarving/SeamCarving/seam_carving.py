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
    vertical_seams = np.copy(image)
    resized_width_image = np.copy(image)
    if width_diff > 0:  # need to scale down the width
        resized_width_image, vertical_seams = scale_down(image, grayscale_original_image, gradients, width_diff,
                                                         forward_implementation, True)
    elif width_diff < 0:  # need to scale up the width
        resized_width_image, vertical_seams = scale_up(image, grayscale_original_image, gradients, abs(width_diff),
                                                       forward_implementation, True)

    resized_width_image = np.rot90(resized_width_image, k=1, axes=(0, 1))  # rotating CCW
    grayscale_resized_image = utils.to_grayscale(resized_width_image)
    horizontal_seams = np.copy(resized_width_image)
    gradients = utils.get_gradients(resized_width_image)
    resized_image = np.copy(resized_width_image)
    if height_diff > 0:  # need to scale down the height
        resized_image, horizontal_seams = scale_down(resized_width_image, grayscale_resized_image, gradients,
                                                     height_diff, forward_implementation, False)
    elif height_diff < 0:  # need to scale up the height
        resized_image, horizontal_seams = scale_up(resized_width_image, grayscale_resized_image, gradients,
                                                   abs(height_diff), forward_implementation, False)

    resized_image = np.rot90(resized_image, k=-1, axes=(0, 1))  # rotating CW
    horizontal_seams = np.rot90(horizontal_seams, k=-1, axes=(0, 1))  # rotating CW

    out_images_dict = {'resized': resized_image, 'vertical_seams': vertical_seams, 'horizontal_seams': horizontal_seams}

    return out_images_dict


def scale_down(image: NDArray,
               grayscale_image: NDArray,
               gradients: NDArray,
               dim_diff: int,
               is_forward: bool,
               is_vertical: bool):
    """
        :param image: ِnparray which represents an image
        :param grayscale_image: ِnparray which represents a grayscale image
        :param gradients: ِnparray which represents a gradients' matrix
        :param dim_diff: the desired difference in width / height
        :param is_forward: a boolean flag that indicates whether forward or basic implementation is used
        :param is_vertical: a boolean flag that indicates whether vertical or horizontal for seams coloring
        :return: resized_image scaled up width by dim_diff and seams image
    """
    indices_matrix, seams_matrix = calculate_seams(grayscale_image, gradients, dim_diff, is_forward)
    resized_image = create_original_without_seams(image, indices_matrix)
    rows_range = np.arange(image.shape[0])
    for seam_number in range(dim_diff):
        image[rows_range, seams_matrix[seam_number, :]] = [255, 0, 0] if is_vertical else [0, 0, 0]
    return resized_image, image


def scale_up(image: NDArray,
             grayscale_image: NDArray,
             gradients: NDArray,
             dim_diff: int,
             is_forward: bool,
             is_vertical: bool):
    """
        :param image: ِnparray which represents an image
        :param grayscale_image: ِnparray which represents a grayscale image
        :param gradients: ِnparray which represents a gradients' matrix
        :param dim_diff: the desired difference in width / height
        :param is_forward: a boolean flag that indicates whether forward or basic implementation is used
        :param is_vertical: a boolean flag that indicates whether vertical or horizontal for seams coloring
        :return: resized_image scaled up width by dim_diff and seams image.
    """
    indices_matrix, seams_matrix = calculate_seams(grayscale_image, gradients, dim_diff, is_forward)
    resized_image = create_original_with_dup_seams(image, seams_matrix, dim_diff)
    rows_range = np.arange(image.shape[0])
    for seam_number in range(dim_diff):
        image[rows_range, seams_matrix[seam_number, :]] = [255, 0, 0] if is_vertical else [0, 0, 0]
    return resized_image, image


def calculate_seams(grayscale_image: NDArray, gradients: NDArray, dim_diff: int, is_forward: bool):
    indices_matrix = np.indices((grayscale_image.shape[0], grayscale_image.shape[1]))[1]
    seams_matrix = np.zeros((dim_diff, grayscale_image.shape[0]), dtype=int)

    for seam_number in range(dim_diff):
        Cl, Cv, Cr = compute_forward_costs(grayscale_image, is_forward)
        cost_matrix = calculate_cost_matrix(grayscale_image, gradients, Cl, Cv, Cr)
        best_gray_seam, best_orig_seam = find_best_seam(cost_matrix, gradients, indices_matrix, Cl, Cv)

        seams_matrix[seam_number, :] = best_orig_seam
        grayscale_image, indices_matrix, gradients = remove_seam(grayscale_image,
                                                                 indices_matrix,
                                                                 gradients,
                                                                 best_gray_seam)
    return indices_matrix, seams_matrix


def calculate_cost_matrix(grayscale_image: NDArray,
                          E: NDArray,
                          left_cost: NDArray,
                          vertical_cost: NDArray,
                          right_cost: NDArray):
    M = np.zeros_like(grayscale_image)
    M[0, :] = E[0, :]
    for row_index in range(1, M.shape[0]):
        shift_right_row = np.concatenate((np.array([256.]), M[row_index - 1, 0:-1]))
        shift_left_row = np.concatenate((M[row_index - 1, 1:], np.array([256.])))
        temp_min = np.fmin(M[row_index - 1, :] + vertical_cost[row_index, :], shift_left_row + right_cost[row_index, :])
        M[row_index] = E[row_index] + np.fmin(temp_min, shift_right_row + left_cost[row_index, :])
    return M


def find_best_seam(M: NDArray, E: NDArray, indices_matrix: NDArray, Cl: NDArray, Cv: NDArray):
    best_gray_seam = np.zeros((M.shape[0],), dtype=int)
    best_gray_seam[-1] = np.argmin(M[-1, :])
    best_orig_seam = np.zeros((M.shape[0],), dtype=int)

    for k in range(1, M.shape[0]):
        i = -k
        best_prev_index = best_gray_seam[i + 1]
        j = best_prev_index
        is_right_edge = best_prev_index == M.shape[1] - 1
        is_left_edge = best_prev_index == 0
        if is_left_edge and is_right_edge:
            min_column_index = j
        elif is_left_edge:
            if M[i, j] == E[i, j] + M[i-1, j] + Cv[i, j]:
                min_column_index = j
            else:
                min_column_index = j+1
        elif is_right_edge:
            if M[i, j] == E[i, j] + M[i-1, j] + Cv[i, j]:
                min_column_index = j
            else:
                min_column_index = j-1
        else:
            if M[i, j] == E[i, j] + M[i-1, j] + Cv[i, j]:
                min_column_index = j
            elif M[i, j] == E[i, j] + M[i-1, j-1] + Cl[i, j]:
                min_column_index = j-1
            else:
                min_column_index = j+1

        best_gray_seam[i-1] = min_column_index
    for row_index in range(M.shape[0]):
        best_orig_seam[row_index] = indices_matrix[row_index, best_gray_seam[row_index]]
    return best_gray_seam, best_orig_seam


def remove_seam(grayscale_image: NDArray, indices_matrix: NDArray, gradients: NDArray, seam: NDArray):
    grayscale_image = remove_seam_from_matrix(grayscale_image, seam)
    indices_matrix = remove_seam_from_matrix(indices_matrix, seam)
    gradients = remove_seam_from_matrix(gradients, seam)
    return grayscale_image, indices_matrix, gradients


def remove_seam_from_matrix(matrix: NDArray, seam: NDArray):
    for row_index in range(matrix.shape[0]):
        matrix[row_index, :] = np.roll(matrix[row_index, :], shift=-1 * seam[row_index], axis=0)
    matrix = np.delete(matrix, 0, 1)  # deleting first column
    for row_index in range(matrix.shape[0]):
        matrix[row_index, :] = np.roll(matrix[row_index, :], shift=seam[row_index], axis=0)
    return matrix


def create_original_without_seams(image: NDArray, indices_matrix: NDArray):
    resized_image = np.zeros((indices_matrix.shape[0], indices_matrix.shape[1], 3))
    for row_index in range(image.shape[0]):
        resized_image[row_index, :] = np.take(image[row_index, :], indices_matrix[row_index, :], axis=0)
    return resized_image


def create_original_with_dup_seams(image: NDArray, seams_matrix: NDArray, dim_diff: int):
    seams_t = seams_matrix.T
    resized_image = np.zeros((image.shape[0], image.shape[1] + dim_diff, 3))
    for row_index in range(image.shape[0]):
        values = np.take(image[row_index, :], seams_t[row_index, :], axis=0)
        resized_image[row_index, :] = np.insert(image[row_index, :],
                                                seams_t[row_index, :],
                                                values, axis=0)
    return resized_image


def compute_forward_costs(grayscale_image: NDArray, is_forward: bool):
    left_cost = np.zeros_like(grayscale_image)
    vertical_cost = np.zeros_like(grayscale_image)
    right_cost = np.zeros_like(grayscale_image)
    if not is_forward:
        return left_cost, vertical_cost, right_cost

    zero_column = np.broadcast_to([0.], [grayscale_image.shape[0], 1])
    zero_row = np.broadcast_to([0.], [1, grayscale_image.shape[1]])
    left_neighbors = np.concatenate([zero_column, grayscale_image[:, 0:-1]], axis=1)
    right_neighbors = np.concatenate([grayscale_image[:, 1:], zero_column], axis=1)
    upper_neighbors = np.concatenate([zero_row, grayscale_image[0:-1, :]], axis=0)
    common_cost = np.abs(left_neighbors - right_neighbors)
    left_cost = common_cost + np.abs(upper_neighbors - left_neighbors)
    vertical_cost = common_cost
    right_cost = common_cost + np.abs(upper_neighbors - right_neighbors)

    return left_cost, vertical_cost, right_cost



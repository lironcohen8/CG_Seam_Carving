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
    vertical_seams = image
    horizontal_seams = image
    resized_width_image = image
    if width_diff > 0:  # need to scale down the width
        resized_width_image, vertical_seams = scale_down(image, grayscale_original_image, gradients, width_diff,
                                                         forward_implementation, True)
    elif width_diff < 0:  # need to scale up the width
        resized_width_image, vertical_seams = scale_up(image, grayscale_original_image, gradients, abs(width_diff),
                                                       forward_implementation, True)

    grayscale_resized_image = utils.to_grayscale(resized_width_image)
    resized_width_image = np.rot90(resized_width_image, k=1, axes=(0, 1))  # rotating CCW
    grayscale_resized_image = np.rot90(grayscale_resized_image, k=1, axes=(0, 1))  # rotating CCW
    gradients = np.rot90(gradients, k=1, axes=(0, 1))  # rotating CCW
    resized_image = resized_width_image
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
        if is_forward:
            cost_matrix = calculate_cost_matrix_forward(grayscale_image, gradients)
            best_seam = find_best_seam_forward(cost_matrix, indices_matrix)
        else:
            cost_matrix = calculate_cost_matrix_basic(grayscale_image, gradients)
            best_seam = find_best_seam_basic(cost_matrix, indices_matrix)

        # print(seams_matrix[:, seam_number])
        seams_matrix[seam_number, :] = best_seam
        grayscale_image, indices_matrix, gradients = remove_seam(grayscale_image, indices_matrix, gradients, best_seam)

    return indices_matrix, seams_matrix


def calculate_cost_matrix_basic(grayscale_image: NDArray, E: NDArray):
    M = np.zeros_like(grayscale_image)
    M[0, :] = E[0, :]
    # column_256 = np.broadcast_to([256.], [grayscale_image.shape[1], 1])
    for row_index in range(1, M.shape[0]):
        shift_right_row = np.concatenate((np.array([256.]), M[row_index - 1, 0:-1]))
        shift_left_row = np.concatenate((M[row_index - 1, 1:], np.array([256.])))
        M[row_index] = E[row_index] + np.fmin(M[row_index - 1, :], shift_left_row, shift_right_row)
    return M


def find_best_seam_basic(cost_matrix: NDArray, indices_matrix: NDArray):
    print("cost: ", cost_matrix.shape)
    print("indices: ", indices_matrix.shape)
    best_orig_seam = np.zeros((cost_matrix.shape[0],), dtype=int)
    best_orig_seam[-1] = np.argmin(cost_matrix[-1, :])
    for i in range(2, cost_matrix.shape[0] + 1):
        row_index = -i
        best_prev_index = best_orig_seam[row_index + 1]
        is_right_edge = best_prev_index == cost_matrix.shape[1] - 1
        is_left_edge = best_prev_index == 0

        if is_left_edge and is_right_edge:  # one column
            min_column_index = cost_matrix[row_index, best_prev_index]
        elif is_left_edge:
            print("LE row index: ", row_index, " best prev index: ", best_prev_index)
            candidates = [cost_matrix[row_index, best_prev_index],
                          cost_matrix[row_index, best_prev_index + 1]]
            min_column_index = best_prev_index + candidates.index(min(candidates))
        elif is_right_edge:
            print("RE row index: ", row_index, " best prev index: ", best_prev_index)
            candidates = [cost_matrix[row_index, best_prev_index - 1],
                          cost_matrix[row_index, best_prev_index]]
            min_column_index = best_prev_index - 1 + candidates.index(min(candidates))
            print("addition: ", candidates.index(min(candidates)))
            print("min_column_index: ", min_column_index)
        else:
            print("BOTH row index: ", row_index, " best prev index: ", best_prev_index)
            candidates = [cost_matrix[row_index, best_prev_index - 1],
                          cost_matrix[row_index, best_prev_index],
                          cost_matrix[row_index, best_prev_index + 1]]
            min_column_index = best_prev_index - 1 + candidates.index(min(candidates))

        best_orig_seam[row_index] = min_column_index
    print("found seam!")
    for row_index in range(cost_matrix.shape[0]):
        best_orig_seam[row_index] = indices_matrix[row_index, best_orig_seam[row_index]]
    return best_orig_seam


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
    resized_image = np.empty((indices_matrix.shape[0], indices_matrix.shape[1], 3))
    print("shape of image is: ", image.shape)
    for row_index in range(image.shape[0]):
        print("in shape: ", image[row_index, :].shape)
        print("in indices: ", indices_matrix[row_index, :].shape)
        resized_image[row_index, :] = np.take(image[row_index, :], indices_matrix[row_index, :], axis=0)
    return resized_image
    # for seam in seams_matrix:
    #   image = remove_seam_from_matrix(image, seam)
    # return image


def create_original_with_dup_seams(image: NDArray, seams_matrix: NDArray, dim_diff: int):
    seams_t = seams_matrix.T
    resized_image = np.empty((image.shape[0], image.shape[1] + dim_diff, 3))
    for row_index in range(image.shape[0]):
        values = np.take(image[row_index, :], seams_t[row_index, :], axis=0)
        resized_image[row_index, :] = np.insert(image[row_index, :],
                                                seams_t[row_index, :],
                                                values, axis=0)


    return resized_image
    # for seam in seams_matrix:
    #   print("range is: ", image.shape[0])
    #  for row_index in range(image.shape[0]):
    #     print("row is: ", row_index)
    #    image[row_index, :] = np.roll(image[row_index, :], shift=seam[row_index])
    # image = np.insert(image, 0, image[0, :], 1)  # duplicating first column
    # for row_index in range(image.shape[0]):
    #   image[row_index, :] = np.roll(image[row_index, :], shift=-1 * seam[row_index])
    # return image


def calculate_cost_matrix_forward(grayscale_image: NDArray, E: NDArray):
    # TODO continue
    pass


def find_best_seam_forward(cost_matrix: NDArray, indices_matrix: NDArray):
    # TODO continue
    pass

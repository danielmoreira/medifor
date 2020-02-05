"""
Implements a content matcher of image pairs.
"""

import math
import numpy
import cv2


# Keeps halving the given image size up to the point of having less pixels than <max_image_size>.
# The image aspect ratio is kept.
# Returns the resized image.
def _decrease_image_if_necessary(image, max_image_size):
    while image.shape[0] * image.shape[1] > max_image_size:
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    return image


# Selects matches that are geometrically consistent with the i-th and j-th matches
# of the given list of image matches.
def _select_i_j_consistent_matches(matches, i, j, query_keypoints, train_keypoints, displacement_tolerance_threshold):
    # holds the selected matches
    selected_matches = []

    # the reference i-th and j-th matches must refer to different pairs of interest points
    if matches[i].queryIdx == matches[j].queryIdx or matches[i].trainIdx == matches[j].trainIdx:
        return selected_matches

    # controls the interest points that are already part of a valid match
    used_query_idx = [matches[i].queryIdx, matches[j].queryIdx]
    used_train_idx = [matches[i].trainIdx, matches[j].trainIdx]

    # query and train interest points in matrix format
    query_points = numpy.array([[query_keypoints[0].pt[0], query_keypoints[0].pt[1]]])
    for k in range(1, len(query_keypoints)):
        query_points = numpy.append(query_points, [[query_keypoints[k].pt[0], query_keypoints[k].pt[1]]], 0)

    train_points = numpy.array([[train_keypoints[0].pt[0], train_keypoints[0].pt[1]]])
    for k in range(1, len(train_keypoints)):
        train_points = numpy.append(train_points, [[train_keypoints[k].pt[0], train_keypoints[k].pt[1]]], 0)

    # puts query and train images in the same scale
    query_i_point = query_points[matches[i].queryIdx]
    query_j_point = query_points[matches[j].queryIdx]
    query_ij_distance = math.sqrt(
        (query_i_point[0] - query_j_point[0]) ** 2 + (query_i_point[1] - query_j_point[1]) ** 2)

    train_i_point = train_points[matches[i].trainIdx]
    train_j_point = train_points[matches[j].trainIdx]
    train_ij_distance = math.sqrt(
        (train_i_point[0] - train_j_point[0]) ** 2 + (train_i_point[1] - train_j_point[1]) ** 2)

    distance_rate = 1.0
    if train_ij_distance > 0.0:
        distance_rate = query_ij_distance / train_ij_distance

    if distance_rate != 0.0:
        if distance_rate > 1.0:
            scale_matrix = numpy.zeros((3, 3))
            scale_matrix[0, 0] = distance_rate
            scale_matrix[1, 1] = distance_rate
            scale_matrix[2, 2] = 1.0
            train_points = cv2.perspectiveTransform(numpy.float32([train_points]), scale_matrix)[0]

        elif distance_rate < 1.0:
            scale_matrix = numpy.zeros((3, 3))
            scale_matrix[0, 0] = 1.0 / distance_rate
            scale_matrix[1, 1] = 1.0 / distance_rate
            scale_matrix[2, 2] = 1.0
            query_points = cv2.perspectiveTransform(numpy.float32([query_points]), scale_matrix)[0]

        # computes the rotation of the train image towards the query image
        query_angle = math.atan2(query_j_point[1] - query_i_point[1], query_j_point[0] - query_i_point[0])
        if query_angle < 0.0:
            query_angle = 2.0 * math.pi + query_angle

        train_angle = math.atan2(train_j_point[1] - train_i_point[1], train_j_point[0] - train_i_point[0])
        if train_angle < 0.0:
            train_angle = 2.0 * math.pi + train_angle

        train_angle_correction = query_angle - train_angle
        sine = math.sin(train_angle_correction)
        cosine = math.cos(train_angle_correction)

        rotation_translation_matrix = numpy.zeros((3, 3))
        rotation_translation_matrix[0, 0] = cosine
        rotation_translation_matrix[0, 1] = sine
        rotation_translation_matrix[1, 0] = -sine
        rotation_translation_matrix[1, 1] = cosine
        rotation_translation_matrix[2, 2] = 1.0

        # computes the translation of the train image towards the query image
        rotation_translation_matrix[0, 2] = query_i_point[0] - train_i_point[0]
        rotation_translation_matrix[1, 2] = query_i_point[1] - train_i_point[1]

        # rotates and translates the train image
        if distance_rate != 0.0:
            train_points = cv2.perspectiveTransform(numpy.float32([train_points]), rotation_translation_matrix)[0]

        # filters the geometrically consistent matches
        for k in range(len(matches)):
            # avoids computing already usd interest points
            if matches[k].queryIdx in used_query_idx or matches[k].trainIdx in used_train_idx:
                continue

            query_x = query_points[matches[k].queryIdx][0]
            query_y = query_points[matches[k].queryIdx][1]

            train_x = train_points[matches[k].trainIdx][0]
            train_y = train_points[matches[k].trainIdx][1]

            if math.sqrt((query_x - train_x) ** 2 + (query_y - train_y) ** 2) < displacement_tolerance_threshold:
                selected_matches.append(k)
                used_query_idx.append(matches[k].queryIdx)
                used_train_idx.append(matches[k].trainIdx)

    # returns the selected matches
    return selected_matches


# Selects matches that are geometrically consistent.
def _select_consistent_matches(matches, query_keypoints, train_keypoints, reference_match_depth,
                               displacement_tolerance_threshold):
    match_count = len(matches)
    if reference_match_depth > match_count:
        reference_match_depth = match_count

    selected_matches = []
    used_matches = []

    for i in range(reference_match_depth - 1):
        if i not in used_matches:
            for j in range(i + 1, reference_match_depth):
                if j not in used_matches:
                    consistent_matches = _select_i_j_consistent_matches(matches, i, j, query_keypoints, train_keypoints,
                                                                        displacement_tolerance_threshold)
                    if len(consistent_matches) > 0:
                        if len(consistent_matches) > len(selected_matches):
                            selected_matches = consistent_matches[:]

                        used_matches = used_matches + consistent_matches

    answer = []
    for i in selected_matches:
        answer.append(matches[i])
    sorted(answer, key=lambda m: m.distance)

    return answer


# Performs geometrically consistent matching of two given images,
# based on their given interest points and descriptions.
# Returns the obtained matches.
def match(keypoints1, descriptions1, keypoints2, descriptions2, nndr_threshold=0.75, reference_match_depth=25,
          displacement_tolerance_threshold=50):
    # if there are more points in image 1 than in 2, swaps them
    did_swap = False
    if len(keypoints1) < len(keypoints2):
        did_swap = True
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1

    # holds the good matches
    good_matches = []

    # finds the good matches between the interest points
    first_and_second_matches = []
    if len(descriptions1) > 0 and len(descriptions2) > 0:
        matcher = cv2.BFMatcher()
        first_and_second_matches = matcher.knnMatch(descriptions1, descriptions2, k=2)

    try:
        for i, (a, b) in enumerate(first_and_second_matches):
            if b.distance != 0 and a.distance / b.distance < nndr_threshold:
                good_matches.append(a)
        sorted(good_matches, key=lambda m: m.distance)
        good_matches = _select_consistent_matches(good_matches, keypoints1, keypoints2, reference_match_depth,
                                                  displacement_tolerance_threshold)
    except:
        good_matches = []  # no good matches

    # re-swaps the matches, if it is the case
    if did_swap:
        for match in good_matches:
            match.queryIdx, match.trainIdx = match.trainIdx, match.queryIdx

    # returns the good matches
    return good_matches


# Warps image 1 towards image 2 and computes the mask of image 2
# that highlights the difference ROI between the two images.
def compute_context_mask(keypoints1, image1, keypoints2, image2, matches, max_image_size=100000):
    gs_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gs_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    img1_points = []
    img2_points = []
    for m in matches:
        img1_points.append(keypoints1[m.queryIdx].pt)
        img2_points.append(keypoints2[m.trainIdx].pt)
    img1_points = numpy.array(img1_points)
    img2_points = numpy.array(img2_points)

    homography, _ = cv2.findHomography(img1_points, img2_points, cv2.LMEDS)
    if homography is not None:
        warp1_img = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))
        gs_warp1_img = cv2.warpPerspective(gs_image1, homography, (gs_image2.shape[1], gs_image2.shape[0]))

        # reduces images if necessary
        gs_warp1_img = _decrease_image_if_necessary(gs_warp1_img, max_image_size)
        gs_image2 = _decrease_image_if_necessary(gs_image2, max_image_size)

        diff_mask = numpy.abs(gs_image2 - gs_warp1_img)
        cv2.medianBlur(diff_mask, 11, diff_mask)
        _, diff_mask = cv2.threshold(diff_mask, 20, 255, cv2.THRESH_BINARY)

        # open operation
        morph_kernel = numpy.ones((11, 11), numpy.uint8)
        diff_mask = cv2.erode(diff_mask, morph_kernel)
        diff_mask = cv2.dilate(diff_mask, morph_kernel)

        # close operation
        diff_mask = cv2.dilate(diff_mask, morph_kernel)
        diff_mask = cv2.erode(diff_mask, morph_kernel)

        # output
        if diff_mask.shape[0] != warp1_img.shape[0] or diff_mask.shape[1] != warp1_img.shape[1]:
            diff_mask = cv2.resize(diff_mask, (warp1_img.shape[1], warp1_img.shape[0]))
        return diff_mask.astype(numpy.uint8), warp1_img

    return numpy.zeros(gs_image2.shape, dtype=numpy.uint8), cv2.resize(image1, (image2.shape[1], image2.shape[0]))

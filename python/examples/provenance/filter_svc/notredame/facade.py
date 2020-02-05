# Implements a facade to call Notre Dame's fitlering implementation.

import sys
import os
import numpy
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/image_lib/')  # image library path
import image_reader
import image_descriptor
import image_matcher


# Reads an image, given its file path.
# Returns the obtained image.
def read_image(image_file_path):
    image = image_reader.read(image_file_path)
    if image is None:
        image = numpy.zeros((64, 64, 3), dtype=numpy.uint8)  # empty image, to keep program running
        print('[WARNING] Image', image_file_path, 'has no content.')

    return image


# Describes a given image.
# Returns its interest points and respective feature vectors (descriptions).
def describe_image(image, mask=None):
    keypoints, descriptions = image_descriptor.surf_detect_rsift_describe(image, mask)

    if len(keypoints) == 0:
        return [], []

    return keypoints, descriptions


# Builds an image rank containing <rank_size> images,
# based on the given previously obtained <description_search> result.
def build_image_rank(description_search, rank_size):
    # builds the image rank, given the description-wise search result
    image_rank = []

    # rank data
    votes = {}
    labels = {}
    max_vote = 0.0

    # for each description-wise result
    for item in description_search:
        for element in item['value']:
            gallery_image_id = int(element[0])
            gallery_image_label = element[1]
            gallery_image_pos = int(element[2])

            if gallery_image_id not in votes.keys():
                votes[gallery_image_id] = 0.0
                labels[gallery_image_id] = gallery_image_label

            # vote is weighted by description position
            votes[gallery_image_id] = votes[gallery_image_id] + (1.0 / gallery_image_pos)

            if votes[gallery_image_id] > max_vote:
                max_vote = votes[gallery_image_id]

    # adds content to the image rank
    for gallery_image_id in votes.keys():
        score = votes[gallery_image_id] / max_vote
        image_rank.append((gallery_image_id, labels[gallery_image_id], score))

    # sorts, trims, and returns the image rank
    image_rank = sorted(image_rank, key=lambda x: x[2], reverse=True)[:rank_size]
    return image_rank


# Obtains the context mask between the given query and the given gallery image.
# Parameters:
# <query_image> - The query image whose content will be masked by the computed context mask.
# <query_keypoints> - The previously detected query keypoints.
# <query_descriptions> - The description of the query keypoints.
# <gallery_image> - The gallery image whose content will be compared to the query.
# <gallery_image_keypoints> - The previously detected gallery image keypoints.
# <gallery_image_descriptions> - The description of the gallery image keypoints.
# Returns the context mask, which highlights the content differences between the query and the gallery image,
# after keypoint-based image alignment. The returned mask refers to the query content and therefore has it size.
# Returns None if no context mask could be computed (i.e., there were not enough matches between the keypoints of
# the query and of the gallery image.
def compute_context_mask(query_image, query_keypoints, query_descriptions,
                         gallery_image, gallery_image_keypoints, gallery_image_descriptions):
    # obtains the geometrically consistent matches between the query and gallery image
    good_matches = image_matcher.match(gallery_image_keypoints, gallery_image_descriptions,
                                       query_keypoints, query_descriptions)

    # if there are enough matches for homography
    if len(good_matches) >= 4:
        return image_matcher.compute_context_mask(gallery_image_keypoints, gallery_image,
                                                  query_keypoints, query_image,
                                                  good_matches)
    else:
        return None, None  # no mask could be computed


# Combines a given set of context masks into a single one.
# Parameter:
# <context_mask_list> - The list of context masks.
# Returns the OR-merged mask.
def merge_context_masks(context_mask_list):
    if len(context_mask_list) > 0:
        output = context_mask_list[0]
        for i in range(1, len(context_mask_list)):
            output = cv2.bitwise_or(output, context_mask_list[i])
        return output

    else:
        return None


# Merges a given set of image ranks.
# <image_ranks> - Takes a list of image ranks and combines them into a single one.
# <rank_size> - Maximum size of the obtained image rank.
# Returns the obtained image rank.
def merge_ranks(image_ranks, rank_size=500):
    if len(image_ranks) == 0:
        return []

    # holds the votes for each gallery image
    rank_dict = {}

    for i in range(len(image_ranks)):
        for item in image_ranks[i]:
            item_key = item[1].split('/')[-1]  # reg and flp have the same key
            item_score = item[2]
            if i > 0:
                item_score = item_score / 2.0  # tier-1 has higher weight

            # if the current item is not in the dictionary yet,
            # or its respective dictionary score is below the current one,
            # updates the dictionary with the current item
            if item_key not in rank_dict or rank_dict[item_key][2] < item_score:
                rank_dict[item_key] = (item[0], item[1], item_score)

    # computes the final rank
    image_rank = []
    for key in rank_dict.keys():
        image_rank.append(rank_dict[key])

    # sorts and trims the image rank
    image_rank = sorted(image_rank, key=lambda x: x[2], reverse=True)[:rank_size]
    return image_rank

import sys
import os

import requests
import numpy as np
from pprint import pprint

from medifor.v1.provenanceservice import ProvenanceService, query_index
from medifor.v1 import provenance_pb2
import PIL.Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/notredame/')
import facade

index_url = ""
GALLERY_FOLDER = "/data/world/"
QUERY_EXPANSION_FAIL_TOLERANCE = 10  # tolerates failure in geometrically consistent matches n-times in a sequence


# Auxiliary method. Queries a given <query_image> using the given <query_func>, whose content
# is masked/selected by <query_mask>.
# The obtained image rank has up to <result_limit> images.
# Parameter <is_simple_query> is TRUE if no query expansion must be performed, FALSE otherwise.
def _query_image(query_image, query_func, result_limit, query_mask=None, is_simple_query=False):
    # describes the given image
    query_keypoints, query_descriptions = facade.describe_image(query_image, mask=query_mask)
    if len(query_descriptions) == 0:
        return []

    # queries the indices
    description_search = query_func(query_descriptions.tolist(), index_url, result_limit)

    # obtains the 1st tier image rank
    image_rank = facade.build_image_rank(description_search, result_limit)

    # if we have a simple query, returns the obtained image rank
    if is_simple_query:
        return image_rank

    # else, performs query expansion
    expansion_ranks = [image_rank]
    context_masks = []
    gallery_selection = []

    # for each image obtained in the 1st-tier rank
    mask_computation_failures = 0
    for item in image_rank:
        # reads the current image
        gallery_image_id_parts = item[1].split('/')
        is_flip = (gallery_image_id_parts[0] == 'flp')
        gallery_image_filename = gallery_image_id_parts[1]

        gallery_image = facade.read_image(GALLERY_FOLDER + '/' + gallery_image_filename)
        if is_flip:
            gallery_image = np.fliplr(gallery_image)

        # describes the current rank image
        gallery_image_keypoints, gallery_image_descriptions = facade.describe_image(gallery_image)

        # obtains the query content mask
        query_mask, warp_gallery_image = facade.compute_context_mask(query_image, query_keypoints, query_descriptions,
                                                                     gallery_image, gallery_image_keypoints,
                                                                     gallery_image_descriptions)

        if query_mask is not None:
            mask_computation_failures = 0  # restart failure count
            context_masks.append(query_mask)
            gallery_selection.append(warp_gallery_image)
        else:
            mask_computation_failures = mask_computation_failures + 1  # one more failure
            if mask_computation_failures >= QUERY_EXPANSION_FAIL_TOLERANCE:  # n failures in a sequence
                break  # time to stop looking for candidates to the query_expansion

    # re-queries the original query, this time using a mask that is the combination of the obtained context masks
    query_mask = facade.merge_context_masks(context_masks)
    expansion_ranks.append(
        _query_image(query_image, query_func, result_limit, query_mask=query_mask, is_simple_query=True))

    # Queries the selected 1st-tier rank images
    for i in range(len(context_masks)):
        expansion_ranks.append(
            _query_image(gallery_selection[i], query_func, result_limit,
                         query_mask=context_masks[i], is_simple_query=True))

    # merges ranks and returns the result
    image_rank = facade.merge_ranks(expansion_ranks, rank_size=result_limit)
    return image_rank


def filter(req, resp, query_func):
    """This is the function that does the filtering.  It makes calls to the index(s) through
    a REST API using the requests library.  You can modify this to your own needs, or use the
    function call provided here.  The image is 'encoded' into a 100 dimensional numpy vector as
    a stand in for the feature vector.  You can encode this as you see fit since you control the
    decoding in the index."""

    # def encode(img):
    #     """Dummy encoding function to take an image and turn it into a 100-D vector for index search"""
    #     seed = np.sum(img)
    #     np.random.seed(seed)
    #     return [[float(v) for v in q] for q in np.random.random((1, 100)).astype("float32")]
    #
    # with PIL.Image.open(req.image.uri) as img:
    #     img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    #
    # # index_results is a list of results across all index shards. Results contained in index_results["value"]
    # index_results = query_func(encode(img), index_url, req.result_limit)
    #
    # matches = []
    # # This section unpacks the results from the index service. Your implementation will likely be different.
    # for img_matches in index_results:
    #     for i, img_id in enumerate(img_matches["value"]['ids']):
    #         new_match = provenance_pb2.ImageMatch()
    #         if not img_id:
    #             new_match.image_id = str(img_matches["value"]["fids"][i])
    #         else:
    #             new_match.image_id = str(img_id)
    #         new_match.score = img_matches["value"]["dists"][i]
    #         matches.append(new_match)
    #
    # resp.matches.extend(matches)

    # reads the given image and queries it with query expansion
    query_image = facade.read_image(req.image.uri)
    image_rank = _query_image(query_image, query_func, req.result_limit, is_simple_query=False)

    # prepares the service output
    matches = []
    for item in image_rank:
        new_match = provenance_pb2.ImageMatch()
        new_match.image_id = str(item[1].split('/')[-1])
        new_match.score = item[2]
        matches.append(new_match)

    resp.matches.extend(matches)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    # Accepts a list of urls. Each url needs to be preceded by the flag.
    p.add_argument('--url', type=str, help='URL of host to talk to', action='append')
    args = p.parse_args()
    if not args.url:
        # Adds a default if left blank
        args.url = ['http://localhost:8080/search']
    index_url = args.url
    svc = ProvenanceService()
    svc.RegisterProvenanceFiltering(filter)
    svc.Run()

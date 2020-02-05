#!/usr/bin/env python3
import sys
import os

from medifor.v1.provenanceservice import IndexSvc
import faiss
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/notredame/')
import facade


def index(img, limit):
    """ 
    Function for to retrieve matching images from index shard. This function can essentially be
    whatever you want as long as the returned value is json serializable. The provenanceservice
    library will take care of the REST calls. In your analytic function you will receive a list of
    whatever is returned by this function (because it assumes multiple shards).
    """

    # # Standard faiss index search function
    # D, I = idx.search(np.array(img).astype('float32'), limit)
    #
    # # TODO process and package the results as needed. The statement below is just meant to be illustrative.
    # # Whatever is returned (e.g., results in this case) will be sent via REST to your filtering analytic as long as it is JSON serializable.
    # results = [{
    #             'fids': [int(x) for x in ids],
    #             'ids': [id_map.get(int(x)) for x in ids],
    #             'dists': [float(x) for x in dists],
    #             } for ids, dists in zip(I,D)][0]
    #
    # return results

    results = []
    for i in range(len(idxs)):
        current_results = facade.query_image_from_index(np.array(img).astype('float32'), idxs[i], id_maps[i])
        for result in current_results:
            results.append(result)

    return results


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    # Arguments for the faiss index service.
    # Port: The port that the service will run on.
    # Index: The location of the faiss index file
    # Map: Used to map index IDs (integers) to actual image IDs (NIST provided MD5 hashes)
    p.add_argument('--port', type=int, default=8080, help='Port to listen on')
    p.add_argument('--index', default=[], required=True, action="append", help="Location of FAISS index file.")
    # p.add_argument('--map', type=str, default='', help='Location of file mapping index IDs to other IDs as needed.')
    p.add_argument('--map', default=[], required=True, action="append", help='Location of file mapping index IDs.')
    args = p.parse_args()

    # Import the index
    # Just taking the first element for this example, but the index flag now supports multiple index files. Use them as you need.
    # idx = faiss.read_index(args.index[0])
    idxs = []
    for idx in args.index:
        idxs.append(faiss.read_index(args.index[0]))

    # Create the ID map
    # id_map = {}
    # if args.map:
    #     with open(args.map, 'rb') as f:
    #         for line in f:
    #             line = line.strip()
    #             if line.startswith('#'): continue
    #             k, v = line.split(':')
    #             id_map[int(k.strip())] = v.strip()
    id_maps = []
    for id_map in args.map:
        id_maps.append(np.load(id_map, allow_pickle=True))

    idxsvc = IndexSvc(__name__, host="::", port=args.port)
    idxsvc.RegisterQuery(index)
    # idxsvc.set_map(id_map)
    idxsvc.run()

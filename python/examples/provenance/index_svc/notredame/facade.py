# Implements a facade to call Notre Dame's fitlering implementation.

# Auxiliary function to perform id position binary search (id mapping).
def _id_binary_search(image_id, image_list, begin_index, end_index):
    if end_index < begin_index:
        return begin_index - 1

    else:
        ref_index = int((begin_index + end_index) / 2)

        if image_id == image_list[ref_index]:
            # ignores mapping to empty descriptions
            while ref_index + 1 < len(image_list) and image_list[ref_index] == image_list[ref_index + 1]:
                ref_index = ref_index + 1

            return ref_index

        elif image_id < image_list[ref_index]:
            return _id_binary_search(image_id, image_list, begin_index, ref_index - 1)

        else:
            return _id_binary_search(image_id, image_list, ref_index + 1, end_index)


# Description-wise querying.
# Queries the given <query-descriptions> in the given <index>, whose metadata is stored in <index_map>.
def query_image_from_index(query_descriptions, index, index_map, knn=8):
    description_search = []

    # searches the query descriptions inside the given index
    distances, indices = index.search(query_descriptions, knn)

    # loads the index mapping of gallery descriptions to indexed images
    gallery_id_offset = index_map[0]
    gallery_image_labels = index_map[1]
    gallery_id_map = index_map[2]

    # organizes the description-wise image search
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            gallery_image_id = _id_binary_search(indices[i][j], gallery_id_map, 0, len(gallery_id_map) - 1)
            description_search.append(
                (str(gallery_image_id + gallery_id_offset), str(gallery_image_labels[gallery_image_id]), str(i + 1),
                 str(distances[i][j])))

    return description_search

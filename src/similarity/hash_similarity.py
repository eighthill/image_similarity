def hash_similarity(
    hash1,
    hash2
):

    distance = hash1 - hash2

    return 1 - distance / 64
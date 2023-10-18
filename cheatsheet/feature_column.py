from collections import namedtuple


class SparseFeat(namedtuple('sparse_feat', [])):
    pass


class DenseFeat(namedtuple('dense_feat', [])):
    pass


class SparseSeqFeat(namedtuple('sparse_sequence_feat', [])):
    pass


class DenseSeqFeat(namedtuple('dense_sequence_feat', [])):
    pass


def create_embedding_matrix():
    pass



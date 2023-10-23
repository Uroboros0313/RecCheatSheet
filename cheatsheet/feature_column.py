from typing import List, Dict
from collections import namedtuple, OrderedDict


class SparseFeat(namedtuple('sparse_feat', [])):
    pass


class DenseFeat(namedtuple('dense_feat', [])):
    pass


class SparseSeqFeat(namedtuple('sparse_sequence_feat', [])):
    pass


class DenseSeqFeat(namedtuple('dense_sequence_feat', [])):
    pass


def build_input_layers(feature_columns) -> Dict:
    pass
    

def build_embedding_lookup_dict() -> Dict:
    pass




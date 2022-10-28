# this code was provided by the authors of: Steven Cao, Nikita Kitaev, and Dan
# Klein. 2020. Multilingual Alignment of Contextual Word Representations. In
# International Conference on Learning Representation

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")


def keep_1to1(alignments):
    if len(alignments) == 0:
        return alignments

    counts1 = np.zeros(np.max(alignments[:,0]) + 1)
    counts2 = np.zeros(np.max(alignments[:,1]) + 1)

    for a in alignments:
        counts1[a[0]] += 1
        counts2[a[1]] += 1

    alignments2 = []
    for a in alignments:
        if counts1[a[0]] == 1 and counts2[a[1]] == 1:
            alignments2.append(a)
    return np.array(alignments2)


def normalize(vecs):
    norm = torch.linalg.norm(vecs)
    norm[norm < 1e-5] = 1
    normalized = vecs / norm
    return normalized


def hubness_CSLS(ann_1, ann_2, k=10):
    """
    Computes hubness r(x) of an embedding x, or the mean similarity of x to
    the K closest neighbors in Y. Used for the CSLS metric:
    CSLS(x, y) = 2cos(x,y) - r(x) - r(y)
    which penalizes words with high hubness, or a dense neighborhood.

    Uses k = 10, similarly to https://arxiv.org/pdf/1710.04087.pdf.
    """
    ann_1, ann_2 = normalize(ann_1), normalize(ann_2)
    sim = torch.mm(ann_1, ann_2.T) # words_1 x words_2
    return torch.topk(sim, k, axis=1)[0].mean(axis=1), torch.topk(sim.T, k, axis=1)[0].mean(axis=1)


def bestk_idx_CSLS(x, vecs, vec_hubness, k=5):
    """
    Looks for the k closest vectors using the CSLS metric, which is cosine
    similarity with a hubness penalty.

    Usage:
        hub_1, hub_2 = hubness_CSLS(vecs_1, vecs_2)
        # get word translations for vecs_1[0]
        best_k = bestk_idx_CSLS(vecs_1[0], vecs_2, hub_2)
    """
    x, vecs = normalize(x), normalize(vecs)
    sim = 2 * torch.matmul(vecs, x) - vec_hubness
    return torch.topk(sim, k)[1]

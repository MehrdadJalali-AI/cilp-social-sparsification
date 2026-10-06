"""NetworKit edge scores computed in a torch-free process.

NetworKit and PyTorch ship separate OpenMP runtimes; importing both in one process segfaults on
this platform (macOS arm64, networkit 11.0.1, torch 2.4.0). run_baselines.py therefore calls this
module as a subprocess:  python3 -m src.revision.nk_scores <und.npy> <n> <kind> <out.npy>
"""
import sys

import networkit as nk
import numpy as np


def main(und_path: str, n: int, kind: str, out_path: str) -> None:
    und = np.load(und_path)
    G = nk.Graph(n)
    for u, v in und.T.tolist():
        G.addEdge(u, v)
    G.indexEdges()
    if kind == "nk_local_degree":
        s = nk.sparsification.LocalDegreeScore(G)
    elif kind == "nk_local_similarity":
        tri = nk.sparsification.TriangleEdgeScore(G)
        tri.run()
        s = nk.sparsification.LocalSimilarityScore(G, tri.scores())
    else:
        raise ValueError(kind)
    s.run()
    sc = s.scores()
    by_pair = {}
    G.forEdges(lambda u, v, w, eid: by_pair.__setitem__((min(u, v), max(u, v)), sc[eid]))
    np.save(out_path, np.array([by_pair[(a, b)] for a, b in und.T.tolist()], dtype=np.float64))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])

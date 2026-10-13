"""Genome codes and the family tree for GAGA species.

Genome code: per-species gene-family copy numbers from GAGA's hierarchical orthogroups (31,540 HOGs x 163 ants),
log1p-transformed. These are the "gene knowledge" layer (which families a genome has, and how many copies),
not raw sequence.

Tree: the GAGA dated phylogeny (MCMCtree, Ma). Under Brownian motion the covariance between two species is the
time from the root to their most recent common ancestor, which gives the tree-only kernel.
"""
from __future__ import annotations

import gzip
import re

import numpy as np
import pandas as pd


def read_genecounts(path: str, ids: list[str] | None = None) -> pd.DataFrame:
    """Species x HOG copy-number table."""
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    df = df.T
    if ids is not None:
        df = df.loc[ids]
    return df


def genome_features(counts: pd.DataFrame, min_species: int = 3) -> pd.DataFrame:
    """log1p copy numbers of the HOGs that vary among the given species and are present in >= min_species."""
    x = np.log1p(counts.astype(float))
    present = (counts > 0).sum(0)
    keep = (present >= min_species) & (x.std(0) > 0)
    return x.loc[:, keep]


# --- Newick -------------------------------------------------------------------------------------------------

class Node:
    __slots__ = ("name", "length", "children", "parent")

    def __init__(self):
        self.name, self.length, self.children, self.parent = "", 0.0, [], None


def parse_newick(text: str) -> Node:
    text = text.strip().rstrip(";")
    tokens = re.findall(r"\(|\)|,|:[^,():;]+|[^,():;]+", text)
    root = cur = Node()
    for t in tokens:
        if t == "(":
            child = Node()
            child.parent = cur
            cur.children.append(child)
            cur = child
        elif t == ",":
            sib = Node()
            sib.parent = cur.parent
            cur.parent.children.append(sib)
            cur = sib
        elif t == ")":
            cur = cur.parent
        elif t.startswith(":"):
            cur.length = float(t[1:])
        else:
            cur.name = t.strip().strip("'")
    return root


def tips(node: Node) -> list[Node]:
    if not node.children:
        return [node]
    return [t for c in node.children for t in tips(c)]


def phylo_covariance(tree_text: str, ids: list[str]) -> pd.DataFrame:
    """Brownian-motion covariance: shared root-to-MRCA path length, for the listed tip IDs."""
    root = parse_newick(tree_text)
    depth: dict[int, float] = {}
    paths: dict[str, list[int]] = {}

    def walk(n: Node, d: float, path: list[int]):
        d = d + (n.length if n is not root else 0.0)
        depth[id(n)] = d
        path = path + [id(n)]
        if not n.children:
            paths[n.name] = path
        for c in n.children:
            walk(c, d, path)

    walk(root, 0.0, [])
    missing = [i for i in ids if i not in paths]
    if missing:
        raise KeyError(f"tips missing from tree: {missing[:5]}")
    C = np.zeros((len(ids), len(ids)))
    for a, ia in enumerate(ids):
        pa = paths[ia]
        for b in range(a, len(ids)):
            pb = paths[ids[b]]
            k = 0
            while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
                k += 1
            C[a, b] = C[b, a] = depth[pa[k - 1]]
    return pd.DataFrame(C, index=ids, columns=ids)


def patristic(C: pd.DataFrame) -> pd.DataFrame:
    d = np.diag(C.values)
    return pd.DataFrame(d[:, None] + d[None, :] - 2 * C.values, index=C.index, columns=C.columns)

"""Input-only feature extraction for DASE7506 MP2.

Every feature here is a function of the question, the candidate texts and the
frozen embeddings only. Nothing reads a reference answer or a question id.

Candidate layout is always [N, 4, 1024]; question q candidate j is X[q, j].
"""
from collections import Counter

import numpy as np

EMB_DIM = 1024
SURFACE_NAMES = ('chars', 'lines', 'digits', 'has_boxed', 'has_hash', 'has_equals',
                 'n_eq_pairs', 'n_latex', 'ends_boxed', 'n_number_tokens')
VOTE_NAMES = ('vote_count', 'n_valid', 'in_top_group', 'parseable', 'n_parseable')


# ----------------------------------------------------------------- surface
def surface_features(texts):
    """texts: list of strings -> [n, |SURFACE_NAMES|]."""
    out = np.zeros((len(texts), len(SURFACE_NAMES)), dtype=np.float64)
    for i, t in enumerate(texts):
        n_digits = sum(ch.isdigit() for ch in t)
        number_tokens = 0
        for tok in t.replace('=', ' ').replace(',', ' ').split():
            s = tok.strip('.$\\{}()[]')
            if s and (s[0].isdigit() or (s[0] in '+-' and len(s) > 1 and s[1].isdigit())):
                number_tokens += 1
        stripped = t.rstrip()
        out[i] = [
            len(t),
            t.count('\n') + 1,
            n_digits,
            1.0 if '\\boxed' in t else 0.0,
            1.0 if '####' in t else 0.0,
            1.0 if '=' in t else 0.0,
            t.count('='),
            t.count('\\'),
            1.0 if (stripped.endswith('$') and '\\boxed' in stripped[-40:]) else 0.0,
            number_tokens,
        ]
    return out


# ----------------------------------------------------------------- embeddings
def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def embedding_features(emb, mode):
    """emb: [N,4,1024]. mode in {raw, l2} -> same shape."""
    if mode == 'raw':
        return emb.astype(np.float64)
    if mode == 'l2':
        return l2_normalize(emb.astype(np.float64))
    raise ValueError(mode)


def relative_vectors(emb_norm):
    """Current vector minus the mean of the four candidates of the same question."""
    mean = emb_norm.mean(axis=1, keepdims=True)
    return emb_norm - mean


def cosine_summary(emb_norm):
    """[N,4,2]: mean and max cosine similarity to the other three candidates."""
    N = emb_norm.shape[0]
    out = np.zeros((N, 4, 2), dtype=np.float64)
    for q in range(N):
        V = emb_norm[q]                       # [4,1024] unit rows
        sim = V @ V.T
        mask = ~np.eye(4, dtype=bool)          # exclude self-similarity
        others = sim[mask].reshape(4, 3)
        out[q, :, 0] = others.mean(axis=1)
        out[q, :, 1] = others.max(axis=1)
    return out


# ----------------------------------------------------------------- votes
def vote_features(bank):
    """[N,4,|VOTE_NAMES|] built from parsed final answers of the same question.

    parse_failures are NOT pooled as a shared answer.
    """
    from common import parse_answer
    N = len(bank)
    out = np.zeros((N, 4, len(VOTE_NAMES)), dtype=np.float64)
    for q, row in enumerate(bank):
        parsed = [parse_answer(c['text']) for c in row['candidates']]
        counts = Counter(p for p in parsed if p is not None)
        n_valid = len(counts)
        n_parse = sum(p is not None for p in parsed)
        top = max(counts.values()) if counts else 0
        for j, p in enumerate(parsed):
            out[q, j] = [
                counts.get(p, 0) if p is not None else 0,
                n_valid,
                1.0 if (p is not None and counts.get(p, 0) == top and top > 0) else 0.0,
                1.0 if p is not None else 0.0,
                n_parse,
            ]
    return out


# ----------------------------------------------------------------- assembler
class FeatureSpec:
    """Recipe: which groups to concatenate, in order."""

    def __init__(self, emb_mode=None, use_relative=False, use_cos=False,
                 use_vote=False, use_surface=False):
        self.emb_mode = emb_mode
        self.use_relative = use_relative
        self.use_cos = use_cos
        self.use_vote = use_vote
        self.use_surface = use_surface

    @property
    def name(self):
        parts = []
        if self.emb_mode:
            parts.append(self.emb_mode)
        if self.use_relative:
            parts.append('rel')
        if self.use_cos:
            parts.append('cos')
        if self.use_vote:
            parts.append('vote')
        if self.use_surface:
            parts.append('surf')
        return '+'.join(parts) or 'empty'

    def dim(self):
        d = 0
        if self.emb_mode:
            d += EMB_DIM
        if self.use_relative:
            d += EMB_DIM
        d += 2 * int(self.use_cos) + len(VOTE_NAMES) * int(self.use_vote) \
            + len(SURFACE_NAMES) * int(self.use_surface)
        return d

    def blocks(self, bank, emb, need_norm_stats=False):
        """Return [N,4,dim] float64 and a list of (name, slice, needs_std)."""
        norm = embedding_features(emb, self.emb_mode) if self.emb_mode else None
        # always compute an L2 view if we need relative/cos
        l2view = norm if (self.emb_mode == 'l2') else (
            l2_normalize(emb.astype(np.float64)) if (self.use_relative or self.use_cos) else None)
        pieces, layout = [], []
        pos = 0
        if self.emb_mode:
            pieces.append(norm)
            layout.append((self.emb_mode, slice(pos, pos + EMB_DIM), True))
            pos += EMB_DIM
        if self.use_relative:
            pieces.append(relative_vectors(l2view))
            layout.append(('rel', slice(pos, pos + EMB_DIM), True))
            pos += EMB_DIM
        if self.use_cos:
            pieces.append(cosine_summary(l2view))
            layout.append(('cos', slice(pos, pos + 2), True))
            pos += 2
        if self.use_vote:
            pieces.append(vote_features(bank))
            layout.append(('vote', slice(pos, pos + len(VOTE_NAMES)), True))
            pos += len(VOTE_NAMES)
        if self.use_surface:
            texts = [c['text'] for row in bank for c in row['candidates']]
            s = surface_features(texts).reshape(len(bank), 4, len(SURFACE_NAMES))
            pieces.append(s)
            layout.append(('surf', slice(pos, pos + len(SURFACE_NAMES)), True))
            pos += len(SURFACE_NAMES)
        X = np.concatenate(pieces, axis=2) if pieces else np.zeros((len(bank), 4, 0))
        return X, layout

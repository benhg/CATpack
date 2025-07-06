import itertools as it
import json

BINS = 16
CHARS = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-,.!?@$#"

def _adjacency_free_pairs(n_bins=BINS):
    """Yield pairs (i,j) with j-i>1 to enforce a zero gap."""
    for i, j in it.combinations(range(n_bins), 2):
        if j - i > 1:          # forbid neighbours
            yield i, j

pairs = list(_adjacency_free_pairs())
assert len(pairs) >= len(CHARS), "Not enough codes!"

# Build code-book
codebook = {}
for ch, (i, j) in zip(CHARS, pairs):
    bits = ['0'] * BINS
    bits[BINS - 1 - i] = '1'   # MSB = lowest-freq bin (keep your ordering)
    bits[BINS - 1 - j] = '1'
    codebook[ch] = ''.join(bits)

# Always keep the all-ones pre/post-amble
codebook['^'] = '1' * BINS

# round-trip dictionary
reverse = {v: k for k, v in codebook.items()}

# --- quick sanity checks -----------------------------------------------------
from itertools import combinations
dmin = min(sum(a != b for a, b in zip(codebook[c1], codebook[c2]))
           for c1, c2 in combinations(CHARS, 2))
#assert dmin >= 4, f"Minimal inter-symbol distance only {dmin}!"

print("Generated", len(codebook)-1, "symbols, min distance =", dmin)

print(json.dumps(codebook, indent=2))

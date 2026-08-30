"""Compare two seed vocabulary files to understand differences in tokens, counts, and ranks."""

import argparse


def load_seed_vocab(path: str) -> dict[str, float]:
    vocab = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2:
                token, count_str = parts
                vocab[token] = float(count_str)
    return vocab


def compute_ranks(vocab: dict[str, float]) -> dict[str, int]:
    """Rank 1 = highest count."""
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    return {token: rank for rank, (token, _) in enumerate(sorted_tokens, 1)}


def main():
    parser = argparse.ArgumentParser(description="Compare two seed vocabulary files")
    parser.add_argument("file_a", help="First seed vocab file")
    parser.add_argument("file_b", help="Second seed vocab file")
    parser.add_argument("-n", type=int, default=20, help="Number of results per section (default: 20)")
    args = parser.parse_args()

    vocab_a = load_seed_vocab(args.file_a)
    vocab_b = load_seed_vocab(args.file_b)
    ranks_a = compute_ranks(vocab_a)
    ranks_b = compute_ranks(vocab_b)
    n = args.n

    tokens_a = set(vocab_a.keys())
    tokens_b = set(vocab_b.keys())
    added = tokens_b - tokens_a
    dropped = tokens_a - tokens_b
    shared = tokens_a & tokens_b

    print(f"File A: {args.file_a}")
    print(f"File B: {args.file_b}")
    print(f"  A tokens: {len(vocab_a):,}  |  B tokens: {len(vocab_b):,}")
    print(f"  Added: {len(added):,}  |  Dropped: {len(dropped):,}  |  Shared: {len(shared):,}")
    print()

    # --- Added tokens (sorted by count in B, descending) ---
    print(f"{'='*80}")
    print(f"ADDED in B (top {n} by count)")
    print(f"{'='*80}")
    added_sorted = sorted(added, key=lambda t: vocab_b[t], reverse=True)
    for token in added_sorted[:n]:
        print(f"  {repr(token):40s}  count={vocab_b[token]:>12,.1f}  rank={ranks_b[token]:>6,}")
    if len(added) > n:
        print(f"  ... and {len(added) - n:,} more")
    print()

    # --- Dropped tokens (sorted by count in A, descending) ---
    print(f"{'='*80}")
    print(f"DROPPED from A (top {n} by count)")
    print(f"{'='*80}")
    dropped_sorted = sorted(dropped, key=lambda t: vocab_a[t], reverse=True)
    for token in dropped_sorted[:n]:
        print(f"  {repr(token):40s}  count={vocab_a[token]:>12,.1f}  rank={ranks_a[token]:>6,}")
    if len(dropped) > n:
        print(f"  ... and {len(dropped) - n:,} more")
    print()

    # --- Biggest percentage change in count (shared tokens only) ---
    print(f"{'='*80}")
    print(f"BIGGEST % CHANGE in count (top {n} increases, top {n} decreases)")
    print(f"{'='*80}")
    pct_changes = []
    for token in shared:
        ca = vocab_a[token]
        cb = vocab_b[token]
        if ca > 0:
            pct = (cb - ca) / ca * 100
        else:
            pct = float('inf') if cb > 0 else 0.0
        pct_changes.append((token, ca, cb, pct))

    pct_changes.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Top {n} increases:")
    for token, ca, cb, pct in pct_changes[:n]:
        print(f"  {repr(token):40s}  {ca:>12,.1f} → {cb:>12,.1f}  ({pct:+.1f}%)")

    print(f"\n  Top {n} decreases:")
    for token, ca, cb, pct in pct_changes[-n:]:
        print(f"  {repr(token):40s}  {ca:>12,.1f} → {cb:>12,.1f}  ({pct:+.1f}%)")
    print()

    # --- Biggest rank change (shared tokens only) ---
    print(f"{'='*80}")
    print(f"BIGGEST RANK CHANGE (top {n} promotions, top {n} demotions)")
    print(f"{'='*80}")
    rank_changes = []
    for token in shared:
        ra = ranks_a[token]
        rb = ranks_b[token]
        rank_changes.append((token, ra, rb, ra - rb))

    rank_changes.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Top {n} promotions (moved up in rank):")
    for token, ra, rb, delta in rank_changes[:n]:
        print(f"  {repr(token):40s}  rank {ra:>6,} → {rb:>6,}  (↑{delta:,})")

    print(f"\n  Top {n} demotions (moved down in rank):")
    for token, ra, rb, delta in rank_changes[-n:]:
        print(f"  {repr(token):40s}  rank {ra:>6,} → {rb:>6,}  (↓{-delta:,})")


if __name__ == "__main__":
    main()

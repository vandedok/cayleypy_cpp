"""Helper functions."""

from typing import Any, Sequence
from collections import Counter
from itertools import combinations, permutations
from sympy.combinatorics import Permutation


def identity_perm(n: int) -> list[int]:
    return list(range(n))


def apply_permutation(p: Any, x: Sequence[Any]) -> list[Any]:
    return [x[p[i]] for i in range(len(p))]


def compose_permutations(p1: Sequence[int], p2: Sequence[int]) -> list[int]:
    """Returns p1∘p2."""
    return apply_permutation(p1, p2)


def inverse_permutation(p: Sequence[int]) -> list[int]:
    n = len(p)
    ans = [0] * n
    for i in range(n):
        ans[p[i]] = i
    return ans


def is_permutation(p: Any) -> bool:
    return sorted(list(p)) == list(range(len(p)))


def transposition(n: int, i1: int, i2: int) -> list[int]:
    """Returns permutation of n elements that is transposition (swap) of i1 and i2."""
    assert 0 <= i1 < n
    assert 0 <= i2 < n
    assert i1 != i2
    perm = list(range(n))
    perm[i1], perm[i2] = i2, i1
    return perm


def permutation_from_cycles(n: int, cycles: list[list[int]], offset: int = 0) -> list[int]:
    """Returns permutation of size n having given cycles."""
    perm = list(range(n))
    cycles_offsetted = [[x - offset for x in y] for y in cycles]

    for cycle in cycles_offsetted:
        for i in range(len(cycle)):
            assert 0 <= cycle[i] < n
            assert perm[cycle[i]] == cycle[i], "Cycles must not intersect."
            perm[cycle[i]] = cycle[(i + 1) % len(cycle)]
    return perm


def permutations_with_cycle_lenghts(n: int, cycle_lengths: list[int]) -> list[list[int]]:
    """
    Generate all permutations of {0,...,n-1} with cycle lengths given by `structure`
    (a list, e.g. [2,1,1]).
    """
    if sum(cycle_lengths) != n:
        raise ValueError("Sum of cycle lengths must equal n")

    lengths_counter = Counter(cycle_lengths)
    all_elems = set(range(n))
    result = []

    def backtrack(last_min, available, lengths_counter):
        # if no lengths left, we should have used all elements
        if not lengths_counter:
            if not available:
                yield []
            return

        # iterate available distinct cycle lengths in increasing order
        for k in sorted(lengths_counter):
            # choose combinations of available elements of size k
            for comb in combinations(sorted(available), k):
                m = comb[0]  # minimal element of this cycle
                if m <= last_min:
                    # ensure strict increase of minima -> canonical ordering
                    continue
                # fix rotation: keep m first, permute the rest
                rest = list(comb[1:])
                for order in permutations(rest):
                    cycle = (m,) + order
                    new_available = available - set(cycle)

                    # update multiset of lengths
                    new_lengths = lengths_counter.copy()
                    new_lengths[k] -= 1
                    if new_lengths[k] == 0:
                        del new_lengths[k]

                    # recurse
                    for tail_cycles in backtrack(m, new_available, new_lengths):
                        yield [cycle] + tail_cycles

    for cycles in backtrack(-1, all_elems, lengths_counter):
        # convert cycles (tuples) to sympy Permutation
        result.append(Permutation(list(cycles), size=n).array_form)
    return result

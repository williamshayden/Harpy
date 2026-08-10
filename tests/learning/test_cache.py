"""Contract tests for bounded learning-only spectrum evidence caching."""

from __future__ import annotations

import numpy as np
import pytest

from harpy.learning.cache import SpectrumEvidenceCache


def test_cache_miss_then_hit_owns_isolated_spectra() -> None:
    """A hit must not expose either caller or cache-owned evidence storage."""
    calls: list[int] = []
    cache = SpectrumEvidenceCache()

    def render() -> np.ndarray:
        calls.append(6_000)
        return np.linspace(0.0, 1.0, 1_961, dtype=np.float32)

    first = cache.get_or_create(6_000, on_miss=render)
    second = cache.get_or_create(6_000, on_miss=lambda: pytest.fail("unexpected miss"))

    assert calls == [6_000]
    assert np.array_equal(first, second)
    assert not np.shares_memory(first, second)
    assert not first.flags.writeable and not second.flags.writeable
    assert len(cache) == 1


@pytest.mark.parametrize("candidate_cents", [True, False, 1_100.0, "6000", 1_099, 10_901])
def test_cache_rejects_invalid_keys_before_a_miss(candidate_cents: object) -> None:
    """Only reachable integer-cent coordinates may allocate a cache entry."""
    cache = SpectrumEvidenceCache()

    with pytest.raises(ValueError, match="candidate_pitch_cents"):
        cache.get_or_create(candidate_cents, on_miss=lambda: pytest.fail("unexpected miss"))

    assert len(cache) == 0
    assert cache.spectrum_bytes == 0


@pytest.mark.parametrize(
    "invalid_spectrum",
    [
        np.ones(1_961, dtype=np.float64),
        np.ones(1_960, dtype=np.float32),
        np.ones((1_961, 1), dtype=np.float32),
        np.full(1_961, np.nan, dtype=np.float32),
        np.full(1_961, np.inf, dtype=np.float32),
    ],
)
def test_cache_rejects_invalid_miss_spectra_without_committing(
    invalid_spectrum: np.ndarray,
) -> None:
    """Malformed renderer output cannot leave a partially populated cache."""
    cache = SpectrumEvidenceCache()

    with pytest.raises(ValueError, match="spectrum"):
        cache.get_or_create(6_000, on_miss=lambda: invalid_spectrum)

    assert len(cache) == 0
    assert cache.spectrum_bytes == 0


def test_cache_miss_failure_leaves_no_entry() -> None:
    """A renderer exception must preserve the empty pre-miss cache state."""
    cache = SpectrumEvidenceCache()

    with pytest.raises(RuntimeError, match="render failed"):
        cache.get_or_create(
            6_000,
            on_miss=lambda: (_ for _ in ()).throw(RuntimeError("render failed")),
        )

    assert len(cache) == 0
    assert cache.spectrum_bytes == 0


def test_cache_normalizes_a_c_contiguous_owned_immutable_spectrum() -> None:
    """Non-contiguous renderer evidence is copied into the canonical cache form."""
    source = np.arange(3_922, dtype=np.float32)[::2]
    cache = SpectrumEvidenceCache()

    result = cache.get_or_create(6_000, on_miss=lambda: source)

    assert result.flags.c_contiguous
    assert result.flags.owndata
    assert not result.flags.writeable
    assert np.array_equal(result, source)
    assert not np.shares_memory(result, source)


def test_cache_has_exact_full_reachable_capacity_and_spectrum_bytes() -> None:
    """Every reachable coordinate fits within the fixed spectrum-only byte budget."""
    cache = SpectrumEvidenceCache()
    spectrum = np.zeros(1_961, dtype=np.float32)

    for candidate_cents in range(1_100, 10_901):
        cache.get_or_create(candidate_cents, on_miss=lambda: spectrum)

    assert len(cache) == 9_801
    assert cache.spectrum_bytes == 76_879_044


def test_cache_detects_mutated_private_entry_before_returning_a_hit() -> None:
    """A corrupted cache-owned array fails closed instead of serving altered evidence."""
    cache = SpectrumEvidenceCache()
    cache.get_or_create(6_000, on_miss=lambda: np.zeros(1_961, dtype=np.float32))
    stored = cache._entries[6_000].spectrum
    stored.setflags(write=True)
    stored[0] = 1.0
    stored.setflags(write=False)

    with pytest.raises(RuntimeError, match="digest-integrity"):
        cache.get_or_create(6_000, on_miss=lambda: pytest.fail("unexpected miss"))

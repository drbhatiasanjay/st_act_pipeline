"""
Comprehensive tests for P0-4: metric aggregation and validation coverage fixes.

All tests are hermetic and deterministic: no dependency on data/staging,
sample_submission.csv, or real .geff files. Synthetic in-memory
IndexedRXGraph instances, synthetic metadata objects, and a bare
TrainingLoop instance (bypassing __init__, following tests/test_train.py's
make_bare_training_loop pattern) stand in for real data throughout.

Tests:
A. Weighted per-sample adjusted metric
B. Edge and division micro-aggregation
C. Metadata per sample
D. Dictionary key mismatch
E. Evaluation exception
F. Validation cap isolation
G. Complete-sample validation cap
H. Interleaved pairs
I. Provenance
J. Caller regression
"""

import math
from pathlib import Path

import polars as pl
import pytest
import torch
import tracksdata as td

from src.evaluation import evaluate_submission
from src.tracking_cellmot import evaluate, node_recall, per_sample_metrics, summarise
from src.train import TrainingLoop


def make_bare_training_loop(**extra_attrs):
    """Bypass __init__ (needs real models/loaders); set only the attributes
    the method under test actually reads. Mirrors tests/test_train.py's
    helper of the same name -- duplicated here (not imported) to keep this
    file hermetic and independent of another test module's internals."""
    loop = TrainingLoop.__new__(TrainingLoop)
    for key, value in extra_attrs.items():
        setattr(loop, key, value)
    return loop


class _FakeMetadata:
    """Duck-types tracksdata's GeffMetadata just enough for
    evaluate_submission(): an object with an `.extra` dict containing
    'estimated_number_of_nodes'."""

    def __init__(self, estimated_number_of_nodes: int):
        self.extra = {'estimated_number_of_nodes': estimated_number_of_nodes}


def _make_graph(node_specs, edges):
    """Build an in-memory IndexedRXGraph. node_specs: list of (t,z,y,x)
    tuples; node i gets id i (tracksdata assigns ids in add_node() order).
    edges: list of (src_id, tgt_id) pairs."""
    g = td.graph.IndexedRXGraph()
    for key in ('t', 'z', 'y', 'x'):
        try:
            g.add_node_attr_key(key, pl.Int64, 0)
        except ValueError:
            pass  # 't' is auto-registered by IndexedRXGraph() itself
    for t, z, y, x in node_specs:
        g.add_node({'t': t, 'z': z, 'y': y, 'x': x})
    for src, tgt in edges:
        g.add_edge(src, tgt, {})
    return g


def _make_fake_val_batch(sample_id: str, t_idx: int) -> dict:
    return {
        "frame_t": torch.zeros(1, 1, 8, 16, 16),
        "frame_t1": torch.zeros(1, 1, 8, 16, 16),
        "sample_id": [sample_id],
        "t_idx": [t_idx],
    }


class _FakeDatasetWithPairs:
    def __init__(self, pairs):
        self.pairs = pairs


class _FakeValLoaderWithPairs:
    """Minimal val_loader stand-in exposing both the iteration protocol
    validate_epoch() needs (__iter__/__len__) and the .dataset.pairs
    attribute the max_validation_samples path reads for safe, deterministic
    complete-sample selection (mirrors CompetitionDataset.pairs)."""

    def __init__(self, batches, pairs):
        self._batches = batches
        self.dataset = _FakeDatasetWithPairs(pairs)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class _FakeAlwaysDetectingUNet3D(torch.nn.Module):
    """Every forward() call produces one clear peak per channel, so
    validate_epoch() completes without tripping the zero-detections circuit
    breaker -- used for tests that need a real (non-exception) return value."""

    def forward(self, x):
        batch = x.shape[0]
        logits = torch.full((batch, 2, 8, 16, 16), -100.0)
        logits[:, :, 4, 8, 8] = 100.0
        features = torch.zeros(batch, 4, 8, 16, 16)
        return logits, features


class _FakeEdgeTransformer(torch.nn.Module):
    """Minimal stand-in for SimpleNodeTransformer: fixed flat edge-logit
    tensor matching greedy_edge_assignment()'s expected (n_t*n_t1,) shape."""

    def forward(self, nodes_t, nodes_t1, features_t, features_t1):
        n_t = nodes_t.shape[0]
        n_t1 = nodes_t1.shape[0]
        return torch.full((n_t * n_t1,), 100.0)  # sigmoid(100) ~= 1.0


def _make_loop(val_loader, hyperparams_overrides=None):
    hyperparams = {
        "detection_threshold": 0.5,
        "max_positive_voxel_fraction": 0.005,
        "nms_radius_um": 5.0,
        "edge_threshold": 0.5,
        "max_batches_per_epoch": None,
    }
    if hyperparams_overrides:
        hyperparams.update(hyperparams_overrides)
    return make_bare_training_loop(
        unet3d=_FakeAlwaysDetectingUNet3D(),
        transformer=_FakeEdgeTransformer(),
        device=torch.device("cpu"),
        _amp_enabled=False,
        # Nonexistent path: geff_path.exists() is False, so GT loading is
        # skipped cleanly (no exception, no evaluation_failure count) --
        # these tests only care about which batches get PROCESSED, not
        # about a real score.
        data_dir=Path("nonexistent_test_dir_for_p04_hermetic_tests"),
        hyperparams=hyperparams,
        val_loader=val_loader,
        epoch_fallback_counts={
            "heatmap_generation_failure": 0,
            "edge_target_generation_failure": 0,
            "edge_loss_computation_failure": 0,
            "evaluation_failure": 0,
        },
    )


def _count_peaks_for_channel_calls(monkeypatch, loop):
    """Monkeypatch loop._peaks_for_channel to record every call; returns the
    list that gets appended to. _peaks_for_channel is called exactly twice
    per PROCESSED batch (channel 0, channel 1) and zero times for a batch
    skipped by the allowed_sample_ids filter -- a direct, code-level proxy
    for "was this batch touched", independent of the eventual score."""
    seen = []
    real_peaks = loop._peaks_for_channel

    def recording(*args, **kwargs):
        seen.append(1)
        return real_peaks(*args, **kwargs)

    monkeypatch.setattr(loop, "_peaks_for_channel", recording)
    return seen


# ============================================================
# Test A: Weighted per-sample adjusted metric
# ============================================================

class TestWeightedPerSampleAdjustedMetric:
    def test_weighted_adjustment_two_samples_different_ratios(self):
        """
        Create two synthetic samples with different TP/FP/FN support and
        node-count ratios. Assert exact agreement with
        per_sample_metrics() + summarise(), and assert disagreement with
        the old global-penalty formula.
        """
        gt1 = _make_graph(
            [(0, i * 100, 0, 0) for i in range(10)]
            + [(1, (i - 10) * 100, 0, 0) for i in range(10, 20)],
            [(src, 10 + (src + off) % 10) for src in range(10) for off in range(5)],
        )
        pred1 = gt1.copy()

        gt2 = _make_graph(
            [(0, i * 100, 100, 0) for i in range(5)]
            + [(1, (i - 5) * 100, 100, 0) for i in range(5, 10)],
            [(src, 5 + (src + off)) for src in range(5) for off in range(6) if src + off < 5],
        )
        pred2 = gt2.copy()
        for i in range(100):
            pred2.add_node({'t': 0, 'z': 10000 + i, 'y': 10000, 'x': 0})

        er1 = evaluate(pred1, gt1)
        er2 = evaluate(pred2, gt2)
        nr1 = node_recall(pred1, gt1)
        nr2 = node_recall(pred2, gt2)
        n_total_1 = gt1.num_nodes()
        n_total_2 = gt2.num_nodes()
        row1 = per_sample_metrics(er1, n_total_1, nr1)
        row2 = per_sample_metrics(er2, n_total_2, nr2)
        summary = summarise([row1, row2])

        assert summary['edge_jaccard'] == (
            (er1.edge_tp + er2.edge_tp)
            / (er1.edge_tp + er1.edge_fp + er1.edge_fn + er2.edge_tp + er2.edge_fp + er2.edge_fn)
        ), "Edge Jaccard must be micro-averaged"

        assert not math.isnan(summary['adj_edge_jaccard']), "adj_edge_jaccard should not be NaN"
        assert summary['adj_edge_jaccard'] > 0, "adj_edge_jaccard should be positive"

        num_pred_total = er1.num_pred_nodes + er2.num_pred_nodes
        total_true = n_total_1 + n_total_2
        global_node_ratio = (num_pred_total - total_true) / total_true if total_true > 0 else 0
        global_adjusted = max(0, summary['edge_jaccard'] * (1 - 0.1 * global_node_ratio))

        assert summary['adj_edge_jaccard'] != global_adjusted or num_pred_total == total_true, \
            "Per-sample weighted aggregation should differ from global penalty approach"


# ============================================================
# Test B: Edge and division micro-aggregation
# ============================================================

class TestEdgeAndDivisionMicroAggregation:
    def test_micro_aggregation_edge_tp_fp_fn(self):
        """Verify evaluate_submission() sums TP/FP/FN across samples BEFORE
        computing Jaccard, by comparing its result against an independent
        manual combination of two separate evaluate() calls."""
        gt1 = _make_graph([(0, 0, 0, 0), (1, 0, 0, 0)], [(0, 1)])
        pred1 = gt1.copy()

        # Sample 2: pred is missing one real GT edge -> a genuine FN.
        gt2 = _make_graph(
            [(0, 500, 0, 0), (1, 500, 0, 0), (2, 500, 0, 0)],
            [(0, 1), (1, 2)],
        )
        pred2 = _make_graph(
            [(0, 500, 0, 0), (1, 500, 0, 0), (2, 500, 0, 0)],
            [(0, 1)],
        )

        ref1 = evaluate(pred1.copy(), gt1)
        ref2 = evaluate(pred2.copy(), gt2)
        expected_edge_jaccard = (
            (ref1.edge_tp + ref2.edge_tp)
            / (ref1.edge_tp + ref1.edge_fp + ref1.edge_fn + ref2.edge_tp + ref2.edge_fp + ref2.edge_fn)
        )
        assert 0 < expected_edge_jaccard < 1, \
            "test setup must produce a genuine partial mismatch, not a trivial 0 or 1"

        result = evaluate_submission([pred1, pred2], [gt1, gt2])

        assert result['edge_jaccard'] == pytest.approx(expected_edge_jaccard)
        assert result['num_datasets'] == 2

    def test_division_micro_aggregation_pooled_before_jaccard(self, monkeypatch):
        """division_jaccard must be computed from POOLED division TP/FP/FN
        across samples (sum(division_tp) / sum(division_tp+fp+fn)), not a
        per-sample average. Monkeypatches src.evaluation.evaluate (the name
        as evaluate_submission() itself resolves it) to return deterministic
        EvaluationResult objects with different division counts per sample
        -- evaluate_submission() remains the function under test; only the
        per-pair evaluate() call is replaced, since hand-constructing real
        division events in a synthetic graph is unnecessary for testing
        aggregation arithmetic specifically."""
        import src.evaluation as evaluation_module
        from src.tracking_cellmot.metrics import EvaluationResult

        # Sample 1: division_tp=3, division_fp=1, division_fn=0 (jaccard=0.75)
        # Sample 2: division_tp=1, division_fp=0, division_fn=2 (jaccard=1/3)
        results_by_call = [
            EvaluationResult(edge_tp=10, edge_fp=0, edge_fn=0,
                              division_tp=3, division_fp=1, division_fn=0, num_pred_nodes=10),
            EvaluationResult(edge_tp=5, edge_fp=0, edge_fn=0,
                              division_tp=1, division_fp=0, division_fn=2, num_pred_nodes=5),
        ]
        call_index = {"n": 0}

        def fake_evaluate(pred, gt, scale=None, max_distance=7.0):
            result = results_by_call[call_index["n"]]
            call_index["n"] += 1
            return result

        monkeypatch.setattr(evaluation_module, "evaluate", fake_evaluate)

        gt1 = _make_graph([(0, 0, 0, 0)], [])
        pred1 = gt1.copy()
        gt2 = _make_graph([(0, 500, 0, 0)], [])
        pred2 = gt2.copy()

        result = evaluation_module.evaluate_submission([pred1, pred2], [gt1, gt2])

        expected_division_jaccard = (3 + 1) / ((3 + 1) + (1 + 0) + (0 + 2))
        assert result['division_jaccard'] == pytest.approx(expected_division_jaccard), (
            f"division_jaccard must be pooled-TP/(pooled-TP+FP+FN), not a "
            f"per-sample average. Expected {expected_division_jaccard}, "
            f"got {result['division_jaccard']}"
        )

        # Regression guard: a naive per-sample AVERAGE of the two samples'
        # division Jaccards must give a DIFFERENT number in this scenario,
        # proving the test can actually distinguish pooled-before-ratio from
        # the wrong per-sample-averaged formula, not just happen to agree.
        jaccard_1 = 3 / (3 + 1 + 0)
        jaccard_2 = 1 / (1 + 0 + 2)
        naive_average = (jaccard_1 + jaccard_2) / 2
        assert result['division_jaccard'] != pytest.approx(naive_average), (
            "test setup must make pooled-vs-averaged aggregation genuinely "
            "differ, else this test cannot distinguish the two formulas"
        )


# ============================================================
# Test C: Metadata per sample
# ============================================================

class TestMetadataPerSample:
    def test_each_sample_uses_own_estimated_nodes(self):
        """Each sample's T_true must come from ITS OWN metadata entry, not
        get mixed up with another sample's. Proven by constructing pred/gt
        dicts with metadata whose insertion order deliberately differs from
        pred_graphs' key order (would silently break under a positional,
        not key-based, pairing bug), and by proving the correctly-paired
        result genuinely differs from a deliberately-swapped pairing."""
        # Sample A has 5 edges (weight=5), sample B has 1 edge (weight=1) --
        # unequal weights are required for a swap to be visible at all: with
        # equal weights the weighted average of {ratio_a, ratio_b} doesn't
        # depend on which sample owns which ratio.
        gt_a = _make_graph(
            [(0, 0, 0, 0), (0, 100, 0, 0), (1, 0, 0, 0), (1, 100, 0, 0), (2, 0, 0, 0), (2, 100, 0, 0)],
            [(0, 2), (0, 3), (1, 2), (1, 3), (2, 4)],
        )
        pred_a = gt_a.copy()
        gt_b = _make_graph([(0, 500, 0, 0), (1, 500, 0, 0)], [(0, 1)])
        pred_b = gt_b.copy()

        pred_dict = {"sample_b": pred_b.copy(), "sample_a": pred_a.copy()}
        gt_dict = {"sample_b": gt_b, "sample_a": gt_a}
        metadata_dict = {"sample_a": _FakeMetadata(2), "sample_b": _FakeMetadata(9999)}

        result = evaluate_submission(pred_dict, gt_dict, gt_metadata=metadata_dict)

        # node_recall() requires its graph to already carry match attributes
        # from a prior evaluate() call -- reuse the SAME matched object for
        # both, rather than a second fresh (unmatched) copy.
        matched_pred_a = pred_a.copy()
        er_a = evaluate(matched_pred_a, gt_a)
        nr_a = node_recall(matched_pred_a, gt_a)

        matched_pred_b = pred_b.copy()
        er_b = evaluate(matched_pred_b, gt_b)
        nr_b = node_recall(matched_pred_b, gt_b)

        row_a_correct = per_sample_metrics(er_a, 2, nr_a)
        row_b_correct = per_sample_metrics(er_b, 9999, nr_b)
        expected_correct = summarise([row_a_correct, row_b_correct])

        row_a_swapped = per_sample_metrics(er_a, 9999, nr_a)
        row_b_swapped = per_sample_metrics(er_b, 2, nr_b)
        expected_swapped = summarise([row_a_swapped, row_b_swapped])

        assert result['adjusted_edge_jaccard'] == pytest.approx(expected_correct['adj_edge_jaccard'])
        assert expected_correct['adj_edge_jaccard'] != pytest.approx(expected_swapped['adj_edge_jaccard']), \
            "test setup must make a metadata swap actually visible, else it can't guard against one"

    def test_gt_node_count_fallback_when_metadata_unavailable(self):
        """Verify fallback to GT node count when estimated_number_of_nodes
        is unavailable, and that it genuinely differs from the
        metadata-driven T_true (proving the fallback path is really taken,
        not coincidentally identical)."""
        gt = _make_graph(
            [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)],
            [(0, 1), (1, 2)],
        )
        pred = gt.copy()
        pred.add_node({'t': 0, 'z': 900, 'y': 0, 'x': 0})  # inflate pred's node count

        result_with_meta = evaluate_submission([pred.copy()], [gt], gt_metadata=[_FakeMetadata(2)])
        result_without_meta = evaluate_submission([pred.copy()], [gt], gt_metadata=None)

        assert not math.isnan(result_with_meta['adjusted_edge_jaccard'])
        assert not math.isnan(result_without_meta['adjusted_edge_jaccard'])
        assert result_with_meta['adjusted_edge_jaccard'] != pytest.approx(
            result_without_meta['adjusted_edge_jaccard']
        ), "metadata T_true (2) and GT node-count fallback (3) must produce visibly different adjustments"


# ============================================================
# Test D: Dictionary key mismatch
# ============================================================

class TestDictionaryKeyMismatch:
    def _pair(self):
        gt = _make_graph([(0, 0, 0, 0), (1, 0, 0, 0)], [(0, 1)])
        pred = gt.copy()
        meta = _FakeMetadata(2)
        return pred, gt, meta

    def test_missing_prediction_sample(self):
        """GT has a sample ID that pred_graphs lacks -> must raise, naming it."""
        pred1, gt1, meta1 = self._pair()
        pred2, gt2, meta2 = self._pair()
        pred_dict = {"s1": pred1}
        gt_dict = {"s1": gt1, "s2": gt2}
        metadata_dict = {"s1": meta1, "s2": meta2}

        with pytest.raises(ValueError, match="s2"):
            evaluate_submission(pred_dict, gt_dict, gt_metadata=metadata_dict)

    def test_missing_gt_sample(self):
        """pred_graphs has a sample ID that GT lacks -> must raise, naming it."""
        pred1, gt1, meta1 = self._pair()
        pred1_extra, _, _ = self._pair()
        pred_dict = {"s1": pred1, "s_extra": pred1_extra}
        gt_dict = {"s1": gt1}
        metadata_dict = {"s1": meta1}

        with pytest.raises(ValueError, match="s_extra"):
            evaluate_submission(pred_dict, gt_dict, gt_metadata=metadata_dict)

    def test_missing_metadata_sample(self):
        """A dict-input sample with no matching metadata entry -> must raise, naming it."""
        pred1, gt1, _ = self._pair()
        pred_dict = {"s1": pred1}
        gt_dict = {"s1": gt1}
        metadata_dict = {}

        with pytest.raises(ValueError, match="s1"):
            evaluate_submission(pred_dict, gt_dict, gt_metadata=metadata_dict)

    def test_unexpected_metadata_sample(self):
        """Extra metadata keys not present in pred_graphs/gt_graphs -> must
        raise, naming the unexpected key, not silently ignored."""
        pred1, gt1, meta1 = self._pair()
        pred_dict = {"s1": pred1}
        gt_dict = {"s1": gt1}
        metadata_dict = {"s1": meta1, "s_bogus": meta1}

        with pytest.raises(ValueError, match="s_bogus"):
            evaluate_submission(pred_dict, gt_dict, gt_metadata=metadata_dict)


# ============================================================
# Test H (optional): Mixed container-type hardening
# ============================================================

class TestMixedContainerTypes:
    """evaluate_submission() must reject mixed pred/gt/metadata container
    types with a clear ValueError, not an AttributeError from calling
    .keys() on a list or a KeyError from indexing a dict with an int."""

    def test_pred_dict_gt_list_raises_value_error(self):
        gt = _make_graph([(0, 0, 0, 0), (1, 0, 0, 0)], [(0, 1)])
        pred = gt.copy()
        with pytest.raises(ValueError, match="both be dicts or both be lists"):
            evaluate_submission({"s1": pred}, [gt])

    def test_pred_list_gt_dict_raises_value_error(self):
        gt = _make_graph([(0, 0, 0, 0), (1, 0, 0, 0)], [(0, 1)])
        pred = gt.copy()
        with pytest.raises(ValueError, match="both be dicts or both be lists"):
            evaluate_submission([pred], {"s1": gt})

    def test_graph_lists_with_metadata_dict_raises_value_error(self):
        gt = _make_graph([(0, 0, 0, 0), (1, 0, 0, 0)], [(0, 1)])
        pred = gt.copy()
        with pytest.raises(ValueError, match="metadata must be a list"):
            evaluate_submission([pred], [gt], gt_metadata={"s1": _FakeMetadata(2)})


# ============================================================
# Test E: Evaluation exception
# ============================================================

class TestEvaluationException:
    def test_one_failing_sample_fails_entire_call(self):
        """One failing sample must fail the ENTIRE evaluate_submission()
        call -- it must not silently return a metric computed from only the
        successful samples."""
        gt1 = _make_graph([(0, 0, 0, 0), (1, 0, 0, 0)], [(0, 1)])
        pred1 = gt1.copy()
        gt2 = _make_graph([(0, 500, 0, 0), (1, 500, 0, 0)], [(0, 1)])

        # Deliberately pass None as the second prediction to force evaluate()
        # to raise inside evaluate_submission()'s per-sample loop.
        with pytest.raises(RuntimeError, match="sample_1"):
            evaluate_submission([pred1, None], [gt1, gt2])


# ============================================================
# Test F: Validation cap isolation
# ============================================================

class TestValidationCapIsolation:
    def test_max_batches_does_not_truncate_validation(self, monkeypatch):
        """max_batches_per_epoch must not truncate validate_epoch(), even
        though it's set far smaller than the real batch count -- it is a
        training-only cap after P0-4."""
        batches = [_make_fake_val_batch("sample_a", t) for t in range(6)]
        loop = _make_loop(batches, hyperparams_overrides={"max_batches_per_epoch": 2})
        seen = _count_peaks_for_channel_calls(monkeypatch, loop)

        val_metrics = loop.validate_epoch()

        # 2 calls per batch (channel 0, channel 1) x all 6 batches, not just
        # the first max_batches_per_epoch=2 batches.
        assert len(seen) == 12
        assert val_metrics["validation_samples_evaluated"] == 1


# ============================================================
# Test G: Complete-sample validation cap
# ============================================================

class TestCompleteSampleValidationCap:
    @staticmethod
    def _two_samples_pairs_and_batches():
        pairs = [
            ("sample_a", 0), ("sample_a", 1), ("sample_a", 2),
            ("sample_b", 0), ("sample_b", 1),
        ]
        batches = [_make_fake_val_batch(sid, t) for sid, t in pairs]
        return pairs, batches

    def test_max_validation_samples_one_selects_first_sample_completely(self, monkeypatch):
        pairs, batches = self._two_samples_pairs_and_batches()
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader, hyperparams_overrides={"max_validation_samples": 1})
        seen = _count_peaks_for_channel_calls(monkeypatch, loop)

        val_metrics = loop.validate_epoch()

        # Only sample_a's 3 batches processed (x2 channels = 6 calls);
        # sample_b's 2 batches must be skipped entirely, not partially.
        assert len(seen) == 6
        assert val_metrics["validation_is_full_fold"] is False
        assert val_metrics["validation_samples_evaluated"] == 1
        assert val_metrics["validation_samples_total"] == 2

    def test_max_validation_samples_two_evaluates_both_completely(self, monkeypatch):
        """cap == total: coverage is complete, so this IS a full fold --
        a complete fold must not be reported as incomplete merely because a
        cap happened to be supplied (the cap's VALUE, not its mere
        presence, is what determines full-fold status)."""
        pairs, batches = self._two_samples_pairs_and_batches()
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader, hyperparams_overrides={"max_validation_samples": 2})
        seen = _count_peaks_for_channel_calls(monkeypatch, loop)

        val_metrics = loop.validate_epoch()

        assert len(seen) == 10  # all 5 batches x2 channels, both samples complete
        assert val_metrics["validation_is_full_fold"] is True
        assert val_metrics["validation_samples_evaluated"] == 2
        assert val_metrics["validation_samples_total"] == 2
        assert val_metrics["validation_sample_cap"] == 2

    def test_max_validation_samples_greater_than_total_reports_full_fold(self, monkeypatch):
        """cap > total: selection is clamped to the real total, and full
        coverage still means full fold."""
        pairs, batches = self._two_samples_pairs_and_batches()
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader, hyperparams_overrides={"max_validation_samples": 1000})
        seen = _count_peaks_for_channel_calls(monkeypatch, loop)

        val_metrics = loop.validate_epoch()

        assert len(seen) == 10  # all 5 batches x2 channels, both samples complete
        assert val_metrics["validation_is_full_fold"] is True
        assert val_metrics["validation_samples_evaluated"] == 2
        assert val_metrics["validation_samples_total"] == 2
        assert val_metrics["validation_sample_cap"] == 1000

    def test_max_validation_samples_none_evaluates_full_fold(self, monkeypatch):
        pairs, batches = self._two_samples_pairs_and_batches()
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader)  # max_validation_samples not set -> None
        seen = _count_peaks_for_channel_calls(monkeypatch, loop)

        val_metrics = loop.validate_epoch()

        assert len(seen) == 10  # all batches processed, no cap applied
        assert val_metrics["validation_is_full_fold"] is True
        assert val_metrics["validation_samples_evaluated"] == 2
        assert val_metrics["validation_samples_total"] == 2
        assert val_metrics["validation_sample_cap"] is None


class TestMaxValidationSamplesValidation:
    """max_validation_samples must reject values that could silently do the
    wrong thing, before validate_epoch() ever touches the val_loader."""

    def test_zero_raises(self):
        pairs, batches = TestCompleteSampleValidationCap._two_samples_pairs_and_batches()
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader, hyperparams_overrides={"max_validation_samples": 0})
        with pytest.raises(ValueError, match="positive"):
            loop.validate_epoch()

    def test_negative_raises(self):
        pairs, batches = TestCompleteSampleValidationCap._two_samples_pairs_and_batches()
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader, hyperparams_overrides={"max_validation_samples": -1})
        with pytest.raises(ValueError, match="positive"):
            loop.validate_epoch()

    def test_bool_raises(self):
        """bool is an int subclass in Python (isinstance(True, int) is
        True) -- must be rejected explicitly, not silently accepted as 1."""
        pairs, batches = TestCompleteSampleValidationCap._two_samples_pairs_and_batches()
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader, hyperparams_overrides={"max_validation_samples": True})
        with pytest.raises(ValueError, match="int"):
            loop.validate_epoch()

    def test_float_raises(self):
        pairs, batches = TestCompleteSampleValidationCap._two_samples_pairs_and_batches()
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader, hyperparams_overrides={"max_validation_samples": 1.5})
        with pytest.raises(ValueError, match="int"):
            loop.validate_epoch()


# ============================================================
# Test H: Interleaved pairs
# ============================================================

class TestInterleavedPairs:
    def test_interleaved_batches_respects_sample_selection(self, monkeypatch):
        """Even when the actual batch STREAM interleaves two samples'
        batches, a selected sample's own sequence (still internally
        chronological) must be fully processed and a non-selected sample's
        batches must be fully skipped -- global batch contiguity across
        samples is not required."""
        interleaved_batches = [
            _make_fake_val_batch("sample_a", 0),
            _make_fake_val_batch("sample_b", 0),
            _make_fake_val_batch("sample_a", 1),
            _make_fake_val_batch("sample_b", 1),
            _make_fake_val_batch("sample_a", 2),
        ]
        # .dataset.pairs reflects the canonical per-sample-contiguous
        # ordering (as CompetitionDataset really provides) used for
        # deterministic "first N sample IDs" selection; the batch stream
        # itself is what's interleaved.
        canonical_pairs = [
            ("sample_a", 0), ("sample_a", 1), ("sample_a", 2),
            ("sample_b", 0), ("sample_b", 1),
        ]
        val_loader = _FakeValLoaderWithPairs(interleaved_batches, canonical_pairs)
        loop = _make_loop(val_loader, hyperparams_overrides={"max_validation_samples": 1})
        seen = _count_peaks_for_channel_calls(monkeypatch, loop)

        val_metrics = loop.validate_epoch()

        # sample_a's 3 interleaved batches processed (x2 channels = 6);
        # sample_b's 2 interleaved batches skipped entirely.
        assert len(seen) == 6
        assert val_metrics["validation_samples_evaluated"] == 1


# ============================================================
# Test I: Provenance
# ============================================================

class TestProvenance:
    def test_provenance_fields_present(self):
        """evaluate_submission() itself does NOT add provenance fields --
        those are added by TrainingLoop.validate_epoch() only. This proves
        that contract directly."""
        gt = _make_graph([(0, 0, 0, 0), (1, 0, 0, 0)], [(0, 1)])
        pred = gt.copy()

        result = evaluate_submission([pred], [gt])

        assert 'validation_is_full_fold' not in result
        assert 'validation_samples_evaluated' not in result
        assert 'edge_jaccard' in result
        assert 'adjusted_edge_jaccard' in result

    def test_training_loop_provenance_full_fold(self):
        """TrainingLoop.validate_epoch() sets validation_is_full_fold=True
        with no cap configured."""
        pairs = [("sample_a", 0), ("sample_a", 1)]
        batches = [_make_fake_val_batch(sid, t) for sid, t in pairs]
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader)

        val_metrics = loop.validate_epoch()

        assert val_metrics['validation_is_full_fold'] is True
        assert val_metrics['validation_samples_evaluated'] == 1
        assert val_metrics['validation_samples_total'] == 1

    def test_training_loop_provenance_capped(self):
        """TrainingLoop.validate_epoch() sets validation_is_full_fold=False
        and records the true total when a cap is explicitly configured."""
        pairs = [("sample_a", 0), ("sample_a", 1), ("sample_b", 0)]
        batches = [_make_fake_val_batch(sid, t) for sid, t in pairs]
        val_loader = _FakeValLoaderWithPairs(batches, pairs)
        loop = _make_loop(val_loader, hyperparams_overrides={"max_validation_samples": 1})

        val_metrics = loop.validate_epoch()

        assert val_metrics['validation_is_full_fold'] is False
        assert val_metrics['validation_samples_evaluated'] == 1
        assert val_metrics['validation_samples_total'] == 2


# ============================================================
# Test J: Caller regression
# ============================================================

class TestCallerRegression:
    def test_evaluate_checkpoint_uses_corrected_submission_eval(self):
        """evaluate_checkpoint.py must call the shared, corrected
        evaluate_submission() rather than reimplementing the aggregation
        formula locally."""
        source = Path("evaluate_checkpoint.py").read_text(encoding="utf-8")
        assert "from src.evaluation import" in source and "evaluate_submission" in source
        assert "ADJUSTMENT_ALPHA" not in source, \
            "must not reimplement the adjustment formula locally"

    def test_verify_eval_fixed_uses_corrected_submission_eval(self):
        """verify_eval_fixed.py is a diagnostic/crash-verification script
        (memory + native-crash behavior during inference) -- confirmed by
        direct inspection to never have called evaluate_submission() at all,
        in this baseline or before it; it never computes a final score.
        This guards against it ever growing a duplicate, stale copy of the
        aggregation formula rather than importing the shared one, should it
        later be extended to compute a score."""
        source = Path("verify_eval_fixed.py").read_text(encoding="utf-8")
        assert "ADJUSTMENT_ALPHA" not in source, \
            "must not reimplement the adjustment formula locally"
        assert "evaluate_submission" not in source, \
            "this script does not compute a final score; if that changes, " \
            "it must import evaluate_submission from src.evaluation, not reimplement it"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

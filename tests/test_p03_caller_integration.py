"""
P0-3 fix (2026-07-17): proves the three affected callers
(TrainingLoop.validate_epoch(), evaluate_checkpoint.py, verify_eval_fixed.py)
actually use the shared src/prediction_graph.py helper -- not a
reimplementation of the same logic inline -- and that the submission/
inference path (run_pipeline.py) is NOT routed through it at all (that path
uses a fundamentally different per-timepoint architecture that was already
confirmed, via the P0-3 audit, to be unaffected by the overlapping-window
defect this fix addresses).

Run: py -m pytest tests/test_p03_caller_integration.py -v
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import evaluate_checkpoint
import run_pipeline
import src.train as train_module
import verify_eval_fixed
from src.prediction_graph import PredictionGraphAssembler


class TestValidateEpochUsesSharedHelper:
    """Test G (part 1): TrainingLoop.validate_epoch() must call the real
    PredictionGraphAssembler, not a reimplemented/inlined version."""

    def test_module_imports_prediction_graph_assembler(self):
        assert train_module.PredictionGraphAssembler is PredictionGraphAssembler

    def test_validate_epoch_source_references_the_assembler(self):
        source = inspect.getsource(train_module.TrainingLoop.validate_epoch)
        assert "PredictionGraphAssembler()" in source
        assert "assembler.process_window(" in source
        assert "assembler.add_edges(" in source
        assert "assembler.validate_window_order(" in source
        # The old unconditional dual-frame add_node block must be gone.
        assert "node_id_map_t" not in source
        assert "node_id_map_t1" not in source


class TestEvaluateCheckpointUsesSharedHelper:
    """Test G (part 2): evaluate_checkpoint.py must call the real
    PredictionGraphAssembler."""

    def test_module_imports_prediction_graph_assembler(self):
        assert evaluate_checkpoint.PredictionGraphAssembler is PredictionGraphAssembler

    def test_run_evaluation_source_references_the_assembler(self):
        source = inspect.getsource(evaluate_checkpoint.run_evaluation)
        assert "PredictionGraphAssembler()" in source
        assert "assembler.process_window(" in source
        assert "assembler.add_edges(" in source
        assert "assembler.validate_window_order(" in source
        assert "node_id_map_t" not in source
        assert "node_id_map_t1" not in source

    def test_val_loader_uses_shuffle_false(self):
        source = inspect.getsource(evaluate_checkpoint.run_evaluation)
        assert "shuffle=False" in source


class TestVerifyEvalFixedUsesSharedHelper:
    """Test G (part 3): verify_eval_fixed.py must call the real
    PredictionGraphAssembler."""

    def test_module_imports_prediction_graph_assembler(self):
        assert verify_eval_fixed.PredictionGraphAssembler is PredictionGraphAssembler

    def test_run_evaluation_source_references_the_assembler(self):
        source = inspect.getsource(verify_eval_fixed.run_evaluation)
        assert "PredictionGraphAssembler()" in source
        assert "assembler.process_window(" in source
        assert "assembler.add_edges(" in source
        assert "assembler.validate_window_order(" in source
        assert "node_id_map_t" not in source
        assert "node_id_map_t1" not in source

    def test_loader_uses_shuffle_false(self):
        source = inspect.getsource(verify_eval_fixed.run_evaluation)
        assert "shuffle=False" in source


class TestSubmissionPathIsolation:
    """Test H: run_pipeline.py (submission/inference) must remain completely
    unrouted through PredictionGraphAssembler -- it uses a fundamentally
    different per-timepoint detection architecture
    (centroids_by_t[t] = ..., one detection pass per real timepoint, no
    overlapping-pair windowing) that the P0-3 audit already confirmed cannot
    exhibit this defect. P0-3's fix must not touch this file at all."""

    def test_run_pipeline_does_not_import_prediction_graph(self):
        assert not hasattr(run_pipeline, "PredictionGraphAssembler")

    def test_run_pipeline_source_has_no_reference_to_the_assembler(self):
        source = inspect.getsource(run_pipeline)
        assert "PredictionGraphAssembler" not in source
        assert "prediction_graph" not in source

    def test_run_dataset_still_uses_per_timepoint_detection_loop(self):
        """Confirms the architecture the P0-3 audit relied on to declare
        this path safe is still intact: one detection call per real
        timepoint, stored once into centroids_by_t[t] -- not a pairwise
        overlapping-window loop."""
        source = inspect.getsource(run_pipeline.run_dataset)
        assert "for t in range(t_dim)" in source
        assert "centroids_by_t[t] = consensus_centroids" in source


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

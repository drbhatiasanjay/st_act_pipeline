"""
Shared production helper for sequential predicted-graph assembly across
overlapping validation/evaluation windows.

P0-3 fix (2026-07-17): TrainingLoop.validate_epoch(), evaluate_checkpoint.py,
and verify_eval_fixed.py each independently accumulated a per-sample
IndexedRXGraph across a DataLoader of overlapping (t, t+1) windows (e.g.
window A=(0,1), window B=(1,2)) -- frame 1 is produced by BOTH windows
(window A's channel-1 output, window B's channel-0 output), and all three
copy-pasted graph-building blocks unconditionally called add_node() for
BOTH channels every batch, creating two distinct graph nodes for what should
be one real detection at frame 1. Confirmed via a P0-3 audit (root-cause
trace + independent reproducer using the real IndexedRXGraph/add_node code):
4 nodes / frame counts {0:1, 1:2, 2:1} for a 3-frame sample, instead of the
correct 3 nodes / {0:1, 1:1, 2:1}. Order-independent (shuffle only changes
node ID numbering, not the duplication itself).

CANONICAL GRAPH IDENTITY RULE: every frame gets exactly ONE canonical set of
graph nodes, decided purely by (sample_id, t_idx) sequence position:
- the FIRST frame seen for a sample is created from that window's channel-0
  detections;
- every SUBSEQUENT frame is created exactly once, from channel 1 of the
  window that first produces it -- a later window's channel-0 peaks for that
  SAME real timepoint are never used to create new nodes (they are not
  assumed to have identical coordinates or even the same count as the
  earlier window's channel-1 detections -- see PredictionGraphAssembler.
  process_window()'s docstring).

This is deliberately NOT coordinate-based deduplication, NOT nearest-
neighbor merging, and NEVER unions two overlapping detection sets. Frame
identity is decided by sequence position alone, established once per frame
and never revisited or compared against later detections.

Sequential ownership REQUIRES chronological per-sample window order (see
validate_window_order()) -- the caller's DataLoader must use shuffle=False.
"""

import logging

import polars as pl
from tracksdata.graph import IndexedRXGraph

logger = logging.getLogger(__name__)


class _SamplePredictionState:
    """One sample's predicted-graph assembly state: its IndexedRXGraph, the
    canonical node IDs/coordinates already established per frame, and the
    last t_idx processed (for the chronological-contract check)."""

    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        self.graph = IndexedRXGraph()
        for key in ('t', 'x', 'y', 'z'):
            try:
                self.graph.add_node_attr_key(key, pl.Int64, 0)
            except ValueError:
                pass  # key already exists

        # Canonical node IDs/coordinates already assigned per frame t_idx.
        # A frame present here has ALREADY been decided -- it is never
        # re-created from a later window's channel-0 peaks.
        self.canonical_node_ids: dict[int, list[int]] = {}
        self.canonical_coords: dict[int, list[tuple[float, float, float]]] = {}

        self.last_t_idx: int | None = None

        # Diagnostic-only raw peak counters -- never used to create nodes or
        # decide frame identity. Kept clearly separate from the graph's real
        # unique node count (see PredictionGraphAssembler.diagnostics()).
        self.raw_channel0_peaks_total = 0
        self.raw_channel1_peaks_total = 0
        self.edges_added_total = 0


class PredictionGraphAssembler:
    """
    Owns predicted-graph assembly state for every sample in one validation/
    evaluation pass. Replaces the three copy-pasted unconditional dual-frame
    node-addition blocks previously in TrainingLoop.validate_epoch(),
    evaluate_checkpoint.py, and verify_eval_fixed.py.

    Usage per batch (one (t_idx, t_idx+1) window):
        source_ids, source_coords, target_ids, target_coords = \\
            assembler.process_window(sample_id, t_idx, peaks_t, peaks_t1)
        # ... sample features at source_coords/target_coords from the
        # CURRENT window's feature tensor, run the transformer + greedy
        # edge assignment against those coordinate sets ...
        assembler.add_edges(sample_id, source_ids, target_ids, edges)

    After the full pass:
        pred_graphs = assembler.pred_graphs()  # -> dict[sample_id, IndexedRXGraph]
        diagnostics = assembler.diagnostics()
    """

    def __init__(self):
        self._samples: dict[str, _SamplePredictionState] = {}

    def _get_or_create_sample(self, sample_id: str) -> _SamplePredictionState:
        if sample_id not in self._samples:
            self._samples[sample_id] = _SamplePredictionState(sample_id)
        return self._samples[sample_id]

    def validate_window_order(self, sample_id: str, t_idx: int) -> None:
        """
        Enforce the chronological ordering contract: the first window
        observed for a sample may start at any t_idx (a sample's real GT
        annotation range need not start at t_idx=0); every SUBSEQUENT window
        for that SAME sample must be exactly contiguous (t_idx == previous
        t_idx + 1). Raises RuntimeError -- never silently produces a partial
        or skipped graph -- on a duplicate, backward, or gapped t_idx.
        Multiple samples may be interleaved in any order; only each
        sample's OWN sequence must stay contiguous.
        """
        state = self._get_or_create_sample(sample_id)
        if state.last_t_idx is not None:
            expected = state.last_t_idx + 1
            if t_idx != expected:
                raise RuntimeError(
                    f"sample_id={sample_id}: chronological window-order contract "
                    f"violated -- expected t_idx={expected} (contiguous with the "
                    f"previously processed window for this sample), got "
                    f"t_idx={t_idx}. Sequential predicted-graph assembly (P0-3) "
                    f"requires each sample's windows to be processed in strict "
                    f"chronological order -- this usually means a DataLoader is "
                    f"not using shuffle=False, or windows for this sample were "
                    f"interleaved out of order across batches."
                )
        state.last_t_idx = t_idx

    def process_window(
        self,
        sample_id: str,
        t_idx: int,
        peaks_t: list,
        peaks_t1: list,
    ) -> tuple[list[int], list[tuple], list[int], list[tuple]]:
        """
        Register one window's raw NMS peaks and apply the canonical graph
        identity rule. Call validate_window_order(sample_id, t_idx) BEFORE
        this on every window.

        peaks_t (channel 0, this window's own detections at frame t_idx):
        used to create NEW canonical nodes only if frame t_idx has never
        been seen before for this sample (i.e. this is the sample's first
        window). Otherwise frame t_idx was already created by the PRIOR
        window's channel-1 output -- peaks_t is counted diagnostically
        (raw_channel0_peaks_total) and otherwise ignored; it is NEVER
        compared against the existing canonical coordinates, matched,
        merged, or used to create additional nodes. The real model is not
        guaranteed to return identical (or even equal-count) overlapping
        predictions between window A's channel-1 and window B's channel-0
        for the same real timepoint -- this rule sidesteps that entirely by
        never using the second observation for node creation at all.

        peaks_t1 (channel 1, this window's detections at frame t_idx+1):
        ALWAYS creates new canonical nodes for frame t_idx+1 -- every frame
        is owned by exactly one channel-1 observation, the first one to
        produce it.

        Returns (source_node_ids, source_coords, target_node_ids,
        target_coords) -- the canonical node IDs/coordinates for frame
        t_idx (source) and frame t_idx+1 (target), for the caller to sample
        features against this window's feature tensor and run edge
        prediction. Coordinates are (z, y, x) tuples exactly as passed in.
        """
        state = self._get_or_create_sample(sample_id)
        state.raw_channel0_peaks_total += len(peaks_t)
        state.raw_channel1_peaks_total += len(peaks_t1)

        if t_idx not in state.canonical_node_ids:
            source_ids: list[int] = []
            source_coords: list[tuple] = []
            for (z, y, x) in peaks_t:
                node_id = state.graph.add_node({
                    't': t_idx, 'x': int(round(x)), 'y': int(round(y)), 'z': int(round(z)),
                })
                source_ids.append(node_id)
                source_coords.append((z, y, x))
            state.canonical_node_ids[t_idx] = source_ids
            state.canonical_coords[t_idx] = source_coords
        else:
            source_ids = state.canonical_node_ids[t_idx]
            source_coords = state.canonical_coords[t_idx]

        target_ids: list[int] = []
        target_coords: list[tuple] = []
        for (z, y, x) in peaks_t1:
            node_id = state.graph.add_node({
                't': t_idx + 1, 'x': int(round(x)), 'y': int(round(y)), 'z': int(round(z)),
            })
            target_ids.append(node_id)
            target_coords.append((z, y, x))
        state.canonical_node_ids[t_idx + 1] = target_ids
        state.canonical_coords[t_idx + 1] = target_coords

        return source_ids, source_coords, target_ids, target_coords

    def add_edges(
        self,
        sample_id: str,
        source_ids: list[int],
        target_ids: list[int],
        edges: list[tuple],
    ) -> int:
        """
        edges: list of (src_idx, tgt_idx, prob) where src_idx/tgt_idx are
        indices INTO source_ids/target_ids (matching
        greedy_edge_assignment()'s output convention against the coordinate/
        feature tensors built from process_window()'s returned coords) --
        NOT graph node IDs directly. Adds each edge using the persistent
        canonical node IDs. Returns the number of edges added.
        """
        state = self._get_or_create_sample(sample_id)
        count = 0
        for src_idx, tgt_idx, _prob in edges:
            state.graph.add_edge(source_ids[src_idx], target_ids[tgt_idx], {})
            count += 1
        state.edges_added_total += count
        return count

    def pred_graphs(self) -> dict[str, IndexedRXGraph]:
        """The per-sample IndexedRXGraphs, exactly the shape
        evaluate_submission() expects (dict[sample_id, IndexedRXGraph])."""
        return {sample_id: state.graph for sample_id, state in self._samples.items()}

    def diagnostics(self) -> dict[str, int]:
        """
        Aggregate, run-wide counters. predicted_nodes_total/
        predicted_edges_total count UNIQUE graph nodes/edges actually
        inserted (post canonical-identity rule) -- NOT raw per-channel peak
        counts, which are reported separately
        (raw_channel0_peaks_total/raw_channel1_peaks_total) so the two are
        never confused as the same thing.
        """
        predicted_nodes_total = sum(len(s.graph.node_ids()) for s in self._samples.values())
        predicted_edges_total = sum(s.edges_added_total for s in self._samples.values())
        raw_channel0_peaks_total = sum(s.raw_channel0_peaks_total for s in self._samples.values())
        raw_channel1_peaks_total = sum(s.raw_channel1_peaks_total for s in self._samples.values())
        return {
            'predicted_nodes_total': predicted_nodes_total,
            'predicted_edges_total': predicted_edges_total,
            'raw_channel0_peaks_total': raw_channel0_peaks_total,
            'raw_channel1_peaks_total': raw_channel1_peaks_total,
        }

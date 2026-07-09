"""
Inference utilities for detection and edge assignment.

Includes test-time augmentation (TTA) for detection and greedy edge assignment.
"""

import torch
import torch.nn as nn


def tta_inference(model: nn.Module,
                  frame_t: torch.Tensor,
                  frame_t1: torch.Tensor,
                  views: list[str] | None = None) -> torch.Tensor:
    """
    Test-time augmentation: average detection logits across 4 views.

    Applies flip transformations (original, flip Y, flip X, flip Y+X) and
    averages the logits to produce a smoother, more robust detection map.

    Args:
        model: UNet3D detection model (must be in eval mode)
        frame_t: (1, Z, Y, X) frame at time t
        frame_t1: (1, Z, Y, X) frame at time t+1
        views: List of views to use. Default: ['original', 'flip_y', 'flip_x', 'flip_yx']

    Returns:
        Averaged logits: (1, 1, Z, Y, X) detection probabilities [0,1]
    """
    if views is None:
        views = ['original', 'flip_y', 'flip_x', 'flip_yx']

    logits_sum = None
    num_views = 0

    with torch.no_grad():
        for view in views:
            # Apply flip transformation
            if view == 'original':
                frame_t_aug = frame_t.clone()
                frame_t1_aug = frame_t1.clone()
            elif view == 'flip_y':
                frame_t_aug = torch.flip(frame_t, dims=[1])  # Flip Y axis (dim 1 is Y)
                frame_t1_aug = torch.flip(frame_t1, dims=[1])
            elif view == 'flip_x':
                frame_t_aug = torch.flip(frame_t, dims=[2])  # Flip X axis (dim 2 is X)
                frame_t1_aug = torch.flip(frame_t1, dims=[2])
            elif view == 'flip_yx':
                frame_t_aug = torch.flip(frame_t, dims=[1, 2])  # Flip both Y and X
                frame_t1_aug = torch.flip(frame_t1, dims=[1, 2])
            else:
                raise ValueError(f"Unknown view: {view}")

            # Concatenate frames and add batch dimension for model input
            x = torch.cat([frame_t_aug, frame_t1_aug], dim=0)  # (2, Z, Y, X)
            x = x.unsqueeze(0)  # (1, 2, Z, Y, X)

            # Forward pass
            logits, _ = model(x)  # (1, 1, Z, Y, X)

            # Reverse flip transformation on output logits
            if view == 'flip_y':
                logits = torch.flip(logits, dims=[3])  # Reverse Y flip (dim 3 is Y in output)
            elif view == 'flip_x':
                logits = torch.flip(logits, dims=[4])  # Reverse X flip (dim 4 is X)
            elif view == 'flip_yx':
                logits = torch.flip(logits, dims=[3, 4])  # Reverse both

            # Accumulate logits
            if logits_sum is None:
                logits_sum = logits.clone()
            else:
                logits_sum = logits_sum + logits

            num_views += 1

    # Average across views
    averaged_logits = logits_sum / num_views

    return averaged_logits


def greedy_edge_assignment(edge_probs: torch.Tensor,
                          nodes_t: torch.Tensor,
                          nodes_t1: torch.Tensor,
                          candidate_edges: torch.Tensor | None = None,
                          threshold: float = 0.5,
                          max_children: int = 2,
                          max_parents: int = 1) -> dict:
    """
    Greedy edge assignment respecting cardinality constraints.

    Sorts candidate edges by probability (descending) and greedily accepts
    edges that respect cardinality limits: each node has ≤max_parents incoming
    and ≤max_children outgoing edges.

    Args:
        edge_probs: (num_candidates,) edge probabilities [0,1]
        nodes_t: (n_t, 3) source node coordinates [z, y, x]
        nodes_t1: (n_t1, 3) target node coordinates [z, y, x]
        candidate_edges: (num_candidates, 2) edge indices as (src_idx, tgt_idx).
                        If None, assumes edges are all pairwise edges in order.
        threshold: Minimum probability to consider an edge (default 0.5)
        max_children: Maximum outgoing edges per node (default 2 for divisions)
        max_parents: Maximum incoming edges per node (default 1)

    Returns:
        Dictionary containing:
        - 'edges': list of accepted (src_idx, tgt_idx, probability) tuples
        - 'edge_graph': dict mapping (src_idx, tgt_idx) to probability
        - 'stats': dict with assignment statistics
    """
    n_t = nodes_t.shape[0]
    n_t1 = nodes_t1.shape[0]

    # Handle empty node sets
    if n_t == 0 or n_t1 == 0:
        return {
            'edges': [],
            'edge_graph': {},
            'stats': {
                'num_nodes_t': n_t,
                'num_nodes_t1': n_t1,
                'num_candidate_edges': 0,
                'num_accepted_edges': 0,
                'mean_prob': 0.0,
            }
        }

    # Generate candidate edges if not provided
    if candidate_edges is None:
        candidate_edges = []
        for i in range(n_t):
            for j in range(n_t1):
                candidate_edges.append((i, j))
        candidate_edges = torch.tensor(candidate_edges, dtype=torch.long)

    # Filter edges above threshold
    valid_mask = edge_probs > threshold
    valid_edges = candidate_edges[valid_mask]
    valid_probs = edge_probs[valid_mask]

    # Sort by probability (descending)
    sorted_indices = torch.argsort(valid_probs, descending=True)
    sorted_edges = valid_edges[sorted_indices]
    sorted_probs = valid_probs[sorted_indices]

    # Greedy assignment
    accepted_edges = []
    children_count = {i: 0 for i in range(n_t)}
    parents_count = {j: 0 for j in range(n_t1)}

    for edge_idx, (src, tgt) in enumerate(sorted_edges):
        src_idx = int(src.item() if isinstance(src, torch.Tensor) else src)
        tgt_idx = int(tgt.item() if isinstance(tgt, torch.Tensor) else tgt)
        prob = float(sorted_probs[edge_idx].item() if isinstance(sorted_probs[edge_idx], torch.Tensor) else sorted_probs[edge_idx])

        # Check cardinality constraints
        if children_count[src_idx] >= max_children:
            continue  # Source node already has max outgoing edges
        if parents_count[tgt_idx] >= max_parents:
            continue  # Target node already has max incoming edges

        # Accept edge
        accepted_edges.append((src_idx, tgt_idx, prob))
        children_count[src_idx] += 1
        parents_count[tgt_idx] += 1

    # Build edge graph dictionary
    edge_graph = {(src, tgt): prob for src, tgt, prob in accepted_edges}

    # Compute statistics
    mean_prob = sum(prob for _, _, prob in accepted_edges) / len(accepted_edges) if accepted_edges else 0.0

    stats = {
        'num_nodes_t': n_t,
        'num_nodes_t1': n_t1,
        'num_candidate_edges': len(candidate_edges),
        'num_valid_edges': len(valid_edges),
        'num_accepted_edges': len(accepted_edges),
        'threshold': threshold,
        'mean_prob': mean_prob,
        'max_children_constraint': max_children,
        'max_parents_constraint': max_parents,
    }

    return {
        'edges': accepted_edges,
        'edge_graph': edge_graph,
        'stats': stats
    }

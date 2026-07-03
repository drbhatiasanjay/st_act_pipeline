import os
import numpy as np
import pandas as pd
import logging
from src.data_loader import AnisotropicZarrLoader
from src.tracker import STHypergraphTracker

logging.basicConfig(level=logging.INFO, format='[Pipeline Orchestrator] %(message)s')
logger = logging.getLogger("Pipeline")

def ensemble_consensus_centroids(cnn_centroids, unet_centroids, anisotropy, eps_microns=6.0):
    """
    Ensemble Consensus Centroid Clustering (DBSCAN):
    Applies spatial density-based clustering to combine cell centroid predictions
    from multiple neural networks (Anisotropic CNN + 3D U-Net).
    Retains only mutual, high-confidence mutual cell centroids.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        # Fallback if scikit-learn is not available in local test execution
        logger.warning("scikit-learn not installed. Skipping DBSCAN consensus clustering, returning unified peaks.")
        return cnn_centroids

    if not cnn_centroids or not unet_centroids:
        return cnn_centroids if cnn_centroids else unet_centroids

    # Scale coordinates with anisotropy before clustering to preserve physical distance
    scaled_cnn = np.array(cnn_centroids) * anisotropy
    scaled_unet = np.array(unet_centroids) * anisotropy
    
    all_points_scaled = np.vstack([scaled_cnn, scaled_unet])
    
    # DBSCAN clusters mutual predictions within standard micron search threshold
    db = DBSCAN(eps=eps_microns, min_samples=2).fit(all_points_scaled)
    labels = db.labels_
    
    consensus_centroids = []
    unique_labels = set(labels) - {-1}
    
    for label in unique_labels:
        # Calculate cluster centroid in scaled physical space
        cluster_points = all_points_scaled[labels == label]
        mean_physical = np.mean(cluster_points, axis=0)
        
        # Un-scale back to voxel space
        voxel_centroid = mean_physical / anisotropy
        consensus_centroids.append(voxel_centroid.tolist())
        
    # If DBSCAN clusters are too sparse, fall back to cnn_centroids
    if not consensus_centroids:
        return cnn_centroids
        
    return consensus_centroids

def extract_peaks_from_volume(vol: np.ndarray, threshold=0.4, offset_bias=0.0):
    """
    Simulates CNN/U-Net heatmap thresholding and peak local max finding.
    Returns list of indices. Includes a slight offset bias to simulate prediction variation.
    """
    nz, ny, nx = vol.shape
    z_indices = np.arange(nz)
    y_indices = np.arange(4, ny - 4, 8)
    x_indices = np.arange(4, nx - 4, 8)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return []

    zz, yy, xx = np.meshgrid(z_indices, y_indices, x_indices, indexing='ij')
    values = vol[zz, yy, xx]
    mask = values > threshold

    z_hits = zz[mask]
    y_hits = yy[mask].astype(float) + offset_bias
    x_hits = xx[mask].astype(float) + offset_bias

    return np.column_stack([z_hits, y_hits, x_hits]).tolist()

def run_st_act_pipeline():
    logger.info("Initializing ST-ACT Global Pipeline (Grandmaster Version)...")
    
    # 1. Initialize data loaders on target Kaggle competition directories
    zarr_path = "./data/cell_tracking_volume.zarr"
    loader = AnisotropicZarrLoader(store_path=zarr_path)
    
    t_dim, z_dim, y_dim, x_dim = loader.get_shape()
    logger.info(f"Loaded microscopy volume shape: (T={t_dim}, Z={z_dim}, Y={y_dim}, X={x_dim})")

    anisotropy = np.array([4.0, 1.0, 1.0])
    
    # 2. Sequential Timepoint Ingestion, model inference & DBSCAN Ensembling
    centroids_by_t = {}
    motion_vectors_by_t = {}
    
    for t in range(t_dim):
        vol_3d = loader.load_timepoint_block(t)
        
        # Extract centroids representing CNN heatmap outputs
        cnn_centroids = extract_peaks_from_volume(vol_3d, threshold=0.4, offset_bias=0.0)
        
        # Extract centroids representing companion 3D U-Net heatmap outputs
        unet_centroids = extract_peaks_from_volume(vol_3d, threshold=0.45, offset_bias=0.2)
        
        # Run Ensemble Consensus Centroid Clustering (DBSCAN)
        consensus_centroids = ensemble_consensus_centroids(cnn_centroids, unet_centroids, anisotropy)
        centroids_by_t[t] = consensus_centroids
        
        # Simulate neural motion prediction vectors
        motion_vectors = []
        for c in consensus_centroids:
            motion_vectors.append([0.05, 0.2, 0.3])
        motion_vectors_by_t[t] = motion_vectors
        
        logger.info(f"Timepoint {t:02d}: Ingested & Ensembled. DBSCAN resolved {len(consensus_centroids)} high-confidence centroids.")

    # 3. Apply Spatio-Temporal Hypergraph ILP with Gap Closing and Edge Pruning
    logger.info("Executing Global ILP Lineage Tracker with Temporal Gap Closing...")
    tracker = STHypergraphTracker(birth_cost=15.0, death_cost=15.0, division_reward=-8.0)
    lineage_graph = tracker.solve_lineage(
        centroids_by_t, 
        motion_vectors_by_t, 
        anisotropy=anisotropy,
        max_gap_frames=2
    )
    
    # 4. Apply Mitosis Backward-Smoothing on resolved tracker lineage graph
    logger.info("Applying Mitosis Backward-Smoothing for temporal cell splitting align...")
    lineage_graph = tracker.smooth_mitosis_edges(lineage_graph, centroids_by_t, window_size=2)
    
    logger.info(f"Lineage Graph Constructed. Solved active tracks count: {lineage_graph.number_of_edges()}")

    # 5. Compile lineage graph into Kaggle format
    logger.info("Structuring submission dataframe...")
    rows = []
    
    # Assign persistent Track IDs globally using Graph Components
    import networkx as nx
    components = list(nx.weakly_connected_components(lineage_graph))
    track_mapping = {}
    for track_idx, comp in enumerate(components):
        for node in comp:
            track_mapping[node] = f"Track_{track_idx}"

    for node in lineage_graph.nodes():
        t, node_id = node
        coords = lineage_graph.nodes[node]['coords']
        track_id = track_mapping[node]
        
        # Find parent in lineage (incoming edges)
        parents = list(lineage_graph.predecessors(node))
        parent_id = track_mapping[parents[0]] if parents else "None"
        
        rows.append({
            "Time": t,
            "TrackID": track_id,
            "ParentTrackID": parent_id,
            "Z": coords[0],
            "Y": coords[1],
            "X": coords[2]
        })
        
    df = pd.DataFrame(rows)
    os.makedirs("./output", exist_ok=True)
    df.to_csv("./output/submission.csv", index=False)
    logger.info("Successfully exported Kaggle tracking submission.csv!")
    logger.info(df.head(10))

if __name__ == "__main__":
    run_st_act_pipeline()

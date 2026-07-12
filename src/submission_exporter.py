"""
Submission exporter for the Kaggle cell tracking competition.

Provides functions to:
1. Export tracksdata graphs to Kaggle-compliant submission CSV
2. Validate submission CSV against the schema

Schema:
    id,dataset,row_type,node_id,t,z,y,x,source_id,target_id

    - id: globally sequential integer (0, 1, 2, ...)
    - dataset: dataset identifier (string)
    - row_type: 'node' or 'edge'
    - node_id: per-dataset-local (resets to 1 for each new dataset)
    - t,z,y,x: time and spatial coordinates (integers, -1 for edges)
    - source_id,target_id: node_ids for edges (-1 for nodes)
"""

import logging
from pathlib import Path

import pandas as pd
import tracksdata as td

logger = logging.getLogger(__name__)


def export_submission(
    graphs_dict: dict[str, td.graph.BaseGraph],
    output_path: str | Path,
    required_dataset_ids: list[str] | None = None,
) -> str | pd.DataFrame:
    """
    Export tracksdata graphs to a Kaggle-compliant submission CSV.

    Parameters
    ----------
    graphs_dict : dict[str, tracksdata.graph.BaseGraph]
        Dictionary mapping dataset_id (str) to tracksdata graph objects.
        Graphs are processed in sorted dataset_id order for deterministic output.

    output_path : str or Path
        Path where the CSV file will be written.

    required_dataset_ids : list[str], optional
        The full set of real test dataset_ids that must appear in the output
        (per the competition rule "Every dataset in the test set must appear
        in the submission"). A dataset with zero detected nodes/edges
        contributes zero rows under this schema (there's no "empty" row
        type), so it would otherwise silently vanish from the CSV with no
        warning -- exactly the kind of silent gap this project has
        repeatedly been bitten by. If given, any required id producing zero
        rows is logged loudly (not fabricated -- inventing fake detections
        would be worse than an honest gap).

    Returns
    -------
    str or pd.DataFrame
        Path to the written CSV file (as string).

    Raises
    ------
    ValueError
        If a graph contains edges with nodes not in node_id_map (dangling edges).
    """
    rows = []
    global_id = 0

    # Process datasets in sorted order for deterministic output
    for dataset_id in sorted(graphs_dict.keys()):
        graph = graphs_dict[dataset_id]

        # Reset per-dataset node_id mapping
        node_id_map = {}
        per_dataset_node_id = 0

        # Process nodes first
        # Use node_ids() to get the list of node identifiers (integers in tracksdata)
        for node in graph.node_ids():
            per_dataset_node_id += 1
            node_id_map[node] = per_dataset_node_id

            # Extract node attributes (t, z, y, x)
            # tracksdata stores these as node attributes (accessible via dict-like interface)
            node_attrs = graph.nodes[node]
            t = int(node_attrs['t'])
            z = int(node_attrs['z'])
            y = int(node_attrs['y'])
            x = int(node_attrs['x'])

            # Create node row: [id, dataset, row_type, node_id, t, z, y, x, source_id=-1, target_id=-1]
            row = {
                'id': global_id,
                'dataset': dataset_id,
                'row_type': 'node',
                'node_id': per_dataset_node_id,
                't': t,
                'z': z,
                'y': y,
                'x': x,
                'source_id': -1,
                'target_id': -1,
            }
            rows.append(row)
            global_id += 1

        # Process edges
        # Use edge_list() to get the list of (source, target) tuples
        for source, target in graph.edge_list():
            # Validate that both nodes are in the node_id_map
            if source not in node_id_map or target not in node_id_map:
                raise ValueError(
                    f"Dangling edge in dataset '{dataset_id}': ({source}, {target}). "
                    f"Source or target node not found in node_id_map."
                )

            source_node_id = node_id_map[source]
            target_node_id = node_id_map[target]

            # Create edge row: [id, dataset, row_type='edge', node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id, target_id]
            row = {
                'id': global_id,
                'dataset': dataset_id,
                'row_type': 'edge',
                'node_id': -1,
                't': -1,
                'z': -1,
                'y': -1,
                'x': -1,
                'source_id': source_node_id,
                'target_id': target_node_id,
            }
            rows.append(row)
            global_id += 1

    if required_dataset_ids is not None:
        present_dataset_ids = {row['dataset'] for row in rows}
        missing = sorted(set(required_dataset_ids) - present_dataset_ids)
        if missing:
            logger.warning(
                f"{len(missing)} required test dataset(s) produced ZERO rows and are "
                f"MISSING from the submission (violates 'every dataset must appear'): "
                f"{missing}. Not fabricating fake detections -- diagnose why these "
                f"datasets got zero predictions before submitting."
            )

    # Convert to DataFrame and save
    df = pd.DataFrame(rows)

    # Ensure column order matches schema
    column_order = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    df = df[column_order]

    # Save to CSV (no index, exact schema)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return str(output_path)


def validate_submission(csv_path: str | Path) -> bool:
    """
    Validate a submission CSV against the schema.

    Parameters
    ----------
    csv_path : str or Path
        Path to the submission CSV file.

    Returns
    -------
    bool
        True if all validation checks pass.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If any validation check fails. Error message includes the specific check
        that failed and relevant details.
    """
    csv_path = Path(csv_path)

    # Check 1: File exists and is readable
    if not csv_path.exists():
        raise FileNotFoundError(f"Submission CSV not found: {csv_path}")

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}") from e

    # Check 2: Header is exactly as expected
    expected_header = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    if list(df.columns) != expected_header:
        raise ValueError(
            f"CSV header mismatch.\n"
            f"Expected: {expected_header}\n"
            f"Got: {list(df.columns)}"
        )

    # Check 3: All rows have 10 columns (no missing values in read)
    if len(df.columns) != 10:
        raise ValueError(f"Expected 10 columns, got {len(df.columns)}")

    # Check for NaN values (which would indicate missing data)
    if df.isnull().any().any():
        raise ValueError(
            f"CSV contains missing values (NaN):\n{df[df.isnull().any(axis=1)]}"
        )

    # Check 4: id column is globally sequential (0, 1, 2, ..., len(df)-1)
    expected_ids = list(range(len(df)))
    actual_ids = df['id'].tolist()
    if actual_ids != expected_ids:
        # Find first mismatch
        for i, (exp, act) in enumerate(zip(expected_ids, actual_ids, strict=False)):
            if exp != act:
                raise ValueError(
                    f"id column not sequential at row {i}: expected {exp}, got {act}"
                )
        raise ValueError(f"id column length mismatch: expected {len(expected_ids)}, got {len(actual_ids)}")

    # Check 5: row_type is either 'node' or 'edge'
    invalid_row_types = set(df['row_type'].unique()) - {'node', 'edge'}
    if invalid_row_types:
        raise ValueError(f"Invalid row_type values: {invalid_row_types}. Must be 'node' or 'edge'.")

    # Check 6: For 'node' rows: source_id=-1, target_id=-1, coordinates are integers
    node_rows = df[df['row_type'] == 'node']
    if len(node_rows) > 0:
        bad_source = node_rows[node_rows['source_id'] != -1]
        if len(bad_source) > 0:
            raise ValueError(
                f"For 'node' rows, source_id must be -1. "
                f"Found {len(bad_source)} violations at indices: {bad_source.index.tolist()[:5]}"
            )

        bad_target = node_rows[node_rows['target_id'] != -1]
        if len(bad_target) > 0:
            raise ValueError(
                f"For 'node' rows, target_id must be -1. "
                f"Found {len(bad_target)} violations at indices: {bad_target.index.tolist()[:5]}"
            )

        # Check coordinates are integers (no decimals, or float values that are whole numbers)
        for col in ['t', 'z', 'y', 'x']:
            if not all(node_rows[col] == node_rows[col].astype(int)):
                raise ValueError(
                    f"For 'node' rows, column '{col}' must have integer values. "
                    f"Found non-integer values."
                )

    # Check 7: For 'edge' rows: node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id/target_id are positive integers
    edge_rows = df[df['row_type'] == 'edge']
    if len(edge_rows) > 0:
        bad_node_id = edge_rows[edge_rows['node_id'] != -1]
        if len(bad_node_id) > 0:
            raise ValueError(
                f"For 'edge' rows, node_id must be -1. "
                f"Found {len(bad_node_id)} violations."
            )

        for col in ['t', 'z', 'y', 'x']:
            bad_coords = edge_rows[edge_rows[col] != -1]
            if len(bad_coords) > 0:
                raise ValueError(
                    f"For 'edge' rows, column '{col}' must be -1. "
                    f"Found {len(bad_coords)} violations."
                )

        bad_source = edge_rows[edge_rows['source_id'] < 0]
        if len(bad_source) > 0:
            raise ValueError(
                f"For 'edge' rows, source_id must be positive integer. "
                f"Found {len(bad_source)} violations with negative source_id."
            )

        bad_target = edge_rows[edge_rows['target_id'] < 0]
        if len(bad_target) > 0:
            raise ValueError(
                f"For 'edge' rows, target_id must be positive integer. "
                f"Found {len(bad_target)} violations with negative target_id."
            )

    # Check 8: node_id resets per dataset (for node rows, track max node_id per dataset)
    node_rows_with_dataset = df[df['row_type'] == 'node'][['dataset', 'node_id']]
    if len(node_rows_with_dataset) > 0:
        for dataset_id in node_rows_with_dataset['dataset'].unique():
            dataset_node_ids = node_rows_with_dataset[
                node_rows_with_dataset['dataset'] == dataset_id
            ]['node_id'].tolist()

            # Check that node_ids for this dataset are sequential starting from 1
            if dataset_node_ids:
                expected_node_ids = list(range(1, len(dataset_node_ids) + 1))
                if sorted(dataset_node_ids) != expected_node_ids:
                    raise ValueError(
                        f"For dataset '{dataset_id}', node_id is not sequential from 1. "
                        f"Expected {expected_node_ids}, got {sorted(set(dataset_node_ids))}"
                    )

    # Check 9: No duplicate (dataset, node_id) pairs within node rows
    node_rows_unique = df[df['row_type'] == 'node'][['dataset', 'node_id']]
    if len(node_rows_unique) > 0:
        duplicates = node_rows_unique.duplicated()
        if duplicates.any():
            dup_rows = node_rows_unique[duplicates]
            raise ValueError(
                f"Found duplicate (dataset, node_id) pairs in node rows:\n{dup_rows}"
            )

    return True

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
        in the submission"). P0-6: fail-closed, not just a warning -- when
        given, this raises ValueError (never fabricates rows) if: the list
        is empty, contains duplicates or non-string/empty entries;
        graphs_dict's keys do not exactly match required_dataset_ids (missing
        and unexpected reported separately); any required dataset has zero
        nodes; or the total node/edge row count across all required datasets
        is zero. None (the default) preserves the original generic,
        schema-only export behavior with no required-ID enforcement.

    Returns
    -------
    str or pd.DataFrame
        Path to the written CSV file (as string).

    Raises
    ------
    ValueError
        If a graph contains edges with nodes not in node_id_map (dangling edges).
    """
    if required_dataset_ids is not None:
        if not isinstance(required_dataset_ids, list | tuple):
            raise ValueError(
                f"required_dataset_ids must be an ordered sequence of strings, got "
                f"{type(required_dataset_ids).__name__}"
            )
        required_dataset_ids = list(required_dataset_ids)
        if len(required_dataset_ids) == 0:
            raise ValueError("required_dataset_ids must not be an empty list.")
        for did in required_dataset_ids:
            if not isinstance(did, str) or did == "":
                raise ValueError(
                    f"required_dataset_ids must contain only non-empty strings, got {did!r}"
                )
        seen_ids: set = set()
        dupe_ids: set = set()
        for did in required_dataset_ids:
            if did in seen_ids:
                dupe_ids.add(did)
            seen_ids.add(did)
        if dupe_ids:
            raise ValueError(f"required_dataset_ids contains duplicate ID(s): {sorted(dupe_ids)}")

        present_dataset_ids = set(graphs_dict.keys())
        required_set = set(required_dataset_ids)
        missing = sorted(required_set - present_dataset_ids)
        unexpected = sorted(present_dataset_ids - required_set)
        if missing or unexpected:
            raise ValueError(
                f"graphs_dict dataset IDs do not exactly match required_dataset_ids. "
                f"Missing (required but absent from graphs_dict): {missing}. "
                f"Unexpected (present in graphs_dict but not required): {unexpected}."
            )

        zero_node_required = sorted(
            did for did in required_dataset_ids if graphs_dict[did].num_nodes() == 0
        )
        if zero_node_required:
            raise ValueError(
                f"Required dataset(s) have ZERO nodes: {zero_node_required}. Not "
                f"fabricating detections -- diagnose why these datasets got zero "
                f"predictions before submitting."
            )

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
            # round(), not bare int(): a smoothed/upstream float coordinate
            # can land infinitesimally below its true integer value from
            # floating-point noise -- int() truncates toward zero and
            # silently exports one voxel too low. Same bug class as
            # run_pipeline.py:convert_nx_to_tracksdata (fixed ace1a60); this
            # is the literal last site before a coordinate becomes a scored
            # row, so it needs its own defense rather than relying on every
            # upstream caller already rounding correctly.
            t = int(round(node_attrs['t']))
            z = int(round(node_attrs['z']))
            y = int(round(node_attrs['y']))
            x = int(round(node_attrs['x']))

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
        total_node_rows = sum(1 for row in rows if row['row_type'] == 'node')
        total_edge_rows = sum(1 for row in rows if row['row_type'] == 'edge')
        if total_node_rows == 0:
            raise ValueError("Total node row count across all required datasets is ZERO.")
        if total_edge_rows == 0:
            raise ValueError("Total edge row count across all required datasets is ZERO.")

    # Convert to DataFrame and save. Pass columns= explicitly: pd.DataFrame(rows) on an
    # empty rows list produces a DataFrame with zero columns, and df[column_order] then
    # raises KeyError since those column names don't exist yet on an empty frame -- a real
    # crash confirmed when every sample in a submission has zero detections.
    column_order = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    df = pd.DataFrame(rows, columns=column_order)

    # Save to CSV (no index, exact schema)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return str(output_path)


def validate_submission(
    csv_path: str | Path,
    required_dataset_ids: list[str] | None = None,
) -> bool:
    """
    Validate a submission CSV against the schema.

    Parameters
    ----------
    csv_path : str or Path
        Path to the submission CSV file.
    required_dataset_ids : list[str], optional
        P0-6: when given, enforces production-submission-grade checks on top
        of the generic schema checks below -- see Part E2's exact list
        (non-header-only CSV, exact dataset-ID equality, at least one node
        per required dataset, positive total node/edge rows, structurally
        valid edges: same-dataset endpoints, source != target, target time =
        source time + 1, no duplicate edges, out-degree <= 2, in-degree <=
        1, non-negative node coordinates, positive integer node/edge IDs,
        no bool-like/non-integer values in ID/coordinate columns). None (the
        default) preserves the original generic schema-validation behavior,
        including accepting a header-only, schema-correct CSV.

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

    if required_dataset_ids is not None:
        if not isinstance(required_dataset_ids, list | tuple):
            raise ValueError(
                f"required_dataset_ids must be an ordered sequence of strings, got "
                f"{type(required_dataset_ids).__name__}"
            )
        required_dataset_ids = list(required_dataset_ids)
        if len(required_dataset_ids) == 0:
            raise ValueError("required_dataset_ids must not be an empty list.")
        for did in required_dataset_ids:
            if not isinstance(did, str) or did == "":
                raise ValueError(
                    f"required_dataset_ids must contain only non-empty strings, got {did!r}"
                )
        seen_req_ids: set = set()
        dupe_req_ids: set = set()
        for did in required_dataset_ids:
            if did in seen_req_ids:
                dupe_req_ids.add(did)
            seen_req_ids.add(did)
        if dupe_req_ids:
            raise ValueError(f"required_dataset_ids contains duplicate ID(s): {sorted(dupe_req_ids)}")

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

    if required_dataset_ids is not None and len(df) == 0:
        raise ValueError(
            "Submission is header-only (zero rows) but required_dataset_ids was given "
            "-- a real submission must contain rows for every required dataset."
        )

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

    if required_dataset_ids is not None:
        present_dataset_ids = set(df['dataset'].unique().tolist()) if len(df) > 0 else set()
        required_set = set(required_dataset_ids)
        missing = sorted(required_set - present_dataset_ids)
        unexpected = sorted(present_dataset_ids - required_set)
        if missing or unexpected:
            raise ValueError(
                f"Submission dataset IDs do not exactly match required_dataset_ids. "
                f"Missing (required but absent): {missing}. "
                f"Unexpected (present but not required): {unexpected}."
            )

        node_rows = df[df['row_type'] == 'node']
        edge_rows = df[df['row_type'] == 'edge']

        for did in required_dataset_ids:
            if len(node_rows[node_rows['dataset'] == did]) == 0:
                raise ValueError(f"Required dataset '{did}' has zero node rows.")

        if len(node_rows) == 0:
            raise ValueError("Total node row count is ZERO.")
        if len(edge_rows) == 0:
            raise ValueError("Total edge row count is ZERO.")

        # Reject bool-like or non-numeric values: an ID/coordinate column
        # must be a genuine integer dtype, never bool or object (e.g. stray
        # "True"/"False" strings that pandas would otherwise silently accept
        # into an object-dtype column).
        numeric_columns = ['id', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
        for col in numeric_columns:
            if df[col].dtype == bool:
                raise ValueError(f"Column '{col}' must not contain boolean values.")
            if not pd.api.types.is_integer_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be integer-typed, got dtype {df[col].dtype}.")

        bad_node_coords = node_rows[(node_rows[['t', 'z', 'y', 'x']] < 0).any(axis=1)]
        if len(bad_node_coords) > 0:
            raise ValueError(
                f"Node rows contain negative time/coordinate values at indices: "
                f"{bad_node_coords.index.tolist()[:5]}"
            )

        bad_edge_endpoints = edge_rows[(edge_rows['source_id'] < 1) | (edge_rows['target_id'] < 1)]
        if len(bad_edge_endpoints) > 0:
            raise ValueError(
                f"Edge rows must have positive (>=1) source_id/target_id, violations at "
                f"indices: {bad_edge_endpoints.index.tolist()[:5]}"
            )

        # Per-(dataset, node_id) -> t lookup, for edge structural validation.
        node_time_lookup: dict[tuple, int] = {
            (row['dataset'], int(row['node_id'])): int(row['t'])
            for _, row in node_rows.iterrows()
        }

        seen_edge_keys: set = set()
        out_degree: dict[tuple, int] = {}
        in_degree: dict[tuple, int] = {}
        for _, row in edge_rows.iterrows():
            did = row['dataset']
            src = int(row['source_id'])
            tgt = int(row['target_id'])

            if (did, src) not in node_time_lookup:
                raise ValueError(f"Edge in dataset '{did}' references missing source node_id={src}.")
            if (did, tgt) not in node_time_lookup:
                raise ValueError(f"Edge in dataset '{did}' references missing target node_id={tgt}.")
            if src == tgt:
                raise ValueError(f"Edge in dataset '{did}' has source_id == target_id == {src} (self-edge).")

            src_t = node_time_lookup[(did, src)]
            tgt_t = node_time_lookup[(did, tgt)]
            if tgt_t != src_t + 1:
                raise ValueError(
                    f"Edge in dataset '{did}' (source_id={src} t={src_t}, target_id={tgt} "
                    f"t={tgt_t}) does not satisfy target_t == source_t + 1."
                )

            edge_key = (did, src, tgt)
            if edge_key in seen_edge_keys:
                raise ValueError(f"Duplicate edge in dataset '{did}': source_id={src}, target_id={tgt}.")
            seen_edge_keys.add(edge_key)

            out_key = (did, src)
            in_key = (did, tgt)
            out_degree[out_key] = out_degree.get(out_key, 0) + 1
            in_degree[in_key] = in_degree.get(in_key, 0) + 1

        bad_out = {k: v for k, v in out_degree.items() if v > 2}
        if bad_out:
            raise ValueError(f"Node(s) exceed max out-degree of 2: {bad_out}")
        bad_in = {k: v for k, v in in_degree.items() if v > 1}
        if bad_in:
            raise ValueError(f"Node(s) exceed max in-degree of 1: {bad_in}")

    return True

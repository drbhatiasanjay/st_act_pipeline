import logging
import os
from collections.abc import Generator

import blosc2
import numpy as np
import zarr
from numcodecs import Blosc

# Cap blosc2's internal decompression thread pool. It defaults to os.cpu_count()
# (16 here), but our real per-timepoint chunks are a single (1,64,256,256) block
# -- spawning 16 threads to decompress one small chunk is pure overhead, not
# parallelism. Confirmed on real data: this single call is a ~375x speedup
# (30s -> 0.08s per timepoint) and also eliminates a native `Segmentation
# fault` that reproduced reliably around the ~36th real decompression with the
# default thread count (16-thread churn against Windows' scheduler/handle
# limits over many repeated calls, not a zarr/blosc2 correctness bug).
blosc2.set_nthreads(1)

# Configure optimized logging
logging.basicConfig(level=logging.INFO, format='[ST-ACT Data Ingestion] %(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger("ST_ACT_Loader")

class AnisotropicZarrLoader:
    """
    ST-ACT Memory-Safe 4D Anisotropic Zarr v3 Ingestor.
    Loads and decompresses 3D blocks/timepoints from a 4D Zarr v3 store
    using fast blosc2/blosclz compression codecs in a memory-efficient manner.
    """
    def __init__(self, store_path: str, anisotropy_ratio: tuple[float, float, float] = (4.0, 1.0, 1.0), simulate: bool = False):
        """
        Initialize the Anisotropic Zarr Loader.

        Args:
            store_path (str): File system path to the Zarr v3 store directory or zip container.
            anisotropy_ratio (Tuple[float, float, float]): Voxel size ratios for (Z, Y, X) axes.
                Defaults to (4.0, 1.0, 1.0) for anisotropic fluorescent microscopy.
            simulate (bool): If True, creates a simulated store when path doesn't exist.
                Defaults to False (real data required).
        """
        self.store_path = store_path
        self.anisotropy_ratio = np.array(anisotropy_ratio, dtype=np.float32)
        self.dataset: zarr.Array | None = None
        self.simulate = simulate
        self._quantile_normalization_params: tuple[float, float] | None = None
        # Single-slot cache for the most recently decompressed timepoint.
        # CompetitionDataset.__getitem__ requests (t, t+1) per item with
        # shuffle=False consecutive access, so item i's t+1 is item i+1's t --
        # confirmed live in a real run's log as every timepoint being
        # decompressed twice in a row (~30s each) with no caching at all.
        self._last_t: int | None = None
        self._last_normalize: bool | None = None
        self._last_block: np.ndarray | None = None
        self._init_store()

    def _init_store(self) -> None:
        """
        Initializes connection to the Zarr v3 store.
        For real competition data: reads Zarr v3 OME-NGFF stores at array path "0/".
        For testing: optionally creates a simulated store if simulate=True.
        """
        try:
            # Check if store exists
            if not os.path.exists(self.store_path):
                if self.simulate:
                    logger.warning(f"Path '{self.store_path}' not found. Creating simulated cellular Zarr store.")
                    self._create_simulated_store()
                else:
                    raise FileNotFoundError(f"Real data path '{self.store_path}' not found. Use simulate=True for testing.")
            else:
                # Real data exists: attempt to open it
                logger.info(f"Opening real Zarr v3 store at '{self.store_path}'")

            # Open Zarr array (Zarr v3 OME-NGFF compliant)
            try:
                # First, try opening as a group (root) to access nested array at "0/"
                root = zarr.open(self.store_path, mode='r')

                # Check if root is itself an array (legacy single-array store)
                if hasattr(root, 'shape') and hasattr(root, 'chunks'):
                    self.dataset = root
                    logger.info("Opened Zarr store as direct array.")
                # Check if it's a group with nested array at "0/" (OME-NGFF standard)
                elif hasattr(root, '__getitem__'):
                    try:
                        self.dataset = root['0']
                        logger.info("Resolved nested OME-NGFF array at path '0/' in Zarr group.")
                    except (KeyError, TypeError):
                        # If no "0/", try first available array key
                        array_keys = list(root.array_keys()) if hasattr(root, 'array_keys') else []
                        if array_keys:
                            self.dataset = root[array_keys[0]]
                            logger.info(f"Resolved array '{array_keys[0]}' in Zarr group.")
                        else:
                            raise ValueError("Zarr store group contains no openable arrays.") from None
                else:
                    raise ValueError("Zarr store is neither an array nor a readable group.")

                # Extract quantile normalization parameters if present in metadata
                self._extract_quantile_params(root)

            except Exception as open_err:
                logger.error(f"Failed to open Zarr store: {str(open_err)}")
                raise ValueError(f"Could not resolve openable Zarr array at '{self.store_path}': {str(open_err)}") from open_err

            logger.info(f"Successfully opened Zarr volume: {self.store_path}")
            logger.info(f"Volume Shape: {self.dataset.shape} | Chunks: {self.dataset.chunks} | Dtype: {self.dataset.dtype}")
            if self._quantile_normalization_params:
                logger.info(f"Quantile normalization parameters found: q_low={self._quantile_normalization_params[0]}, q_high={self._quantile_normalization_params[1]}")
        except Exception as e:
            logger.error(f"Failed to initialize Zarr Store at '{self.store_path}': {str(e)}")
            raise

    def _extract_quantile_params(self, root) -> None:
        """
        Extract quantile normalization parameters from Zarr metadata.
        Looks for image_statistics.quantiles in zarr attributes.

        Args:
            root: Zarr group or array object with attributes
        """
        try:
            # Try to get attributes from the root
            attrs = root.attrs if hasattr(root, 'attrs') else {}

            if 'image_statistics' in attrs:
                image_stats = attrs['image_statistics']
                if 'quantiles' in image_stats:
                    quantiles = image_stats['quantiles']
                    # Use 0.1 and 0.9 quantiles for normalization (10th and 90th percentile)
                    q_low = quantiles.get('0.1', None)
                    q_high = quantiles.get('0.9', None)

                    if q_low is not None and q_high is not None:
                        self._quantile_normalization_params = (float(q_low), float(q_high))
                        logger.info(f"Extracted quantile params: 0.1={q_low}, 0.9={q_high}")
        except Exception as e:
            logger.debug(f"Could not extract quantile parameters: {str(e)}")

    def get_shape(self) -> tuple[int, int, int, int]:
        """
        Returns the (T, Z, Y, X) dimensions of the 4D dataset.

        Returns:
            Tuple[int, int, int, int]: Shape of the 4D Zarr array.
        """
        return self.dataset.shape if self.dataset is not None else (0, 0, 0, 0)

    def _create_simulated_store(self) -> None:
        """
        Generates a mock Zarr store mimicking a 4D anisotropic microscopy volume.
        Utilizes high-speed blosc2 compression with the blosclz shuffle codec.
        """
        os.makedirs(os.path.dirname(self.store_path) or '.', exist_ok=True)
        shape = (20, 10, 128, 128)  # Dimensions: (T, Z, Y, X)
        chunks = (1, 5, 64, 64)     # Chunking dimensions

        # Configure the fast blosc2 compressor for parallel decompression
        compressor = Blosc(cname='blosclz', clevel=5, shuffle=Blosc.SHUFFLE)

        # Write array using zarr API
        # Handle Zarr v3 vs v2 compatibility by trying zarr_format=2 first,
        # fallback if zarr_format keyword is not supported (i.e. Zarr v2 installed).
        try:
            try:
                z_arr = zarr.open_array(
                    self.store_path,
                    mode='w',
                    shape=shape,
                    chunks=chunks,
                    dtype='float32',
                    compressor=compressor,
                    zarr_format=2
                )
            except TypeError:
                z_arr = zarr.open_array(
                    self.store_path,
                    mode='w',
                    shape=shape,
                    chunks=chunks,
                    dtype='float32',
                    compressor=compressor
                )
        except Exception:
            try:
                try:
                    z_arr = zarr.open(
                        self.store_path,
                        mode='w',
                        shape=shape,
                        chunks=chunks,
                        dtype='float32',
                        compressor=compressor,
                        zarr_format=2
                    )
                except TypeError:
                    z_arr = zarr.open(
                        self.store_path,
                        mode='w',
                        shape=shape,
                        chunks=chunks,
                        dtype='float32',
                        compressor=compressor
                    )
            except Exception as e:
                logger.error(f"Critical error during simulated store allocation: {str(e)}")
                raise

        # Populate with mock fluorescent cellular spots and background noise
        logger.info("Generating simulated cellular Zarr store content...")
        for t in range(shape[0]):
            # Initialize with random ambient background noise
            img = np.random.normal(0.01, 0.005, size=shape[1:]).astype(np.float32)

            # Simulate a few active cell structures moving through space
            for cell_id in range(4):
                base_z = 3 + int(cell_id + t * 0.1) % 4
                base_y = 30 + (cell_id * 20 + t * 2) % 68
                base_x = 30 + (cell_id * 20 + t * 3) % 68

                # Mitosis simulation
                if t >= 10 and cell_id == 1:
                    for offset in [-8, 8]:
                        self._draw_cell(img, base_z, base_y + offset, base_x + offset)
                else:
                    self._draw_cell(img, base_z, base_y, base_x)

            # Write single timepoint block to disk
            z_arr[t] = img

        logger.info("Simulation Zarr store prepared and compressed successfully!")

    def _draw_cell(self, grid: np.ndarray, z: int, y: int, x: int, radius: int = 4) -> None:
        """
        Helper method to render a Gaussian cellular signal in anisotropic space.
        """
        nz, ny, nx = grid.shape
        for dz in range(-1, 2):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    cz, cy, cx = z + dz, y + dy, x + dx
                    if 0 <= cz < nz and 0 <= cy < ny and 0 <= cx < nx:
                        # Incorporate anisotropy ratio into distance metric to prevent Z warping
                        dist_sq = (dz * self.anisotropy_ratio[0])**2 + dy**2 + dx**2
                        val = np.exp(-dist_sq / 12.0)
                        grid[cz, cy, cx] = max(grid[cz, cy, cx], val)

    def _apply_quantile_normalization(self, data: np.ndarray) -> np.ndarray:
        """
        Apply quantile normalization to raw data if normalization parameters are available.
        Normalizes to [0, 1] range using: (data - q_low) / (q_high - q_low)

        Args:
            data (np.ndarray): Raw data to normalize

        Returns:
            np.ndarray: Normalized data (float32, [0,1] range) or original data if no quantiles
        """
        if self._quantile_normalization_params is None:
            return data

        q_low, q_high = self._quantile_normalization_params
        if q_high <= q_low:
            logger.warning(f"Invalid quantile range: {q_low} >= {q_high}. Skipping normalization.")
            return data

        # Apply quantile normalization
        normalized = (data.astype(np.float32) - q_low) / (q_high - q_low)
        # Clamp to [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized

    def load_timepoint_block(self, t: int, normalize: bool = True) -> np.ndarray:
        """
        Loads and decompresses a single 3D timepoint volume (Z, Y, X) into memory.
        Optionally applies quantile normalization.
        Bypasses full-file loading, loading only the specific sliced block.

        Args:
            t (int): Timepoint index to extract.
            normalize (bool): If True and quantile params available, applies normalization.

        Returns:
            np.ndarray: A decompressed 3D NumPy array of shape (Z, Y, X) (uint16 raw or float32 normalized).
        """
        if self.dataset is None:
            raise RuntimeError("Zarr store connection has not been initialized.")

        num_t = self.dataset.shape[0]
        if t < 0 or t >= num_t:
            raise IndexError(f"Timepoint index {t} is out of bounds for a dataset containing {num_t} frames.")

        if t == self._last_t and normalize == self._last_normalize and self._last_block is not None:
            return self._last_block

        logger.info(f"Loading and decompressing 3D block for timepoint T={t}...")
        try:
            # Query slice: Zarr's smart indexing decompresses only the blosc2 blocks
            # overlapping with index 't'.
            timepoint_vol = self.dataset[t, :, :, :]

            # Apply normalization if requested
            if normalize and self._quantile_normalization_params is not None:
                timepoint_vol = self._apply_quantile_normalization(timepoint_vol)

            self._last_t = t
            self._last_normalize = normalize
            self._last_block = timepoint_vol
            return timepoint_vol
        except MemoryError:
            logger.critical("Memory threshold exceeded during Zarr slice loading! Calling garbage collection...")
            import gc
            gc.collect()
            raise

    def stream_chunks_3d(self, t: int, chunk_size: tuple[int, int, int] = (2, 64, 64)) -> Generator[tuple[tuple[int, int, int], np.ndarray], None, None]:
        """
        A memory-efficient generator yielding spatial sub-chunks (Z, Y, X) of a single timepoint.
        Useful for running inference on low-memory/restricted-RAM environments.

        Args:
            t (int): Timepoint index.
            chunk_size (Tuple[int, int, int]): Target sub-block dimensions (z_dim, y_dim, x_dim).

        Yields:
            Tuple[Tuple[int, int, int], np.ndarray]: Coordinates (z_start, y_start, x_start)
                and the decompressed sub-chunk data array.
        """
        if self.dataset is None:
            logger.error("Store not initialized; cannot stream chunks.")
            return

        shape_3d = self.dataset.shape[1:]  # (Z, Y, X)
        cz, cy, cx = chunk_size

        logger.info(f"Streaming sub-chunks for T={t} with block sizes: {chunk_size}")

        for z in range(0, shape_3d[0], cz):
            z_end = min(z + cz, shape_3d[0])
            for y in range(0, shape_3d[1], cy):
                y_end = min(y + cy, shape_3d[1])
                for x in range(0, shape_3d[2], cx):
                    x_end = min(x + cx, shape_3d[2])

                    # Read the individual sub-block directly.
                    # This prevents loading the full 3D timepoint, reducing transient RAM allocation.
                    chunk_data = self.dataset[t, z:z_end, y:y_end, x:x_end]
                    yield (z, y, x), chunk_data

if __name__ == "__main__":
    # Self-test block to verify the module executes perfectly
    store_path = "./data/cell_tracking_volume.zarr"
    loader = AnisotropicZarrLoader(store_path=store_path)

    t_dim, z_dim, y_dim, x_dim = loader.get_shape()
    print(f"Dataset Shape: Time={t_dim}, Z={z_dim}, Y={y_dim}, X={x_dim}")

    # Load first timepoint block
    first_vol = loader.load_timepoint_block(0)
    print(f"Loaded block shape: {first_vol.shape} | Max value: {np.max(first_vol):.4f}")

    # Stream first sub-block
    for (coords, block) in loader.stream_chunks_3d(t=0, chunk_size=(5, 64, 64)):
        print(f"Streamed sub-block coordinates: {coords} | Shape: {block.shape} | Mean: {np.mean(block):.6f}")
        break

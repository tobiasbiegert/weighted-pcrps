"""
Script to compute block permutation test p-values for score differences.
"""
from absl import app
from absl import flags
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

# Define command-line flags
SCORE_A_PATH = flags.DEFINE_string(
    'score_a_path',
    None,
    help='Path to first score dataset in Zarr format.',
)
SCORE_B_PATH = flags.DEFINE_string(
    'score_b_path',
    None,
    help='Path to second score dataset in Zarr format.',
)
OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    help='Where to save p-values in Zarr format.',
)
VARIABLES = flags.DEFINE_list(
    'variables',
    None,
    help='Variables to test.',
)
CHUNK_SIZE_LON = flags.DEFINE_integer(
    'chunk_size_lon',
    60,
    help='Chunk size for longitude.',
)
CHUNK_SIZE_LAT = flags.DEFINE_integer(
    'chunk_size_lat',
    11,
    help='Chunk size for latitude.',
)
NUM_PERMUTATION = flags.DEFINE_integer(
    'num_permutation',
    1000,
    help='Number of permutation replicates.',
)
SEED = flags.DEFINE_integer(
    'seed',
    42,
    help='Global RNG seed.',
)

def block_permutation_test(
    score_a: np.ndarray,
    score_b: np.ndarray,
    seed_offset: int,
    block_length: int,
    *,
    b: int = 1000,
    root_seed: int = 42,
) -> np.float32:  
    """
    Compute a one-sided block permutation test for two score series.

    Parameters
    ----------
    score_a : numpy.ndarray
        Score values from model A over initialization times.
    score_b : numpy.ndarray
        Score values from model B over initialization times.
    seed_offset : int
        Unique offset for the random seed at a given grid point and lead time.
    block_length : int
        Number of consecutive initialization times per block.
    b : int, default 1000
        Number of permutation replicates.
    root_seed : int, default 42
        Global random seed.

    Returns
    -------
    numpy.float32
        Empirical one-sided p-value.
    """
    if score_a.shape != score_b.shape or score_a.ndim != 1:
        raise ValueError('Inputs must be 1-D arrays of identical length.')

    d = score_a - score_b
    n = d.size
    d_mean = d.mean()

    # Compute number of blocks accounting for possible offset up to block_length-1
    n_blocks = int(np.ceil((n + block_length - 1) / block_length))

    # Draw a b×n_blocks matrix of +/- signs with a deterministic seed
    rng = np.random.default_rng(root_seed + int(seed_offset))
    block_signs = 1 - 2 * rng.integers(0, 2, size=(b, n_blocks), dtype=np.int8)

    # Offsets to slide block boundaries across replicates
    offsets = np.arange(b, dtype=int) % block_length
    
    # Determine block index for each observation in each replicate
    block_indices = (np.arange(n)[None, :] + offsets[:, None]) // block_length
    
    # Expand block signs to full-series signs per replicate
    signs = block_signs[np.arange(b)[:, None], block_indices]

    # Compute replicate means
    m = np.einsum('ij,j->i', signs, d) / n

    # one‑sided p‑value
    n_lt = np.count_nonzero(m < d_mean)
    n_gt = np.count_nonzero(m > d_mean)
    n_eq = b - n_lt - n_gt
    p = (n_lt + 0.5 * n_eq + 0.5) / (b + 1)

    return np.float32(p)

def block_permutation_test_dataset(
    ds_a: xr.Dataset,
    ds_b: xr.Dataset,
    *,
    b: int = 1000,
    seed: int = 42,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """
    Apply the block permutation test to two score datasets.

    The test is applied separately at each longitude, latitude, and lead time. 
    Any additional non-time dimensions, such as ``threshold`` or ``quantile``, are vectorized over automatically.

    Parameters
    ----------
    ds_a : xarray.Dataset
        Score dataset from model A.
    ds_b : xarray.Dataset
        Score dataset from model B.
    b : int, default 1000
        Number of permutation replicates.
    seed : int, default 42
        Global random seed.
    variables : list of str, optional
        Variables to test. If omitted, all data variables in ``ds_a`` are used.

    Returns
    -------
    xarray.Dataset
        Dataset of p-values for each longitude, latitude, and lead time, together with 
        any additional non-time dimensions present in the score variables.
    """
    if variables is None:
        variables = list(ds_a.data_vars)

    # Align both datasets on all coords
    ds_a, ds_b = xr.align(ds_a[variables], ds_b[variables], join='exact')

    # Build a unique seed_offset per grid‐point & lead time
    seed_da = xr.DataArray(
        np.arange(
            ds_a.sizes['longitude']
            * ds_a.sizes['latitude']
            * ds_a.sizes['prediction_timedelta'],
            dtype='uint32',
        ).reshape(
            ds_a.sizes['longitude'],
            ds_a.sizes['latitude'],
            ds_a.sizes['prediction_timedelta'],
        ),
        dims=('longitude', 'latitude', 'prediction_timedelta'),
        coords={
            'longitude': ds_a['longitude'],
            'latitude': ds_a['latitude'],
            'prediction_timedelta': ds_a['prediction_timedelta'],
        },
    )

    # Convert prediction_timedelta into block_length in steps (here 12h init_time steps)
    lead_days = (ds_a['prediction_timedelta'] / np.timedelta64(1, 'D')).astype(int)
    block_length = (lead_days * 2).rename('block_length')
    block_length_da = block_length.broadcast_like(seed_da)

    # Vectorise the block_permutation_test along 'time'
    p_da = xr.apply_ufunc(
        block_permutation_test,
        ds_a,
        ds_b,
        seed_da,
        block_length_da,
        kwargs=dict(b=b, root_seed=seed),
        input_core_dims=[['time'], ['time'], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32],
    )

    return p_da

def is_lower_tail_threshold(label: str) -> bool:
    """
    Return True if the threshold label corresponds to a lower-tail threshold.
    """
    if label == 'monthly_min':
        return True
    if label == 'monthly_max':
        return False
    if isinstance(label, str) and label.startswith('q'):
        return int(label[1:]) < 50
    return False

def prepare_score_dataset(ds: xr.Dataset, variables: list[str]) -> xr.Dataset:
    """
    Prepare score variables for permutation testing.

    Monthly record scores are stored directly for their relevant tails.
    Lower-tail fixed-threshold scores are constructed as PCRPS minus the upper twPCRPS.
    For variables with a ``quantile`` dimension, lower-tail quantiles are constructed analogously.
    All other variables are passed through unchanged.

    Parameters
    ----------
    ds : xarray.Dataset
        Input score dataset.
    variables : list of str
        Variables to prepare.

    Returns
    -------
    xarray.Dataset
        Dataset containing the prepared score variables.
    """
    prepared = {}

    for var in variables:
        da = ds[var]

        if 'threshold' in da.dims:
            # Monthly minima already contain lower-tail scores.
            lower_mask = xr.DataArray(
                np.array([
                    str(t) != 'monthly_min' and is_lower_tail_threshold(str(t))
                    for t in da.threshold.values
                ]),
                dims=('threshold',),
                coords={'threshold': da.threshold},
            )

            if bool(lower_mask.any()):
                pcrps_var = var.replace('_tw_pcrps', '_pcrps')
                if pcrps_var not in ds:
                    raise ValueError(
                        f'Variable {var} requires {pcrps_var} in the same dataset.'
                    )

                pcrps_da = ds[pcrps_var]
                prepared[var] = xr.where(lower_mask, pcrps_da - da, da)
            else:
                prepared[var] = da

        elif 'quantile' in da.dims:
            pcrps_var = var.replace('_qw_pcrps', '_pcrps')
            if pcrps_var not in ds:
                raise ValueError(
                    f'Variable {var} requires {pcrps_var} in the same dataset.'
                )

            quantile_coord = da['quantile']

            lower_mask = xr.DataArray(
                quantile_coord.values < 0.5,
                dims=('quantile',),
                coords={'quantile': quantile_coord},
            )

            pcrps_da = ds[pcrps_var]
            prepared[var] = xr.where(lower_mask, pcrps_da - da, da)

        else:
            prepared[var] = da

    return xr.Dataset(prepared)

def main(argv):

    if not VARIABLES.value:
        raise ValueError('Provide at least one variable via --variables.')

    chunking_dict = {
        'longitude': CHUNK_SIZE_LON.value,
        'latitude': CHUNK_SIZE_LAT.value,
        'prediction_timedelta': 1,
        'time': -1,
        'threshold': -1,
        'quantile': -1        
    }

    score_a_raw = xr.open_zarr(SCORE_A_PATH.value, decode_timedelta=True)
    score_b_raw = xr.open_zarr(SCORE_B_PATH.value, decode_timedelta=True)

    score_a = prepare_score_dataset(score_a_raw, VARIABLES.value)
    score_b = prepare_score_dataset(score_b_raw, VARIABLES.value)

    chunking_a = {k: v for k, v in chunking_dict.items() if k in score_a.dims}
    chunking_b = {k: v for k, v in chunking_dict.items() if k in score_b.dims}

    score_a = score_a.chunk(chunking_a)
    score_b = score_b.chunk(chunking_b)

    p_values = block_permutation_test_dataset(
        score_a,
        score_b,
        variables=VARIABLES.value,
        b=NUM_PERMUTATION.value,
        seed=SEED.value,
    )

    if 'threshold' in p_values.coords:
        p_values = p_values.assign_coords(
            threshold=('threshold', p_values.threshold.values.astype('U'))
        )
    
    with ProgressBar():
        p_values.to_zarr(
            OUTPUT_PATH.value,
            mode='w',
            consolidated=True,
        )

if __name__ == '__main__':
    flags.mark_flags_as_required(['score_a_path', 'score_b_path', 'output_path', 'variables'])
    app.run(main)
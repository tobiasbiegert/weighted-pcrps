"""
Script to compute PCRPS^(0) results and optionally qwPCRPS^(0) and twPCRPS^(0).
"""
from absl import app
from absl import flags
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

# Define command-line flags
TARGET_PATH = flags.DEFINE_string(
    'target_path',
    None,
    help='Path to target dataset in Zarr format.',
)
TIME_START = flags.DEFINE_string(
    'time_start',
    None,
    help='ISO 8601 initialization timestamp (inclusive) at which to start.',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    None,
    help='ISO 8601 initialization timestamp (inclusive) at which to stop.',
)
VARIABLES = flags.DEFINE_list(
    'variables',
    ['2m_temperature'],
    help='Variables to compute PCRPS^(0) for.',
)
OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    help='Where to save results.',
)
CHUNK_SIZE_LAT = flags.DEFINE_integer(
    'chunk_size_lat',
    11,
    help='Chunk size for latitude dimension.',
)
CHUNK_SIZE_LON = flags.DEFINE_integer(
    'chunk_size_lon',
    16,
    help='Chunk size for longitude dimension.',
)
QW_PCRPS = flags.DEFINE_bool(
    'qw_pcrps',
    False,
    help='If True, also compute quantile-weighted PCRPS^(0).',
)
QUANTILES = flags.DEFINE_list(
    'quantiles',
    [],
    help='Optional list of quantiles in (0,1) for qwPCRPS^(0). If empty and --qw_pcrps=True, use 0.01,...,0.99.',
)
TW_PCRPS = flags.DEFINE_bool(
    'tw_pcrps',
    False,
    help='If True, also compute threshold-weighted PCRPS^(0).',
)
QUANTILE_THRESHOLDS_PATH = flags.DEFINE_string(
    'quantile_thresholds_path',
    None,
    help='Path to thresholds based on historical quantiles for twPCRPS^(0).',
)
MONTHLY_MAX_RECORDS_PATH = flags.DEFINE_string(
    'monthly_max_records_path',
    None,
    help='Path to thresholds based on historical monthly maximum records for twPCRPS^(0).',
)
MONTHLY_MIN_RECORDS_PATH = flags.DEFINE_string(
    'monthly_min_records_path',
    None,
    help='Path to thresholds based on historical monthly minimum records for twPCRPS^(0).',
)
LEVELS = flags.DEFINE_list(
    'levels',
    [500, 700, 850],
    help='Comma delimited list of pressure levels to select.'
)
REFERENCE_FORECAST_PATH = flags.DEFINE_string(
    'reference_forecast_path',
    None,
    help='Path to forecast dataset used to define initialization times and lead times.',
)
LEAD_TIMES_HOURS = flags.DEFINE_list(
    'lead_times_hours',
    None,
    help='List of lead times in hours.',
)


def open_zarr_auto(path, **kwargs):
    """
    Open a Zarr store from either a GCS or local path.

    Parameters
    ----------
    path : str
        Path to the Zarr store.
    **kwargs
        Additional keyword arguments passed to ``xarray.open_zarr``.

    Returns
    -------
    xarray.Dataset
        Opened Zarr dataset.
    """
    if str(path).startswith('gs://'):
        return xr.open_zarr(store=path, storage_options={'token': 'anon'}, **kwargs)
    return xr.open_zarr(store=path, **kwargs)


def pcrps0(y):
    """
    Compute PCRPS^(0) using a sort-based O(n log n) approach.

    PCRPS^(0) is defined as

    .. math::

        \\text{PCRPS}^{(0)}(y) = \\frac{1}{2 n^2} \\sum_{i,j} |y_i - y_j|,

    where :math:`n` is the number of observations.

    Parameters
    ----------
    y : array_like
        Observations.

    Returns
    -------
    numpy.ndarray
        PCRPS^(0) value(s) with the same leading shape as ``y``.
    """
    y_sorted = np.sort(y, axis=-1)
    n = y_sorted.shape[-1]
    ranks = np.arange(1, n + 1, dtype=np.float32)
    weights = 2 * ranks - n - 1
    return (np.sum(y_sorted * weights, axis=-1) / (n**2)).astype(np.float32)


def qw_pcrps0_upper(y, q):
    """
    Compute upper-tail quantile-weighted PCRPS^(0).

    Parameters
    ----------
    y : array_like
        Observations.
    q : array_like
        Quantile levels in (0, 1).

    Returns
    -------
    numpy.ndarray
        Upper-tail quantile-weighted PCRPS^(0) for each quantile level.
    """
    y = np.asarray(y)
    q = np.asarray(q)

    # sort along last axis
    x = np.sort(y, axis=-1)
    n = x.shape[-1]
    m = q.shape[-1]

    # Probabilities and weights
    p = (np.arange(1, n + 1, dtype=np.float32)) / n
    p_star = np.maximum(p[None, :], q[:, None])
    p_prev = np.empty_like(p)
    p_prev[0] = 0.0
    p_prev[1:] = p[:-1]
    p_prev_star = np.maximum(p_prev[None, :], q[:, None])

    w = p_star - p_prev_star
    v = p_star**2 - p_prev_star**2

    # Suffix sums for w and w * x
    W_suffix = np.cumsum(w[:, ::-1], axis=-1)[:, ::-1]

    leading_shape = x.shape[:-1]
    prefix = (1,) * len(leading_shape)

    # x with quantile axis
    x_q = x[..., None, :]

    # broadcast w, v to leading dims
    w_b = w.reshape(prefix + w.shape)
    v_b = v.reshape(prefix + v.shape)

    WX_suffix = np.cumsum((w_b * x_q)[..., ::-1], axis=-1)[..., ::-1]

    # V and Vx
    V  = np.sum(v, axis=-1)
    Vx = np.sum(v_b * x_q, axis=-1)

    # Indices k(i) for runs of equal x along last axis
    first = np.ones_like(x[..., :1], dtype=bool)
    rest = x[..., 1:] != x[..., :-1]
    change = np.concatenate([first, rest], axis=-1)

    base_idx = np.arange(n, dtype=int)
    start_positions = np.where(change, base_idx, 0)
    k = np.maximum.accumulate(start_positions, axis=-1)

    # Gather W_suffix and WX_suffix at k
    W_suffix_full = np.broadcast_to(
        W_suffix.reshape((1,) * len(leading_shape) + W_suffix.shape),
        leading_shape + W_suffix.shape,
    )

    k_exp = np.expand_dims(k, axis=-2)
    Wk = np.take_along_axis(W_suffix_full, k_exp, axis=-1)
    WXk = np.take_along_axis(WX_suffix,       k_exp, axis=-1)

    # Per-observation qwCRPS contributions, then mean over time
    Vx_exp = Vx[..., :, None]
    V_exp = V.reshape(prefix + (m, 1))
    x_qb = np.broadcast_to(x_q, leading_shape + (m, n))

    crps_i = 2.0 * WXk - 2.0 * x_qb * Wk - Vx_exp + x_qb * V_exp

    return np.mean(crps_i, axis=-1).astype(np.float32)

def pcrps0_along_time(da):
    """
    Apply ``pcrps0`` along the ``time`` dimension.

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        Input data with a ``time`` dimension.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        PCRPS^(0) values with the ``time`` dimension removed.
    """
    return xr.apply_ufunc(
        pcrps0,
        da,
        input_core_dims=[['time']],
        output_core_dims=[[]],
        dask='parallelized',
        output_dtypes=[np.float32]
    )

def qw_pcrps0_along_time(da, q):
    """
    Apply ``qw_pcrps0_upper`` along the ``time`` dimension.

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        Input data with a ``time`` dimension.
    q : xarray.DataArray
        Quantile levels with a ``quantile`` dimension.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Quantile-weighted PCRPS^(0) with the ``time`` dimension removed and an additional ``quantile`` dimension.
    """
    return xr.apply_ufunc(
        qw_pcrps0_upper,
        da,
        q,
        input_core_dims=[['time'], ['quantile']],
        output_core_dims=[['quantile']],
        dask='parallelized',
        output_dtypes=[np.float32],
    )

def main(argv):
    del argv

    lead_times = [np.timedelta64(int(h), 'h') for h in LEAD_TIMES_HOURS.value]

    forecast_ref = open_zarr_auto(
        REFERENCE_FORECAST_PATH.value,
        decode_timedelta=True,
    ).sel(time=slice(TIME_START.value, TIME_STOP.value))

    forecast_ref = forecast_ref.sel(prediction_timedelta=lead_times)

    target_start = forecast_ref.time[0] + forecast_ref.prediction_timedelta.min()
    target_stop = forecast_ref.time[-1] + forecast_ref.prediction_timedelta.max()

    targets = open_zarr_auto(TARGET_PATH.value)[VARIABLES.value].sel(
        time=slice(target_start, target_stop)
    )
    if 'level' in targets.dims:
        targets = targets.sel(level=[int(level) for level in LEVELS.value])

    base_chunks = {
        'time': -1,
        'latitude': CHUNK_SIZE_LAT.value,
        'longitude': CHUNK_SIZE_LON.value,
    }
    if 'level' in targets.dims:
        base_chunks['level'] = 1

    threshold_chunks = {
        **{k: v for k, v in base_chunks.items() if k != 'time'},
        'threshold': -1,
    }

    time_threshold_chunks = {
        **base_chunks,
        'threshold': -1,
    }

    targets = targets.chunk(base_chunks)

    lead_coord = xr.DataArray(
        lead_times,
        dims=('prediction_timedelta',),
        coords={'prediction_timedelta': lead_times},
        name='prediction_timedelta',
    )

    if QW_PCRPS.value:
        if QUANTILES.value:
            q_values = np.array([float(q) for q in QUANTILES.value], dtype=np.float32)
        else:
            q_values = np.arange(1, 100, dtype=np.float32) / 100.0
        qs = xr.DataArray(
            q_values,
            dims=('quantile',),
            coords={'quantile': q_values},
            name='quantile',
        )
    else:
        qs = None

    quantile_thresholds = None
    record_min_thresholds = None
    record_max_thresholds = None

    if TW_PCRPS.value:
        if (
            QUANTILE_THRESHOLDS_PATH.value is None
            and MONTHLY_MIN_RECORDS_PATH.value is None
            and MONTHLY_MAX_RECORDS_PATH.value is None
        ):
            raise ValueError(
                'TW_PCRPS=True but none of --quantile_thresholds_path, --monthly_min_records_path, or --monthly_max_records_path are set.'
            )

        if QUANTILE_THRESHOLDS_PATH.value is not None:
            quantile_thresholds = open_zarr_auto(QUANTILE_THRESHOLDS_PATH.value)[VARIABLES.value]

            if 'level' in quantile_thresholds.dims:
                quantile_thresholds = quantile_thresholds.sel(
                    level=[int(level) for level in LEVELS.value]
                )

            quantile_thresholds = quantile_thresholds.chunk(threshold_chunks)

        months = targets.time.dt.month

        if MONTHLY_MIN_RECORDS_PATH.value is not None:
            rec_min = open_zarr_auto(MONTHLY_MIN_RECORDS_PATH.value)[VARIABLES.value]

            if 'level' in rec_min.dims:
                rec_min = rec_min.sel(level=[int(level) for level in LEVELS.value])

            record_min_thresholds = (
                rec_min.sel(month=months)
                .expand_dims(threshold=['monthly_min'], axis=-1)
                .reset_coords('month', drop=True)
                .chunk(time_threshold_chunks)
            )

        if MONTHLY_MAX_RECORDS_PATH.value is not None:
            rec_max = open_zarr_auto(MONTHLY_MAX_RECORDS_PATH.value)[VARIABLES.value]

            if 'level' in rec_max.dims:
                rec_max = rec_max.sel(level=[int(level) for level in LEVELS.value])

            record_max_thresholds = (
                rec_max.sel(month=months)
                .expand_dims(threshold=['monthly_max'], axis=-1)
                .reset_coords('month', drop=True)
                .chunk(time_threshold_chunks)
            )

    pcrps0_list = []
    qw_pcrps0_list = []
    tw_pcrps0_list = []

    for lt in lead_times:
        valid_time = forecast_ref.time + lt
        target_lt = targets.sel(time=valid_time)

        pcrps0_list.append(pcrps0_along_time(target_lt))

        if QW_PCRPS.value:
            qw_pcrps0_list.append(qw_pcrps0_along_time(target_lt, qs))

        if TW_PCRPS.value:
            tw_parts = []

            if record_min_thresholds is not None:
                record_min_lt = record_min_thresholds.sel(time=valid_time)
                target_lt_tw = (
                    record_min_lt.astype(np.float64) - target_lt.astype(np.float64)
                ).clip(min=0).chunk(time_threshold_chunks)
                tw_parts.append(pcrps0_along_time(target_lt_tw))

            if quantile_thresholds is not None:
                target_lt_tw = target_lt.clip(min=quantile_thresholds).chunk(time_threshold_chunks)
                tw_parts.append(pcrps0_along_time(target_lt_tw))

            if record_max_thresholds is not None:
                record_max_lt = record_max_thresholds.sel(time=valid_time)
                target_lt_tw = (
                    target_lt.astype(np.float64) - record_max_lt.astype(np.float64)
                ).clip(min=0).chunk(time_threshold_chunks)
                tw_parts.append(pcrps0_along_time(target_lt_tw))

            if len(tw_parts) == 1:
                tw_pcrps0_list.append(tw_parts[0])
            else:
                tw_pcrps0_list.append(xr.concat(tw_parts, dim='threshold'))

    result_parts = []

    pcrps0_result = xr.concat(pcrps0_list, dim=lead_coord).rename({var: f'{var}_pcrps' for var in VARIABLES.value})
    result_parts.append(pcrps0_result)

    if QW_PCRPS.value:
        qw_pcrps0_result = xr.concat(qw_pcrps0_list, dim=lead_coord).rename({var: f'{var}_qw_pcrps' for var in VARIABLES.value})
        result_parts.append(qw_pcrps0_result)

    if TW_PCRPS.value:
        tw_pcrps0_result = xr.concat(tw_pcrps0_list, dim=lead_coord).rename({var: f'{var}_tw_pcrps' for var in VARIABLES.value})
        tw_chunks = {
            'prediction_timedelta': 1,
            'latitude': CHUNK_SIZE_LAT.value,
            'longitude': CHUNK_SIZE_LON.value,
            'threshold': -1,
        }
        if 'level' in tw_pcrps0_result.dims:
            tw_chunks['level'] = 1

        tw_pcrps0_result = tw_pcrps0_result.chunk(tw_chunks)
        for var in VARIABLES.value:
            tw_pcrps0_result[f'{var}_tw_pcrps'].attrs = {
                'monthly_threshold_method': 'threshold_relative_empirical',
                'stored_tail': 'upper for fixed thresholds and monthly_max; lower for monthly_min',
            }
        result_parts.append(tw_pcrps0_result)

    result = xr.merge(result_parts)

    if 'threshold' in result.coords:
        result = result.assign_coords(threshold=('threshold', result.threshold.values.astype('U')))

    with ProgressBar():
        result.to_zarr(
            OUTPUT_PATH.value,
            mode='w',
            consolidated=True,
        )

if __name__ == '__main__':
    flags.mark_flags_as_required([
        'target_path',
        'reference_forecast_path',
        'time_start',
        'time_stop',
        'output_path',
        'lead_times_hours',
    ])
    app.run(main)
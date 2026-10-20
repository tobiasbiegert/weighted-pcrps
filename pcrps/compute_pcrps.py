"""
Script to compute PCRPS results and optionally qwPCRPS and twPCRPS, together with timing measurements.
"""
import logging
from absl import app
from absl import flags
import apache_beam as beam
import xarray as xr
import xarray_beam as xbeam
import numpy as np
import dask.array as da
from pcrps.pcrps_helper import get_easyuq_tasks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define command-line flags
PREDICTION_PATH = flags.DEFINE_string(
    'prediction_path',
    None,
    help='Path to forecasts to evaluate in Zarr format',
)
TARGET_PATH = flags.DEFINE_string(
    'target_path',
    None,
    help='Path to ground-truth to evaluate in Zarr format',
)
TIME_START = flags.DEFINE_string(
    'time_start',
    None,
    help='ISO 8601 timestamp (inclusive) at which to start',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    None,
    help='ISO 8601 timestamp (inclusive) at which to stop',
)
VARIABLES = flags.DEFINE_list(
    'variables',
    ['2m_temperature'],
    help='Variables to compute PCRPS for'
)
LEVELS = flags.DEFINE_list(
    'levels',
    [500, 700, 850],
    help='Comma delimited list of pressure levels to select.'
)
OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    help='Where to save results.'
)
TEMPORAL_OUTPUT_PATH = flags.DEFINE_string(
    'temporal_output_path',
    None,
    help='Optional second output containing individual scores. Requires --temporal_mean=True.'
)
TEMPORAL_THRESHOLDS = flags.DEFINE_list(
    'temporal_thresholds',
    [],
    help='Threshold labels to retain in the second output. If empty, retain all computed thresholds.'
)
TEMPORAL_QUANTILES = flags.DEFINE_list(
    'temporal_quantiles',
    [],
    help='qwPCRPS quantiles to retain in the second output. If empty, retain all computed quantiles.'
)
CHUNK_SIZE_LAT = flags.DEFINE_integer(
    'chunk_size_lat',
    -1,
    help='Chunk size for latitude dimension'
)
CHUNK_SIZE_LON = flags.DEFINE_integer(
    'chunk_size_lon',
    -1,
    help='Chunk size for longitude dimension'
)
RUNNER = flags.DEFINE_string(
    'runner',
    None,
    help='beam.runners.Runner'
)
SKIP_NON_HEADLINE = flags.DEFINE_bool(
    'skip_non_headline',
    False,
    help='If True, skip variable-level combinations not defined in VARIABLE_LEVEL_MAPPING (WB2 headline scores).'
)
RECHUNK = flags.DEFINE_bool(
    'rechunk',
    False,
    help='Whether to apply rechunk transform to the dataset before processing.'
)
TEMPORAL_MEAN = flags.DEFINE_bool(
    'temporal_mean',
    False,
    help='Whether to directly compute PCRPS by averaging over time.'
)
QW_PCRPS = flags.DEFINE_bool(
    'qw_pcrps',
    False,
    help='If True, compute quantile-weighted PCRPS in addition to standard PCRPS.'
)
QUANTILES = flags.DEFINE_list(
    'quantiles',
    [],
    help='Optional list of quantiles in (0,1) for qwPCRPS. If empty and --qw_pcrps=True, use default 0.01,0.02,...,0.99.'
)
TW_PCRPS = flags.DEFINE_bool(
    'tw_pcrps',
    False,
    help='If True, compute threshold-weighted PCRPS in addition to standard PCRPS.'
)
QUANTILE_THRESHOLDS_PATH = flags.DEFINE_string(
    'quantile_thresholds_path',
    None,
    help='Path to thresholds based on historical quantiles for twPCRPS.'
)
MONTHLY_MAX_RECORDS_PATH = flags.DEFINE_string(
    'monthly_max_records_path',
    None,
    help='Path to thresholds based on historical monthly max records for twPCRPS.'
)
MONTHLY_MIN_RECORDS_PATH = flags.DEFINE_string(
    'monthly_min_records_path',
    None,
    help='Path to thresholds based on historical monthly min records for twPCRPS.'
)
LEAD_TIMES_HOURS = flags.DEFINE_list(
    'lead_times_hours',
    [],
    help='Optional list of lead times in hours. If empty, use all available lead times.'
)
VARIABLE_LEVEL_MAPPING = {
    '2m_temperature': None,
    'mean_sea_level_pressure': None,
    '10m_wind_speed': None,
    'temperature': 850,
    'specific_humidity': 700,
    'geopotential': 500,
    'wind_speed': 850
}

def compute_pcrps(key, prediction_chunk, targets=None, variable_level_mapping=None, skip_non_headline=True, compute_qw=False, quantiles=None, compute_tw=False, thresholds=None):
    """
    Compute PCRPS and optionally qwPCRPS and/or twPCRPS for a single chunk.

    Parameters
    ----------
    key : xarray_beam.ChunkKey
        Metadata key identifying this chunk.
    prediction_chunk : xarray.Dataset
        Forecast data for a single chunk.
    targets : xarray.Dataset, optional
        Target data.
    variable_level_mapping : dict, optional
        Mapping from variable name to WB2 headline level. If provided and ``skip_non_headline`` is True,
        upper-air variables at levels different from the mapped headline level are skipped and filled with NaNs.
    skip_non_headline : bool, default True
        If True, skip upper-level chunks whose level does not match the specified headline level in ``variable_level_mapping``.
        Skipped chunks return NaN-filled arrays with the appropriate shape.
    compute_qw : bool, default False
        If True, also compute quantile-weighted PCRPS at the specified quantile levels.
    quantiles : array-like, optional
        Sequence of quantile levels in (0, 1) at which qwPCRPS is evaluated. Required if ``compute_qw`` is True.
    compute_tw : bool, default False
        If True, also compute threshold-weighted PCRPS at the specified threshold levels.
    thresholds : xarray.Dataset or xarray.DataArray, optional
        Threshold data used for twPCRPS. Required if ``compute_tw`` is True.

    Returns
    -------
    tuple
        The input key together with a dataset containing PCRPS values and timing measurements.
    """
    # Import necessary modules
    import numpy as np
    import xarray as xr
    from pcrps.pcrps_helper import compute_easyuq, get_easyuq_tasks

    # Restrict forecast times so that forecast_time + lead_time lies within the target window (not necessary for 2020 data)
    start_time_bound = (targets.time[0] - prediction_chunk.prediction_timedelta).values[0]
    end_time_bound = (targets.time[-1] - prediction_chunk.prediction_timedelta).values[-1]
    prediction_chunk = prediction_chunk.sel(time=slice(start_time_bound, end_time_bound))

    # Extract the variable name from the key
    var, = key.vars

    # Get the lead time of the chunk. This assumes that 'prediction_timedelta' contains only a single lead time value. Therefore the chunk size along this dimension must be 1.
    lead_time = prediction_chunk.prediction_timedelta.values[0]

    # Use the same timing-task labels as compute_easyuq
    tasks = get_easyuq_tasks(
        compute_qw=compute_qw,
        compute_tw=compute_tw,
        thresholds=thresholds,
    )
    n_tasks = len(tasks)

    # Skip TP24hr for lead times of less than 24 hours
    if (var == 'total_precipitation_24hr') & (lead_time < 86400000000000):
        logging.info(f'Skipping {key}')
        # PCRPS placeholder
        pcrps = np.full((
            len(prediction_chunk.longitude),
            len(prediction_chunk.latitude),
            len(prediction_chunk.prediction_timedelta),
            len(prediction_chunk.time)
        ), np.nan, dtype=np.float32)
        pcrps_coords = {
            'longitude': prediction_chunk.longitude,
            'latitude': prediction_chunk.latitude,
            'prediction_timedelta': prediction_chunk.prediction_timedelta,
            'time': prediction_chunk.time
        }

        # Dummy timing placeholder
        time_dummy = np.full((
            len(prediction_chunk.longitude),
            len(prediction_chunk.latitude),
            len(prediction_chunk.prediction_timedelta),
            n_tasks
        ), np.nan, dtype=np.float32)
        time_coords = {
            'longitude': prediction_chunk.longitude,
            'latitude': prediction_chunk.latitude,
            'prediction_timedelta': prediction_chunk.prediction_timedelta,
            'task': tasks
        }

        vars_dict = {
            f'{var}_pcrps': (('longitude','latitude','prediction_timedelta','time'), pcrps),
            f'{var}_time': (('longitude','latitude','prediction_timedelta','task'), time_dummy),
        }

        if compute_qw:
            qw_pcrps = np.full((
                len(prediction_chunk.longitude),
                len(prediction_chunk.latitude),
                len(prediction_chunk.prediction_timedelta),
                len(prediction_chunk.time),
                len(quantiles),
            ), np.nan, dtype=np.float32)
            vars_dict[f'{var}_qw_pcrps'] = (
                ('longitude', 'latitude', 'prediction_timedelta', 'time', 'quantile'),
                qw_pcrps,
            )
            pcrps_coords['quantile'] = quantiles

        if compute_tw:
            tw_pcrps = np.full((
                    len(prediction_chunk.longitude),
                    len(prediction_chunk.latitude),
                    len(prediction_chunk.prediction_timedelta),
                    len(prediction_chunk.time),
                    len(thresholds.threshold),
                ), np.nan, dtype=np.float32)
            vars_dict[f'{var}_tw_pcrps'] = (
                ('longitude', 'latitude', 'prediction_timedelta', 'time', 'threshold'),
                tw_pcrps,
            )
            pcrps_coords['threshold'] = thresholds.threshold

        result_dataset = xr.Dataset(vars_dict, coords={**pcrps_coords, **time_coords})

        return key, result_dataset

    # Align target chunk with the valid prediction times
    valid_time = prediction_chunk.time + lead_time
    target_chunk = targets.sel(time=valid_time, latitude=prediction_chunk.latitude, longitude=prediction_chunk.longitude)
    if compute_tw:
        threshold_chunk = thresholds.sel(time=valid_time, latitude=prediction_chunk.latitude, longitude=prediction_chunk.longitude)

    if 'level' in prediction_chunk.dims:
        # Upper-level variable
        level = prediction_chunk.level.values[0]
        headline_level = variable_level_mapping.get(var)
        # Skip if we only want the WB2 headline score levels
        if (level != headline_level) & skip_non_headline:
            logging.info(f'Skipping {key}')
            pcrps = np.full((
                len(prediction_chunk.longitude),
                len(prediction_chunk.latitude),
                len(prediction_chunk.prediction_timedelta),
                len(prediction_chunk.level),
                len(prediction_chunk.time)
            ), np.nan, dtype=np.float32)
            pcrps_coords = {
                'longitude': prediction_chunk.longitude,
                'latitude': prediction_chunk.latitude,
                'prediction_timedelta': prediction_chunk.prediction_timedelta,
                'level': prediction_chunk.level,
                'time': prediction_chunk.time
            }

            # Dummy timing placeholder
            time_dummy = np.full((
                len(prediction_chunk.longitude),
                len(prediction_chunk.latitude),
                len(prediction_chunk.prediction_timedelta),
                len(prediction_chunk.level),
                n_tasks
            ), np.nan, dtype=np.float32)
            time_coords = {
                'longitude': prediction_chunk.longitude,
                'latitude': prediction_chunk.latitude,
                'prediction_timedelta': prediction_chunk.prediction_timedelta,
                'level': prediction_chunk.level,
                'task': tasks
            }

            vars_dict = {
                f'{var}_pcrps': (('longitude','latitude','prediction_timedelta','level','time'), pcrps),
                f'{var}_time': (('longitude','latitude','prediction_timedelta','level','task'), time_dummy),
            }

            if compute_qw and quantiles is not None:
                qw_pcrps = np.full((
                    len(prediction_chunk.longitude),
                    len(prediction_chunk.latitude),
                    len(prediction_chunk.prediction_timedelta),
                    len(prediction_chunk.level),
                    len(prediction_chunk.time),
                    len(quantiles),
                ), np.nan, dtype=np.float32)
                vars_dict[f'{var}_qw_pcrps'] = (
                    ('longitude', 'latitude', 'prediction_timedelta', 'level', 'time', 'quantile'),
                    qw_pcrps,
                )
                pcrps_coords['quantile'] = quantiles

            if compute_tw:
                tw_pcrps = np.full(
                    (
                        len(prediction_chunk.longitude),
                        len(prediction_chunk.latitude),
                        len(prediction_chunk.prediction_timedelta),
                        len(prediction_chunk.level),
                        len(prediction_chunk.time),
                        len(thresholds.threshold),
                    ), np.nan, dtype=np.float32)
                vars_dict[f'{var}_tw_pcrps'] = (
                    ('longitude', 'latitude', 'prediction_timedelta', 'level', 'time', 'threshold'),
                    tw_pcrps,
                )
                pcrps_coords['threshold'] = thresholds.threshold

            result_dataset = xr.Dataset(vars_dict, coords={**pcrps_coords, **time_coords})

        else:
            logging.info(f'Processing {key}')
            preds = prediction_chunk.sel(level=level)[var].compute()
            obs = target_chunk.sel(level=level)[var].compute()
            if compute_tw:
                ts = threshold_chunk.sel(level=level)[var].compute()
            else:
                ts = None
            logging.info(f'Loading complete for {key}')
            result_dataset = compute_easyuq(preds, obs, var, lead_time, level=level, compute_qw=compute_qw, quantiles=quantiles, compute_tw=compute_tw, thresholds=ts)
            logging.info(f'PCRPS results for {key}')
    else:
        # Surface variable
        logging.info(f'Processing {key}')
        preds = prediction_chunk[var].compute()
        obs = target_chunk[var].compute()
        if compute_tw:
            ts = threshold_chunk[var].compute()
        else:
            ts = None
        logging.info(f'Loading complete for {key}')
        result_dataset = compute_easyuq(preds, obs, var, lead_time, compute_qw=compute_qw, quantiles=quantiles, compute_tw=compute_tw, thresholds=ts)
        logging.info(f'PCRPS results for {key}')

    return key, result_dataset

def select_temporal_scores(key, scores, thresholds=None, quantiles=None):
    """
    Select thresholds and quantiles for the additional temporal output.

    Parameters
    ----------
    key : xarray_beam.ChunkKey or None
        Metadata key identifying the chunk. Use None when selecting a template.
    scores : xarray.Dataset
        Individual scores or their output template.
    thresholds : list of str, optional
        Threshold labels to retain. If None, retain all thresholds.
    quantiles : array-like, optional
        qwPCRPS quantiles to retain. If None, retain all quantiles.

    Returns
    -------
    tuple
        The unchanged key and the selected dataset, including timing measurements.
    """
    selection = {}
    if thresholds is not None:
        selection['threshold'] = thresholds
    if quantiles is not None:
        selection['quantile'] = quantiles

    return key, scores.sel(selection)

def average_temporal_scores(key, scores):
    """
    Average scores over the complete time series within one chunk.

    Parameters
    ----------
    key : xarray_beam.Key
        Metadata key identifying the chunk.
    scores : xarray.Dataset
        Individual scores and timing measurements. The chunk must contain
        the complete evaluation period.

    Returns
    -------
    tuple
        The key without its time offset and the averaged dataset.
        Timing measurements, which have no time dimension, remain unchanged.
    """
    averaged = scores.mean('time', skipna=False, keep_attrs=True)
    if 'time' in key.offsets:
        key = key.with_offsets(time=None)
    return key, averaged

def main(argv):
    if TEMPORAL_OUTPUT_PATH.value is not None:
        if not TEMPORAL_MEAN.value:
            raise ValueError('--temporal_output_path requires --temporal_mean=True.')
        if TEMPORAL_OUTPUT_PATH.value.rstrip('/') == OUTPUT_PATH.value.rstrip('/'):
            raise ValueError('The main and temporal output paths must be different.')

        temporal_thresholds = TEMPORAL_THRESHOLDS.value or None
        temporal_quantiles = (
            np.array([float(q) for q in TEMPORAL_QUANTILES.value], dtype=np.float32)
            if TEMPORAL_QUANTILES.value else None
        )

    # Open model outputs and get input chunks
    predictions, input_chunks = xbeam.open_zarr(PREDICTION_PATH.value, decode_timedelta=True)

    # Handle different variable name and scale for FuXi
    if ('total_precipitation_24hr' not in predictions.data_vars and 'total_precipitation_24hr_from_6hr' in predictions.data_vars):
        predictions = predictions.rename({'total_precipitation_24hr_from_6hr': 'total_precipitation_24hr'})
        predictions["total_precipitation_24hr"] = predictions["total_precipitation_24hr"] / 1000.0

    logging.info('Selecting variables, lead times and time.')

    predictions = predictions[VARIABLES.value].sel(time=slice(TIME_START.value, TIME_STOP.value))

    if LEAD_TIMES_HOURS.value:
        lead_times = [np.timedelta64(int(h), 'h') for h in LEAD_TIMES_HOURS.value]
        predictions = predictions.sel(prediction_timedelta=lead_times)

    # Define working chunks
    working_chunks = {
        'longitude': CHUNK_SIZE_LON.value,
        'latitude': CHUNK_SIZE_LAT.value,
        'prediction_timedelta': 1,
        'level': 1,
        'time': -1
    }

    if 'level' not in predictions.dims:
        if 'level' in input_chunks:
            input_chunks.pop('level')
        working_chunks.pop('level')
    else:
        predictions = predictions.sel(level=[int(level) for level in LEVELS.value])

    # Define output chunks
    output_chunks = working_chunks.copy()

    if TEMPORAL_MEAN.value:
        output_chunks.pop('time', None)

    # Open target data
    logging.info('Opening targets.')
    tmin = predictions.time[0] + predictions.prediction_timedelta.min()
    tmax = predictions.time[-1] + predictions.prediction_timedelta.max()
    targets = xr.open_zarr(TARGET_PATH.value).sel(time=slice(tmin, tmax))[VARIABLES.value]

    # Decide which quantiles to use for qwPCRPS
    if QW_PCRPS.value:
        if QUANTILES.value:
            quantiles = np.array([float(q) for q in QUANTILES.value], dtype=np.float32)
        else:
            quantiles = np.arange(1, 100, dtype=np.float32) / 100.0  # 0.01..0.99
    else:
        quantiles = None

    thresholds = None
    if TW_PCRPS.value:
        logging.info('Opening threshold datasets for twPCRPS.')

        # Require that at least one threshold source is provided
        if (
            QUANTILE_THRESHOLDS_PATH.value is None
            and MONTHLY_MIN_RECORDS_PATH.value is None
            and MONTHLY_MAX_RECORDS_PATH.value is None
        ):
            raise ValueError(
                'TW_PCRPS=True but none of --quantile_thresholds_path, '
                '--monthly_min_records_path, or --monthly_max_records_path are set. '
                'Provide at least one of them.'
            )

        threshold_pieces = []

        # Use verification times for monthly record-based thresholds
        months = targets.time.dt.month

        # 1) Monthly min thresholds
        if MONTHLY_MIN_RECORDS_PATH.value is not None:
            logging.info('Opening monthly min record thresholds.')
            rec_min = xr.open_zarr(MONTHLY_MIN_RECORDS_PATH.value)[VARIABLES.value]
            rec_min_thr = (
                rec_min.sel(month=months)
                .expand_dims(threshold=['monthly_min'], axis=-1)
                .reset_coords('month', drop=True)
            )
            threshold_pieces.append(rec_min_thr)

        # 2) Quantile-based thresholds
        if QUANTILE_THRESHOLDS_PATH.value is not None:
            logging.info('Opening quantile-based thresholds.')
            ts = xr.open_zarr(QUANTILE_THRESHOLDS_PATH.value)[VARIABLES.value]

            ts_thr = ts.expand_dims(time=targets.time)

            threshold_pieces.append(ts_thr)

        # 3) Monthly max thresholds
        if MONTHLY_MAX_RECORDS_PATH.value is not None:
            logging.info('Opening monthly max record thresholds.')
            rec_max = xr.open_zarr(MONTHLY_MAX_RECORDS_PATH.value)[VARIABLES.value]
            rec_max_thr = (
                rec_max.sel(month=months)
                .expand_dims(threshold=['monthly_max'], axis=-1)
                .reset_coords('month', drop=True)
            )
            threshold_pieces.append(rec_max_thr)

        # Concat different thresholds
        thresholds = xr.concat(threshold_pieces, dim='threshold')

        # Validate the threshold labels once, before processing chunks
        threshold_names = thresholds.threshold.values.tolist()
        if not threshold_names:
            raise ValueError('TW_PCRPS=True requires non-empty threshold data.')
        if len(set(threshold_names)) != len(threshold_names):
            raise ValueError('Threshold labels must be unique.')

        # Chunk thresholds
        thr_chunks = working_chunks.copy()
        thr_chunks['threshold'] = -1
        thresholds = thresholds.chunk({d: s for d, s in thr_chunks.items() if d in thresholds.dims})

    # Use the same timing-task labels for the output template
    tasks = get_easyuq_tasks(
        compute_qw=QW_PCRPS.value,
        compute_tw=TW_PCRPS.value,
        thresholds=thresholds,
    )
    n_tasks = len(tasks)

    # Create template
    templates = []
    temporal_templates = []

    for var in VARIABLES.value:
        coords = {
            'longitude': predictions.longitude,
            'latitude': predictions.latitude,
            'prediction_timedelta': predictions.prediction_timedelta,
            'task': tasks
        }
        if not TEMPORAL_MEAN.value:
            coords['time'] = predictions.time
        if QW_PCRPS.value:
            coords['quantile'] = quantiles
        if TW_PCRPS.value:
            coords['threshold'] = thresholds.threshold

        data_vars = {
            f'{var}_pcrps': (
                ('longitude', 'latitude', 'prediction_timedelta', 'time'),
                da.full((
                    len(predictions.longitude),
                    len(predictions.latitude),
                    len(predictions.prediction_timedelta),
                    len(predictions.time),
                ), np.nan, dtype=np.float32),
            ),
            f'{var}_time': (
                ('longitude', 'latitude', 'prediction_timedelta', 'task'),
                da.full((
                    len(predictions.longitude),
                    len(predictions.latitude),
                    len(predictions.prediction_timedelta),
                    n_tasks,
                ), np.nan, dtype=np.float32),
            ),
        }

        if QW_PCRPS.value:
            data_vars[f'{var}_qw_pcrps'] = (
                ('longitude', 'latitude', 'prediction_timedelta', 'time', 'quantile'),
                da.full((
                    len(predictions.longitude),
                    len(predictions.latitude),
                    len(predictions.prediction_timedelta),
                    len(predictions.time),
                    len(quantiles),
                ), np.nan, dtype=np.float32),
            )

        if TW_PCRPS.value:
            data_vars[f'{var}_tw_pcrps'] = (
                ('longitude', 'latitude', 'prediction_timedelta', 'time', 'threshold'),
                da.full(
                    (
                        len(predictions.longitude),
                        len(predictions.latitude),
                        len(predictions.prediction_timedelta),
                        len(predictions.time),
                        len(thresholds.threshold),
                    ),
                    np.nan,
                    dtype=np.float32,
                ),
            )

        template = xr.Dataset(data_vars, coords=coords)
        if TW_PCRPS.value:
            template[f'{var}_tw_pcrps'].attrs = {
                'monthly_threshold_method': 'threshold_relative_idr',
                'stored_tail': 'upper for fixed thresholds and monthly_max; lower for monthly_min',
            }

        # If variable has level dimension, add it and make sure dim order matches compute_pc output
        if 'level' in predictions[var].dims:
            template = template.expand_dims({'level': predictions.level})

            template[f'{var}_pcrps'] = template[f'{var}_pcrps'].transpose('longitude', 'latitude', 'prediction_timedelta', 'level', 'time')
            template[f'{var}_time'] = template[f'{var}_time'].transpose('longitude', 'latitude', 'prediction_timedelta', 'level', 'task')

            if QW_PCRPS.value:
                template[f'{var}_qw_pcrps'] = template[f'{var}_qw_pcrps'].transpose(
                    'longitude', 'latitude', 'prediction_timedelta', 'level', 'time', 'quantile',
                )

            if TW_PCRPS.value:
                template[f'{var}_tw_pcrps'] = template[f'{var}_tw_pcrps'].transpose(
                    'longitude', 'latitude', 'prediction_timedelta', 'level', 'time', 'threshold'
                )

        # Keep a template for the optional temporal-score output
        if TEMPORAL_OUTPUT_PATH.value is not None:
            _, temporal_template = select_temporal_scores(
                None,
                template.assign_coords(time=predictions.time),
                thresholds=temporal_thresholds,
                quantiles=temporal_quantiles,
            )
            temporal_templates.append(temporal_template)

        # Optionally discard time dimension
        if TEMPORAL_MEAN.value:
            template[f'{var}_pcrps'] = template[f'{var}_pcrps'].isel(time=0, drop=True)

            if QW_PCRPS.value:
                template[f'{var}_qw_pcrps'] = template[f'{var}_qw_pcrps'].isel(time=0, drop=True)

            if TW_PCRPS.value:
                template[f'{var}_tw_pcrps'] = template[f'{var}_tw_pcrps'].isel(time=0, drop=True)

        # Chunking
        chunk_spec = {dim: size for dim, size in working_chunks.items() if dim in template.dims}
        template = template.chunk(chunk_spec)

        templates.append(template)

    final_template = xr.merge(templates)
    if TEMPORAL_OUTPUT_PATH.value is not None:
        temporal_template = xr.merge(temporal_templates).chunk(working_chunks)
    logging.info('Templates created.')

    # Define and run the Apache Beam pipeline
    with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
        pcrps_pipeline = (
            root
            | 'DatasetToChunks' >> xbeam.DatasetToChunks(
                predictions,
                split_vars=True,
                chunks=input_chunks if RECHUNK.value else working_chunks,
            )
        )
        # Optionally rechunk
        if RECHUNK.value:
            pcrps_pipeline = (
                pcrps_pipeline
                | 'Rechunk' >> xbeam.Rechunk(
                    dim_sizes=predictions.sizes,
                    source_chunks=input_chunks,
                    target_chunks=working_chunks,
                    itemsize=4,
                )
            )
        # Compute PCRPS
        pcrps_pipeline = (
            pcrps_pipeline
            | 'ComputePCRPS' >> beam.MapTuple(
                compute_pcrps,
                targets=targets,
                variable_level_mapping=VARIABLE_LEVEL_MAPPING,
                skip_non_headline=SKIP_NON_HEADLINE.value,
                compute_qw=QW_PCRPS.value,
                quantiles=quantiles,
                compute_tw=TW_PCRPS.value,
                thresholds=thresholds
            )
        )
        # Optionally also write selected individual scores from this computation
        if TEMPORAL_OUTPUT_PATH.value is not None:
            (
                pcrps_pipeline
                | 'SelectTemporalScores' >> beam.MapTuple(
                    select_temporal_scores,
                    thresholds=temporal_thresholds,
                    quantiles=temporal_quantiles,
                )
                | 'TemporalChunksToZarr' >> xbeam.ChunksToZarr(
                    TEMPORAL_OUTPUT_PATH.value,
                    template=xbeam.make_template(temporal_template),
                    zarr_chunks=working_chunks,
                )
            )
        # # Optionally average over time
        # if TEMPORAL_MEAN.value:
        #     pcrps_pipeline = (
        #         pcrps_pipeline
        #         | 'TemporalMean' >> xbeam.Mean(
        #             dim='time',
        #             skipna=False,
        #         )
        #     )
        # Optionally average over time within each completed chunk
        if TEMPORAL_MEAN.value:
            pcrps_pipeline = (
                pcrps_pipeline
                | 'TemporalMean' >> beam.MapTuple(
                    average_temporal_scores
                )
            )
        # Write to Zarr
        (
            pcrps_pipeline
            | 'ChunksToZarr' >> xbeam.ChunksToZarr(
                OUTPUT_PATH.value,
                template=xbeam.make_template(final_template),
                zarr_chunks=output_chunks,
            )
        )

if __name__ == '__main__':
    flags.mark_flags_as_required(['prediction_path', 'target_path', 'output_path'])
    app.run(main)
import numpy as np
import time
import xarray as xr
from isodisreg import idr

def get_points(predictions):
    return np.array(predictions.points)

def get_cdf(predictions):
    return np.array(predictions.ecdf)

def modify_points(points):
    return np.hstack([points[0], np.diff(points)])

def _pcrps(y, p, w, x):
    """
    Compute PCRPS for a single observation and a single IDR predictive distribution.

    Parameters
    ----------
    y : float
        Observed value.
    p : array-like
        CDF values at support points x.
    w : array-like
        Bin weights derived from the CDF.
    x : array-like
        Support points of the predictive distribution.

    Returns
    -------
    float
        PCRPS value for this observation and IDR predictive distribution.
    """
    return 2*np.sum(w*(np.array((y<=x))-p+0.5*w)*np.array(x-y))

def _qw_pcrps_upper(y, p, x, q):
    """
    Compute upper-tail quantile-weighted PCRPS (qwPCRPS) for a single observation and
    a single IDR predictive distribution at given quantile levels.

    Parameters
    ----------
    y : float
        Observed value.
    p : array-like
        CDF values at support points x.
    x : array-like
        Support points of the predictive distribution.
    q : float or array-like
        One or more quantiles in (0, 1).

    Returns
    -------
    np.ndarray, shape (n_q,)
        qwPCRPS values for each quantile in `q`.
    """
    q = np.atleast_1d(np.asarray(q))
    p_star = np.maximum(p[None, :], q[:, None])

    p_prev = np.empty_like(p)
    p_prev[0] = 0.0
    p_prev[1:] = p[:-1]
    p_prev_star = np.maximum(p_prev[None, :], q[:, None])

    indicator = (y <= x).astype(float)[None, :]
    terms = indicator * (p_star - p_prev_star) - 0.5 * (p_star**2 - p_prev_star**2)

    return 2 * np.sum(terms * (x - y)[None, :], axis=1)

def _tw_pcrps_upper(y, p, w, x, t):
    """
    Compute upper-tail threshold-weighted PCRPS (twPCRPS) for a single observation and
    a single IDR predictive distribution at given threshold levels.

    Parameters
    ----------
    y : float
        Observed value.
    p : array-like, shape (n_points,)
        CDF values at support points x.
    w : array-like
        Bin weights derived from the CDF.
    x : array-like, shape (n_points,)
        Support points of the predictive distribution.
    t : array-like
        One or more thresholds.

    Returns
    -------
    np.ndarray, shape (n_q,)
        twPCRPS values for each threshold in `t`.
    """
    t = np.atleast_1d(t)
    x_clip = np.maximum(x[None, :], t[:, None])
    y_clip = np.maximum(y, t)

    w2 = w[None, :]
    p2 = p[None, :]
    indicator = (y_clip[:, None] <= x_clip).astype(float)
    terms = w2 * (indicator - p2 + 0.5 * w2) * (x_clip - y_clip[:, None])

    return 2 * np.sum(terms, axis=1)

def get_easyuq_tasks(
    compute_pcrps=True,
    compute_qw=False,
    compute_tw=False,
    thresholds=None,
):
    """
    Return the timing-task labels for the requested scores and thresholds.

    Parameters
    ----------
    compute_pcrps : bool, default True
        Whether ordinary PCRPS is requested.
    compute_qw : bool, default False
        Whether quantile-weighted PCRPS is requested.
    compute_tw : bool, default False
        Whether threshold-weighted PCRPS is requested.
    thresholds : xarray.Dataset or xarray.DataArray, optional
        Threshold data with a threshold coordinate. Required if compute_tw is True.
        The labels monthly_min and monthly_max identify the separately fitted monthly record scores;
        all other labels identify fixed-threshold scores.

    Returns
    -------
    list of str
        Labels identifying the required fitting and score-evaluation timing measurements.
    """
    if not (compute_pcrps or compute_qw or compute_tw):
        raise ValueError("At least one score must be requested.")

    threshold_names = []
    if compute_tw:
        threshold_names = thresholds.threshold.values.tolist()

    monthly_names = ("monthly_min", "monthly_max")
    has_fixed_thresholds = any(
        name not in monthly_names for name in threshold_names
    )

    tasks = []
    if compute_pcrps or compute_qw or has_fixed_thresholds:
        tasks.append("idr_time")
    if compute_pcrps:
        tasks.append("pcrps_time")
    if compute_qw:
        tasks.append("qw_pcrps_time")
    if has_fixed_thresholds:
        tasks.append("fixed_tw_pcrps_time")

    for name in monthly_names:
        if name in threshold_names:
            tasks.extend([
                f"{name}_idr_time",
                f"{name}_tw_pcrps_time",
            ])

    return tasks

def _fit_easyuq(y, preds_point_df):
    """
    Fit IDR and extract its in-sample predictive distributions, recording fitting time.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Outcomes corresponding to the supplied forecasts. These may be original outcomes or transformed threshold-relative outcomes.
    preds_point_df : pandas.DataFrame, shape (n, 1)
        Original or transformed deterministic forecasts for one grid point and lead time. Rows must correspond to the entries in y.

    Returns
    -------
    x : list of numpy.ndarray
        Support points of each predictive distribution.
    p : list of numpy.ndarray
        Predictive CDF values at the corresponding support points.
    w : list of numpy.ndarray
        Probability masses at the support points.
    elapsed : numpy.float32
        Elapsed wall-clock time in seconds for fitting IDR, obtaining the in-sample distributions, and extracting their arrays.
    """
    start_time = time.time()

    fitted_idr = idr(y, preds_point_df)
    predictions = fitted_idr.predict(preds_point_df, digits=8).predictions

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(modify_points, p))

    elapsed = np.float32(time.time() - start_time)
    return x, p, w, elapsed

def compute_easyuq(preds, obs, var, lead_time, level=None, compute_qw=False, quantiles=None, compute_tw=False, thresholds=None, compute_pcrps=True):
    """
    Fit EasyUQ at each latitude/longitude point and compute the requested scores.

    Ordinary PCRPS, qwPCRPS, and fixed-threshold twPCRPS use the original fit.
    Monthly record scores use separate fits to threshold-relative forecasts and outcomes.

    Parameters
    ----------
    preds : xarray.DataArray
        Data array of model outputs for the variable of interest.
    obs : xarray.DataArray
        Observed data array for the variable of interest.
    var : str
        Name of the variable.
    lead_time : numpy.timedelta64
        Prediction lead time.
    level : int, optional
        Vertical level (if applicable) for the variable.
    compute_qw : bool, optional
        If True, also compute quantile-weighted PCRPS (qwPCRPS) at the specified quantile levels.
    quantiles : array-like, optional
        Sequence of quantile levels in [0, 1] at which qwPCRPS is evaluated.
    compute_tw : bool, optional
        If True, also compute threshold-weighted PCRPS (twPCRPS) at the specified thresholds.
    thresholds : xarray.DataArray, optional
        Threshold data with latitude, longitude, time, and threshold dimensions.
        The labels monthly_min and monthly_max identify monthly record thresholds; all other labels identify fixed thresholds.
        The monthly_min score is stored directly for the lower tail; other scores are stored for the upper tail.
    compute_pcrps : bool, optional
        If True (the default), compute ordinary PCRPS. If False, the original
        fit is still calculated when required for qwPCRPS or fixed thresholds.

    Returns
    -------
    xarray.Dataset
        Dataset containing the requested scores for each grid point and forecast
        initialization time, together with corresponding fitting and evaluation times.
    """
    tasks = get_easyuq_tasks(
        compute_pcrps=compute_pcrps,
        compute_qw=compute_qw,
        compute_tw=compute_tw,
        thresholds=thresholds,
    )

    if compute_tw:
        threshold_names = thresholds.threshold.values.tolist()
        fixed_indices = [
            j for j, name in enumerate(threshold_names)
            if name not in ('monthly_min', 'monthly_max')
        ]

    # Initialize an empty list to store the datasets for each grid point
    pcrps_datasets = []
    # Iterate over each grid point in chunk
    for lat in preds.latitude.values:
        for lon in preds.longitude.values:
            # Select data
            preds_point = preds.sel(latitude=lat, longitude=lon, prediction_timedelta=lead_time)
            preds_point_df = preds_point.to_dataframe()[[var]]

            # Get valid times
            valid_time = preds_point.time + lead_time

            # Extract corresponding observations
            y = obs.sel(latitude=lat, longitude=lon, time=valid_time).values

            # Check observations before fitting IDR
            if y.ndim != 1:
                raise ValueError('obs must be a 1-D array')
            if not np.all(np.isfinite(y)):
                raise ValueError('obs contains non-finite values')
            if len(y) != len(preds_point_df):
                raise ValueError('obs must have the same length as the forecasts')

            # Expand dimensions
            expand_dims = {
                'longitude': [lon],
                'latitude': [lat],
                'prediction_timedelta': preds.prediction_timedelta,
            }
            if level is not None:
                expand_dims['level'] = [level]

            vars_dict = {}
            timings = {}

            # Fit IDR once for all scores that use the original pairs
            if 'idr_time' in tasks:
                x, p, w, idr_time = _fit_easyuq(y, preds_point_df)
                timings['idr_time'] = idr_time

            # PCRPS
            if compute_pcrps:
                start_time_pcrps = time.time()
                pcrps = np.float32(list(map(_pcrps, y, p, w, x)))
                pcrps_time = np.float32(time.time() - start_time_pcrps)
                timings['pcrps_time'] = pcrps_time

                da_pcrps = xr.DataArray(
                    pcrps,
                    dims=['time'],
                    coords={'time': preds_point.time},
                )
                da_pcrps = da_pcrps.expand_dims(expand_dims)
                vars_dict[f'{var}_pcrps'] = da_pcrps

            # qwPCRPS
            if compute_qw:
                q = np.asarray(quantiles, dtype=np.float32)

                start_time_qw_pcrps = time.time()
                qw_pcrps = np.float32(list(map(_qw_pcrps_upper, y, p, x, np.broadcast_to(q, (y.size, q.size)))))
                qw_pcrps_time = np.float32(time.time() - start_time_qw_pcrps)
                timings['qw_pcrps_time'] = qw_pcrps_time

                da_qw_pcrps = xr.DataArray(
                    qw_pcrps,
                    dims=['time', 'quantile'],
                    coords={'time': preds_point.time, 'quantile': q},
                )
                da_qw_pcrps = da_qw_pcrps.expand_dims(expand_dims)
                vars_dict[f'{var}_qw_pcrps'] = da_qw_pcrps

            # twPCRPS
            if compute_tw:
                t = thresholds.sel(latitude=lat, longitude=lon, time=valid_time).transpose('time', 'threshold').values

                tw_pcrps = np.empty((y.size, len(threshold_names)), dtype=np.float32)

                # Fixed thresholds: reuse the original fit
                if fixed_indices:
                    start_time_tw_pcrps = time.time()
                    tw_pcrps[:, fixed_indices] = np.float32(list(map(_tw_pcrps_upper, y, p, w, x, t[:, fixed_indices])))
                    tw_pcrps_time = np.float32(time.time() - start_time_tw_pcrps)
                    timings['fixed_tw_pcrps_time'] = tw_pcrps_time

                # Monthly thresholds: fit each record type separately
                for threshold_name in ('monthly_min', 'monthly_max'):
                    if threshold_name not in threshold_names:
                        continue

                    j = threshold_names.index(threshold_name)
                    t_i = np.asarray(t[:, j], dtype=np.float64)
                    forecast = preds_point_df[var].to_numpy(dtype=np.float64)

                    if threshold_name == 'monthly_min':
                        x_tilde = t_i - forecast
                        y_tilde = t_i - np.asarray(y, dtype=np.float64)
                    else:
                        x_tilde = forecast - t_i
                        y_tilde = np.asarray(y, dtype=np.float64) - t_i

                    transformed_preds_df = preds_point_df.copy()
                    transformed_preds_df[var] = x_tilde

                    x_tw, p_tw, w_tw, idr_time_tw = _fit_easyuq(y_tilde, transformed_preds_df)
                    timings[f'{threshold_name}_idr_time'] = idr_time_tw

                    # Evaluate the transformed distributions at threshold zero
                    start_time_tw_pcrps = time.time()
                    tw_pcrps[:, j] = np.float32(list(map(_tw_pcrps_upper, y_tilde, p_tw, w_tw, x_tw, np.zeros(y.size))))[:, 0]
                    tw_pcrps_time = np.float32(time.time() - start_time_tw_pcrps)
                    timings[f'{threshold_name}_tw_pcrps_time'] = tw_pcrps_time

                da_tw_pcrps = xr.DataArray(
                    tw_pcrps,
                    dims=['time', 'threshold'],
                    coords={'time': preds_point.time, 'threshold': thresholds.threshold},
                )
                da_tw_pcrps = da_tw_pcrps.expand_dims(expand_dims)
                da_tw_pcrps.attrs = {
                    'monthly_threshold_method': 'threshold_relative_idr',
                    'stored_tail': 'upper for fixed thresholds and monthly_max; lower for monthly_min',
                }
                vars_dict[f'{var}_tw_pcrps'] = da_tw_pcrps

            # Assemble all timing measurements in the task order
            time_array = np.asarray([timings[task] for task in tasks], dtype=np.float32)
            time_da = xr.DataArray(
                time_array,
                dims=['task'],
                coords={'task': tasks},
            )
            time_da = time_da.expand_dims(expand_dims)
            vars_dict[f'{var}_time'] = time_da

            pcrps_datasets.append(xr.Dataset(vars_dict))

    # # Merge datasets
    # pcrps_dataset = xr.merge(pcrps_datasets)

    # Concatenate grid-point datasets along the spatial dimensions
    n_lon = preds.sizes["longitude"]
    grid = [
        pcrps_datasets[i:i + n_lon]
        for i in range(0, len(pcrps_datasets), n_lon)
    ]

    pcrps_dataset = xr.combine_nested(
        grid,
        concat_dim=["latitude", "longitude"],
        coords="minimal",
        join="exact",
        combine_attrs="override",
    )

    return pcrps_dataset
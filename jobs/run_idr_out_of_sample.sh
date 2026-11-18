#!/usr/bin/env bash
set -eu

# Change to the repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Load machine-specific directory paths from local_paths.env.
if [ -f "${REPO_ROOT}/local_paths.env" ]; then
  set -a
  source "${REPO_ROOT}/local_paths.env"
  set +a
else
  echo "Missing local_paths.env in ${REPO_ROOT}"
  exit 1
fi

# Save progress and wall-clock timings.
LOG_DIR="${PCRPS_RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"

# -------------------------
# Fixed settings
# -------------------------
TIME_START="2020-01-01"
TIME_STOP="2020-12-16"

CHUNK_LON=8
CHUNK_LAT=11

LEAD_TIMES_HOURS="24,72,120,168,240"
VARS="2m_temperature,mean_sea_level_pressure,10m_wind_speed"

QW_PCRPS="True"
TW_PCRPS="True"

QUANTILE_THRESHOLDS="${PCRPS_THRESHOLDS_DIR}/era5_quantiles_1979_2019.zarr"
MAXREC="${PCRPS_THRESHOLDS_DIR}/era5_record_max_month_1979_2019.zarr"
MINREC="${PCRPS_THRESHOLDS_DIR}/era5_record_min_month_1979_2019.zarr"

TEMPORAL_MEAN="True"

ERA5_TARGET="${PCRPS_DATA_DIR}/era5_240x121_eval_times.zarr"
ERA5_FIT_TARGET="${PCRPS_DATA_DIR}/era5_240x121_oos.zarr"

DIRECT_WORKERS=40
DIRECT_MODE="multi_processing"

TOTAL_JOBS=2
JOB=0

# /dev/null for quiet console
BEAM_OUT="/dev/null"

# -------------------------
# Run one job + print minimal status + timing
# -------------------------
run_job () {
  IDX=$1
  NAME=$2
  PRED=$3
  FIT_PRED=$4
  OUT=$5

  # Skip if the output directories already exists
  if [ -d "${OUT}" ]; then
    echo ""
    echo "============================================================"
    echo "SKIP  Job ${IDX}/${TOTAL_JOBS}: ${NAME} (output already exists)"
    echo "Mean output:  ${OUT}"
    echo "============================================================"
    return 0
  fi

  echo ""
  echo "============================================================"
  echo "START Job ${IDX}/${TOTAL_JOBS}: ${NAME}"
  echo "Started:  $(date '+%F %T')"
  echo "Mean output:  ${OUT}"
  echo "============================================================"

  START=$(date +%s)

  # Run the command, but suppress its output:
  python -m pcrps.compute_pcrps \
    --prediction_path="${PRED}" \
    --target_path="${ERA5_TARGET}" \
    --fit_prediction_path="${FIT_PRED}" \
    --fit_target_path="${ERA5_FIT_TARGET}" \
    --output_path="${OUT}" \
    --variables="${VARS}" \
    --time_start="${TIME_START}" \
    --time_stop="${TIME_STOP}" \
    --chunk_size_lon="${CHUNK_LON}" \
    --chunk_size_lat="${CHUNK_LAT}" \
    --qw_pcrps="${QW_PCRPS}" \
    --tw_pcrps="${TW_PCRPS}" \
    --quantile_thresholds_path="${QUANTILE_THRESHOLDS}" \
    --monthly_max_records_path="${MAXREC}" \
    --monthly_min_records_path="${MINREC}" \
    --temporal_mean="${TEMPORAL_MEAN}" \
    --lead_times_hours="${LEAD_TIMES_HOURS}" \
    --runner=DirectRunner \
    -- \
    --direct_num_workers="${DIRECT_WORKERS}" \
    --direct_running_mode="${DIRECT_MODE}" \
    > "${BEAM_OUT}" 2>&1

  END=$(date +%s)
  DUR=$((END - START))

  echo "DONE  Job ${IDX}/${TOTAL_JOBS}: ${NAME}"
  echo "Finished: $(date '+%F %T')"
  echo "Duration: ${DUR} seconds"
  echo "============================================================"
}

# -------------------------
# Jobs
# -------------------------
JOB=$((JOB+1))
run_job "${JOB}" "pangu vs era5" \
  "${PCRPS_DATA_DIR}/pangu_240x121_2020.zarr" \
  "${PCRPS_DATA_DIR}/pangu_240x121_2018_2019.zarr" \
  "${PCRPS_RESULTS_DIR}/pangu_240x121_vs_era5_oos.zarr"

JOB=$((JOB+1))
run_job "${JOB}" "hres vs era5" \
  "${PCRPS_DATA_DIR}/hres_240x121_2020.zarr" \
  "${PCRPS_DATA_DIR}/hres_240x121_2018_2019.zarr" \
  "${PCRPS_RESULTS_DIR}/hres_240x121_vs_era5_oos.zarr"

echo ""
echo "############################################################"
echo "ALL JOBS FINISHED"
echo "############################################################"

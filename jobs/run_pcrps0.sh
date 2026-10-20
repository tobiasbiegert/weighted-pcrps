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

LEAD_TIMES_HOURS=24,72,120,168,240

CHUNK_LON=16
CHUNK_LAT=11

QW_PCRPS=True
TW_PCRPS=True

REFERENCE_FORECAST_PATH="gs://weatherbench2/datasets/fuxi/2020-240x121_equiangular_with_poles_conservative.zarr"

ERA5_TARGET="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
IFS_ANALYSIS_TARGET="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr"

QUANTILE_THRESHOLDS_PATH="${PCRPS_THRESHOLDS_DIR}/era5_quantiles_1979_2019.zarr"
MONTHLY_MAX_RECORDS_PATH="${PCRPS_THRESHOLDS_DIR}/era5_record_max_month_1979_2019.zarr"
MONTHLY_MIN_RECORDS_PATH="${PCRPS_THRESHOLDS_DIR}/era5_record_min_month_1979_2019.zarr"

TOTAL_JOBS=2
JOB=0

run_job () {
  IDX=$1
  NAME=$2
  TARGET=$3
  OUT=$4
  VARS=$5

  # Skip if output Zarr already exists
  if [ -d "${OUT}" ]; then
    echo ""
    echo "============================================================"
    echo "SKIP  Job ${IDX}/${TOTAL_JOBS}: ${NAME} (output exists)"
    echo "Output:   ${OUT}"
    echo "============================================================"
    return 0
  fi

  echo ""
  echo "============================================================"
  echo "START Job ${IDX}/${TOTAL_JOBS}: ${NAME}"
  echo "Started:  $(date '+%F %T')"
  echo "Output:   ${OUT}"
  echo "============================================================"

  START=$(date +%s)

  python -m pcrps.compute_pcrps0 \
    --target_path="${TARGET}" \
    --reference_forecast_path="${REFERENCE_FORECAST_PATH}" \
    --time_start="${TIME_START}" \
    --time_stop="${TIME_STOP}" \
    --variables="${VARS}" \
    --output_path="${OUT}" \
    --chunk_size_lon="${CHUNK_LON}" \
    --chunk_size_lat="${CHUNK_LAT}" \
    --qw_pcrps="${QW_PCRPS}" \
    --tw_pcrps="${TW_PCRPS}" \
    --quantile_thresholds_path="${QUANTILE_THRESHOLDS_PATH}" \
    --monthly_max_records_path="${MONTHLY_MAX_RECORDS_PATH}" \
    --monthly_min_records_path="${MONTHLY_MIN_RECORDS_PATH}" \
    --lead_times_hours="${LEAD_TIMES_HOURS}"

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
run_job "${JOB}" "era5" \
  "${ERA5_TARGET}" \
  "${PCRPS_RESULTS_DIR}/pcrps0/era5_pcrps0.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr"

JOB=$((JOB+1))
run_job "${JOB}" "ifs_analysis" \
  "${IFS_ANALYSIS_TARGET}" \
  "${PCRPS_RESULTS_DIR}/pcrps0/ifs_analysis_pcrps0.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed"

echo ""
echo "############################################################"
echo "ALL JOBS FINISHED"
echo "############################################################"
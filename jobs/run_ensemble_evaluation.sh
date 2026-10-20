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

# -------------------------
# Fixed settings
# -------------------------
TIME_START="2020-01-01"
TIME_STOP="2020-12-16"

LEAD_TIMES_HOURS="24,72,120,168,240"

CRPS_METHOD="ecdf"
TAIL="upper"

IFS_ENS_PATH="gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr"
GENCAST_PATH="gs://weatherbench2/datasets/gencast/2020-240x121_equiangular_with_poles_conservative.zarr"

ERA5_PATH="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
IFS_ANALYSIS_PATH="gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr"

QUANTILE_THRESHOLDS_PATH="${PCRPS_THRESHOLDS_DIR}/era5_quantiles_1979_2019.zarr"
MAXREC="${PCRPS_THRESHOLDS_DIR}/era5_record_max_month_1979_2019.zarr"
MINREC="${PCRPS_THRESHOLDS_DIR}/era5_record_min_month_1979_2019.zarr"
QUANTILE_THRESHOLDS="q000,q001,q005,q095,q099,q100"

TOTAL_JOBS=8
JOB=0

run_job () {
  IDX=$1
  NAME=$2
  PRED=$3
  TARGET=$4
  OUT=$5
  VARS=$6
  ENS_DIM=$7
  TW=$8

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

  CMD=(
    python scripts/evaluate_ensemble.py
    --prediction_path="${PRED}"
    --target_path="${TARGET}"
    --output_path="${OUT}"
    --variables="${VARS}"
    --time_start="${TIME_START}"
    --time_stop="${TIME_STOP}"
    --crps_method="${CRPS_METHOD}"
    --ensemble_member_dim="${ENS_DIM}"
    --lead_times_hours="${LEAD_TIMES_HOURS}"
    --tw_crps="${TW}"
  )

  if [ "${TW}" = "True" ]; then
    CMD+=(
      --quantile_thresholds_path="${QUANTILE_THRESHOLDS_PATH}"
      --monthly_max_records_path="${MAXREC}"
      --monthly_min_records_path="${MINREC}"
      --quantile_thresholds="${QUANTILE_THRESHOLDS}"
      --tail="${TAIL}"
    )
  fi

  "${CMD[@]}"

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
run_job "${JOB}" "ifs_ens vs era5 crps" \
  "${IFS_ENS_PATH}" \
  "${ERA5_PATH}" \
  "${PCRPS_RESULTS_DIR}/ifs_ens_vs_era5_crps.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr" \
  "number" \
  "False"

JOB=$((JOB+1))
run_job "${JOB}" "ifs_ens vs ifs_analysis crps" \
  "${IFS_ENS_PATH}" \
  "${IFS_ANALYSIS_PATH}" \
  "${PCRPS_RESULTS_DIR}/ifs_ens_vs_ifs_analysis_crps.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed" \
  "number" \
  "False"

JOB=$((JOB+1))
run_job "${JOB}" "gencast vs era5 crps" \
  "${GENCAST_PATH}" \
  "${ERA5_PATH}" \
  "${PCRPS_RESULTS_DIR}/gencast_vs_era5_crps.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr" \
  "sample" \
  "False"

JOB=$((JOB+1))
run_job "${JOB}" "gencast vs ifs_analysis crps" \
  "${GENCAST_PATH}" \
  "${IFS_ANALYSIS_PATH}" \
  "${PCRPS_RESULTS_DIR}/gencast_vs_ifs_analysis_crps.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed" \
  "sample" \
  "False"

JOB=$((JOB+1))
run_job "${JOB}" "ifs_ens vs era5 tw_crps" \
  "${IFS_ENS_PATH}" \
  "${ERA5_PATH}" \
  "${PCRPS_RESULTS_DIR}/ifs_ens_vs_era5_tw_crps.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr" \
  "number" \
  "True"

JOB=$((JOB+1))
run_job "${JOB}" "ifs_ens vs ifs_analysis tw_crps" \
  "${IFS_ENS_PATH}" \
  "${IFS_ANALYSIS_PATH}" \
  "${PCRPS_RESULTS_DIR}/ifs_ens_vs_ifs_analysis_tw_crps.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed" \
  "number" \
  "True"

JOB=$((JOB+1))
run_job "${JOB}" "gencast vs era5 tw_crps" \
  "${GENCAST_PATH}" \
  "${ERA5_PATH}" \
  "${PCRPS_RESULTS_DIR}/gencast_vs_era5_tw_crps.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr" \
  "sample" \
  "True"

JOB=$((JOB+1))
run_job "${JOB}" "gencast vs ifs_analysis tw_crps" \
  "${GENCAST_PATH}" \
  "${IFS_ANALYSIS_PATH}" \
  "${PCRPS_RESULTS_DIR}/gencast_vs_ifs_analysis_tw_crps.zarr" \
  "2m_temperature,mean_sea_level_pressure,10m_wind_speed" \
  "sample" \
  "True"

echo ""
echo "############################################################"
echo "ALL JOBS FINISHED"
echo "############################################################"
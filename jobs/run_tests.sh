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
CHUNK_LON=60
CHUNK_LAT=11
NUM_PERMUTATION=1000
SEED=42

TOTAL_JOBS=9
JOB=0

run_job () {
  IDX=$1
  NAME=$2
  SCORE_A=$3
  SCORE_B=$4
  OUT=$5
  VARS=$6

  # Skip if output already exists
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

  python -m pcrps.test_scores \
      --score_a_path="${SCORE_A}" \
      --score_b_path="${SCORE_B}" \
      --output_path="${OUT}" \
      --variables="${VARS}" \
      --chunk_size_lon="${CHUNK_LON}" \
      --chunk_size_lat="${CHUNK_LAT}" \
      --num_permutation="${NUM_PERMUTATION}" \
      --seed="${SEED}"

  END=$(date +%s)
  DUR=$((END - START))

  echo "DONE  Job ${IDX}/${TOTAL_JOBS}: ${NAME}"
  echo "Finished: $(date '+%F %T')"
  echo "Duration: ${DUR} seconds"
  echo "============================================================"
}

# -------------------------
# Variable sets
# -------------------------
VARS_3="mean_sea_level_pressure_pcrps,2m_temperature_pcrps,10m_wind_speed_pcrps,mean_sea_level_pressure_tw_pcrps,2m_temperature_tw_pcrps,10m_wind_speed_tw_pcrps,mean_sea_level_pressure_qw_pcrps,2m_temperature_qw_pcrps,10m_wind_speed_qw_pcrps"
VARS_4="mean_sea_level_pressure_pcrps,2m_temperature_pcrps,10m_wind_speed_pcrps,total_precipitation_24hr_pcrps,mean_sea_level_pressure_tw_pcrps,2m_temperature_tw_pcrps,10m_wind_speed_tw_pcrps,total_precipitation_24hr_tw_pcrps,mean_sea_level_pressure_qw_pcrps,2m_temperature_qw_pcrps,10m_wind_speed_qw_pcrps,total_precipitation_24hr_qw_pcrps"

# -------------------------
# Paths to temporal score files
# -------------------------
FUXI_ERA5="${PCRPS_RESULTS_DIR}/fuxi_240x121_vs_era5_temporal.zarr"

GRAPHCAST_ERA5="${PCRPS_RESULTS_DIR}/graphcast_240x121_vs_era5_temporal.zarr"
GRAPHCAST_OPERATIONAL_IFS="${PCRPS_RESULTS_DIR}/graphcast_operational_240x121_vs_ifs_analysis_temporal.zarr"

PANGU_ERA5="${PCRPS_RESULTS_DIR}/pangu_240x121_vs_era5_temporal.zarr"
PANGU_OPERATIONAL_IFS="${PCRPS_RESULTS_DIR}/pangu_operational_240x121_vs_ifs_analysis_temporal.zarr"

HRES_ERA5="${PCRPS_RESULTS_DIR}/hres_240x121_vs_era5_temporal.zarr"
HRES_IFS="${PCRPS_RESULTS_DIR}/hres_240x121_vs_ifs_analysis_temporal.zarr"

# -------------------------
# Jobs
# -------------------------
JOB=$((JOB+1))
run_job "${JOB}" "fuxi vs graphcast (era5)" \
  "${FUXI_ERA5}" \
  "${GRAPHCAST_ERA5}" \
  "${PCRPS_RESULTS_DIR}/tests/fuxi_graphcast_vs_era5_p.zarr" \
  "${VARS_4}"

JOB=$((JOB+1))
run_job "${JOB}" "fuxi vs pangu (era5)" \
  "${FUXI_ERA5}" \
  "${PANGU_ERA5}" \
  "${PCRPS_RESULTS_DIR}/tests/fuxi_pangu_vs_era5_p.zarr" \
  "${VARS_3}"

JOB=$((JOB+1))
run_job "${JOB}" "fuxi vs hres (era5)" \
  "${FUXI_ERA5}" \
  "${HRES_ERA5}" \
  "${PCRPS_RESULTS_DIR}/tests/fuxi_hres_vs_era5_p.zarr" \
  "${VARS_4}"

JOB=$((JOB+1))
run_job "${JOB}" "graphcast vs pangu (era5)" \
  "${GRAPHCAST_ERA5}" \
  "${PANGU_ERA5}" \
  "${PCRPS_RESULTS_DIR}/tests/graphcast_pangu_vs_era5_p.zarr" \
  "${VARS_3}"

JOB=$((JOB+1))
run_job "${JOB}" "graphcast vs hres (era5)" \
  "${GRAPHCAST_ERA5}" \
  "${HRES_ERA5}" \
  "${PCRPS_RESULTS_DIR}/tests/graphcast_hres_vs_era5_p.zarr" \
  "${VARS_4}"

JOB=$((JOB+1))
run_job "${JOB}" "pangu vs hres (era5)" \
  "${PANGU_ERA5}" \
  "${HRES_ERA5}" \
  "${PCRPS_RESULTS_DIR}/tests/pangu_hres_vs_era5_p.zarr" \
  "${VARS_3}"

JOB=$((JOB+1))
run_job "${JOB}" "graphcast_operational vs pangu_operational (ifs_analysis)" \
  "${GRAPHCAST_OPERATIONAL_IFS}" \
  "${PANGU_OPERATIONAL_IFS}" \
  "${PCRPS_RESULTS_DIR}/tests/graphcast_operational_pangu_operational_vs_ifs_analysis_p.zarr" \
  "${VARS_3}"

JOB=$((JOB+1))
run_job "${JOB}" "graphcast_operational vs hres (ifs_analysis)" \
  "${GRAPHCAST_OPERATIONAL_IFS}" \
  "${HRES_IFS}" \
  "${PCRPS_RESULTS_DIR}/tests/graphcast_operational_hres_vs_ifs_analysis_p.zarr" \
  "${VARS_3}"

JOB=$((JOB+1))
run_job "${JOB}" "pangu_operational vs hres (ifs_analysis)" \
  "${PANGU_OPERATIONAL_IFS}" \
  "${HRES_IFS}" \
  "${PCRPS_RESULTS_DIR}/tests/pangu_operational_hres_vs_ifs_analysis_p.zarr" \
  "${VARS_3}"

echo ""
echo "############################################################"
echo "ALL JOBS FINISHED"
echo "############################################################"
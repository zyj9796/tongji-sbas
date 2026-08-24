#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "用法: $0 输入SRTM_HGT 输出DEM 输出DEM参数" >&2
  exit 2
fi

input_hgt=$1
output_dem=$2
output_par=$3
compat_dir=/tmp/gamma_gdal_compat

mkdir -p "$(dirname "$output_dem")" "$compat_dir"
ln -sfn /lib/libgdal.so.30 "$compat_dir/libgdal.so.26"
export LD_LIBRARY_PATH="$compat_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# SRTMGL1 stores orthometric heights referenced to EGM96. GAMMA radar geometry
# requires ellipsoidal height, so dem_import adds the bundled EGM96 geoid grid.
dem_import \
  "$input_hgt" \
  "$output_dem" \
  "$output_par" \
  0 1 \
  "$DIFF_HOME/scripts/egm96.dem" \
  "$DIFF_HOME/scripts/egm96.dem_par" \
  0 - - 0 - 1

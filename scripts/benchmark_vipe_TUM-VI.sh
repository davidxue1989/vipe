#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# TUM-VI runner (images only) — supports chunked and unchunked inputs
#
# Unchunked per-sequence layout:
#   ${seq}/images/                 (all frames)
#   ${seq}/timestamp.csv           ("#timestamp [ns],filename")
#   ${seq}/groundtruth.csv
#
# Chunked per-sequence layout:
#   ${seq}/images/part_0001/, part_0002/, ... (each ~300 frames)
#   ${seq}/timestamp_0001.csv, timestamp_0002.csv, ...
#   ${seq}/groundtruth.csv
#
# What this script does per sequence:
#   - If chunked:
#       * Run `vipe infer` ONLY for chunks that do NOT already have pose .npz
#         (unless --reprocess). Chunks with existing .npz are NOT re-inferred.
#       * Convert a chunk to partial TUM if (a) --reprocess OR (b) partial is missing.
#       * Stitch ALL available partials (including from past runs) into vipe_estimate_tum.txt
#   - If unchunked: single-run behavior; skip `vipe infer` when .npz exists unless --reprocess
#   - Always converts groundtruth.csv -> groundtruth.txt (TUM)
#   - Mirrors results to ./output_vipe[_EXPNAME]/poses_and_groundtruth/<seq>/
#   - Optionally runs your evo benchmark (benchmark_dx.py)
#
# Usage:
#   bash run_vipe_tumvi_images.sh [-e EXPNAME] [-p PIPELINE] [-v]
#                                 [--input-root PATH]
#                                 [--only NAME ...] [--only-file FILE]
#                                 [--skip NAME ...] [--skip-file FILE]
#                                 [--poses-kind twc|tcw]
#                                 [--reprocess] [--skip-infer]
#                                 [--bench-py PATH] [--no-browser] [--plots]
# ------------------------------------------------------------

# Defaults
ROOT_IN="/home/dxue/workspaces/isaac_ros-dev/TUM-VI_dx"
EXPNAME=""
PIPELINE="TUM-VI"
VISUALIZE=0
ONLY_LIST=()
ONLY_FILE=""
SKIP_LIST=()
SKIP_FILE=""
POSES_KIND="twc"       # or 'tcw' (if tcw, convert by inverting per-frame)
REPROCESS=0
SKIP_INFER=0
BENCH_PY="/home/dxue/workspaces/isaac_ros-dev/src/rss-ros-envs/scripts/evo_benchmark/benchmark_dx.py"
NO_BROWSER=0
SHOW_PLOTS=0

print_help() {
  sed -n '1,200p' "$0"
  exit 0
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--expname) EXPNAME="${2:-}"; shift 2;;
    -p|--pipeline) PIPELINE="${2:-default}"; shift 2;;
    -v|--vis|--visualize) VISUALIZE=1; shift;;
    --input-root) ROOT_IN="${2:-$ROOT_IN}"; shift 2;;
    --only) ONLY_LIST+=("${2:-}"); shift 2;;
    --only-file) ONLY_FILE="${2:-}"; shift 2;;
    --skip) SKIP_LIST+=("${2:-}"); shift 2;;
    --skip-file) SKIP_FILE="${2:-}"; shift 2;;
    --poses-kind) POSES_KIND="${2:-twc}"; shift 2;;
    --reprocess) REPROCESS=1; shift;;
    --skip-infer|--no-infer) SKIP_INFER=1; shift;;
    --bench-py) BENCH_PY="${2:-$BENCH_PY}"; shift 2;;
    --no-browser) NO_BROWSER=1; shift;;
    --plots) SHOW_PLOTS=1; shift;;
    -h|--help) print_help;;
    *) echo "Unknown arg: $1"; print_help;;
  esac
done

# Merge names from --only-file into ONLY_LIST
if [[ -n "${ONLY_FILE}" ]]; then
  [[ -f "${ONLY_FILE}" ]] || { echo "[ERR] --only-file not found: ${ONLY_FILE}"; exit 1; }
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    ONLY_LIST+=("$line")
  done < "${ONLY_FILE}"
fi

# Merge names from --skip-file into SKIP_LIST
if [[ -n "${SKIP_FILE}" ]]; then
  [[ -f "${SKIP_FILE}" ]] || { echo "[ERR] --skip-file not found: ${SKIP_FILE}"; exit 1; }
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    SKIP_LIST+=("$line")
  done < "${SKIP_FILE}"
fi

# Build sets (assoc arrays) for exact-match filtering
declare -A ONLY_SET=()
declare -A SKIP_SET=()
if [[ ${#ONLY_LIST[@]} -gt 0 ]]; then
  for n in "${ONLY_LIST[@]}"; do ONLY_SET["$n"]=1; done
fi
if [[ ${#SKIP_LIST[@]} -gt 0 ]]; then
  for n in "${SKIP_LIST[@]}"; do SKIP_SET["$n"]=1; done
fi

# Output roots
if [[ -n "${EXPNAME}" ]]; then
  ROOT_OUT="./output_vipe_${EXPNAME}"
else
  ROOT_OUT="./output_vipe"
fi
mkdir -p "${ROOT_OUT}"

DEST_ROOT="${ROOT_OUT}/poses_and_groundtruth"
mkdir -p "${DEST_ROOT}"

# --- Python helpers ---
PY_HELPER_DIR="${ROOT_OUT}/py_helpers"
mkdir -p "${PY_HELPER_DIR}"

# 1) Convert ViPE pose npz + timestamp.csv (ns) -> TUM txt (seconds)
PY_CONVERT_PRED="${PY_HELPER_DIR}/npz_to_tum_from_timestamp_csv.py"
cat > "${PY_CONVERT_PRED}" << 'PYCODE'
#!/usr/bin/env python3
import argparse, sys, os, csv, numpy as np

def load_ns_timestamps_from_csv(path):
    ts_ns = []
    with open(path, 'r', newline='') as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if row[0].startswith('#'):
                continue
            try:
                ts_ns.append(int(row[0]))
            except Exception:
                continue
    return ts_ns

def mat_to_quat_xyzw(R):
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q

def invert_h(T):
    R = T[:3,:3]
    t = T[:3,3]
    Rt = R.T
    tinv = -Rt @ t
    Tinv = np.eye(4, dtype=T.dtype)
    Tinv[:3,:3] = Rt
    Tinv[:3,3]  = tinv
    return Tinv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='Path to ViPE pose .npz')
    ap.add_argument('--timestamp-csv', required=True, help='Path to timestamp.csv with "#timestamp [ns],filename"')
    ap.add_argument('--out', required=True, help='Output TUM trajectory path')
    # Option A: keep hyphenated flag, set dest so code uses args.poses_kind (underscore)
    ap.add_argument('--poses-kind', '-K', dest='poses_kind', choices=['twc','tcw'], default='twc')
    ap.add_argument('--seq-name', default='')
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        arr = None
        for k in ['poses','T','arr_0','pose','video']:
            if k in data:
                arr = data[k]
                break
        if arr is None:
            keys = list(data.keys())
            if not keys:
                print(f"[ERR] No arrays found in {args.npz}", file=sys.stderr); sys.exit(2)
            arr = data[keys[0]]
    else:
        arr = data
    arr = np.array(arr)
    if arr.ndim != 3 or arr.shape[1:] != (4,4):
        print(f"[ERR] Expected shape (N,4,4), got {arr.shape}", file=sys.stderr); sys.exit(2)

    ts_ns = load_ns_timestamps_from_csv(args.timestamp_csv)
    if len(ts_ns) == 0:
        print(f"[ERR] No timestamps found in {args.timestamp_csv}", file=sys.stderr); sys.exit(2)
    ts = [t/1e9 for t in ts_ns]

    n = min(len(ts), arr.shape[0])
    if arr.shape[0] != len(ts):
        print(f"[WARN] Frames mismatch: npz={arr.shape[0]} vs timestamp.csv={len(ts)}; using first {n}", file=sys.stderr)

    with open(args.out, 'w') as f:
        f.write("# ViPE predicted trajectory (TUM format)\n")
        if args.seq_name:
            f.write(f"# sequence: '{args.seq_name}'\n")
        f.write("# columns: timestamp tx ty tz qx qy qz qw\n")
        for i in range(n):
            T = arr[i]
            if args.poses_kind == 'tcw':
                T = invert_h(T)
            R = T[:3,:3].astype(np.float64)
            t = T[:3,3].astype(np.float64)
            qx, qy, qz, qw = mat_to_quat_xyzw(R)
            f.write(f"{ts[i]:.6f} {t[0]:.7f} {t[1]:.7f} {t[2]:.7f} {qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f}\n")

    print(f"[OK] Wrote {args.out} with {n} poses")

if __name__ == "__main__":
    main()
PYCODE
chmod +x "${PY_CONVERT_PRED}"

# 2) Convert TUM-VI groundtruth.csv -> TUM groundtruth.txt
PY_CONVERT_GT="${PY_HELPER_DIR}/groundtruth_csv_to_tum.py"
cat > "${PY_CONVERT_GT}" << 'PYCODE'
#!/usr/bin/env python3
import argparse, sys, csv

# Input CSV columns (TUM-VI):
#   "#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m],
#    q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []"
#
# Output TUM format (seconds):
#   timestamp tx ty tz qx qy qz qw
# where quaternion order is xyzw.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='groundtruth.csv path')
    ap.add_argument('--out', required=True, help='Output groundtruth.txt (TUM format)')
    ap.add_argument('--seq-name', default='')
    args = ap.parse_args()

    rows = []
    with open(args.csv, 'r', newline='') as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if row[0].startswith('#'):
                continue
            try:
                ts_ns = int(row[0])
                px = float(row[1]); py = float(row[2]); pz = float(row[3])
                qw = float(row[4]); qx = float(row[5]); qy = float(row[6]); qz = float(row[7])
            except Exception:
                continue
            ts = ts_ns / 1e9
            rows.append((ts, px, py, pz, qx, qy, qz, qw))  # xyzw for TUM

    rows.sort(key=lambda x: x[0])

    with open(args.out, 'w') as f:
        f.write("# Groundtruth trajectory (TUM format)\n")
        if args.seq_name:
            f.write(f"# sequence: '{args.seq_name}'\n")
        f.write("# columns: timestamp tx ty tz qx qy qz qw\n")
        for (ts, px, py, pz, qx, qy, qz, qw) in rows:
            f.write(f"{ts:.6f} {px:.7f} {py:.7f} {pz:.7f} {qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f}\n")

    print(f"[OK] Wrote {args.out} with {len(rows)} poses")

if __name__ == "__main__":
    main()
PYCODE
chmod +x "${PY_CONVERT_GT}"

# --- Helpers (bash) ---
find_pose_npz() {
  local raw_dir="$1"
  local pdir="${raw_dir}/pose"
  if [[ -d "${pdir}" ]]; then
    shopt -s nullglob
    local npzs=( "${pdir}"/*.npz )
    shopt -u nullglob
    if (( ${#npzs[@]} > 0 )); then
      local newest
      newest="$(ls -t "${pdir}"/*.npz | head -n1)"
      echo "${newest}"
      return 0
    fi
  fi
  if [[ -f "${raw_dir}/pose/video.npz" ]]; then
    echo "${raw_dir}/pose/video.npz"
    return 0
  fi
  return 1
}

open_html() {
  local f="$1"
  if (( NO_BROWSER == 1 )); then
    echo "Browser opening disabled (--no-browser). Result ready at: $f"
    return
  fi
  if command -v google-chrome >/dev/null 2>&1; then
    google-chrome --no-sandbox "$f" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$f" >/dev/null 2>&1 || true
  else
    echo "Result ready at: $f"
  fi
}

is_chunked_seq() {
  local seq_dir="$1"
  local img_dir="${seq_dir}/images"
  shopt -s nullglob
  local parts=( "${img_dir}"/part_* )
  shopt -u nullglob
  if (( ${#parts[@]} == 0 )); then
    return 1
  fi
  # require at least one matching timestamp_*.csv, otherwise treat as unchunked
  shopt -s nullglob
  local tcs=( "${seq_dir}"/timestamp_*.csv )
  shopt -u nullglob
  (( ${#tcs[@]} > 0 ))
}

# Sort helper: prints chunk ids like 0001 0002 ... in numeric order
list_chunk_ids() {
  local seq_dir="$1"
  shopt -s nullglob
  for f in "${seq_dir}"/timestamp_*.csv; do
    basename "$f" | sed -E 's/^timestamp_([0-9]{4})\.csv$/\1/' || true
  done | sort
  shopt -u nullglob
}

echo "[INFO] Input root: ${ROOT_IN}"
echo "[INFO] Output root: ${ROOT_OUT}"
echo "[INFO] Pipeline: ${PIPELINE}"
echo "[INFO] Visualize: $([[ ${VISUALIZE} -eq 1 ]] && echo ON || echo OFF)"
echo "[INFO] Poses-kind: ${POSES_KIND} (twc = camera in world; tcw = invert)"
echo "[INFO] Reprocess: $([[ ${REPROCESS} -eq 1 ]] && echo ON || echo OFF)"
echo "[INFO] Skip infer: $([[ ${SKIP_INFER} -eq 1 ]] && echo ON || echo OFF)"
echo "[INFO] Benchmark py: ${BENCH_PY}"
echo "[INFO] Scanning for sequences with required files ..."

overall_start=$(date +%s)
shopt -s nullglob
seq_count=0
for seq_dir in "${ROOT_IN}"/*; do
  [[ -d "${seq_dir}" ]] || continue
  seq_name="$(basename "${seq_dir}")"

  if [[ ${#ONLY_LIST[@]} -gt 0 && -z "${ONLY_SET[$seq_name]+x}" ]]; then
    continue
  fi
  if [[ ${#SKIP_LIST[@]} -gt 0 && -n "${SKIP_SET[$seq_name]+x}" ]]; then
    echo "[SKIP] ${seq_name}: matched --skip/--skip-file"
    continue
  fi

  img_dir="${seq_dir}/images"
  ts_csv="${seq_dir}/timestamp.csv"
  gt_csv="${seq_dir}/groundtruth.csv"

  have_inputs=1
  [[ -d "${img_dir}" ]] || { echo "[SKIP] ${seq_name}: missing images/"; have_inputs=0; }
  [[ -f "${gt_csv}" ]] || { echo "[SKIP] ${seq_name}: missing groundtruth.csv"; have_inputs=0; }

  if (( have_inputs == 1 )); then
    out_seq_dir="${ROOT_OUT}/${seq_name}"
    raw_dir="${out_seq_dir}/vipe_raw"
    mkdir -p "${raw_dir}"

    echo "------------------------------------------------------------"
    echo "[SEQ] ${seq_name}"

    log="${out_seq_dir}/vipe_infer.log"
    [[ -f "${log}" ]] || : > "${log}"

    if is_chunked_seq "${seq_dir}"; then
      echo "[MODE] Chunked sequence detected." | tee -a "${log}"
      readarray -t chunk_ids < <(list_chunk_ids "${seq_dir}")
      if (( ${#chunk_ids[@]} == 0 )); then
        echo "[ERR] No timestamp_*.csv found although images/part_* exist. Skipping ${seq_name}." | tee -a "${log}"
        continue
      fi

      # Per-chunk run & convert (revised logic)
      for cid in "${chunk_ids[@]}"; do
        part_dir="${img_dir}/part_${cid}"
        tsc="${seq_dir}/timestamp_${cid}.csv"
        if [[ ! -d "${part_dir}" ]]; then
          echo "[WARN] Missing ${part_dir} for timestamp_${cid}.csv; skipping this chunk." | tee -a "${log}"
          continue
        fi
        if [[ ! -f "${tsc}" ]]; then
          echo "[WARN] Missing ${tsc}; skipping this chunk." | tee -a "${log}"
          continue
        fi

        raw_part="${raw_dir}/part_${cid}"
        mkdir -p "${raw_part}"
        part_log="${out_seq_dir}/vipe_infer_part_${cid}.log"
        [[ -f "${part_log}" ]] || : > "${part_log}"

        # Do NOT re-run infer if an npz already exists and --reprocess is NOT set
        npz_path=""
        if [[ ${REPROCESS} -eq 0 ]]; then
          if npz_found="$(find_pose_npz "${raw_part}")"; then
            npz_path="${npz_found}"
          fi
        fi

        # If npz missing, run infer (unless user asked to skip)
        if [[ -z "${npz_path}" && ${SKIP_INFER} -ne 1 ]]; then
          cmd=(vipe infer -p "${PIPELINE}" -o "${raw_part}")
          [[ ${VISUALIZE} -eq 1 ]] && cmd+=(-v)
          cmd+=(--image-dir "${part_dir}")

          start_human="$(date -Iseconds)"; start_epoch=$(date +%s)
          echo "[TIME] start=${start_human} epoch=${start_epoch}" | tee -a "${part_log}"
          echo "[RUN]  ${cmd[*]}" | tee -a "${part_log}"

          set +e
          if command -v /usr/bin/time >/dev/null 2>&1; then
            { /usr/bin/time -f "[TIME] elapsed=%E user=%U sys=%S maxrss_kb=%M" "${cmd[@]}"; } 2>&1 | tee -a "${part_log}"
            infer_rc=$?
          else
            "${cmd[@]}" 2>&1 | tee -a "${part_log}"
            infer_rc=$?
          fi
          set -e

          end_human="$(date -Iseconds)"; end_epoch=$(date +%s)
          dur=$(( end_epoch - start_epoch ))
          printf "[TIME] end=%s epoch=%d duration_s=%d\n" "$end_human" "$end_epoch" "$dur" | tee -a "${part_log}"

          if (( infer_rc != 0 )); then
            echo "[WARN] part_${cid} vipe infer failed (code=${infer_rc}); will try to convert if any npz exists." | tee -a "${part_log}"
          fi

          if npz_found="$(find_pose_npz "${raw_part}")"; then
            npz_path="${npz_found}"
          else
            npz_path=""
          fi
        else
          if [[ -n "${npz_path}" ]]; then
            echo "[SKIP] part_${cid} vipe infer: pose npz already exists (${npz_path}); use --reprocess to force." | tee -a "${part_log}"
          else
            echo "[SKIP] part_${cid} vipe infer: user specified --skip-infer; attempting post-process only." | tee -a "${part_log}"
          fi
        fi

        # Convert to partial TUM only if (a) reprocess, or (b) missing partial
        partial_out="${out_seq_dir}/vipe_estimate_tum_part_${cid}.txt"
        if [[ -n "${npz_path}" ]]; then
          if [[ ${REPROCESS} -eq 1 || ! -f "${partial_out}" ]]; then
            echo "[PRED-part] Convert ${npz_path} + ${tsc} -> ${partial_out} (poses-kind=${POSES_KIND})" | tee -a "${part_log}"
            if ! python3 "${PY_CONVERT_PRED}" --npz "${npz_path}" \
                                              --timestamp-csv "${tsc}" \
                                              --out "${partial_out}" \
                                              --poses-kind "${POSES_KIND}" \
                                              --seq-name "${seq_name}:part_${cid}"; then
              echo "[WARN] part_${cid}: conversion failed; this chunk may be missing from stitching." | tee -a "${part_log}"
            fi
          else
            echo "[SKIP] part_${cid} conversion: ${partial_out} exists; use --reprocess to force." | tee -a "${part_log}"
          fi
        else
          echo "[WARN] part_${cid}: no pose .npz available; cannot convert to partial TUM." | tee -a "${part_log}"
        fi
      done

      # Convert groundtruth (full sequence GT)
      gt_tum="${out_seq_dir}/groundtruth.txt"
      echo "[GT] Convert ${gt_csv} -> ${gt_tum}" | tee -a "${log}"
      python3 "${PY_CONVERT_GT}" --csv "${gt_csv}" --out "${gt_tum}" --seq-name "${seq_name}" \
        2>&1 | tee -a "${log}"

      # Stitch ALL available partial predictions (not just newly created ones)
      pred_out="${out_seq_dir}/vipe_estimate_tum.txt"
      : > "${pred_out}"
      {
        echo "# ViPE predicted trajectory (TUM format)"
        echo "# sequence: '${seq_name}' (stitched)"
        echo "# columns: timestamp tx ty tz qx qy qz qw"
      } >> "${pred_out}"

      appended=0
      for cid in "${chunk_ids[@]}"; do
        f="${out_seq_dir}/vipe_estimate_tum_part_${cid}.txt"
        if [[ -f "${f}" ]]; then
          awk 'BEGIN{skipped=0} { if ($0 ~ /^#/) next; print $0 }' "${f}" >> "${pred_out}"
          appended=$(( appended + 1 ))
        else
          echo "[WARN] Missing partial file for part_${cid}; it will be absent from stitched output." | tee -a "${log}"
        fi
      done

      if (( appended == 0 )); then
        echo "[WARN] No partial predictions stitched for ${seq_name}; final prediction missing." | tee -a "${log}"
        rm -f "${pred_out}" || true
      else
        # Optional: ensure monotonic timestamps
        tmp_sorted="${pred_out}.sorted.tmp"
        { head -n 3 "${pred_out}"; awk 'NR>3 && $0 !~ /^#/' "${pred_out}" | sort -n -k1,1; } > "${tmp_sorted}"
        mv -f "${tmp_sorted}" "${pred_out}"
        echo "[OK] Stitched ${appended} chunk(s) into ${pred_out}"
      fi

    else
      echo "[MODE] Unchunked sequence." | tee -a "${log}"

      # --- Single-run behavior (previous flow) ---
      npz_path=""
      if [[ ${REPROCESS} -eq 0 ]]; then
        if npz_found="$(find_pose_npz "${raw_dir}")"; then
          npz_path="${npz_found}"
        fi
      fi

      run_infer=1
      if [[ ${SKIP_INFER} -eq 1 ]]; then
        run_infer=0
      elif [[ ${REPROCESS} -eq 0 && -n "${npz_path}" ]]; then
        run_infer=0
        echo "[SKIP] vipe infer: ${npz_path} already exists (use --reprocess to force)." | tee -a "${log}"
      fi

      if [[ ${run_infer} -eq 1 ]]; then
        cmd=(vipe infer -p "${PIPELINE}" -o "${raw_dir}")
        [[ ${VISUALIZE} -eq 1 ]] && cmd+=(-v)
        cmd+=(--image-dir "${img_dir}")

        start_human="$(date -Iseconds)"; start_epoch=$(date +%s)
        echo "[TIME] start=${start_human} epoch=${start_epoch}" | tee -a "${log}"
        echo "[RUN]  ${cmd[*]}" | tee -a "${log}"

        set +e
        if command -v /usr/bin/time >/dev/null 2>&1; then
          { /usr/bin/time -f "[TIME] elapsed=%E user=%U sys=%S maxrss_kb=%M" "${cmd[@]}"; } 2>&1 | tee -a "${log}"
          infer_rc=$?
        else
          "${cmd[@]}" 2>&1 | tee -a "${log}"
          infer_rc=$?
        fi
        set -e

        end_human="$(date -Iseconds)"; end_epoch=$(date +%s)
        dur=$(( end_epoch - start_epoch ))
        printf "[TIME] end=%s epoch=%d duration_s=%d\n" "$end_human" "$end_epoch" "$dur" | tee -a "${log}"

        if (( infer_rc != 0 )); then
          echo "[WARN] vipe infer failed with exit code ${infer_rc}. Will attempt post-processing anyway." | tee -a "${log}"
        fi

        if npz_found="$(find_pose_npz "${raw_dir}")"; then
          npz_path="${npz_found}"
        else
          npz_path=""
        fi
      else
        if [[ ${SKIP_INFER} -eq 1 ]]; then
          echo "[SKIP] vipe infer: user specified --skip-infer; attempting post-process only." | tee -a "${log}"
        fi
      fi

      # groundtruth
      gt_tum="${out_seq_dir}/groundtruth.txt"
      echo "[GT] Convert ${gt_csv} -> ${gt_tum}" | tee -a "${log}"
      python3 "${PY_CONVERT_GT}" --csv "${gt_csv}" --out "${gt_tum}" --seq-name "${seq_name}" \
        2>&1 | tee -a "${log}"

      # predictions
      if [[ -z "${npz_path}" ]]; then
        echo "[WARN] Missing ViPE pose .npz under ${raw_dir}/pose/. Prediction TUM will be skipped." | tee -a "${log}"
      else
        if [[ ! -f "${ts_csv}" ]]; then
          echo "[ERR] Missing ${ts_csv} in unchunked mode; skipping prediction convert." | tee -a "${log}"
        else
          pred_out="${out_seq_dir}/vipe_estimate_tum.txt"
          echo "[PRED] Convert ${npz_path} + ${ts_csv} -> ${pred_out} (poses-kind=${POSES_KIND})" | tee -a "${log}"
          python3 "${PY_CONVERT_PRED}" --npz "${npz_path}" --timestamp-csv "${ts_csv}" \
                                       --out "${pred_out}" --poses-kind "${POSES_KIND}" \
                                       --seq-name "${seq_name}" \
                                       2>&1 | tee -a "${log}"
        fi
      fi
      # --- end single-run ---
    fi

    # Mirror into benchmark layout
    dest_dir="${DEST_ROOT}/${seq_name}"
    mkdir -p "${dest_dir}"
    if [[ -f "${out_seq_dir}/groundtruth.txt" ]]; then
      cp -f "${out_seq_dir}/groundtruth.txt" "${dest_dir}/groundtruth.txt"
    fi
    if [[ -f "${out_seq_dir}/vipe_estimate_tum.txt" ]]; then
      cp -f "${out_seq_dir}/vipe_estimate_tum.txt" "${dest_dir}/vipe_estimate_tum.txt"
    fi

    seq_count=$((seq_count+1))
  fi
done
shopt -u nullglob

# -----------------------------
# Run evo benchmark (if present)
# -----------------------------
RESULT_HTML="${DEST_ROOT}/benchmark_results.html"
if [[ -f "${BENCH_PY}" ]]; then
  echo "Running benchmark: ${BENCH_PY}"

  PLOT_ARGS=()
  (( SHOW_PLOTS == 1 )) && PLOT_ARGS+=( --plots )
  if [[ ${#ONLY_LIST[@]} -gt 0 ]]; then
    for tok in "${ONLY_LIST[@]}"; do
      PLOT_ARGS+=( --plot-only "$tok" )
    done
  fi

  if ! python3 "${BENCH_PY}" "${DEST_ROOT}" "${PLOT_ARGS[@]}"; then
    echo "Warning: benchmark.py returned non-zero status"
  fi

  if [[ -f "${RESULT_HTML}" ]]; then
    echo "Opening: ${RESULT_HTML}"
    open_html "${RESULT_HTML}"
  else
    html_guess="$(find "${DEST_ROOT}" -maxdepth 1 -type f -name '*.html' | head -n1 || true)"
    if [[ -n "${html_guess}" ]]; then
      echo "Opening: ${html_guess}"
      open_html "${html_guess}"
    else
      echo "No HTML results found to open (looked under ${DEST_ROOT})."
    fi
  fi
else
  echo "Benchmark script not found at ${BENCH_PY} — skipping evo benchmark."
fi

overall_end=$(date +%s)
overall_dur=$(( overall_end - overall_start ))
echo "============================================================"
echo "[DONE] Processed ${seq_count} sequence(s)."
echo "[OUT]  Per-sequence: ${ROOT_OUT}/<sequence>/{groundtruth.txt, vipe_estimate_tum.txt, vipe_infer*.log, vipe_raw/...}"
echo "[OUT]  Benchmark:    ${DEST_ROOT}/<sequence>/{groundtruth.txt, vipe_estimate_tum.txt}"
echo "[NOTE] Result HTML (if created): ${RESULT_HTML}"
echo "[TOTAL] duration_s=${overall_dur}"

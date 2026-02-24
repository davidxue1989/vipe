#!/usr/bin/env bash
set -euo pipefail

# Use PyTorch's segmented allocator to reduce large contiguous alloc failures
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------------------------------------
# Run ViPE on Freiburg3 sequences (that have ./rgb/video.mp4),
# then convert vipe_raw/pose/video.npz (N x 4 x 4) into TUM format:
#   timestamp tx ty tz qx qy qz qw
# (timestamps from rgb.txt) and write with a header.
#
# Also builds a "poses_and_groundtruth" tree compatible with your
# updated evo benchmark script (benchmark_dx.py) which expects:
#   DEST_ROOT = <output_root>/poses_and_groundtruth/<seq>/
#                 - vipe_estimate_tum.txt  (prediction, with header)
#                 - groundtruth.txt        (copied)
#   (We also write vipe_poses.txt for convenience/back-compat.)
#
# Usage:
#   bash run_vipe_freiburg3.sh [-e EXPNAME] [-p PIPELINE] [-v]
#                              [--input-root PATH]
#                              [--only NAME ...] [--only-file FILE]
#                              [--skip NAME ...] [--skip-file FILE]
#                              [--poses-kind twc|tcw]
#                              [--reprocess] [--skip-infer] [--scale-all]
#                              [--bench-py PATH] [--no-browser] [--plots]
#
# Examples:
#   bash run_vipe_freiburg3.sh -p default
#   bash run_vipe_freiburg3.sh -e vipeA -p default -v --only rgbd_dataset_freiburg3_walking_halfsphere
#   bash run_vipe_freiburg3.sh --poses-kind tcw --plots
#   bash run_vipe_freiburg3.sh --skip rgbd_dataset_freiburg3_walking_static
# ------------------------------------------------------------

# Defaults
ROOT_IN="/DiskBadem_1/data/TUM-RGBD/rgbd_dataset/dataset/freiburg3"
EXPNAME=""             # optional; if empty => output_vipe (no postfix)
PIPELINE="default"
VISUALIZE=0
ONLY_LIST=()           # exact folder names to include
ONLY_FILE=""           # file with names (one per line)
SKIP_LIST=()           # exact folder names to exclude
SKIP_FILE=""           # file with names (one per line)
POSES_KIND="twc"       # or 'tcw' (if tcw, we invert per-frame)
REPROCESS=0            # if 1, always run vipe infer even if outputs exist
SKIP_INFER=0           # NEW: if 1, skip vipe infer entirely
BENCH_PY="/home/dxue/workspaces/isaac_ros-dev/src/rss-ros-envs/scripts/evo_benchmark/benchmark_dx.py"
NO_BROWSER=0           # 1 => don't open result HTML
SHOW_PLOTS=0           # 1 => show evo plots
SCALE_ALL=0            # NEW: pass --scale-all to benchmark script

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
    --skip-infer|--no-infer) SKIP_INFER=1; shift;;   # NEW
    --bench-py) BENCH_PY="${2:-$BENCH_PY}"; shift 2;;
    --no-browser) NO_BROWSER=1; shift;;
    --plots) SHOW_PLOTS=1; shift;;
    --scale-all) SCALE_ALL=1; shift;;                # NEW
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

# Output root: with or without postfix based on -e
if [[ -n "${EXPNAME}" ]]; then
  ROOT_OUT="./output_vipe_${EXPNAME}"
else
  ROOT_OUT="./output_vipe"
fi
mkdir -p "${ROOT_OUT}"

DEST_ROOT="${ROOT_OUT}/poses_and_groundtruth"
mkdir -p "${DEST_ROOT}"

# --- Python helper: NPZ (video.npz) + rgb.txt -> TUM trajectory (with header) ---
PY_HELPER="${ROOT_OUT}/npz_to_tum.py"
cat > "${PY_HELPER}" << 'PYCODE'
#!/usr/bin/env python3
import argparse, sys, os, numpy as np

def load_rgb_timestamps(rgb_txt_path):
    ts = []
    with open(rgb_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            try:
                ts.append(float(parts[0]))
            except Exception:
                continue
    return ts

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
    ap.add_argument('--npz', required=True, help='Path to vipe_raw/pose/video.npz')
    ap.add_argument('--rgb', required=True, help='Path to rgb.txt in the sequence folder')
    ap.add_argument('--out', required=True, help='Output TUM trajectory path')
    ap.add_argument('--poses-kind', dest='poses_kind', choices=['twc','tcw'], default='twc')
    ap.add_argument('--seq-name', default='', help='Sequence name to include in header (optional)')
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        arr = None
        for k in ['poses','T','arr_0','pose','video']:
            if k in data:
                arr = data[k]; break
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

    ts = load_rgb_timestamps(args.rgb)
    if len(ts) == 0:
        print(f"[ERR] No timestamps found in {args.rgb}", file=sys.stderr); sys.exit(2)

    n = min(len(ts), arr.shape[0])
    if arr.shape[0] != len(ts):
        print(f"[WARN] Frames mismatch: npz={arr.shape[0]} vs rgb.txt={len(ts)}; using first {n}", file=sys.stderr)

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
chmod +x "${PY_HELPER}"

# --- Helpers ---
find_rgb_txt() {
  local base="$1"
  local candidates=("${base}/rgb.txt" "${base}/rgb/rgb.txt")
  for p in "${candidates[@]}"; do
    [[ -f "$p" ]] && { echo "$p"; return 0; }
  done
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

echo "[INFO] Input root: ${ROOT_IN}"
echo "[INFO] Output root: ${ROOT_OUT}"
echo "[INFO] Pipeline: ${PIPELINE}"
echo "[INFO] Visualize: $([[ ${VISUALIZE} -eq 1 ]] && echo ON || echo OFF)"
echo "[INFO] Poses-kind: ${POSES_KIND} (twc = camera in world; tcw = invert)"
echo "[INFO] Reprocess: $([[ ${REPROCESS} -eq 1 ]] && echo ON || echo OFF)"
echo "[INFO] Skip infer: $([[ ${SKIP_INFER} -eq 1 ]] && echo ON || echo OFF)"   # NEW
echo "[INFO] Benchmark py: ${BENCH_PY}"
echo "[INFO] Scale-all APE: $([[ ${SCALE_ALL} -eq 1 ]] && echo ON || echo OFF)" # NEW
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"                # NEW
echo "[INFO] Scanning for sequences with ./rgb/video.mp4 ..."

overall_start=$(date +%s)
shopt -s nullglob
seq_count=0
for seq_dir in "${ROOT_IN}"/*; do
  [[ -d "${seq_dir}" ]] || continue
  seq_name="$(basename "${seq_dir}")"

  # Exact-match include filter (if provided)
  if [[ ${#ONLY_LIST[@]} -gt 0 && -z "${ONLY_SET[$seq_name]+x}" ]]; then
    continue
  fi
  # Exact-match skip filter (always checked if provided)
  if [[ ${#SKIP_LIST[@]} -gt 0 && -n "${SKIP_SET[$seq_name]+x}" ]]; then
    echo "[SKIP] ${seq_name}: matched --skip/--skip-file"
    continue
  fi

  video="${seq_dir}/rgb/video.mp4"
  gt_src="${seq_dir}/groundtruth.txt"
  rgb_txt="$(find_rgb_txt "${seq_dir}")" || rgb_txt=""
  if [[ -f "${video}" && -f "${gt_src}" && -n "${rgb_txt}" ]]; then
    out_seq_dir="${ROOT_OUT}/${seq_name}"
    raw_dir="${out_seq_dir}/vipe_raw"
    mkdir -p "${raw_dir}"

    echo "------------------------------------------------------------"
    echo "[SEQ] ${seq_name}"

    # Decide whether to run vipe infer
    npz_path="${raw_dir}/pose/video.npz"
    run_infer=1
    if [[ ${SKIP_INFER} -eq 1 ]]; then
      run_infer=0
      echo "[SKIP] vipe infer: user specified --skip-infer; attempting post-process only." | tee -a "${out_seq_dir}/vipe_infer.log"
    elif [[ ${REPROCESS} -eq 0 && -f "${npz_path}" ]]; then
      run_infer=0
      echo "[SKIP] vipe infer: ${npz_path} already exists (use --reprocess to force)." | tee -a "${out_seq_dir}/vipe_infer.log"
    fi

    log="${out_seq_dir}/vipe_infer.log"
    [[ -f "${log}" ]] || : > "${log}"

    if [[ ${run_infer} -eq 1 ]]; then
      cmd=(vipe infer -p "${PIPELINE}" -o "${raw_dir}")
      [[ ${VISUALIZE} -eq 1 ]] && cmd+=(-v)
      cmd+=("${video}")

      start_human="$(date -Iseconds)"
      start_epoch=$(date +%s)
      echo "[TIME] start=${start_human} epoch=${start_epoch}" | tee "${log}"
      echo "[ENV ] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" | tee -a "${log}"
      echo "[RUN]  ${cmd[*]}" | tee -a "${log}"

      if command -v /usr/bin/time >/dev/null 2>&1; then
        { /usr/bin/time -f "[TIME] elapsed=%E user=%U sys=%S maxrss_kb=%M" "${cmd[@]}"; } \
          2>&1 | tee -a "${log}"
      else
        "${cmd[@]}" 2>&1 | tee -a "${log}"
      fi

      end_human="$(date -Iseconds)"
      end_epoch=$(date +%s)
      dur=$(( end_epoch - start_epoch ))
      printf "[TIME] end=%s epoch=%d duration_s=%d\n" "$end_human" "$end_epoch" "$dur" | tee -a "${log}"
    fi

    # Copy groundtruth to per-seq output folder (always)
    cp -f "${gt_src}" "${out_seq_dir}/groundtruth.txt"
    echo "[OK] Copied groundtruth -> ${out_seq_dir}/groundtruth.txt" | tee -a "${log}"

    # Convert NPZ + rgb.txt -> TUM (always attempt if npz exists)
    if [[ ! -f "${npz_path}" ]]; then
      echo "[WARN] Missing ${npz_path}. Prediction conversion skipped." | tee -a "${log}"
    else
      pred_out="${out_seq_dir}/vipe_estimate_tum.txt"
      echo "[CONVERT] ${npz_path} + ${rgb_txt} -> ${pred_out} (poses-kind=${POSES_KIND})" | tee -a "${log}"
      python "${PY_HELPER}" --npz "${npz_path}" --rgb "${rgb_txt}" \
                            --out "${pred_out}" --poses-kind "${POSES_KIND}" \
                            --seq-name "${seq_name}" \
                            2>&1 | tee -a "${log}"
    fi

    # Mirror into benchmark layout expected by benchmark_dx.py:
    # <output_root>/poses_and_groundtruth/<seq>/{vipe_estimate_tum.txt, groundtruth.txt}
    dest_dir="${DEST_ROOT}/${seq_name}"
    mkdir -p "${dest_dir}"
    cp -f "${out_seq_dir}/groundtruth.txt" "${dest_dir}/groundtruth.txt"
    if [[ -f "${out_seq_dir}/vipe_estimate_tum.txt" ]]; then
      cp -f "${out_seq_dir}/vipe_estimate_tum.txt" "${dest_dir}/vipe_estimate_tum.txt"
    fi

    seq_count=$((seq_count+1))
  else
    [[ -f "${video}" ]] || echo "[SKIP] ${seq_name}: missing ./rgb/video.mp4"
    [[ -f "${gt_src}" ]] || echo "[SKIP] ${seq_name}: missing ./groundtruth.txt"
    [[ -n "${rgb_txt}" ]] || echo "[SKIP] ${seq_name}: missing rgb.txt"
  fi
done
shopt -u nullglob

# -----------------------------
# Run evo benchmark (if present)
# -----------------------------
RESULT_HTML="${DEST_ROOT}/benchmark_results.html"
if [[ -f "${BENCH_PY}" ]]; then
  echo "Running benchmark: ${BENCH_PY}"

  BENCH_ARGS=()
  (( SHOW_PLOTS == 1 )) && BENCH_ARGS+=( --plots )
  (( SCALE_ALL == 1 )) && BENCH_ARGS+=( --scale-all )
  if [[ ${#ONLY_LIST[@]} -gt 0 ]]; then
    for tok in "${ONLY_LIST[@]}"; do
      BENCH_ARGS+=( --plot-only "$tok" )
    done
  fi

  if ! python3 "${BENCH_PY}" "${DEST_ROOT}" "${BENCH_ARGS[@]}"; then
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
overall_dur=$(( overall_end - overall_start ))   # fixed arithmetic
echo "============================================================"
echo "[DONE] Processed ${seq_count} sequence(s)."
echo "[OUT]  Per-sequence: ${ROOT_OUT}/<sequence>/{groundtruth.txt, vipe_estimate_tum.txt, vipe_infer.log, vipe_raw/...}"
echo "[OUT]  Benchmark:    ${DEST_ROOT}/<sequence>/{groundtruth.txt, vipe_estimate_tum.txt}"
echo "[NOTE] Result HTML (if created): ${RESULT_HTML}"
echo "[TOTAL] duration_s=${overall_dur}"

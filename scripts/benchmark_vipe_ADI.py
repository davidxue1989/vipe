#!/usr/bin/env python3
"""
ADI ViPE Runner & Benchmark (Python version)

This script processes visual odometry sequences through ViPE inference,
converts outputs to TUM format, optionally runs evo benchmarking, and
uploads output to S3 for easy transfer to other devices.

Supports both chunked sequences (images/part_XXXX/) and unchunked sequences.
Reads camera intrinsics from camera_original.yaml when available.

Example Usage:
--------------
# Basic run: infer only for sequences missing .npz, then post-process all:
    python benchmark_vipe_ADI.py --input-root /path/to/dataset -p ADI

# Specify experiment name (used as S3 subfolder) and custom local output dir:
    python benchmark_vipe_ADI.py --input-root /path/to/dataset -p ADI \
        -e adi_feb_2026 -o /home/ubuntu/results/vipe_adi_feb_2026

# Process specific sequences only:
    python benchmark_vipe_ADI.py --input-root /path/to/dataset --only seq1 seq2

# Force re-inference for ALL sequences (even if .npz exists):
    python benchmark_vipe_ADI.py --input-root /path/to/dataset --reprocess

# Skip ALL inference, only post-process sequences that have .npz:
    python benchmark_vipe_ADI.py --input-root /path/to/dataset --skip-infer

# Full example with custom paths and benchmarking:
    python scripts/benchmark_vipe_ADI.py \
        -e adi_gt_intr \
        -p ADI \
        --input-root /home/ubuntu/data/ADI_data_feb_2026_extracted \
        --scale-all

# Disable S3 upload:
    python benchmark_vipe_ADI.py --input-root /path/to/dataset -p ADI --no-s3-upload

# Upload existing results to S3 without re-processing:
    python benchmark_vipe_ADI.py -e adi_feb_2026 -o /path/to/existing/output --upload-only

# Use custom S3 settings:
    python benchmark_vipe_ADI.py --input-root /path/to/dataset -p ADI \
        --s3-bucket my-bucket --s3-prefix my/prefix --s3-profile my-profile

# Open browser to view benchmark results:
    python benchmark_vipe_ADI.py --input-root /path/to/dataset --browser
"""
import os
import sys
import glob
import shutil
import argparse
import subprocess
import time
import re
import math
import logging
from datetime import datetime
from pathlib import Path

import hashlib
import json

import numpy as np
import yaml
import hydra
from vipe import get_config_path, make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils.logging import configure_logging

# -------------------------------------------------------------------------
# CONSTANTS & DEFAULTS
# -------------------------------------------------------------------------
DEFAULT_ROOT_IN = "/home/ubuntu/data/ADI_data_feb_2026_extracted"
DEFAULT_BENCH_PY = "/home/dxue/workspaces/isaac_ros-dev/src/rss-ros-envs/scripts/evo_benchmark/benchmark_dx.py"
ENV_VARS = os.environ.copy()
# Set PyTorch allocation config to match bash script
ENV_VARS["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup logging to print to stderr/stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# S3 UPLOAD HELPERS
# -------------------------------------------------------------------------

def compute_file_md5(filepath):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def s3_file_matches(local_path, bucket, key, profile):
    """Check if S3 file matches local file (using ETag/MD5).

    Returns True if files match (no upload needed), False otherwise.
    """
    try:
        result = subprocess.run(
            ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key, "--profile", profile],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return False

        metadata = json.loads(result.stdout)
        etag = metadata.get('ETag', '').strip('"')

        # Multipart uploads have ETags with '-' suffix; can't compare easily
        if '-' in etag:
            return False

        local_md5 = compute_file_md5(local_path)
        return local_md5 == etag
    except Exception:
        return False


def upload_to_s3(local_dir, bucket, prefix, profile, timeout=1800):
    """Upload a local directory to S3 using 'aws s3 sync'.

    Returns True on success, False on failure.
    """
    s3_uri = f"s3://{bucket}/{prefix}"
    logger.info(f"\nUploading output folder to S3...")
    logger.info(f"  Local:   {local_dir}")
    logger.info(f"  Bucket:  {bucket}")
    logger.info(f"  Prefix:  {prefix}")
    logger.info(f"  Profile: {profile}")

    # Summarise what will be uploaded
    files = []
    for root, _dirs, fnames in os.walk(local_dir):
        for fn in fnames:
            files.append(os.path.join(root, fn))
    total_size = sum(os.path.getsize(f) for f in files)
    logger.info(f"  Files: {len(files)} ({total_size / (1024*1024):.1f} MB total)")

    try:
        result = subprocess.run(
            ["aws", "s3", "sync", local_dir, s3_uri, "--profile", profile],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            logger.info(f"  Upload successful!")
            logger.info(f"  Download with: aws s3 sync {s3_uri} {os.path.basename(local_dir)} --profile {profile}")
            return True
        else:
            logger.error(f"  Upload failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.error("  ERROR: AWS CLI not installed. Install with: pip install awscli")
        return False
    except Exception as e:
        logger.error(f"  Upload error: {e}")
        return False


# -------------------------------------------------------------------------
# MATH & CONVERSION HELPERS (Integrated from the embedded script)
# -------------------------------------------------------------------------

def load_timestamps_from_csv(path):
    """
    Returns list of timestamps in SECONDS (float).
    Parses comma or whitespace separated files. Handles ns -> sec conversion logic.
    """
    ts_sec = []
    if not os.path.isfile(path):
        return ts_sec

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Remove inline comments
            line = line.split('#', 1)[0].strip()
            if not line:
                continue

            # Split logic
            if ',' in line:
                parts = line.split(',')
                # Try to get timestamp_sec (last column) if it exists, otherwise first column
                if len(parts) >= 3:
                    tok = parts[2].strip()  # timestamp_sec column
                elif len(parts) >= 2:
                    tok = parts[1].strip()  # timestamp_nsec or second column
                else:
                    tok = parts[0].strip()  # fallback to first column
            else:
                parts = line.split()
                if not parts:
                    continue
                tok = parts[0].strip()

            if tok.lower().startswith('timestamp') or tok.lower() == 'frame':
                continue

            # Numeric check
            if not re.fullmatch(r'[+-]?\d+(\.\d+)?', tok):
                continue

            try:
                if '.' in tok:
                    ts = float(tok)
                else:
                    val = int(tok)
                    # Heuristic: >1e12 treated as nanoseconds
                    ts = val / 1e9 if abs(val) > 1_000_000_000_000 else float(val)
                ts_sec.append(ts)
            except ValueError:
                continue
    return ts_sec

def mat_to_quat_xyzw(R):
    """Converts 3x3 rotation matrix to quaternion [x, y, z, w]."""
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
    norm = np.linalg.norm(q)
    return q / (norm + 1e-12)

def invert_h(T):
    """Inverts a 4x4 homogeneous transformation matrix."""
    R = T[:3,:3]
    t = T[:3,3]
    Rt = R.T
    tinv = -Rt @ t
    Tinv = np.eye(4, dtype=T.dtype)
    Tinv[:3,:3] = Rt
    Tinv[:3,3]  = tinv
    return Tinv

def convert_npz_to_tum(npz_path, csv_path, out_path, poses_kind='twc', seq_name=''):
    """
    Reads ViPE .npz and timestamp.csv, writes TUM trajectory file.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        arr = None
        # Try finding the pose array
        potential_keys = ['poses', 'T', 'arr_0', 'pose', 'video']
        if isinstance(data, np.lib.npyio.NpzFile):
            for k in potential_keys:
                if k in data:
                    arr = data[k]
                    break
            if arr is None and len(data.files) > 0:
                arr = data[data.files[0]]
        else:
            arr = data
        
        arr = np.array(arr)
        if arr.ndim != 3 or arr.shape[1:] != (4,4):
            logger.error(f"[ERR] Expected shape (N,4,4), got {arr.shape} in {npz_path}")
            return False

        ts = load_timestamps_from_csv(csv_path)
        if not ts:
            logger.error(f"[ERR] No usable timestamps found in {csv_path}")
            return False

        n = min(len(ts), arr.shape[0])
        if arr.shape[0] != len(ts):
            logger.warning(f"[WARN] Frames mismatch: npz={arr.shape[0]} vs csv={len(ts)}; using first {n}")

        with open(out_path, 'w') as f:
            f.write("# ViPE predicted trajectory (TUM format)\n")
            if seq_name:
                f.write(f"# sequence: '{seq_name}'\n")
            f.write("# columns: timestamp tx ty tz qx qy qz qw\n")
            
            for i in range(n):
                T = arr[i]
                if poses_kind == 'tcw':
                    T = invert_h(T)
                
                # Extract translation and rotation
                t = T[:3, 3].astype(np.float64)
                R = T[:3, :3].astype(np.float64)
                qx, qy, qz, qw = mat_to_quat_xyzw(R)
                
                f.write(f"{ts[i]:.6f} {t[0]:.7f} {t[1]:.7f} {t[2]:.7f} {qx:.7f} {qy:.7f} {qz:.7f} {qw:.7f}\n")
        return True

    except Exception as e:
        logger.error(f"[ERR] Conversion exception: {e}")
        return False

# -------------------------------------------------------------------------
# SYSTEM HELPERS
# -------------------------------------------------------------------------

def find_pose_npz(raw_dir):
    """
    Looks for valid .npz files in raw_dir/pose/ or raw_dir/pose/video.npz.
    Returns path or None.
    """
    pdir = os.path.join(raw_dir, "pose")
    if os.path.isdir(pdir):
        # Sort by modification time, newest first
        npzs = glob.glob(os.path.join(pdir, "*.npz"))
        if npzs:
            npzs.sort(key=os.path.getmtime, reverse=True)
            return npzs[0]
            
    video_npz = os.path.join(raw_dir, "pose", "video.npz")
    if os.path.isfile(video_npz):
        return video_npz
    
    return None

def open_html_in_browser(path, no_browser=False):
    if no_browser:
        logger.info(f"Browser opening disabled. Result ready at: {path}")
        return

    # Cross-platform-ish attempt
    if shutil.which("google-chrome"):
        subprocess.call(["google-chrome", "--no-sandbox", path], stderr=subprocess.DEVNULL)
    elif shutil.which("xdg-open"):
        subprocess.call(["xdg-open", path], stderr=subprocess.DEVNULL)
    else:
        logger.info(f"Result ready at: {path} (no browser opener found)")

def run_command_with_tee(cmd, log_file, env=None, verbose=True):
    """
    Run a command, writing output to both terminal (if verbose) and log file.
    Returns the exit code.
    """
    with open(log_file, "a") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1  # Line buffered
        )
        for line in process.stdout:
            f.write(line)
            if verbose:
                print(line, end='', flush=True)
        process.wait()
        return process.returncode

def get_chunk_ids(seq_dir):
    """Returns sorted list of chunk IDs (strings '0001', '0002') based on timestamp_*.csv presence."""
    ids = []
    for f in glob.glob(os.path.join(seq_dir, "timestamp_*.csv")):
        fname = os.path.basename(f)
        match = re.match(r'^timestamp_(\d{4})\.csv$', fname)
        if match:
            ids.append(match.group(1))
    return sorted(ids)

def is_chunked_seq(seq_dir):
    img_dir = os.path.join(seq_dir, "images")
    if not os.path.exists(img_dir): 
        return False
    parts = glob.glob(os.path.join(img_dir, "part_*"))
    if not parts:
        return False
    
    # Must have at least one timestamp csv
    ts_csvs = glob.glob(os.path.join(seq_dir, "timestamp_*.csv"))
    return len(ts_csvs) > 0

def read_camera_intrinsics(seq_dir):
    """
    Read camera intrinsics from camera_original.yaml in the sequence directory.
    Returns (intrinsics, distortion) where:
      - intrinsics: list of [fx, cx, fy, cy] or None
      - distortion: list of distortion coefficients or None
    """
    cam_yaml_path = os.path.join(seq_dir, "camera_original.yaml")
    if not os.path.isfile(cam_yaml_path):
        return None, None
    
    try:
        with open(cam_yaml_path, 'r') as f:
            cam_data = yaml.safe_load(f)
        
        if 'cam0' not in cam_data:
            return None, None
        
        cam0 = cam_data['cam0']
        intrinsics = None
        distortion = None
        
        # Extract intrinsics and convert to ViPE format
        # camera_original.yaml has: [fx, cx, fy, cy]
        # ViPE expects: [fx, fy, cx, cy]
        if 'intrinsics' in cam0 and 'data' in cam0['intrinsics']:
            data = cam0['intrinsics']['data']
            if len(data) >= 4:
                fx, cx, fy, cy = float(data[0]), float(data[1]), float(data[2]), float(data[3])
                # Reorder to ViPE format: [fx, fy, cx, cy]
                intrinsics = [fx, fy, cx, cy]
        
        # Extract distortion coefficients
        if 'distortion_coefficients' in cam0 and 'data' in cam0['distortion_coefficients']:
            distortion = [float(x) for x in cam0['distortion_coefficients']['data']]
        
        return intrinsics, distortion
    except Exception as e:
        logger.warning(f"Failed to read camera_original.yaml: {e}")
        return None, None

def run_vipe_inference(img_dir, raw_dir, pipeline, visualize, seq_dir, no_override_intrinsics=False):
    """
    Run ViPE inference directly using the Python API.
    
    Args:
        img_dir: Path to the image directory for this sequence/chunk.
        raw_dir: Path to write ViPE raw outputs.
        pipeline: Name of the ViPE pipeline config (e.g. "ADI").
        visualize: Whether to enable ViPE visualization.
        seq_dir: Path to the sequence directory (used to find camera_original.yaml).
        no_override_intrinsics: If True, do NOT override intrinsics_gt in the
            pipeline config with values from camera_original.yaml. The pipeline
            config values (or lack thereof) will be used as-is. Default False
            (YAML intrinsics override config when available).
    
    Returns True on success, False on failure.
    """
    try:
        # Read camera intrinsics if available
        intrinsics, distortion = read_camera_intrinsics(seq_dir)
        
        # Build overrides
        overrides = [
            f"pipeline={pipeline}",
            f"pipeline.output.path={raw_dir}",
            "pipeline.output.save_artifacts=true"
        ]
        
        # Check if pipeline config already has intrinsics_gt
        with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
            base_cfg = hydra.compose("default", overrides=[f"pipeline={pipeline}"])
        
        has_intrinsics_gt = hasattr(base_cfg.pipeline.slam, 'intrinsics_gt') and base_cfg.pipeline.slam.intrinsics_gt is not None
        
        # Intrinsics override logic:
        #   --no-override-intrinsics: never inject YAML intrinsics; use pipeline config as-is
        #   default:                  YAML intrinsics override config (use + if key missing,
        #                             plain override if key exists)
        if no_override_intrinsics:
            if intrinsics is not None:
                logger.info(f"Intrinsics found in camera_original.yaml but --no-override-intrinsics is set; using pipeline config as-is")
        elif intrinsics is not None:
            intrinsics_str = "[" + ",".join(str(v) for v in intrinsics) + "]"
            if has_intrinsics_gt:
                # Key already exists in config — plain override replaces it
                overrides.append(f"pipeline.slam.intrinsics_gt={intrinsics_str}")
                logger.info(f"Overriding config intrinsics_gt with camera_original.yaml: {intrinsics_str}")
            else:
                # Key does not exist in config — use + prefix to add it
                overrides.append(f"+pipeline.slam.intrinsics_gt={intrinsics_str}")
                logger.info(f"Adding intrinsics_gt from camera_original.yaml: {intrinsics_str}")
        else:
            if has_intrinsics_gt:
                logger.info(f"No camera_original.yaml intrinsics; using pipeline config intrinsics_gt")
            else:
                logger.warning(f"No intrinsics found in camera_original.yaml or pipeline config")
        
        # Note: ViPE's pinhole camera model does NOT support distortion coefficients.
        # The SLAM system expects only [fx, fy, cx, cy] for pinhole cameras.
        # Distortion coefficients from camera_original.yaml are intentionally ignored.
        
        if visualize:
            overrides.append("pipeline.output.save_viz=true")
            overrides.append("pipeline.slam.visualize=true")
        else:
            overrides.append("pipeline.output.save_viz=true")
        
        # Frame directory stream configuration
        overrides.extend([
            "streams=frame_dir_stream",
            f"streams.base_path={img_dir}"
        ])
        
        # Initialize Hydra config with final overrides
        with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
            args = hydra.compose("default", overrides=overrides)
        
        # Create pipeline
        vipe_pipeline = make_pipeline(args.pipeline)
        
        # Create video stream
        video_stream = ProcessedVideoStream(FrameDirStream(Path(img_dir)), []).cache(desc="Reading image frames")
        
        # Run pipeline
        vipe_pipeline.run(video_stream)
        
        return True
    except Exception as e:
        logger.error(f"ViPE inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# -------------------------------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ADI ViPE Runner & Benchmark")
    parser.add_argument('-e', '--expname', default="",
                        help='Experiment name used as the S3 subfolder under --s3-prefix '
                             '(e.g. s3://<bucket>/lab-data/vipe-results/<expname>/). '
                             'Falls back to the local output directory name if not set.')
    parser.add_argument('-p', '--pipeline', default="ADI", help="ViPE pipeline config name")
    parser.add_argument('-v', '--vis', '--visualize', action='store_true', dest='visualize', help="Enable ViPE visualization")
    parser.add_argument('--input-root', default=DEFAULT_ROOT_IN, help="Input dataset root")
    parser.add_argument('-o', '--output-dir', default='./output_vipe',
                        help='Local output directory path (default: ./output_vipe)')
    parser.add_argument('--only', nargs='+', default=[], help="Process only these sequence names")
    parser.add_argument('--only-file', help="File containing list of sequences to process")
    parser.add_argument('--skip', nargs='+', default=[], help="Skip these sequence names")
    parser.add_argument('--skip-file', help="File containing list of sequences to skip")
    parser.add_argument('--poses-kind', default="twc", choices=['twc', 'tcw'], help="Pose format: twc (camera in world) or tcw (world in camera)")
    # Inference control (mutually exclusive in practice):
    #   Default:      Infer only if .npz is missing; post-process all sequences.
    #   --reprocess:  Force re-inference for ALL sequences, even if .npz exists.
    #   --skip-infer: Never run inference; only post-process sequences with existing .npz.
    parser.add_argument('--reprocess', action='store_true',
                        help="Force re-inference for all sequences (even if .npz exists)")
    parser.add_argument('--skip-infer', '--no-infer', action='store_true', dest='skip_infer',
                        help="Skip all inference; only post-process sequences with existing .npz")
    
    parser.add_argument('--bench-py', default=DEFAULT_BENCH_PY, help="Path to benchmark_dx.py")
    parser.add_argument('--browser', action='store_true', help="Open browser after benchmark (default: no browser)")
    parser.add_argument('-q', '--quiet', action='store_true', help="Suppress vipe inference output in terminal (still logs to file)")
    parser.add_argument('--plots', action='store_true', dest='show_plots', help="Pass --plots to benchmark script")
    parser.add_argument('--scale-all', action='store_true', help="Pass --scale-all to benchmark script")

    # Intrinsics override control
    parser.add_argument('--no-override-intrinsics', action='store_true', dest='no_override_intrinsics',
                        help='Do NOT override pipeline config intrinsics_gt with values from '
                             'camera_original.yaml. By default (flag absent), per-sequence YAML '
                             'intrinsics take priority over the pipeline config.')

    # S3 upload options
    parser.add_argument('--no-s3-upload', action='store_true',
                        help='Disable S3 upload of output folder')
    parser.add_argument('--s3-bucket', default='rss-slam',
                        help='S3 bucket name for upload (default: rss-slam)')
    parser.add_argument('--s3-prefix', default='lab-data/vipe-results',
                        help='S3 key prefix/folder (default: lab-data/vipe-results)')
    parser.add_argument('--s3-profile', default='rss',
                        help='AWS CLI profile for S3 upload (default: rss)')
    parser.add_argument('--upload-only', action='store_true',
                        help='Skip processing, just upload existing output folder to S3')

    args = parser.parse_args()

    # 1. Output Roots
    # If --expname is set, append it to the output dir (e.g. ./output_vipe/adi_gt_intr/)
    # so both local and S3 paths use the same experiment name.
    if args.expname:
        root_out = os.path.join(args.output_dir, args.expname)
    else:
        root_out = args.output_dir
    s3_expname = args.expname if args.expname else os.path.basename(os.path.abspath(root_out))
    
    dest_root = os.path.join(root_out, "poses_and_groundtruth")

    # Handle --upload-only: skip all processing, just upload existing output
    if args.upload_only:
        if not os.path.isdir(root_out):
            logger.error(f"Output directory not found: {root_out}")
            sys.exit(1)
        s3_prefix = f"{args.s3_prefix}/{s3_expname}"
        success = upload_to_s3(root_out, args.s3_bucket, s3_prefix, args.s3_profile)
        sys.exit(0 if success else 1)

    os.makedirs(root_out, exist_ok=True)
    os.makedirs(dest_root, exist_ok=True)

    # 2. Filter Lists
    only_set = set(args.only)
    if args.only_file and os.path.exists(args.only_file):
        with open(args.only_file, 'r') as f:
            for line in f:
                if line.strip(): only_set.add(line.strip())
    
    skip_set = set(args.skip)
    if args.skip_file and os.path.exists(args.skip_file):
        with open(args.skip_file, 'r') as f:
            for line in f:
                if line.strip(): skip_set.add(line.strip())

    # 3. Info Dump
    logger.info(f"[INFO] Input root: {args.input_root}")
    logger.info(f"[INFO] Output root: {root_out}")
    logger.info(f"[INFO] Pipeline: {args.pipeline}")
    logger.info(f"[INFO] Visualize: {'ON' if args.visualize else 'OFF'}")
    logger.info(f"[INFO] Poses-kind: {args.poses_kind} (twc = camera in world; tcw = invert)")
    infer_mode = "SKIP ALL" if args.skip_infer else ("FORCE ALL" if args.reprocess else "IF MISSING")
    logger.info(f"[INFO] Inference mode: {infer_mode}")
    logger.info(f"[INFO] Verbose output: {'OFF' if args.quiet else 'ON'}")
    logger.info(f"[INFO] Benchmark py: {args.bench_py}")
    logger.info(f"[INFO] Open browser: {'ON' if args.browser else 'OFF'}")
    logger.info(f"[INFO] Scale-all APE: {'ON' if args.scale_all else 'OFF'}")
    logger.info(f"[INFO] S3 upload: {'OFF' if args.no_s3_upload else 'ON'} (bucket={args.s3_bucket}, prefix={args.s3_prefix}, profile={args.s3_profile})")
    logger.info(f"[INFO] PYTORCH_CUDA_ALLOC_CONF={ENV_VARS.get('PYTORCH_CUDA_ALLOC_CONF', '')}")
    logger.info(f"[INFO] Scanning for sequences with required files ...")

    overall_start = time.time()
    seq_count = 0
    
    # Get all subdirectories in input root
    if not os.path.isdir(args.input_root):
        logger.error(f"Input root does not exist: {args.input_root}")
        sys.exit(1)

    sequences = sorted([d for d in os.listdir(args.input_root) if os.path.isdir(os.path.join(args.input_root, d))])

    for seq_name in sequences:
        # Filter Logic
        if only_set and seq_name not in only_set:
            continue
        if skip_set and seq_name in skip_set:
            logger.info(f"[SKIP] {seq_name}: matched skip list")
            continue

        seq_dir = os.path.join(args.input_root, seq_name)
        img_dir = os.path.join(seq_dir, "images")
        ts_csv = os.path.join(seq_dir, "timestamp.csv")
        gt_txt = os.path.join(seq_dir, "groundtruth.txt")

        if not os.path.isdir(img_dir):
            logger.warning(f"[SKIP] {seq_name}: missing images/")
            continue
        if not os.path.isfile(gt_txt):
            logger.warning(f"[SKIP] {seq_name}: missing groundtruth.txt")
            continue

        # Prepare outputs
        out_seq_dir = os.path.join(root_out, seq_name)
        raw_dir = os.path.join(out_seq_dir, "vipe_raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        main_log_file = os.path.join(out_seq_dir, "vipe_infer.log")
        # Ensure log exists
        Path(main_log_file).touch()

        def log_msg(msg):
            print(msg)
            with open(main_log_file, "a") as f:
                f.write(msg + "\n")

        log_msg("------------------------------------------------------------")
        log_msg(f"[SEQ] {seq_name}")

        chunked = is_chunked_seq(seq_dir)
        
        # ==================== CHUNKED MODE ====================
        if chunked:
            log_msg("[MODE] Chunked sequence detected.")
            chunk_ids = get_chunk_ids(seq_dir)
            if not chunk_ids:
                log_msg(f"[ERR] No timestamp_*.csv found despite part folders. Skipping.")
                continue

            # Per-chunk processing
            for cid in chunk_ids:
                part_dir = os.path.join(img_dir, f"part_{cid}")
                tsc = os.path.join(seq_dir, f"timestamp_{cid}.csv")
                
                if not os.path.isdir(part_dir) or not os.path.isfile(tsc):
                    log_msg(f"[WARN] Missing inputs for chunk {cid}, skipping.")
                    continue

                raw_part = os.path.join(raw_dir, f"part_{cid}")
                os.makedirs(raw_part, exist_ok=True)
                part_log = os.path.join(out_seq_dir, f"vipe_infer_part_{cid}.log")
                Path(part_log).touch()

                # Determine if we need to run inference
                # Check for existing npz
                npz_path = find_pose_npz(raw_part)
                
                # Decide whether to run inference:
                #   --skip-infer: never infer
                #   --reprocess: always infer
                #   default: infer only if npz missing
                should_infer = False
                if args.skip_infer:
                    should_infer = False
                elif args.reprocess:
                    should_infer = True
                elif not npz_path:
                    should_infer = True

                if should_infer:
                    # Run Inference
                    start_human = datetime.now().isoformat()
                    start_epoch = time.time()
                    
                    with open(part_log, "a") as plog:
                        plog.write(f"[TIME] start={start_human} epoch={int(start_epoch)}\n")
                        plog.write(f"[ENV ] PYTORCH_CUDA_ALLOC_CONF={ENV_VARS.get('PYTORCH_CUDA_ALLOC_CONF', '')}\n")
                        plog.write(f"[RUN]  Running ViPE inference on {part_dir}\n")
                    
                    log_msg(f"[RUN] Running ViPE inference on {part_dir}")
                    
                    try:
                        success = run_vipe_inference(part_dir, raw_part, args.pipeline, args.visualize, seq_dir, args.no_override_intrinsics)
                        if not success:
                            log_msg(f"[WARN] ViPE inference failed")
                            with open(part_log, "a") as plog:
                                plog.write(f"[WARN] ViPE inference failed\n")
                    except Exception as e:
                        with open(part_log, "a") as plog:
                            plog.write(f"[ERR] Exception during inference: {e}\n")
                        log_msg(f"[ERR] Exception during inference: {e}")

                    end_human = datetime.now().isoformat()
                    end_epoch = time.time()
                    dur = int(end_epoch - start_epoch)
                    with open(part_log, "a") as plog:
                        plog.write(f"[TIME] end={end_human} epoch={int(end_epoch)} duration_s={dur}\n")
                    log_msg(f"[TIME] Completed in {dur}s")

                    npz_path = find_pose_npz(raw_part)
                else:
                    if args.skip_infer:
                        msg = f"[SKIP] part_{cid} infer: --skip-infer specified"
                    elif npz_path:
                        msg = f"[SKIP] part_{cid} infer: npz exists ({npz_path}); use --reprocess to force"
                    else:
                        msg = f"[SKIP] part_{cid} infer: no npz and inference not requested"
                    log_msg(msg)
                    with open(part_log, "a") as plog: plog.write(msg + "\n")

                # Convert to TUM
                partial_out = os.path.join(out_seq_dir, f"vipe_estimate_tum_part_{cid}.txt")
                if npz_path:
                    if args.reprocess or not os.path.exists(partial_out):
                        log_msg(f"[PRED-part] Convert {npz_path} -> {partial_out}")
                        success = convert_npz_to_tum(npz_path, tsc, partial_out, args.poses_kind, f"{seq_name}:part_{cid}")
                        if not success:
                            log_msg(f"[WARN] part_{cid} conversion failed.")
                    else:
                        with open(part_log, "a") as plog: plog.write(f"[SKIP] conversion: {partial_out} exists\n")
                else:
                    with open(part_log, "a") as plog: plog.write(f"[WARN] No npz to convert for part_{cid}\n")

            # Stitching Logic
            gt_tum = os.path.join(out_seq_dir, "groundtruth.txt")
            log_msg(f"[GT] Copy {gt_txt} -> {gt_tum}")
            shutil.copy2(gt_txt, gt_tum)

            pred_out = os.path.join(out_seq_dir, "vipe_estimate_tum.txt")
            
            # Stitch files
            with open(pred_out, "w") as fout:
                fout.write("# ViPE predicted trajectory (TUM format)\n")
                fout.write(f"# sequence: '{seq_name}' (stitched)\n")
                fout.write("# columns: timestamp tx ty tz qx qy qz qw\n")
                
                appended_count = 0
                all_lines = []
                for cid in chunk_ids:
                    part_f = os.path.join(out_seq_dir, f"vipe_estimate_tum_part_{cid}.txt")
                    if os.path.exists(part_f):
                        with open(part_f, "r") as fin:
                            for line in fin:
                                if not line.startswith('#'):
                                    all_lines.append(line.strip())
                        appended_count += 1
                    else:
                        log_msg(f"[WARN] Missing partial file for part_{cid}")

                # Monotonic Sort
                # Parse lines to (timestamp, full_line) tuples
                parsed_lines = []
                for l in all_lines:
                    parts = l.split()
                    if parts:
                        try:
                            parsed_lines.append((float(parts[0]), l))
                        except ValueError:
                            pass
                
                # Sort by timestamp
                parsed_lines.sort(key=lambda x: x[0])
                
                for _, line in parsed_lines:
                    fout.write(line + "\n")
            
            if appended_count > 0:
                log_msg(f"[OK] Stitched {appended_count} chunk(s) into {pred_out}")
            else:
                log_msg(f"[WARN] No partial predictions stitched for {seq_name}; final prediction missing.")
                # Remove empty file like shell script does
                try:
                    os.remove(pred_out)
                except OSError:
                    pass

        # ==================== UNCHUNKED MODE ====================
        else:
            log_msg("[MODE] Unchunked sequence.")
            
            # Check for existing npz
            npz_path = find_pose_npz(raw_dir)
            
            # Decide whether to run inference:
            #   --skip-infer: never infer
            #   --reprocess: always infer  
            #   default: infer only if npz missing
            should_infer = False
            if args.skip_infer:
                log_msg("[SKIP] infer: --skip-infer specified")
            elif args.reprocess:
                should_infer = True
            elif npz_path:
                log_msg(f"[SKIP] infer: {npz_path} already exists; use --reprocess to force")
            else:
                should_infer = True

            if should_infer:
                log_msg(f"[RUN] Running ViPE inference on {img_dir}")
                start_epoch = time.time()
                
                try:
                    success = run_vipe_inference(img_dir, raw_dir, args.pipeline, args.visualize, seq_dir, args.no_override_intrinsics)
                    if not success:
                        log_msg(f"[WARN] ViPE inference failed")
                except Exception as e:
                    log_msg(f"[ERR] Inference failed: {e}")
                
                dur = int(time.time() - start_epoch)
                log_msg(f"[TIME] Completed in {dur}s")

                npz_path = find_pose_npz(raw_dir)

            # Post-Process
            gt_tum = os.path.join(out_seq_dir, "groundtruth.txt")
            log_msg(f"[GT] Copy {gt_txt} -> {gt_tum}")
            shutil.copy2(gt_txt, gt_tum)

            if npz_path:
                if not os.path.exists(ts_csv):
                    log_msg(f"[ERR] Missing {ts_csv}")
                else:
                    pred_out = os.path.join(out_seq_dir, "vipe_estimate_tum.txt")
                    log_msg(f"[PRED] Convert {npz_path} -> {pred_out}")
                    convert_npz_to_tum(npz_path, ts_csv, pred_out, args.poses_kind, seq_name)
            else:
                log_msg(f"[WARN] No npz found in {raw_dir}, skipping conversion.")

        # ==================== MIRRORING ====================
        dest_seq_dir = os.path.join(dest_root, seq_name)
        os.makedirs(dest_seq_dir, exist_ok=True)
        
        src_gt = os.path.join(out_seq_dir, "groundtruth.txt")
        src_pred = os.path.join(out_seq_dir, "vipe_estimate_tum.txt")
        
        if os.path.exists(src_gt):
            shutil.copy2(src_gt, os.path.join(dest_seq_dir, "groundtruth.txt"))
        if os.path.exists(src_pred):
            shutil.copy2(src_pred, os.path.join(dest_seq_dir, "vipe_estimate_tum.txt"))

        seq_count += 1

    # 4. Benchmarking
    overall_dur = int(time.time() - overall_start)
    print("============================================================")
    print(f"[DONE] Processed {seq_count} sequence(s) in {overall_dur}s.")

    result_html = os.path.join(dest_root, "benchmark_results.html")
    
    if os.path.isfile(args.bench_py):
        print(f"Running benchmark: {args.bench_py}")
        bench_cmd = [sys.executable, args.bench_py, dest_root]
        if args.show_plots: bench_cmd.append("--plots")
        if args.scale_all: bench_cmd.append("--scale-all")
        if only_set:
            for item in only_set:
                bench_cmd.extend(["--plot-only", item])
        
        try:
            subprocess.run(bench_cmd, check=False)
        except Exception as e:
            print(f"Warning: benchmark execution failed: {e}")

        # Check for result HTML
        final_html = None
        if os.path.isfile(result_html):
            final_html = result_html
        else:
            # Fallback search
            candidates = glob.glob(os.path.join(dest_root, "*.html"))
            if candidates:
                final_html = candidates[0]
        
        if final_html:
            print(f"Opening: {final_html}")
            open_html_in_browser(final_html, no_browser=not args.browser)
        else:
            print(f"No HTML results found in {dest_root}")
    else:
        print(f"Benchmark script not found at {args.bench_py} - skipping.")

    # 5. S3 Upload (enabled by default, runs after benchmarking so results are included)
    if not args.no_s3_upload:
        s3_prefix = f"{args.s3_prefix}/{s3_expname}"
        upload_to_s3(root_out, args.s3_bucket, s3_prefix, args.s3_profile)
    else:
        logger.info("[INFO] S3 upload disabled (--no-s3-upload).")

if __name__ == "__main__":
    main()
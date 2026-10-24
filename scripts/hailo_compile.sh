#!/usr/bin/env bash
# W5: compile a trained YOLO26s ONNX to a Hailo-8L HEF with DFC 3.34 + model zoo v2.19.1.
# Runs on the workshop (x86_64); deploying to the Pi and on-device eval are separate
# (scripts/eval_detector.py --backend hef, scripts/pi_detector_ctl.py).
#
#   scripts/hailo_compile.sh --tag smoke [--calib DIR] [--gpu N] [--opt-level N] [--dry-run]
#     --calib      calibration images (>= 1024, the zoo alls calibset_size); default SDG v1
#     --gpu        CUDA_VISIBLE_DEVICES under PCI_BUS_ID order (0 P100, 1 3080 Ti, 2 V100);
#                  "" compiles on CPU (much slower)
#     --opt-level  override the zoo alls (optimization_level=4 + adaround, 116 adaround
#                  blocks ~= days on one P100) with this level and no adaround; the
#                  derived script is $WORK/yolo26s.alls and is recorded in the sidecar
#
# Inputs: $LB/data/models/yolo26s_<tag>/weights/best.onnx, exported end2end by
# `python -m workshop.export_detector onnx --tag <tag>` (the zoo yolo26 parser ends
# at the six one2one head convs). Outputs:
#   $LB/data/hailo/compiled/yolo26s_<tag>.hef  + .json sidecar (lawnberry-detector/1)
#   $LB/data/hailo/work/<tag>/{compile.log,yolo26s.har,yolo26s.hef}
#
# Zoo facts (v2.19.1): results land in the working dir (results_dir=./); --classes is
# ignored for yolo26 (no on-chip NMS) and there is no performance alls for it.
set -euo pipefail

TAG=""
CALIB=""
GPU="0"
OPT_LEVEL=""
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --calib) CALIB="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --opt-level) OPT_LEVEL="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n $TAG ]] || { echo "usage: $0 --tag T [--calib DIR] [--gpu N] [--opt-level N] [--dry-run]" >&2; exit 2; }
[[ -z $OPT_LEVEL || $OPT_LEVEL =~ ^[0-4]$ ]] || { echo "--opt-level must be 0-4" >&2; exit 2; }

REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
LB="${LB:-$HOME/nvme2/lawnberrypiserver}"
WHEEL="$HOME/Downloads/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl"
WHEEL_SHA256=f539ebb5997149ec68ca443a547196a03d28c624fbb072fdcd22a7d37fad9fb1
ONNX="${ONNX:-$LB/data/models/yolo26s_$TAG/weights/best.onnx}"
CALIB="${CALIB:-$LB/data/sdg/v1/images}"
WORK="$LB/data/hailo/work/$TAG"
OUT="$LB/data/hailo/compiled/yolo26s_$TAG.hef"
SPEC="${OUT%.hef}.json"
ENV="${DFC_ENV:-$LB/env-dfc}"
ZOO="$LB/hailo_model_zoo"
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"
ENV_MARK="$ENV/.lawnberry-dfc-$WHEEL_SHA256"
MIN_CALIB=1024

warn() { echo "[compile] WARN: $*" >&2; }
step() { echo "[compile] \$ $*"; }
fatal() { echo "[compile] FATAL: $*" >&2; exit 1; }

# Every command is echoed; with DRY=1 nothing executes and missing inputs are
# reported (WARN) instead of aborting.
run() {
  step "$*"
  [[ $DRY -eq 1 ]] && return 0
  "$@"
}
need() {  # need <path> <what>
  if [[ ! -e $1 ]]; then
    if [[ $DRY -eq 1 ]]; then warn "missing: $1 ($2)"; else fatal "missing $1 ($2)"; fi
  fi
}

# --- 1. inputs -----------------------------------------------------------
need "$WHEEL" "DFC 3.34.0 wheel from the Hailo developer zone"
need "$ONNX" "end2end ONNX (python -m workshop.export_detector onnx --tag $TAG)"
need "$CALIB" "calibration image directory"
if [[ -f $WHEEL && $DRY -eq 0 ]]; then
  echo "$WHEEL_SHA256  $WHEEL" | sha256sum -c --quiet - || fatal "DFC wheel sha256 mismatch"
fi
if [[ -d $CALIB ]]; then
  n_calib=$(find "$CALIB" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
  echo "[compile] calibration images: $n_calib in $CALIB"
  if (( n_calib < MIN_CALIB )); then
    if [[ $DRY -eq 1 ]]; then warn "need >= $MIN_CALIB calibration images"; else fatal "need >= $MIN_CALIB calibration images, found $n_calib"; fi
  fi
fi

# --- 2. ONNX end nodes ----------------------------------------------------
if [[ -f $ONNX ]]; then
  run bash -c "cd '$REPO' && '$LB/env/bin/python' -m workshop.export_detector check --onnx '$ONNX'"
fi

# --- 3. model zoo -----------------------------------------------------------
if [[ ! -d $ZOO ]]; then
  run git clone --depth 1 -b v2.19.1 https://github.com/hailo-ai/hailo_model_zoo "$ZOO"
fi

# --- 4/5. DFC env -----------------------------------------------------------
if [[ ! -x $ENV/bin/python ]]; then
  run "$CONDA" create -y -p "$ENV" python=3.11
fi
if [[ ! -f $ENV_MARK ]]; then
  run "$ENV/bin/python" -m pip install "$WHEEL"
  run "$ENV/bin/python" -m pip install "tensorflow[and-cuda]==2.18.0"
  run "$ENV/bin/python" -m pip install -e "$ZOO"
  run "$ENV/bin/python" -m pip check
  run touch "$ENV_MARK"
fi

# --- 6. preflight -----------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[compile] CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' (PCI_BUS_ID order)"
run "$ENV/bin/python" -c 'import os, hailo_sdk_client, tensorflow as tf; g=tf.config.list_physical_devices("GPU"); print("GPU", g); assert g or os.environ["CUDA_VISIBLE_DEVICES"] == "", "no GPU visible (pass --gpu \"\" to compile on CPU)"; tf.linalg.matmul(tf.ones((256,256)), tf.ones((256,256))).numpy()'
if [[ $DRY -eq 0 ]]; then
  zoo_version="$("$ENV/bin/hailomz" --version 2>&1 | tail -1)"
  echo "[compile] hailomz --version: $zoo_version"
  [[ $zoo_version == *2.19.1* ]] || fatal "expected model zoo v2.19.1"
else
  step "$ENV/bin/hailomz --version"
fi

# --- 7. compile ---------------------------------------------------------------
run mkdir -p "$WORK" "$(dirname "$OUT")"
SCRIPT_ARGS=()
ALLS_USED="zoo:cfg/alls/generic/yolo26s.alls"
if [[ -n $OPT_LEVEL ]]; then
  ALLS="$WORK/yolo26s.alls"
  ZOO_ALLS="$ZOO/hailo_model_zoo/cfg/alls/generic/yolo26s.alls"
  need "$ZOO_ALLS" "zoo yolo26s model script"
  if [[ $DRY -eq 0 ]]; then
    grep -q 'optimization_level=4' "$ZOO_ALLS" || fatal "zoo alls has no optimization_level=4 to override"
    sed -e "s/optimization_level=4/optimization_level=$OPT_LEVEL/" \
        -e '/post_quantization_optimization(adaround/d' "$ZOO_ALLS" > "$ALLS"
    echo "[compile] model script (opt level $OPT_LEVEL, no adaround):"; cat "$ALLS"
  fi
  SCRIPT_ARGS=(--model-script "$ALLS")
  ALLS_USED="$ALLS"
fi
step "cd $WORK && $ENV/bin/hailomz compile yolo26s --ckpt $ONNX --hw-arch hailo8l --calib-path $CALIB ${SCRIPT_ARGS[*]} | tee $WORK/compile.log"
if [[ $DRY -eq 0 ]]; then
  (cd "$WORK" && "$ENV/bin/hailomz" compile yolo26s \
    --ckpt "$ONNX" --hw-arch hailo8l --calib-path "$CALIB" "${SCRIPT_ARGS[@]}" 2>&1 | tee "$WORK/compile.log")
  [[ -f $WORK/yolo26s.hef ]] || fatal "compile produced no $WORK/yolo26s.hef (see compile.log)"
fi

# --- 8. HEF + sidecar spec ----------------------------------------------------
run cp "$WORK/yolo26s.hef" "$OUT"
run bash -c "cd '$REPO' && '$LB/env/bin/python' -m workshop.export_detector spec --names-from '$ONNX' --hef '$OUT' --tag '$TAG' --calib '$CALIB' --model-script '$ALLS_USED' --out '$SPEC'"

# --- 9. done ------------------------------------------------------------------
if [[ $DRY -eq 0 ]]; then
  sha256sum "$OUT" "$SPEC"
fi
echo "[compile] done: $OUT"

#!/usr/bin/env bash
# W5: compile a trained YOLO26 ONNX to a Hailo-8L HEF with the Hailo DFC.
#
# BLOCKED INPUT: the DFC wheel (hailo_dataflow_compiler-3.34*.whl) is only on
# Hailo's developer zone (free login) -- not PyPI, GitHub releases, or
# Hugging Face. Place it in ~/Downloads on the workshop and this script runs
# end-to-end. Every other input is staged; `--dry-run` verifies that.
#
#   scripts/hailo_compile.sh --tag yard1 [--dry-run]
#
# Steps: clone the model zoo if absent, create $LB/env-dfc (python 3.11; DFC
# 3.34 supports <=3.11), install the wheel + zoo requirements, version-check,
# compile with SDG-frame calibration, emit $LB/data/hailo/compiled/<tag>.hef,
# scp to the Pi, smoke with hailortcli run, evaluate per-class recall on the
# real val split.
#
# NOTE: the accelerator is Hailo-8L (part HM21LB1C2LAE), so --hw-arch hailo8l.
# The yolo26 runner flag is --yolov26 per the zoo's yolo-like runners; if
# v2.19.1 spells it differently, `hailomz compile --help` is the authority --
# --dry-run prints the exact command for review.
set -uo pipefail

TAG=""
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n $TAG ]] || { echo "usage: $0 --tag <tag> [--dry-run]" >&2; exit 2; }

LB="${LB:-$HOME/nvme2/lawnberrypiserver}"
WHELL_GLOB="$HOME/Downloads/hailo_dataflow_compiler-3.34*.whl"
CKPT="$LB/data/models/yolo26s_${TAG}/weights/best.onnx"
VAL_DATA="$LB/data/models/yolo26s_${TAG}/dataset/data.yaml"
CALIB="$LB/data/sdg/v1/images"
ZOO="$LB/hailo_model_zoo"
OUT_DIR="$LB/data/hailo/compiled"
OUT="$OUT_DIR/${TAG}.hef"
PI_JUMP="orangepi@192.168.1.140"
PI="kp@192.168.50.20"
# quoted on purpose: the remote shell expands ~ (hailortcli runs there)
# shellcheck disable=SC2088
PI_DIR='~/hailo-compiled'
CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"

warn() { echo "[compile] WARN: $*" >&2; }
step() { echo "[compile] \$ $*"; }

# Every command is echoed; with DRY=1 nothing executes and missing inputs are
# reported (WARN) instead of aborting.
run() {
  step "$*"
  [[ $DRY -eq 1 ]] && return 0
  "$@"
}
need() {  # need <path> <what>
  if [[ ! -e $1 ]]; then
    if [[ $DRY -eq 1 ]]; then warn "missing: $1 ($2)"; else echo "[compile] FATAL: missing $1 ($2)" >&2; exit 1; fi
  fi
}

# --- inputs -------------------------------------------------------------
# intentional glob -> array (single wheel expected)
# shellcheck disable=SC2206
WHELLS=( $WHELL_GLOB )
[[ -f ${WHELLS[0]} ]] || need "${WHELLS[0]}" "DFC wheel: download from Hailo developer zone"
need "$CKPT" "ONNX checkpoint (workshop/train_detector.py)"
need "$CALIB" "calibration images (SDG v1)"
need "$VAL_DATA" "run dataset data.yaml (train_detector.py)"

# --- 1. model zoo -------------------------------------------------------
if [[ ! -d $ZOO ]]; then
  run git clone --depth 1 -b v2.19.1 https://github.com/hailo-ai/hailo_model_zoo "$ZOO"
fi

# --- 2. DFC env ---------------------------------------------------------
if [[ ! -x $LB/env-dfc/bin/python ]]; then
  run "$CONDA" create -y -p "$LB/env-dfc" python=3.11
fi
run "$LB/env-dfc/bin/python" -m pip install "${WHELLS[0]}"
if [[ -f $ZOO/requirements.txt ]]; then
  run "$LB/env-dfc/bin/python" -m pip install -r "$ZOO/requirements.txt"
fi
run "$ZOO/hailomz" configure version-check

# --- 3. compile ---------------------------------------------------------
run mkdir -p "$OUT_DIR"
run "$ZOO/hailomz" compile \
  --ckpt "$CKPT" \
  --yolov26 \
  --hw-arch hailo8l \
  --calib-path "$CALIB" \
  --classes 24 \
  --run_postprocess \
  --performance \
  --name "yolo26s_${TAG}" \
  --output_dir "$OUT_DIR"
if [[ $DRY -eq 0 && -f "$OUT_DIR/yolo26s_${TAG}.hef" ]]; then
  mv "$OUT_DIR/yolo26s_${TAG}.hef" "$OUT"
fi

# --- 4. deploy + smoke on the Pi ----------------------------------------
run ssh -o BatchMode=yes -J "$PI_JUMP" "$PI" mkdir -p "$PI_DIR"
run scp -o BatchMode=yes -o ProxyJump="$PI_JUMP" "$OUT" "$PI:$PI_DIR/"
run ssh -o BatchMode=yes -J "$PI_JUMP" "$PI" \
  "hailortcli run $PI_DIR/${TAG}.hef -c 1 2>&1 | tail -5"

# --- 5. evaluate the compiled graph on the real val set ------------------
run "$ZOO/hailomz" evaluate \
  --name "yolo26s_${TAG}" \
  --hef-path "$OUT" \
  --data "$VAL_DATA" \
  --target full \
  --devices 1 \
  --batch-size 1 \
  || warn "evaluate expects COCO-format val annotations; convert YOLO labels first"

echo "[compile] done: $OUT"

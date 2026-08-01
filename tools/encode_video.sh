#!/usr/bin/env bash
# encode_video.sh — ffmpeg PNG sequences -> H.264 mp4 per shot.
#
# Usage:
#   bash tools/encode_video.sh VIZ_DIR [FPS] [SHOTS...]
#
# Arguments:
#   VIZ_DIR   Root viz directory (contains renders/<shot>/ subdirectories)
#   FPS       Frame rate for output video (default: 24)
#   SHOTS     Shot names to encode (default: q_iso centerline surface_cp)
#
# For each shot, reads VIZ_DIR/renders/<shot>/shot_<shot>_%06d.png and writes
# VIZ_DIR/renders/<shot>/<shot>.mp4 (H.264, yuv420p, CRF 18, preset slow).
#
# Side-by-side composite: if all three default shots exist, also writes
# VIZ_DIR/renders/composite.mp4 with q_iso|centerline stacked over surface_cp.
#
# Requires: ffmpeg (https://ffmpeg.org/). If absent, prints a message and exits 0.

set -euo pipefail

# -- Check ffmpeg presence ---------------------------------------------------
if ! command -v ffmpeg &>/dev/null; then
    echo "[encode_video] ffmpeg not found in PATH."
    echo "  Install via: brew install ffmpeg  (macOS)"
    echo "               apt-get install ffmpeg  (Ubuntu/Debian)"
    echo "               conda install -c conda-forge ffmpeg"
    echo "  No video files written."
    exit 0
fi

# -- Arguments ---------------------------------------------------------------
VIZ_DIR="${1:?Usage: $0 VIZ_DIR [FPS] [SHOTS...]}"
FPS="${2:-24}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true
if [[ $# -eq 0 ]]; then
    SHOTS=(q_iso centerline surface_cp)
else
    SHOTS=("$@")
fi

RENDERS_DIR="${VIZ_DIR}/renders"

if [[ ! -d "${RENDERS_DIR}" ]]; then
    echo "[encode_video] renders directory not found: ${RENDERS_DIR}"
    echo "  Run tools/render_frames.py first."
    exit 1
fi

encode_shot() {
    local shot="$1"
    local indir="${RENDERS_DIR}/${shot}"
    local pattern="${indir}/shot_${shot}_%06d.png"
    local outfile="${indir}/${shot}.mp4"

    if [[ ! -d "${indir}" ]]; then
        echo "[encode_video] shot '${shot}': directory not found: ${indir} — skipping"
        return
    fi

    local n_pngs
    n_pngs=$(find "${indir}" -name "shot_${shot}_*.png" | wc -l)
    if [[ "${n_pngs}" -eq 0 ]]; then
        echo "[encode_video] shot '${shot}': no PNGs found in ${indir} — skipping"
        return
    fi

    echo "[encode_video] shot '${shot}': ${n_pngs} frames -> ${outfile}"
    ffmpeg -y \
        -framerate "${FPS}" \
        -i "${pattern}" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -crf 18 \
        -preset slow \
        -movflags +faststart \
        "${outfile}" \
        2>&1 | tail -5
    echo "[encode_video]   wrote ${outfile}"
}

# -- Encode each shot --------------------------------------------------------
for shot in "${SHOTS[@]}"; do
    encode_shot "${shot}"
done

# -- Optional composite (all three default shots) ----------------------------
DEFAULT_SHOTS=("q_iso" "centerline" "surface_cp")
all_exist=true
for s in "${DEFAULT_SHOTS[@]}"; do
    if [[ ! -f "${RENDERS_DIR}/${s}/${s}.mp4" ]]; then
        all_exist=false
        break
    fi
done

if [[ "${all_exist}" == "true" ]]; then
    COMPOSITE="${RENDERS_DIR}/composite.mp4"
    echo "[encode_video] creating composite (q_iso+centerline | surface_cp)..."
    # hstack q_iso and centerline (top row), pad surface_cp to same width
    ffmpeg -y \
        -i "${RENDERS_DIR}/q_iso/q_iso.mp4" \
        -i "${RENDERS_DIR}/centerline/centerline.mp4" \
        -i "${RENDERS_DIR}/surface_cp/surface_cp.mp4" \
        -filter_complex "
            [0:v][1:v]hstack=inputs=2[top];
            [2:v]scale=iw*2:ih*2[bot_scaled];
            [top][bot_scaled]vstack=inputs=2[out]
        " \
        -map "[out]" \
        -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow \
        -movflags +faststart \
        "${COMPOSITE}" \
        2>&1 | tail -5
    echo "[encode_video]   wrote ${COMPOSITE}"
fi

echo "[encode_video] done."

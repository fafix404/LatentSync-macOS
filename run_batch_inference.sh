#!/bin/bash
#
# Usage: ./run_batch_inference.sh [repertoire_videos] [repertoire_sortie]
#
# Lance scripts/inference.py sur toutes les vidéos (.mp4) d'un répertoire.
# Par défaut, l'audio est extrait de la même vidéo que l'entrée vidéo.

set -euo pipefail

VIDEO_DIR="${1:-videos}"
OUT_DIR="${2:-outputs}"
INFERENCE_STEPS=8
GUIDANCE_SCALE=1.9

mkdir -p "$OUT_DIR"

shopt -s nullglob
videos=("$VIDEO_DIR"/*.mp4)
shopt -u nullglob

if [ ${#videos[@]} -eq 0 ]; then
    echo "Aucune vidéo .mp4 trouvée dans $VIDEO_DIR"
    exit 1
fi

for video in "${videos[@]}"; do
    filename=$(basename "$video")
    name="${filename%.*}"
    out_path="$OUT_DIR/${name}_output.mp4"

    echo "=== Traitement de : $filename ==="

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.inference \
        --unet_config_path "configs/unet/stage2.yaml" \
        --inference_ckpt_path "checkpoints/v1.5/latentsync_unet.pt" \
        --enable_deepcache \
        --video_path "$video" \
        --audio_path "$video" \
        --video_out_path "$out_path" \
        --inference_steps "$INFERENCE_STEPS" \
        --guidance_scale "$GUIDANCE_SCALE"

    echo "-> Sortie : $out_path"
    echo
done

echo "Terminé. Toutes les vidéos ont été traitées."
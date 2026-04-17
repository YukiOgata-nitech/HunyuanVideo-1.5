#!/bin/bash
set -e

# ============================================================
# HunyuanVideo-1.5 推論スクリプト
# 使い方: bash run_inference.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# venv を有効化
source "$SCRIPT_DIR/.venv/bin/activate"

# CUDA 設定（バージョン不一致対策）
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# ============================================================
# 基本設定
# ============================================================
MODEL_PATH="./ckpts"
OUTPUT_DIR="./outputs"
mkdir -p "$OUTPUT_DIR"

PROMPT='A medium tracking shot follows a male athlete in a tight race suit and spiked shoes as he sprints down lane 4 of a bright athletics track. He approaches a white hurdle, plants his lead leg, and launches himself over it in a fluid arc, arms pumping for balance. His trailing leg snaps through cleanly as he lands and immediately drives into the next stride. The stadium is packed with cheering spectators under blazing floodlights. The camera stays low and side-on, capturing the explosive power and rhythm of each jump. Slow-motion moments highlight the peak of each hurdle clearance. Cinematic photography realistic style, shallow depth of field, motion blur on the background crowd.'
IMAGE_PATH=none          # I2V の場合はここに画像パスを指定。T2V は none のまま
SEED=1
RESOLUTION=480p          # 480p or 720p
ASPECT_RATIO=16:9
VIDEO_LENGTH=121         # フレーム数（4n+1: 例 49, 97, 121）

# ============================================================
# 推論高速化設定
# ============================================================
N_GPU=1                  # 使用GPU数
SAGE_ATTN=true           # SageAttention による高速化
ENABLE_CACHE=true        # 特徴量キャッシュ
CACHE_TYPE=deepcache     # deepcache / teacache / taylorcache
CFG_DISTILLED=false      # CFG蒸留モデル使用（~2x高速、専用チェックポイントが必要）
ENABLE_STEP_DISTILL=false # ステップ蒸留（480p I2V のみ、~75%高速）
SPARSE_ATTN=false        # スパースアテンション（720p + flex-block-attn 必要）
OFFLOADING=true          # CPUオフローディング（VRAMが少ない場合はtrue、多い場合はfalseで高速化）
OVERLAP_GROUP_OFFLOADING=true  # オフローディング中のオーバーラップ処理（CPU RAMを多く使うが高速）

# ============================================================
# 品質設定
# ============================================================
REWRITE=false            # プロンプト書き換え（vLLMサーバーが必要）
ENABLE_SR=false          # 超解像（専用チェックポイントが必要）

# ============================================================
# タイムスタンプ付き出力ファイル名
# ============================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="$OUTPUT_DIR/output_${TIMESTAMP}.mp4"

# ============================================================
# モデルの存在確認
# ============================================================
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] モデルが見つかりません: $MODEL_PATH"
    echo "先に以下のコマンドでモデルをダウンロードしてください:"
    echo "  hf download tencent/HunyuanVideo-1.5 --local-dir ./ckpts"
    exit 1
fi

# ============================================================
# 実行
# ============================================================
echo "======================================"
echo "  HunyuanVideo-1.5 推論開始"
echo "======================================"
echo "  プロンプト : $PROMPT"
echo "  画像パス   : $IMAGE_PATH"
echo "  解像度     : $RESOLUTION  ($ASPECT_RATIO)"
echo "  フレーム数 : $VIDEO_LENGTH"
echo "  使用GPU数  : $N_GPU"
echo "  出力先     : $OUTPUT_PATH"
echo "======================================"

torchrun --nproc_per_node=$N_GPU generate.py \
  --prompt "$PROMPT" \
  --image_path "$IMAGE_PATH" \
  --resolution "$RESOLUTION" \
  --aspect_ratio "$ASPECT_RATIO" \
  --video_length "$VIDEO_LENGTH" \
  --seed "$SEED" \
  --rewrite "$REWRITE" \
  --cfg_distilled "$CFG_DISTILLED" \
  --enable_step_distill "$ENABLE_STEP_DISTILL" \
  --sparse_attn "$SPARSE_ATTN" \
  --use_sageattn "$SAGE_ATTN" \
  --enable_cache "$ENABLE_CACHE" \
  --cache_type "$CACHE_TYPE" \
  --offloading "$OFFLOADING" \
  --overlap_group_offloading "$OVERLAP_GROUP_OFFLOADING" \
  --sr "$ENABLE_SR" \
  --output_path "$OUTPUT_PATH" \
  --model_path "$MODEL_PATH"

echo ""
echo "完了! 出力ファイル: $OUTPUT_PATH"

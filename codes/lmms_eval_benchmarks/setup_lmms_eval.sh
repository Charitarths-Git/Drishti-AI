#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_lmms_eval.sh — Install lmms-eval + dependencies for teacher evaluation
# ─────────────────────────────────────────────────────────────────────────────
#
# Hardware: 8× NVIDIA RTX A6000 (49 GB each) | CUDA 12.1+
#
# Usage:
#   chmod +x setup_lmms_eval.sh
#   ./setup_lmms_eval.sh
#
# What this does:
#   1. Installs lmms-eval from GitHub (latest, with video support)
#   2. Installs flash-attn for fast inference
#   3. Installs model-specific dependencies
#   4. Verifies the installation
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "═══════════════════════════════════════════════════════════"
echo "  lmms-eval Setup for Teacher Model Evaluation"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Clone and install lmms-eval ──────────────────────────────────────────
echo ""
echo "  [1/5] Installing lmms-eval from source..."

if [ -d "lmms-eval" ]; then
    echo "  lmms-eval directory exists, pulling latest..."
    cd lmms-eval
    git pull
    cd ..
else
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
fi

cd lmms-eval
pip install -e ".[video]" --quiet
cd ..

echo "  [OK] lmms-eval installed"

# ── 2. Install flash-attn ───────────────────────────────────────────────────
echo ""
echo "  [2/5] Installing flash-attn (3× faster attention on Ampere)..."

pip install flash-attn --no-build-isolation --quiet 2>/dev/null || {
    echo "  [WARN] flash-attn install failed — will use default attention"
    echo "         This is OK but inference will be ~3× slower"
    echo "         To fix: apt install build-essential && pip install flash-attn --no-build-isolation"
}

# ── 3. Install model-specific deps ─────────────────────────────────────────
echo ""
echo "  [3/5] Installing model-specific dependencies..."

# Core model dependencies
pip install transformers>=4.49.0 accelerate>=0.30.0 --quiet

# InternVL3 needs these
pip install einops timm sentencepiece --quiet

# Qwen2.5-VL needs these
pip install qwen-vl-utils>=0.0.8 --quiet

# Tarsier2 (LLaVA-style)
pip install protobuf --quiet

# Video processing
pip install av decord opencv-python --quiet

# General
pip install huggingface-hub>=0.23.0 --quiet
pip install bitsandbytes --quiet  # for quantization if needed

echo "  [OK] Model dependencies installed"

# ── 4. Install evaluation requirements ──────────────────────────────────────
echo ""
echo "  [4/5] Installing evaluation requirements..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/requirements_eval.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements_eval.txt" --quiet
fi

echo "  [OK] Evaluation requirements installed"

# ── 5. Verify installation ─────────────────────────────────────────────────
echo ""
echo "  [5/5] Verifying installation..."

python -c "
import sys
checks = []

# lmms-eval
try:
    import lmms_eval
    checks.append(('lmms-eval', True, getattr(lmms_eval, '__version__', 'installed')))
except ImportError:
    checks.append(('lmms-eval', False, 'NOT FOUND'))

# torch + CUDA
try:
    import torch
    cuda = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda else 0
    checks.append(('PyTorch+CUDA', cuda, f'{torch.__version__}, {n_gpus} GPUs'))
except ImportError:
    checks.append(('PyTorch', False, 'NOT FOUND'))

# flash-attn
try:
    import flash_attn
    checks.append(('flash-attn', True, flash_attn.__version__))
except ImportError:
    checks.append(('flash-attn', False, 'not installed (optional)'))

# transformers
try:
    import transformers
    checks.append(('transformers', True, transformers.__version__))
except ImportError:
    checks.append(('transformers', False, 'NOT FOUND'))

# decord
try:
    import decord
    checks.append(('decord', True, 'installed'))
except ImportError:
    checks.append(('decord', False, 'NOT FOUND'))

print()
for name, ok, ver in checks:
    sym = 'OK' if ok else '!!'
    print(f'  [{sym}] {name}: {ver}')

all_ok = all(ok for _, ok, _ in checks if 'optional' not in str(_))
print()
if all_ok:
    print('  All checks passed. Ready to evaluate teacher models.')
else:
    print('  Some checks failed. Fix the issues above before running evaluations.')
    sys.exit(1)
"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. python download_teacher_models.py"
echo "    2. python run_teacher_eval.py --dry_run"
echo "    3. python run_teacher_eval.py"
echo "═══════════════════════════════════════════════════════════"

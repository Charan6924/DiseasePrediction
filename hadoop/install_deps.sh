#!/bin/bash
# Install Python dependencies needed by mapper.py on the HPCC.
# Run once per session after SSHing in (pip --user persists across sessions).
# Usage: bash hadoop/install_deps.sh

set -euo pipefail

# Load Python — adjust version if HPCC offers a different one
module load python/3.9 2>/dev/null && echo "Loaded python/3.9" || echo "module load skipped (already loaded or unavailable)"

echo "Installing Python packages..."
pip install --user --quiet scikit-learn xgboost joblib numpy

echo ""
echo "Installed versions:"
python3 - <<'EOF'
import sklearn, xgboost, joblib, numpy
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  xgboost      : {xgboost.__version__}")
print(f"  joblib       : {joblib.__version__}")
print(f"  numpy        : {numpy.__version__}")
EOF

echo ""
echo "Deps ready. Run: bash hadoop/setup_hdfs.sh"

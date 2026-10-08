#!/usr/bin/env bash
# Runs INSIDE the container for old-code runs (main @ 6a75c37), after the image entrypoint.
# The old train_step/val_step/test_step take an `epoch` kwarg and need the patched
# mmengine loops.py and base_model.py from the old worktree (README step 4 of that
# commit). transforms_3d.py was already copied by the image entrypoint from
# /workspace/replace_mmdetection_files, which is the old worktree's copy here.
set -euo pipefail
MMENGINE_DIR="$(python -c 'import mmengine, os; print(os.path.dirname(mmengine.__file__))')"
cp /workspace/replace_mmdetection_files/loops.py      "$MMENGINE_DIR/runner/loops.py"
cp /workspace/replace_mmdetection_files/base_model.py "$MMENGINE_DIR/model/base_model/base_model.py"
echo "old_prelude: patched $MMENGINE_DIR/{runner/loops.py,model/base_model/base_model.py}"
exec "$@"

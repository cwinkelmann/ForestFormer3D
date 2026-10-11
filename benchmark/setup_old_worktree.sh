#!/usr/bin/env bash
# Create/verify the old-code worktree (main @ $FF3D_OLD_COMMIT, default 6a75c37) at
# $FF3D_OLD_ROOT. Idempotent: a second run recognizes an existing worktree already at the
# right commit and exits 0 without touching it; if $FF3D_OLD_ROOT exists but is not a git
# worktree, or is a worktree at the wrong commit, it exits 1 rather than overwriting or
# deleting anything.
#
# $FF3D_OLD_ROOT is a SIBLING directory of $FF3D_ROOT (see the FF3D_OLD_ROOT comment in
# common.sh): it must not live under $FF3D_ROOT/work_dirs, which ff3d_docker_old separately
# bind-mounts, or the mount would nest inside itself.
#
# The old README asked for three manual copies into site-packages
# (replace_mmdetection_files/{loops.py,base_model.py,transforms_3d.py}). None is done here:
#   - transforms_3d.py: the image's baked docker/entrypoint.sh copies
#     /workspace/replace_mmdetection_files/transforms_3d.py into mmdet3d at container start.
#   - loops.py / base_model.py: copied by benchmark/old_prelude.sh (mounted into the old
#     container by ff3d_docker_old) because the fixed image entrypoint no longer copies them.
# The old README also asked to run fix_spconv_checkpoint.py before test.py; the benchmark
# instead feeds the old test.py the RAW layout (it permutes weights in memory) -- see
# ff3d_prepare_checkpoint in common.sh / run_release_eval.sh.
#
# One line of the old code's own tools/test.py IS edited (see below): it calls torch.load()/
# torch.save() but never imports torch, which fails on the real GPU host. This is the only
# source change this script makes to a file 6a75c37 itself committed.
set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "$FF3D_ROOT"
git rev-parse --verify --quiet "${FF3D_OLD_COMMIT}^{commit}" >/dev/null \
  || git fetch origin main
git rev-parse --verify --quiet "${FF3D_OLD_COMMIT}^{commit}" >/dev/null \
  || ff3d_die "commit $FF3D_OLD_COMMIT not found even after fetching origin/main"
want="$(git rev-parse --short "$FF3D_OLD_COMMIT")"

if [ -e "$FF3D_OLD_ROOT" ]; then
  [ -f "$FF3D_OLD_ROOT/.git" ] \
    || ff3d_die "$FF3D_OLD_ROOT exists but is not a git worktree (no .git file); refusing to touch it"
  head="$(git -C "$FF3D_OLD_ROOT" rev-parse --short HEAD 2>/dev/null)" \
    || ff3d_die "$FF3D_OLD_ROOT/.git exists but 'git rev-parse HEAD' failed there; refusing to touch it"
  [ "$head" = "$want" ] \
    || ff3d_die "$FF3D_OLD_ROOT is a worktree but at $head, not $FF3D_OLD_COMMIT ($want); refusing to touch it"
  ff3d_log "worktree exists at $FF3D_OLD_ROOT ($head), already at $FF3D_OLD_COMMIT -- nothing to do"
else
  mkdir -p "$(dirname "$FF3D_OLD_ROOT")"
  git worktree add --detach "$FF3D_OLD_ROOT" "$FF3D_OLD_COMMIT"
  head="$(git -C "$FF3D_OLD_ROOT" rev-parse --short HEAD)"
  [ "$head" = "$want" ] || ff3d_die "worktree HEAD $head != $FF3D_OLD_COMMIT ($want)"
fi

# Sanity checks that this really is the old code path the benchmark relies on.
grep -q 'permute(1, 2, 3, 4, 0)' "$FF3D_OLD_ROOT/tools/test.py" \
  || ff3d_die "old tools/test.py does not permute weights in memory; wrong commit?"
grep -q "test_cfg\['output_dir'\] = cfg.work_dir" "$FF3D_OLD_ROOT/tools/test.py" \
  || ff3d_die "old tools/test.py does not route output to work_dir; wrong commit?"
for f in loops.py base_model.py transforms_3d.py; do
  [ -f "$FF3D_OLD_ROOT/replace_mmdetection_files/$f" ] || ff3d_die "old worktree lacks replace_mmdetection_files/$f"
done

ff3d_log "old worktree ready: $FF3D_OLD_ROOT @ $head"

# The image ENTRYPOINT is /workspace/docker/entrypoint.sh (baked in at build time by
# 'COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh' in Dockerfile). ff3d_docker_old
# bind-mounts $FF3D_OLD_ROOT onto /workspace, and this old worktree (commit $FF3D_OLD_COMMIT)
# has NO docker/ directory at all (confirmed: 'git ls-tree -r --name-only 6a75c37 -- docker/'
# is empty). A docker bind mount REPLACES the target directory's contents -- it does not merge
# with what the image baked in there -- so without this copy, /workspace/docker/entrypoint.sh
# would not exist inside an ff3d_docker_old container and the container would fail to start
# ("exec: /workspace/docker/entrypoint.sh: no such file"). We therefore materialize a copy of
# the current entrypoint.sh into the old worktree so ff3d_docker_old's containers can start at
# all. Everything that copied entrypoint.sh then touches already exists in the old worktree:
# replace_mmdetection_files/transforms_3d.py is byte-identical between 6a75c37 and this branch
# (empty `git diff 6a75c37 HEAD -- replace_mmdetection_files/transforms_3d.py`), and
# segmentator/csrc exists (without a build/ symlink yet, which is exactly what the entrypoint
# creates).
old_entrypoint="$FF3D_OLD_ROOT/docker/entrypoint.sh"
mkdir -p "$FF3D_OLD_ROOT/docker"
if [ ! -f "$old_entrypoint" ] || ! cmp -s "$FF3D_ROOT/docker/entrypoint.sh" "$old_entrypoint"; then
  cp "$FF3D_ROOT/docker/entrypoint.sh" "$old_entrypoint"
  chmod +x "$old_entrypoint"
  ff3d_log "copied docker/entrypoint.sh into old worktree: $old_entrypoint"
else
  ff3d_log "old worktree already has an up-to-date docker/entrypoint.sh: $old_entrypoint"
fi
ff3d_log "NOTE: this file is untracked in the old worktree's git status (6a75c37 predates"
ff3d_log "docker/ entirely) -- that is expected, not a sign of a dirty/corrupt worktree."

# The committed 6a75c37 tools/test.py calls torch.load()/torch.save() (lines ~123/139) but
# never imports torch itself -- confirmed on the real GPU host, this dies with
# "NameError: name 'torch' is not defined" during release-eval inference. This is the ONE
# source edit this script makes to the old code's own committed files (everything else --
# transforms_3d.py, loops.py, base_model.py, docker/entrypoint.sh -- is applied via separate
# files/copies, never by editing what 6a75c37 committed). Idempotent: only inserted if
# missing; uses awk (not sed -i, whose in-place syntax differs between GNU and BSD sed) so it
# behaves the same on the Mac and on the GPU host.
old_test_py="$FF3D_OLD_ROOT/tools/test.py"
if grep -qx 'import torch' "$old_test_py"; then
  ff3d_log "old tools/test.py already has 'import torch' -- nothing to do"
else
  awk '{ print } !done && /^import argparse$/ { print "import torch"; done=1 }' \
    "$old_test_py" > "$old_test_py.tmp"
  mv "$old_test_py.tmp" "$old_test_py"
  grep -qx 'import torch' "$old_test_py" \
    || ff3d_die "failed to insert 'import torch' into $old_test_py (no 'import argparse' line?)"
  ff3d_log "inserted 'import torch' into old tools/test.py, right after 'import argparse'"
fi

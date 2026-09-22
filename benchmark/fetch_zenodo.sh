#!/usr/bin/env bash
# Fetch Zenodo record 16742708 (ForAINetV2 dataset + epoch_3000_fix.pth checkpoint) and
# unpack it into the checkout layout.
#
#   bash benchmark/fetch_zenodo.sh             # fetch, verify md5, unpack, place
#   bash benchmark/fetch_zenodo.sh --list      # print key<TAB>size<TAB>md5<TAB>url and exit
#   bash benchmark/fetch_zenodo.sh --dry-run   # print the actions that would be taken
#
# Cache-first: if FF3D_ZENODO_CACHE (default /raid/cwinkelmann/zenodo-16742708) contains a
# files.tsv (tab-separated: key, url, md5:<hex>, size) AND every zip it lists, that cache is
# used as the source of the archives -- nothing is downloaded, and the cache directory is
# never written to (no markers, no temp dirs there). Only when the cache is absent or
# incomplete does this script fall back to querying the Zenodo REST API and downloading
# into $FF3D_ROOT/work_dirs/.zenodo/downloads/.
#
# Layout produced under $FF3D_ROOT (verified archive contents):
#   train_val_data.zip      -> data/ForAINetV2/train_val_data/*.ply
#   test_data.zip            -> data/ForAINetV2/test_data/*.ply
#   clean_forestformer.zip   -> work_dirs/clean_forestformer/epoch_3000_fix.pth
#
# Idempotent: a zip whose "$FF3D_ROOT/work_dirs/.zenodo/<key>.unpacked-<md5>" marker exists
# is skipped on later runs (whether the source was the cache or a download).
set -euo pipefail

FF3D_ROOT="${FF3D_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ZENODO_API="${ZENODO_API:-https://zenodo.org/api/records/16742708}"
FF3D_ZENODO_CACHE="${FF3D_ZENODO_CACHE:-/raid/cwinkelmann/zenodo-16742708}"

FF3D_ZENODO_DIR="$FF3D_ROOT/work_dirs/.zenodo"
FF3D_DOWNLOADS="${FF3D_DOWNLOADS:-$FF3D_ZENODO_DIR/downloads}"
FF3D_DATA="$FF3D_ROOT/data/ForAINetV2"
FF3D_CKPT_DIR="$FF3D_ROOT/work_dirs/clean_forestformer"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*"; }   # portable (GNU and BSD date)
die() { log "ERROR: $*" >&2; exit 1; }

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
  else md5 -q "$1"; fi
}

# --- listing -----------------------------------------------------------------

# files.tsv columns: key, url, md5:<hex>, size. Emitted uniformly as key<TAB>size<TAB>md5<TAB>url.
list_from_cache() {
  local key url md5field size md5
  while IFS=$'\t' read -r key url md5field size; do
    [ -n "$key" ] || continue
    md5="${md5field#md5:}"
    printf '%s\t%s\t%s\t%s\n' "$key" "$size" "$md5" "$url"
  done < "$FF3D_ZENODO_CACHE/files.tsv"
}

list_from_api() {
  curl -sSL --fail "$ZENODO_API" | python3 -c '
import json, sys
rec = json.load(sys.stdin)
for f in rec["files"]:
    md5 = f["checksum"].split(":", 1)[-1]
    print("\t".join([f["key"], str(f["size"]), md5, f["links"]["self"]]))
'
}

cache_ready() {
  # All-or-nothing: a files.tsv that names even one zip not actually present in the cache
  # dir makes this return 1, and main() falls back to the whole API path for every file.
  local tsv="$FF3D_ZENODO_CACHE/files.tsv"
  [ -f "$tsv" ] || return 1
  local key url md5field size
  while IFS=$'\t' read -r key url md5field size; do
    [ -n "$key" ] || continue
    [ -f "$FF3D_ZENODO_CACHE/$key" ] || return 1
  done < "$tsv"
  return 0
}

# --- download (API mode only) -------------------------------------------------

download() {  # download <url> <dest> <md5>
  local url="$1" dest="$2" md5="$3" rc
  if [ -f "$dest" ] && [ "$(md5_of "$dest")" = "$md5" ]; then
    log "already present: $dest"
    return 0
  fi
  log "downloading $url -> $dest"
  set +e
  curl -sS -L --fail --retry 5 --retry-delay 15 -C - -o "$dest" "$url" < /dev/null
  rc=$?
  set -e
  # 33 = server does not honour the byte range, 22 = HTTP 416 (range past the end because
  # the file on disk is already complete). Both leave the existing file as-is; the md5
  # check in process_entry decides whether it is actually complete and correct.
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 22 ] && [ "$rc" -ne 33 ]; then
    die "curl failed with exit $rc for $url"
  fi
  [ -f "$dest" ] || die "curl exit $rc and no file at $dest for $url"
}

# --- unpack --------------------------------------------------------------------

# unpack_zip <key> <archive>: extracts into a scratch dir under $FF3D_ZENODO_DIR (never
# the cache) and moves the payload into the checkout layout.
unpack_zip() {
  local key="$1" archive="$2" dest tmp
  case "$key" in
    train_val_data.zip)     dest="$FF3D_DATA/train_val_data" ;;
    test_data.zip)          dest="$FF3D_DATA/test_data" ;;
    clean_forestformer.zip) dest="$FF3D_CKPT_DIR" ;;
    *)
      log "no unpack rule for $key; leaving it archived only"
      return 0
      ;;
  esac
  mkdir -p "$dest" "$FF3D_ZENODO_DIR"
  tmp="$(mktemp -d "$FF3D_ZENODO_DIR/unpack-XXXXXX")"
  if ! unzip -q "$archive" -d "$tmp"; then
    rm -rf "$tmp"
    die "$key: unzip failed for $archive"
  fi
  find "$tmp" -type f \( -name '*.ply' -o -name '*.pth' \) -not -path '*/__MACOSX/*' -exec mv -f {} "$dest"/ \;
  rm -rf "$tmp"
}

# process_entry <key> <size> <md5> <url> <archive>: verifies md5 against the listing, then
# unpacks (unless the marker for this key+md5 already exists under $FF3D_ZENODO_DIR).
process_entry() {
  local key="$1" size="$2" md5="$3" url="$4" archive="$5" got marker
  [ -f "$archive" ] || die "$key: archive not found at $archive"
  got="$(md5_of "$archive")"
  if [ "$got" != "$md5" ]; then
    die "$key: md5 mismatch for $archive: expected $md5 got $got"
  fi
  log "verified md5 $md5: $key"
  marker="$FF3D_ZENODO_DIR/$key.unpacked-$md5"
  if [ -f "$marker" ]; then
    log "already unpacked: $key"
    return 0
  fi
  unpack_zip "$key" "$archive"
  mkdir -p "$FF3D_ZENODO_DIR"
  touch "$marker"
  log "unpacked: $key"
}

dry_run_entry() {  # dry_run_entry <key> <md5> <action-description>
  local key="$1" md5="$2" action="$3" marker
  marker="$FF3D_ZENODO_DIR/$key.unpacked-$md5"
  if [ -f "$marker" ]; then
    log "[dry-run] $key: already unpacked (marker $marker), would skip"
  else
    log "[dry-run] $key: would $action"
  fi
}

# --- main ------------------------------------------------------------------

main() {
  local list_only=0 dry_run=0 arg
  for arg in "$@"; do
    case "$arg" in
      --list) list_only=1 ;;
      --dry-run) dry_run=1 ;;
      *) die "unknown argument: $arg (supported: --list, --dry-run)" ;;
    esac
  done

  # --list and --dry-run must be side-effect-free: no mkdir here. $FF3D_ZENODO_DIR is
  # created lazily, only by the code paths that actually write (download/unpack/marker).
  local mode listing
  if cache_ready; then
    mode=cache
    log "using local cache: $FF3D_ZENODO_CACHE" >&2
    listing="$(list_from_cache)"
  else
    mode=api
    log "no usable cache at $FF3D_ZENODO_CACHE; querying $ZENODO_API" >&2
    listing="$(list_from_api)"
  fi
  [ -n "$listing" ] || die "no files found (mode=$mode)"

  if [ "$list_only" = 1 ]; then
    printf '%s\n' "$listing"
    return 0
  fi

  # dry-run touches nothing on disk, not even the downloads dir.
  [ "$mode" = "api" ] && [ "$dry_run" != 1 ] && mkdir -p "$FF3D_DOWNLOADS"

  while IFS=$'\t' read -r key size md5 url; do
    [ -n "$key" ] || continue
    if [ "$dry_run" = 1 ]; then
      if [ "$mode" = "cache" ]; then
        dry_run_entry "$key" "$md5" "verify md5 against the cache and unpack $FF3D_ZENODO_CACHE/$key"
      else
        dry_run_entry "$key" "$md5" "download $url -> $FF3D_DOWNLOADS/$key, verify md5, unpack"
      fi
      continue
    fi
    if [ "$mode" = "cache" ]; then
      process_entry "$key" "$size" "$md5" "$url" "$FF3D_ZENODO_CACHE/$key"
    else
      local archive="$FF3D_DOWNLOADS/$key"
      download "$url" "$archive" "$md5"
      process_entry "$key" "$size" "$md5" "$url" "$archive"
    fi
  done <<< "$listing"

  [ "$dry_run" = 1 ] && return 0

  log "train_val_data: $(find "$FF3D_DATA/train_val_data" -name '*.ply' 2>/dev/null | wc -l | tr -d ' ') ply"
  log "test_data: $(find "$FF3D_DATA/test_data" -name '*.ply' 2>/dev/null | wc -l | tr -d ' ') ply"
  if [ -f "$FF3D_CKPT_DIR/epoch_3000_fix.pth" ]; then
    log "checkpoint: $FF3D_CKPT_DIR/epoch_3000_fix.pth"
  else
    log "checkpoint: missing"
  fi
}

main "$@"

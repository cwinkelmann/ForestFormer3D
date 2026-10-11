#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# prepare_safe_testfile_names.sh
#
#  * Repairs any "...fixednamefixedname..."   -> "...fixedname"
#  * Adds ONE trailing "fixedname" to names that still end with "_<digits>"
#  * Keeps original order in meta_data/test_list_initial.txt
#  * Renames files/dirs in test_data/ accordingly; the extension is stripped
#    before matching and re-attached afterwards, so plot_1.ply -> plot_1fixedname.ply
#  Works with bash 3.2 (no associative arrays).
# ---------------------------------------------------------------------------
set -euo pipefail
FIXTAG="${FIXTAG:-fixedname}"

ROOT="${ROOT:-data/ForAINetV2}"
LIST="${ROOT}/meta_data/test_list_initial.txt"
LIST_BAK="${ROOT}/meta_data/test_list_initial_original.txt"
DATA="${ROOT}/test_data"

[[ -f "$LIST" ]] || { echo "missing list: $LIST" >&2; exit 1; }
[[ -d "$DATA" ]] || { echo "missing data dir: $DATA" >&2; exit 1; }

cp "$LIST" "$LIST_BAK"
echo "backup created -> $(basename "$LIST_BAK")"

# convert one basename (no extension) to its safe form
convert_name() {
    local name="$1"
    local clean
    clean=$(printf '%s' "$name" | sed -E "s/(${FIXTAG})+/${FIXTAG}/g")
    if [[ "$clean" =~ _[0-9]+$ ]] && [[ "$clean" != *"${FIXTAG}" ]]; then
        printf '%s\n' "${clean}${FIXTAG}"
    else
        printf '%s\n' "$clean"
    fi
}

# ---------- (A) new list, preserving order; mapping file "orig<TAB>safe" ----
map_file=$(mktemp)
tmp_list="${LIST}.tmp"
: > "$tmp_list"
while IFS= read -r scan || [[ -n "${scan:-}" ]]; do
    [[ -z "$scan" ]] && continue
    safe=$(convert_name "$scan")
    printf '%s\n' "$safe" >> "$tmp_list"
    printf '%s\t%s\n' "$scan" "$safe" >> "$map_file"
done < "$LIST"
mv "$tmp_list" "$LIST"
echo "$(basename "$LIST") updated ( $(wc -l < "$LIST" | tr -d ' ') lines )"

lookup_map() {   # prints the mapped safe name for $1, or nothing
    awk -F '\t' -v key="$1" '$1 == key { print $2; exit }' "$map_file"
}

# ---------- (B) depth-first walk over test_data --------------------------------
find "$DATA" -depth -mindepth 1 | while IFS= read -r path; do
    name="$(basename "$path")"
    dir="$(dirname "$path")"
    if [[ -f "$path" && "$name" == *.* ]]; then
        base="${name%.*}"
        ext=".${name##*.}"
    else
        base="$name"
        ext=""
    fi
    target="$(convert_name "$base")"
    mapped="$(lookup_map "$base")"
    [[ -n "$mapped" ]] && target="$mapped"
    target="${target}${ext}"
    if [[ "$name" != "$target" ]]; then
        if [[ -e "$dir/$target" ]]; then
            echo "skip (target exists): $dir/$target"
        else
            mv "$path" "$dir/$target"
            echo "mv $name -> $target"
        fi
    fi
done

rm -f "$map_file"
echo "All names are now safe and consistent."

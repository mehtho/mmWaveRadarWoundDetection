#!/bin/bash

rm -rf "Official Testing Data/j/0.0"

find "Official Testing Data" -type d -print0 |
while IFS= read -r -d '' dir; do
  # Count .npy files in this directory
  count=$(find "$dir" -maxdepth 1 -type f -name "*.npy" | wc -l)
  [ "$count" -eq 9 ] || continue

  parent=$(basename "$dir")
  echo "Renaming in: $dir (parent name: $parent)"

  i=0

  find "$dir" -maxdepth 1 -type f -name "*.npy" -print0 | sort -z |
  while IFS= read -r -d '' f; do
    new="$dir/${parent}_${i}.npy"
    echo "  mv -- \"$f\" \"$new\""
    mv -- "$f" "$new"
    i=$((i+1))
  done
done


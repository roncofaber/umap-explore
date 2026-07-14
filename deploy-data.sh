#!/usr/bin/env bash
# Upload pre-computed HDF5 embeddings to the fly.io volume.
# Usage: ./deploy-data.sh [app-name]
#   app-name defaults to "umap-explore"

set -euo pipefail

APP="${1:-umap-explore}"
LOCAL_DIR="data/embeddings"
REMOTE_DIR="/app/data/embeddings"

H5_FILES=("$LOCAL_DIR"/*.h5)
if [[ ! -e "${H5_FILES[0]}" ]]; then
  echo "No .h5 files found in $LOCAL_DIR — run 'python precompute.py --action all' first."
  exit 1
fi

echo "Uploading ${#H5_FILES[@]} file(s) to $APP:$REMOTE_DIR"
echo ""

# Delete existing files on the volume (fly sftp put refuses to overwrite)
echo "→ Clearing old files on volume..."
fly ssh console -C "sh -c 'rm -f $REMOTE_DIR/*.h5'" -a "$APP" 2>/dev/null || true

# Upload each file
for f in "${H5_FILES[@]}"; do
  name="$(basename "$f")"
  size_mb=$(du -m "$f" | cut -f1)
  echo "→ Uploading $name (${size_mb} MB)..."
  fly sftp put "$f" "$REMOTE_DIR/$name" -a "$APP"
done

echo ""
echo "Done. $APP is using the updated embeddings."

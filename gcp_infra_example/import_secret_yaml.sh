#!/usr/bin/env bash
set -euo pipefail

# Import all key/value pairs from a flat YAML file into Secret Manager:
# - Create secrets if missing (with chosen replication policy)
# - If secret exists, add a new version
# - Optional: bind secretAccessor to a runtime service account
#
# Usage:
#   ./import_secret_yaml.sh \
#     --file ./infra/secret_env.yaml \
#     --project YOUR_PROJECT_ID \
#     [--prefix prod_] \
#     [--replication-policy automatic] \
#     [--bind-sa my-job-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com] \
#     [--dry-run]
#
# Notes:
# - Requires: gcloud, yq (mikefarah version)
# - Secret version size limit: 64 KiB
# - This script does NOT print secret values (to avoid leaks)

FILE=""
PROJECT=""
PREFIX=""
REPLICATION="automatic"
BIND_SA=""
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file) FILE="$2"; shift 2;;
    --project) PROJECT="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --replication-policy) REPLICATION="$2"; shift 2;;
    --bind-sa) BIND_SA="$2"; shift 2;;
    --dry-run) DRY_RUN="1"; shift 1;;
    -h|--help)
      cat <<EOF
Usage:
  $0 --file ./infra/secret_env.yaml --project YOUR_PROJECT_ID [--prefix prod_] [--replication-policy automatic] [--bind-sa SA_EMAIL] [--dry-run]
EOF
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# --- validations ---
if [[ -z "$FILE" || -z "$PROJECT" ]]; then
  echo "ERROR: --file and --project are required"
  exit 1
fi
if [[ ! -f "$FILE" ]]; then
  echo "ERROR: file not found: $FILE"
  exit 1
fi
if ! command -v yq >/dev/null 2>&1; then
  echo "ERROR: yq not found. Install via 'brew install yq' or see https://github.com/mikefarah/yq"
  exit 1
fi
if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not found."
  exit 1
fi

echo ">>> Project:             $PROJECT"
echo ">>> YAML file:           $FILE"
echo ">>> Replication policy:  $REPLICATION"
echo ">>> Prefix:              ${PREFIX:-<none>}"
echo ">>> Bind SA:             ${BIND_SA:-<none>}"
[[ "$DRY_RUN" == "1" ]] && echo ">>> DRY RUN:             ON"
echo

create_secret_if_missing () {
  local name="$1"
  if gcloud secrets describe "$name" --project "$PROJECT" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY] create secret: $name"
  else
    echo "Creating secret: $name"
    gcloud secrets create "$name" \
      --project "$PROJECT" \
      --replication-policy="$REPLICATION" \
      >/dev/null
  fi

  # Optional: bind accessor for runtime SA right after creation
  if [[ -n "$BIND_SA" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY] bind roles/secretmanager.secretAccessor on $name to $BIND_SA"
    else
      gcloud secrets add-iam-policy-binding "$name" \
        --project "$PROJECT" \
        --member="serviceAccount:${BIND_SA}" \
        --role="roles/secretmanager.secretAccessor" \
        >/dev/null
    fi
  fi
}

add_secret_version () {
  local name="$1"
  local value="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY] add version to: $name (len=${#value})"
  else
    # keep exact payload (no trailing newline)
    echo -n "$value" | gcloud secrets versions add "$name" \
      --project "$PROJECT" \
      --data-file=- \
      >/dev/null
    echo "Added version to: $name"
  fi
}

# Iterate YAML keys
# Expecting flat mapping: KEY: value
# Use yq to output "KEY=VALUE" lines
while IFS="=" read -r key val; do
  [[ -z "$key" ]] && continue

  # Trim spaces
  key="$(echo -n "$key" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  val="$(echo -n "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

  # Skip if value is empty
  if [[ -z "$val" ]]; then
    echo "⚠️  Skipping $key (empty value)"
    continue
  fi

  secret_name="${PREFIX}${key}"

  create_secret_if_missing "$secret_name"
  add_secret_version "$secret_name" "$val"

done < <(yq -r 'to_entries | .[] | "\(.key)=\(.value)"' "$FILE")

echo
echo "All secrets processed."
echo "Tip: Grant runtime SA access (if not bound above):"
echo "  gcloud secrets add-iam-policy-binding SECRET_NAME --project $PROJECT --member \"serviceAccount:YOUR_RUNTIME_SA\" --role roles/secretmanager.secretAccessor"

#!/bin/bash
# run_all_routes.sh
# Run ALL available routes in the fail2drive_split directory.
# Skips routes that already have a processed .pkl file.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
F2D_SPLIT="/home/ace428/Soham/HyDrive/Bench2Drive/Fail2Drive/fail2drive_split"
PROCESSED_DIR="$ROOT/processed"

mkdir -p "$PROCESSED_DIR"

# Discover all route names from XML files
mapfile -t ALL_ROUTES < <(ls "$F2D_SPLIT"/*.xml | xargs -n 1 basename | sed 's/.xml//' | sort)

echo "Discovered ${#ALL_ROUTES[@]} total routes."

FAILED=()
SKIPPED=0
RUN=0

for ROUTE in "${ALL_ROUTES[@]}"; do
    # Skip debug routes
    if [[ "$ROUTE" == *"debug"* ]]; then
        echo "  - Skipping debug route: $ROUTE"
        ((SKIPPED++))
        continue
    fi

    OUT_PKL="$PROCESSED_DIR/${ROUTE}_processed.pkl"
    RAW_PKL="$ROOT/close_loop_log/save/${ROUTE}/divergence_logs/route_${ROUTE}.pkl"

    # 1. Skip if already processed
    if [ -f "$OUT_PKL" ]; then
        echo "  - Skipping already processed: $ROUTE"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo ""
    echo "════════════════════════════════════════"
    echo "Processing Route ($((RUN + 1))): $ROUTE"
    echo "════════════════════════════════════════"

    # 2. Run the simulation
    # We use the existing debug_b2d.sh which handles simulation + saving
    if bash "$SCRIPT_DIR/debug_b2d.sh" "$ROUTE"; then
        echo "  ✓ Simulation finished"
    else
        echo "  ✗ Simulation FAILED"
        FAILED+=("$ROUTE")
        continue
    fi

    # 3. Postprocess labels
    if [ -f "$RAW_PKL" ]; then
        conda run -n sparsedrive python "$ROOT/tests/process_logs.py" \
            --input "$RAW_PKL" \
            --out "$PROCESSED_DIR"
        echo "  ✓ Postprocessed → $OUT_PKL"
        RUN=$((RUN + 1))
    else
        echo "  ✗ Raw pkl not found at $RAW_PKL (Check if simulation actually saved data)"
        FAILED+=("$ROUTE")
    fi
done

echo ""
echo "════════════════════════════════════════"
echo "Batch complete."
echo "Total routes: ${#ALL_ROUTES[@]}"
echo "Already processed (skipped): $SKIPPED"
echo "Newly processed: $RUN"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed routes: ${#FAILED[@]}"
    for r in "${FAILED[@]}"; do echo "  - $r"; done
fi

#!/bin/bash
# Run multiple Fail2Drive routes sequentially and postprocess each one.
# Usage: bash scripts/run_batch.sh
# Each route saves to its own directory; no overwrites.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

# ── Routes to run ───────────────────────────────────────────────────────────
# PRIMARY: Base routes - planner has seen similar situations, failures reflect
# genuine planning uncertainty. These are the main dataset for the paper claim.
ROUTES=(
    Base_HardBrake_0035
    Base_HardBrake_0036
    Base_HardBrake_0037
    Base_HardBrake_0038
    Base_HardBrake_0039
    Base_PedestrianCrowd_0065
    Base_PedestrianCrowd_0066
    Base_PedestrianCrowd_0067
    Base_PedestrianCrowd_0068
    Base_PedestrianCrowd_0069
    Base_PedestriansOnRoad_0085
    Base_PedestriansOnRoad_0086
    Base_PedestriansOnRoad_0087
    Base_PedestriansOnRoad_0088
    Base_RightOfWay_0055
    Base_RightOfWay_0056
    Base_RightOfWay_0057
    Base_RightOfWay_0058
    Base_RightOfWay_0059
    Base_Animals_0075
    Base_Animals_0076
    Base_Animals_0077
    Base_Animals_0078
    Base_Animals_0079

    # SECONDARY (OOD): Small held-out Generalization set - test if signal
    # transfers to distribution-shifted scenarios. Report separately.
    Generalization_Animals_1082
    Generalization_HardBrake_1035
    Generalization_HardBrake_1036
    Generalization_PedestrianCrowd_1065
    Generalization_PedestrianCrowd_1066
    Generalization_RightOfWay_1055
)
# ────────────────────────────────────────────────────────────────────────────

PROCESSED_DIR="$ROOT/processed"
mkdir -p "$PROCESSED_DIR"

FAILED=()

for ROUTE in "${ROUTES[@]}"; do
    echo ""
    echo "════════════════════════════════════════"
    echo "Running: $ROUTE"
    echo "════════════════════════════════════════"

    RAW_PKL="$ROOT/close_loop_log/save/${ROUTE}/divergence_logs/route_${ROUTE}.pkl"
    OUT_PKL="$PROCESSED_DIR/${ROUTE}_processed.pkl"

    # Skip if already processed
    if [ -f "$OUT_PKL" ]; then
        echo "  Already processed, skipping."
        continue
    fi

    # Run the route
    if bash "$SCRIPT_DIR/debug_b2d.sh" "$ROUTE"; then
        echo "  ✓ Route finished"
    else
        echo "  ✗ Route FAILED, skipping postprocess"
        FAILED+=("$ROUTE")
        continue
    fi

    # Postprocess labels
    if [ -f "$RAW_PKL" ]; then
        conda run -n sparsedrive python "$ROOT/tools/postprocess_labels.py" \
            --input "$RAW_PKL" \
            --output "$OUT_PKL"
        echo "  ✓ Postprocessed → $OUT_PKL"
    else
        echo "  ✗ Raw pkl not found at $RAW_PKL"
        FAILED+=("$ROUTE")
    fi
done

echo ""
echo "════════════════════════════════════════"
echo "Batch complete."
echo "Processed files: $(ls $PROCESSED_DIR/*_processed.pkl 2>/dev/null | wc -l)"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed routes:"
    for r in "${FAILED[@]}"; do echo "  - $r"; done
fi

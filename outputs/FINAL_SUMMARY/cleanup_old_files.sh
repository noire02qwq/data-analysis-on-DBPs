#!/bin/bash
# Cleanup script for obsolete output files
# Run this script to clean up old bayes optimization and temporary files

echo "=== Cleaning up old bayes optimization directories ==="
for dir in outputs/*bayes*; do
    if [ -d "$dir" ]; then
        echo "Removing: $dir"
        rm -rf "$dir"
    fi
done

echo ""
echo "=== Cleaning up old temporary trial directories ==="
find outputs -type d -name "trial_*" -exec rm -rf {} + 2>/dev/null

echo ""
echo "=== Cleaning up old incomplete outputs ==="
# Keep: caww29_unified, FINAL_SUMMARY, catboost_regressor (latest), finetune_*, lsww29_*
# Remove: old bayes, old temporary, old incomplete

echo "Cleanup complete!"

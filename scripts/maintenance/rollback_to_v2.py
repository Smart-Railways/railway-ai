"""
Rollback to Legacy V2 Model Runbook & Verification Utility.
Safely restores models/legacy/calibrated_xgboost_v2_legacy.pkl to models/production/calibrated_xgboost.pkl.
"""
import shutil
import hashlib
import os
import subprocess
import sys

LEGACY_PATH = "models/legacy/calibrated_xgboost_v2_legacy.pkl"
PROD_PATH = "models/production/calibrated_xgboost.pkl"
EXPECTED_LEGACY_SHA = "3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386"

def get_sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def execute_rollback(dry_run=True):
    print(f"Checking legacy artifact: {LEGACY_PATH}...")
    assert os.path.exists(LEGACY_PATH), "Legacy artifact missing!"
    legacy_sha = get_sha256(LEGACY_PATH)
    assert legacy_sha == EXPECTED_LEGACY_SHA, f"Legacy hash mismatch: {legacy_sha}"
    print("Legacy artifact verified byte-for-byte identical.")

    if dry_run:
        print("[DRY-RUN] Rollback artifact is valid. Run with --execute to perform actual rollback.")
        return

    print(f"Restoring {LEGACY_PATH} to {PROD_PATH}...")
    shutil.copy2(LEGACY_PATH, PROD_PATH)
    assert get_sha256(PROD_PATH) == EXPECTED_LEGACY_SHA, "Restored production hash mismatch!"
    print("Rollback complete. Running regression test suite...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"])
    if res.returncode == 0:
        print("ALL TESTS PASSED. Rollback successful and verified.")
    else:
        print("TEST REGRESSION DETECTED! Investigation required.")

if __name__ == "__main__":
    dry_run = "--execute" not in sys.argv
    execute_rollback(dry_run=dry_run)

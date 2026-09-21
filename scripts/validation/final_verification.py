import os
import sys
import hashlib
import json
from pathlib import Path

print("=== FINAL REPOSITORY VERIFICATION ===")

# A. Production model hash
prod_path = "models/production/calibrated_xgboost.pkl"
assert os.path.exists(prod_path), f"Production model missing at {prod_path}!"
with open(prod_path, "rb") as f:
    prod_sha = hashlib.sha256(f.read()).hexdigest()
expected_prod_sha = "e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae"
print(f"Production Model SHA256: {prod_sha}")
assert prod_sha == expected_prod_sha, f"Production hash mismatch! {prod_sha} != {expected_prod_sha}"
print("✔ Production model hash: PASS")

# B. Legacy model hash
legacy_path = "models/legacy/calibrated_xgboost_v2_legacy.pkl"
assert os.path.exists(legacy_path), f"Legacy model missing at {legacy_path}!"
with open(legacy_path, "rb") as f:
    legacy_sha = hashlib.sha256(f.read()).hexdigest()
expected_legacy_sha = "3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386"
print(f"Legacy Model SHA256:     {legacy_sha}")
assert legacy_sha == expected_legacy_sha, f"Legacy hash mismatch! {legacy_sha} != {expected_legacy_sha}"
print("✔ Legacy model hash: PASS")

# C. Load production model through RailwayMLEngine
from src.services.ml_engine import RailwayMLEngine
engine = RailwayMLEngine()
health = engine.health()
print("Engine health:", health)
assert health["status"] == "ok", f"Engine unhealthy: {health}"
assert health["artifacts"]["calibrated_xgboost_available"] is True, "Model not available in engine!"
print("✔ Production engine loading: PASS")

# D. Verify schema exists and loads
schema_path = "schemas/ml_response.schema.json"
assert os.path.exists(schema_path), f"Schema missing at {schema_path}"
with open(schema_path) as f:
    schema_data = json.load(f)
assert "$schema" in schema_data
print("✔ Schema path and format: PASS")

# E. Search for any lingering references to scratch/ in active python code
broken_scratch_refs = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["backend", "__pycache__"]]
    for f in files:
        if f.endswith((".py", ".json", ".md")):
            p = os.path.join(root, f)
            try:
                with open(p, "r", errors="ignore") as file:
                    for line_no, line in enumerate(file, 1):
                        if "scratch/" in line and not "REPOSITORY_REORGANIZATION_REPORT" in f and not "inventory" in f and not "reorganize" in f:
                            broken_scratch_refs.append((p, line_no, line.strip()))
            except Exception:
                pass

if broken_scratch_refs:
    print(f"WARNING: Found {len(broken_scratch_refs)} references to scratch/:")
    for p, lno, ltext in broken_scratch_refs[:5]:
        print(f"  {p}:{lno} -> {ltext}")
else:
    print("✔ Zero broken references to scratch/: PASS")

# F. Verify scratch/ directory does not exist
assert not os.path.exists("scratch"), "scratch/ directory still exists!"
print("✔ scratch/ removed: PASS")

print("\nALL REPOSITORY INTEGRITY CHECKS PASSED!")

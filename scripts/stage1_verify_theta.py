"""Independently reproduce each frozen theta's reported calibration objective by
re-running Snow17RegionSetup.simulation()/objectivefunction() directly with the
saved calibrated_parameters vector (no optimizer involved)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from stage1_calibrate import Snow17RegionSetup, BOUNDS, OUT_DIR

for region in ["North", "Central", "South"]:
    theta = json.loads((OUT_DIR / "frozen_parameters" / f"theta_{region[0]}.json").read_text())
    setup = Snow17RegionSetup(region)
    vec = [theta["calibrated_parameters"][k] for k in BOUNDS.keys()]
    sim = setup.simulation(vec)
    obj = setup.objectivefunction(sim, setup.evaluation())
    recorded = theta["one_minus_nse"]
    match = abs(obj - recorded) < 1e-6
    print(f"{region}: recorded_1-NSE={recorded:.6f}, reproduced_1-NSE={obj:.6f}, match={match}")

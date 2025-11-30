# ROOT CAUSE SUMMARY

## What We Know For Certain

1. **Math is perfect**: All thrust↔omega conversions work correctly ✅
2. **Configuration**: mass=0.73 kg, motor_tau=0.0001s ✅  
3. **Expected behavior** with action=[0.2, 0.2, 0.2, 0.2]:
   - Thrust per motor: 2.5065 N
   - Acceleration: +3.92 m/s² (upward)
   
4. **Actual behavior**:
   - Acceleration: -5.67 m/s² (downward!)
   - Implied thrust: ~0.755 N per motor (only 30% of expected)

## Critical Discovery

The **ratio is exactly 0.30**: `0.755 / 2.5065 = 0.301`

This suggests the action is being scaled by ~30% somewhere.

## Hypothesis: Mass Mismatch

If the YAML config says `mass: 0.73` but the C++ code is using a **different mass** for normalization, this could explain it!

Let's check:
- If C++ thinks mass = **0.22 kg** (30% of 0.73):
  - act_mean = 0.22 * 9.81 / 4 = 0.539 N
  - act_std = 0.22 * 2 * 9.81 / 4 = 1.078 N
  - With action=0.2: thrust = 0.2 * 1.078 + 0.539 = **0.755 N** ✅

**This matches perfectly!**

## Next Step

The C++ code may be:
1. Not loading the mass from YAML correctly
2. Using a hardcoded different mass value
3. Mass being overridden somewhere

**USER: Please check what mass value is ACTUALLY being used in the C++ backend**

Run this in Docker:

```python
python3 <<'EOF'
import os
os.environ['FLIGHTMARE_PATH'] = '/root/flightmare'

from flightgym import QuadrotorEnv_v1

cfg_yaml = """
quadrotor_env:
  sim_dt: 0.02
  max_t: 5.0

quadrotor_dynamics:
  mass: 0.73
  arm_l: 0.17
  motor_omega_min: 150.0
  motor_omega_max: 3000.0
  motor_tau: 0.0001
  thrust_map: [1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325]
  kappa: 0.016
  omega_max: [6.0, 6.0, 6.0]

rl:
  pos_coeff: -0.002
  ori_coeff: -0.002
  lin_vel_coeff: -0.0002
  ang_vel_coeff: -0.0002
  act_coeff: -0.0002
"""

env = QuadrotorEnv_v1(cfg_yaml, False)
print(env)  # This should print the environment parameters including mass
EOF
```

This will print the actual mass being used!

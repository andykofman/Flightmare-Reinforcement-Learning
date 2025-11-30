# CRITICAL BUG FOUND: Motor Tau Value

## Summary

With `motor_tau = 0.0001` seconds (extremely fast response):
- `motor_tau_inv = 10,000`
- `c = exp(-0.02 * 10,000) ≈ 0`
- Motors respond almost instantly

## The Problem

Motors start at `motor_omega_ = 0` RPM when environment resets.

With the current polynomial `thrust =  1.33e-6 * omega² + 0.0038 * omega - 1.769`:
- At omega=0: **thrust = -1.769 N** (NEGATIVE!)
- Motors are clamped to [150, 3000] RPM
- But thrust is calculated AFTER omega, not before

## Calculation with motor_tau = 0.0001

Step 1 (t=0.02s):
- Desired thrust with action=0.2: **2.507 N/motor**
- Desired omega: **858.84 RPM**
- Previous omega: **0 RPM** (reset)
- c = exp(-200) ≈ 0
- **New omega: 0 * 0 + 858.84 * 1 = 858.84 RPM** ✅
- **Actual thrust: 2.507 N** ✅

Wait, this should work...

## Re-checking Actual Observed Behavior

From test output:
- Expected acceleration: +3.92 m/s²
- Observed acceleration: -5.67 m/s²
- This implies actual thrust: **0.755 N/motor** (only 30% of expected!)

## Hypothesis: Thrust Polynomial Bug

The reverse calculation from omega to thrust might be broken!

Let me verify the `motorOmegaToThrust` function is correct...

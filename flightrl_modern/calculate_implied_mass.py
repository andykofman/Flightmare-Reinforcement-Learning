#!/usr/bin/env python3
"""
Calculate what mass is actually being used based on observed behavior
"""

# Test by directly computing what thrust we expect
MASS_CONFIGURED = 0.73
GZ = -9.81

act_mean_expected = MASS_CONFIGURED * (-GZ) / 4
act_std_expected = MASS_CONFIGURED * 2 * (-GZ) / 4

print(f"If mass={MASS_CONFIGURED}:")
print(f"  act_mean should be: {act_mean_expected:.4f} N")
print(f"  act_std should be: {act_std_expected:.4f} N")
print(f"  With action=0.2: thrust = {0.2 * act_std_expected + act_mean_expected:.4f} N")

# Now what if mass is actually being used as default 1.0?
MASS_MAYBE = 1.0
act_mean_if_1kg = MASS_MAYBE * (-GZ) / 4  
act_std_if_1kg = MASS_MAYBE * 2 * (-GZ) / 4

print(f"\nIf mass={MASS_MAYBE} (default):")
print(f"  act_mean would be: {act_mean_if_1kg:.4f} N")
print(f"  act_std would be: {act_std_if_1kg:.4f} N")
print(f"  With action=0.2: thrust = {0.2 * act_std_if_1kg + act_mean_if_1kg:.4f} N")

# Calculate implied mass from observed behavior
# We observed ~0.755 N with action=0.2
# So: 0.755 = 0.2 * act_std + act_mean
# And: act_mean = mass * 9.81 / 4
# And: act_std = mass * 9.81 / 2
# Therefore: 0.755 = 0.2 * (mass * 9.81 / 2) + (mass * 9.81 / 4)
# 0.755 = mass * 9.81 * (0.1 + 0.25)
# 0.755 = mass * 9.81 * 0.35
# mass = 0.755 / (9.81 * 0.35)

MASS_IMPLIED = 0.755 / (9.81 * 0.35)
print(f"\nIMPLIED mass from observed thrust (0.755 N): {MASS_IMPLIED:.4f} kg")
print(f"Ratio to configured: {MASS_IMPLIED / MASS_CONFIGURED:.2%}")

print("\n" + "="*60)
print("CONCLUSION:")
if abs(MASS_IMPLIED - MASS_CONFIGURED) < 0.01:
    print(f"✅ Mass is correct ({MASS_CONFIGURED} kg)")
elif abs(MASS_IMPLIED - 1.0) < 0.01:
    print(f"❌ BUG: Using default mass (1.0 kg) instead of configured ({MASS_CONFIGURED} kg)")
else:
    print(f"❌ BUG: Using unknown mass ({MASS_IMPLIED:.4f} kg)")
    print(f"   Expected: {MASS_CONFIGURED} kg")

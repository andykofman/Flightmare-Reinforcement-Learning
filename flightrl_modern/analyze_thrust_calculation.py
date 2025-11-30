#!/usr/bin/env python3
"""
Detailed thrust calculation analysis to find the bug
"""
import numpy as np

# From quadrotor_env.yaml
MASS = 0.73  # kg
GRAVITY = 9.81  # m/s²
MOTOR_OMEGA_MIN = 150.0
MOTOR_OMEGA_MAX = 3000.0
THRUST_MAP = np.array([1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325])

print("="*80)
print("  THRUST CALCULATION ANALYSIS")
print("="*80)

# From quadrotor_env.cpp line 39-40:
act_mean = MASS * GRAVITY / 4  # Hover thrust per motor
act_std = MASS * 2 * GRAVITY / 4  # Action scaling

print(f"\nQuadrotor Parameters:")
print(f"  Mass: {MASS} kg")
print(f"  Gravity: {GRAVITY} m/s²")
print(f"  Required hover thrust: {MASS * GRAVITY:.4f} N")

print(f"\nAction Normalization:")
print(f"  act_mean (per motor): {act_mean:.4f} N")
print(f"  act_std (per motor): {act_std:.4f} N")

# Thrust map polynomial: thrust = a*omega² + b*omega + c
a, b, c = THRUST_MAP
print(f"\nThrust Map Polynomial:")
print(f"  thrust = {a:.4e} * omega² + {b:.4f} * omega + {c:.4f}")

# Calculate omega for hover thrust
# Given thrust, solve: a*omega² + b*omega + c - thrust = 0
def thrust_to_omega(thrust):
    """Solve quadratic: a*omega² + b*omega + (c - thrust) = 0"""
    discriminant = b**2 - 4*a*(c - thrust)
    if discriminant < 0:
        return None
    omega = (-b + np.sqrt(discriminant)) / (2*a)
    return omega

# What omega is needed for hover thrust per motor?
hover_omega = thrust_to_omega(act_mean)
print(f"\nHover Calculation (action=0):")
print(f"  Required thrust per motor: {act_mean:.4f} N")
print(f"  Required omega: {hover_omega:.2f} RPM")
print(f"  Motor omega range: [{MOTOR_OMEGA_MIN}, {MOTOR_OMEGA_MAX}] RPM")

if hover_omega:
    if hover_omega < MOTOR_OMEGA_MIN:
        print(f"  ❌ ERROR: Hover omega ({hover_omega:.2f}) < min ({MOTOR_OMEGA_MIN})")
        print(f"  → Motors CANNOT produce hover thrust!")
    elif hover_omega > MOTOR_OMEGA_MAX:
        print(f"  ❌ ERROR: Hover omega ({hover_omega:.2f}) > max ({MOTOR_OMEGA_MAX})")
    else:
        print(f"  ✅ Hover omega within range")

# Test various actions
print("\n" + "="*80)
print("  ACTION TO THRUST MAPPING")
print("="*80)

test_actions = [
    ("Zero (hover)", 0.0),
    ("+20% (should rise)", 0.2),
    ("+50% (strong rise)", 0.5),
    ("-20% (descent)", -0.2),
]

for name, action_val in test_actions:
    # Denormalize: thrust = action * act_std + act_mean
    thrust_per_motor = action_val * act_std + act_mean
    total_thrust = 4 * thrust_per_motor
    
    # Convert to omega
    omega = thrust_to_omega(thrust_per_motor)
    
    # Check clamping
    if omega:
        omega_clamped = np.clip(omega, MOTOR_OMEGA_MIN, MOTOR_OMEGA_MAX)
        # Convert back to thrust after clamping
        actual_thrust = a * omega_clamped**2 + b * omega_clamped + c
        total_actual = 4 * actual_thrust
    else:
        omega_clamped = MOTOR_OMEGA_MIN
        actual_thrust = 0
        total_actual = 0
    
    # Calculate net force
    weight = MASS * GRAVITY
    net_force = total_actual - weight
    acceleration = net_force / MASS
    
    print(f"\n{name} (action={action_val:+.1f}):")
    print(f"  Denormalized thrust: {thrust_per_motor:.4f} N/motor, {total_thrust:.4f} N total")
    print(f"  Required omega: {omega:.2f} RPM" if omega else "  Required omega: INVALID")
    print(f"  Clamped omega: {omega_clamped:.2f} RPM")
    print(f"  Actual thrust after clamp: {actual_thrust:.4f} N/motor, {total_actual:.4f} N total")
    print(f"  Net force: {net_force:+.4f} N")
    print(f"  Acceleration: {acceleration:+.4f} m/s²")
    
    if acceleration > 0.5:
        print(f"  → Should RISE")
    elif acceleration < -0.5:
        print(f"  → Should FALL")
    else:
        print(f"  → Should HOVER")

# Check if thrust polynomial makes sense
print("\n" + "="*80)
print("  THRUST POLYNOMIAL SANITY CHECK")
print("="*80)

test_omegas = [150, 500, 1000, 1500, 2000, 2500, 3000]
print("\nOmega (RPM) → Thrust (N):")
for omega in test_omegas:
    thrust = a * omega**2 + b * omega + c
    print(f"  {omega:4.0f} RPM → {thrust:6.3f} N")

# Check if negative thrust at low omega
thrust_at_min = a * MOTOR_OMEGA_MIN**2 + b * MOTOR_OMEGA_MIN + c
print(f"\nThrust at motor_omega_min ({MOTOR_OMEGA_MIN} RPM): {thrust_at_min:.4f} N")
if thrust_at_min < 0:
    print("  ❌ ERROR: Negative thrust at minimum omega!")
    print("  → This polynomial is INVALID for this omega range!")

print("\n" + "="*80)
print("  DIAGNOSIS")
print("="*80)

# Calculate what SHOULD happen with action=0.2
action_02_thrust = 0.2 * act_std + act_mean
action_02_total = 4 * action_02_thrust
action_02_accel = (action_02_total - weight) / MASS

print(f"\nWith action=[0.2, 0.2, 0.2, 0.2]:")
print(f"  Expected total thrust: {action_02_total:.4f} N")
print(f"  Weight: {weight:.4f} N") 
print(f"  Expected acceleration: {action_02_accel:+.4f} m/s²")

if action_02_accel > 0:
    print(f"  → Quad SHOULD rise!")
    print(f"\n  🐛 BUG CONFIRMED: Physics validation shows it falls instead")
    print(f"     Possible causes:")
    print(f"     1. Thrust polynomial produces negative thrust")
    print(f"     2. Omega clamping limits thrust too much")
    print(f"     3. Motor dynamics (tau) prevents thrust application")
    print(f"     4. Thrust not being applied correctly in C++")

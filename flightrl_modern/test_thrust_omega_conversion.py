#!/usr/bin/env python3
"""
Test if motorThrustToOmega and motorOmegaToThrust are inverse functions
"""
import numpy as np

# From config
thrust_map = np.array([1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325])
a, b, c = thrust_map

print("Testing thrust <-> omega conversion")
print(f"Polynomial: thrust = {a:.4e} * omega² + {b:.4f} * omega + {c:.4f}")

def motorOmegaToThrust(omega):
    """C++ implementation"""
    return a * omega**2 + b * omega + c

def motorThrustToOmega(thrust):
    """C++ implementation"""
    scale = 1.0 / (2.0 * a)
    offset = -b * scale
    discriminant = b**2 - 4.0 * a * (c - thrust)
    if discriminant < 0:
        return None
    root = np.sqrt(discriminant)
    omega = offset + scale * root
    return omega

print("\n" + "="*60)
print("Testing Roundtrip Conversion")
print("="*60)

test_omegas = [150, 500, 858.84, 1000, 2000, 3000]

for omega_orig in test_omegas:
    thrust = motorOmegaToThrust(omega_orig)
    omega_back = motorThrustToOmega(thrust)
    error = abs(omega_back - omega_orig) if omega_back else None
    
    print(f"\nOmega: {omega_orig:7.2f} RPM")
    print(f"  → Thrust: {thrust:7.3f} N")
    print(f"  → Omega:  {omega_back:7.2f} RPM" if omega_back else "  → Omega:  INVALID")
    if error is not None:
        print(f"  Error: {error:.6f} RPM {'✅' if error < 0.01 else '❌'}")
    

print("\n" + "="*60)
print("Testing Specific Action=0.2 Case")
print("="*60)

# With action=0.2
MASS = 0.73
GRAVITY = 9.81
act_std = MASS * 2 * GRAVITY / 4
act_mean = MASS * GRAVITY / 4

thrust_desired = 0.2 * act_std + act_mean
print(f"\nAction: 0.2")
print(f"Desired thrust: {thrust_desired:.4f} N")

omega_calc = motorThrustToOmega(thrust_desired)
print(f"Calculated omega: {omega_calc:.2f} RPM")

thrust_actual = motorOmegaToThrust(omega_calc)
print(f"Actual thrust from omega: {thrust_actual:.4f} N")

print(f"Match: {'✅' if abs(thrust_actual - thrust_desired) < 0.001 else '❌'}")

# Now simulate what happens starting from omega=0
print("\n" + "="*60)
print("Simulating Motor Dynamics Starting from omega=0")
print("="*60)

motor_tau = 0.0001
motor_tau_inv = 1.0 / motor_tau
sim_dt = 0.02

motor_omega = 0.0  # Initial
motor_omega_desired = omega_calc

print(f"\nMotor tau: {motor_tau} s (tau_inv = {motor_tau_inv})")
print(f"Simulation timestep: {sim_dt} s")
print(f"Initial motor omega: {motor_omega} RPM")
print(f"Desired motor omega: {motor_omega_desired:.2f} RPM")

# First-order system response
c_exp = np.exp(-sim_dt * motor_tau_inv)
print(f"\nExponential coefficient: c = exp(-{sim_dt} * {motor_tau_inv}) = {c_exp:.10f}")

motor_omega_new = c_exp * motor_omega + (1.0 - c_exp) * motor_omega_desired
print(f"New motor omega after 1 timestep: {motor_omega_new:.2f} RPM")

thrust_new = motorOmegaToThrust(motor_omega_new)
print(f"Thrust from new omega: {thrust_new:.4f} N")

print(f"\nExpected: {thrust_desired:.4f} N")
print(f"Actual:   {thrust_new:.4f} N")
print(f"Match: {'✅' if abs(thrust_new - thrust_desired) < 0.001 else '❌'}")

if abs(thrust_new - thrust_desired) > 0.001:
    print(f"\n❌ BUG: Thrust doesn't match after motor dynamics!")
    print(f"   Difference: {thrust_new - thrust_desired:.4f} N")

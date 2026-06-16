import numpy as np
from ddp_solver import (VehicleParams, VehicleDynamics, se2_exp, se2_log,
                         DDPSolver, ddp_solve, trajectory_cost)

np.set_printoptions(precision=4, suppress=True)

# --- Setup ---
params = VehicleParams()
dyn = VehicleDynamics(params)

dt = 0.05
N = 15

T0 = np.eye(3)          # start at origin, heading along +x
v0 = np.array([5.0, 0.0, 0.0])   # 5 m/s forward, no lateral/yaw

# Reference: drive straight, target speed 8 m/s, target y=0
v_target = 8.0

def stage_cost(T, v, u, k):
    x, y = T[0, 2], T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    vx, vy, omega = v
    delta, throttle = u

    cost = 0.0
    cost += 5.0 * (y ** 2)              # stay near y=0
    cost += 2.0 * (theta ** 2)          # stay aligned with x-axis
    cost += 1.0 * (vx - v_target) ** 2  # track target speed
    cost += 0.5 * vy ** 2
    cost += 0.1 * omega ** 2
    cost += 0.05 * delta ** 2
    cost += 0.01 * throttle ** 2
    return cost

def terminal_cost(T, v):
    x, y = T[0, 2], T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    vx, vy, omega = v
    cost = 0.0
    cost += 20.0 * (y ** 2)
    cost += 10.0 * (theta ** 2)
    cost += 5.0 * (vx - v_target) ** 2
    cost += 2.0 * vy ** 2
    cost += 2.0 * omega ** 2
    return cost

# --- Initial guess: zero steering, mild throttle ---
U_init = np.zeros((N, 2))
U_init[:, 1] = 0.3   # mild throttle

solver = DDPSolver(dyn, stage_cost, terminal_cost, dt)

print("Initial rollout cost:")
Ts0, Vs0 = dyn.simulate(T0, v0, U_init, dt)
print("  J =", trajectory_cost(solver, Ts0, Vs0, U_init))
print("  final pos:", Ts0[-1][:2, 2], " final v:", Vs0[-1])
print()

print("Running DDP...")
Ts, Vs, Us, J_hist = ddp_solve(solver, T0, v0, U_init, max_iters=20, verbose=True)

print()
print("Final cost:", J_hist[-1])
print("Final position (x,y):", Ts[-1][:2, 2])
print("Final heading (rad):", np.arctan2(Ts[-1][1, 0], Ts[-1][0, 0]))
print("Final velocity [vx,vy,omega]:", Vs[-1])
print()
print("Optimized controls (delta, throttle):")
print(Us)

# --- Sanity checks ---
print()
print("=== Sanity checks ===")

# 1. SE(2) membership check for every pose in trajectory
max_det_err = 0.0
max_orth_err = 0.0
for T in Ts:
    R = T[:2, :2]
    max_det_err = max(max_det_err, abs(np.linalg.det(R) - 1.0))
    max_orth_err = max(max_orth_err, np.max(np.abs(R @ R.T - np.eye(2))))
    assert np.allclose(T[2, :], [0, 0, 1])
print(f"max |det(R)-1| over trajectory: {max_det_err:.2e}")
print(f"max |R R^T - I| over trajectory: {max_orth_err:.2e}")

# 2. exp/log inverse check
test_xi = np.array([0.3, -0.1, 0.7])
T_test = se2_exp(test_xi)
xi_back = se2_log(T_test)
print(f"exp/log round-trip error: {np.max(np.abs(test_xi - xi_back)):.2e}")

# 3. zero-slip Pacejka check
F0 = dyn.lateral_force(0.0, params.Fz_f)
print(f"lateral force at zero slip angle: {F0:.2e}")
F0L = dyn.longitudinal_force(0.0, params.Fz_r)
print(f"longitudinal force at zero slip ratio: {F0L:.2e}")

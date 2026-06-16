# Formula Student — SE(2) DDP Trajectory Optimizer

A from-scratch, pure NumPy implementation of a **Differential Dynamic Programming (DDP)** trajectory optimizer for a Formula Student race car, built on rigorous Lie-group geometry.

> Built in four phases with Claude as a Senior Principal Engineer in Autonomous Vehicle Dynamics and Optimal Control.

---

## Live Interactive Simulator

**[Launch in browser →](https://mattyaldridge07-ship-it.github.io/DDPTrajectoryOptimiser/sim/)**

Adjust all vehicle, tire, and control parameters live. Powered by the SE(2) exponential-map integrator running in pure JavaScript — no server needed.

---

## Architecture

```
fs-ddp/
├── solver/
│   ├── ddp_solver.py     # Phases 1–3: dynamics, backward pass, forward pass
│   └── test_basic.py     # Smoke test — runs a 15-step straight-line scenario
├── sim/
│   └── index.html        # Interactive browser simulator (Phase 1 forward sim)
└── docs/
    └── math_notes.md     # Key equations and geometric derivations
```

---

## Four Phases

### Phase 1 — SE(2) Vehicle Dynamics & Pacejka Tire Model
- Rigid-body bicycle model with full nonlinear **Pacejka Magic Formula** tires (lateral + longitudinal)
- Vehicle pose represented as an element of the matrix Lie group **SE(2)** — no Euler angles, no singularities
- Discrete-time integration via the **SE(2) exponential map**: `T_{k+1} = T_k @ exp(dt * v_{k+1})`, guaranteeing exact manifold membership at every step

### Phase 2 — Lie-Algebraic Backward Pass (Riccati Equations)
- Error state `δx ∈ R⁶` defined via the **logarithmic map** (`se2_log`) as a local tangent-space perturbation
- Central finite-difference Jacobians/Hessians mapped correctly into the body-frame chart
- Q-function expansions `Q_xx (6×6)`, `Q_uu (2×2)`, `Q_ux (2×6)` with Tikhonov regularization on `Q_uu`
- Produces feedforward `k (2,)` and feedback gains `K (2×6)` per timestep

### Phase 3 — Forward Pass & Manifold Retraction
- New controls: `u_new = u_bar + α·k + K @ δx` where `δx = state_error(T, v, T_bar, v_bar)`
- Pose retraction back onto SE(2) via `dynamics.step()` → `se2_exp` (never drifts off the manifold)
- **Backtracking line search** with Armijo condition using the predicted cost reduction `dV` from the backward pass
- Adaptive regularization schedule (grow on failure, shrink on success)

### Phase 4 — Augmented Lagrangian Constraints
- **Track boundary** constraints: `|e_lat(T)| ≤ half_width` (two inequalities via centerline projection)
- **Friction circle** constraints: `Fx² + Fy² ≤ (μ·Fz)²` for front and rear axles
- C¹ AL penalty `ψ(g, λ, ρ)` folded into the stage cost — allows infeasible initial trajectories
- Outer loop: dual ascent `λ ← max(0, λ + ρg)` + penalty growth `ρ ← β·ρ` until feasibility

---

## Quickstart

```bash
git clone https://github.com/mattyaldridge07-ship-it/DDPTrajectoryOptimiser.git
cd DDPTrajectoryOptimiser
python solver/test_basic.py
```

**Requirements:** Python 3.8+, NumPy only. No other dependencies.

Expected output:
```
Initial rollout cost:
  J = 51.72
Running DDP...
[iter 0] alpha=1  J: 51.720502 -> 41.701198  (dJ=1.002e+01)
...
Converged at iter N
max |det(R)-1| over trajectory: 0.00e+00   ← exact SE(2) membership
exp/log round-trip error:       1.39e-17   ← machine precision
lateral force at zero slip angle: 0.00e+00 ← Pacejka verified
```

---

## Key Mathematical Objects

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| `T ∈ SE(2)` | Vehicle pose (3×3 homogeneous matrix) | manifold, dim 3 |
| `v = [vx, vy, ω]` | Body-frame velocities | R³ |
| `δx` | Lie-algebraic error state | R⁶ |
| `u = [δ, throttle]` | Control: steering + drive/brake | R² |
| `A` | Discrete dynamics Jacobian (tangent-space) | 6×6 |
| `B` | Control influence matrix | 6×2 |
| `K` | DDP feedback gain | 2×6 |

---

## Enabling GitHub Pages (for the live simulator)

1. Go to **Settings → Pages** in your repo
2. Set source: **Deploy from branch**, branch: `main`, folder: `/sim`
3. Wait ~60 seconds → live at `https://mattyaldridge07-ship-it.github.io/DDPTrajectoryOptimiser/`

# Mathematical Notes — SE(2) DDP

## SE(2) Exponential Map

Given a body-frame twist ξ = [vx, vy, ω] integrated over dt:

```
exp(ξ) = [[cos θ,  -sin θ,  V·[vx,vy]_x],
           [sin θ,   cos θ,  V·[vx,vy]_y],
           [0,       0,      1           ]]

where θ = ω·dt,   V = [[A, -B], [B, A]]
      A = sin θ / θ,   B = (1 - cos θ) / θ      (Taylor-expanded for θ → 0)
```

Pose update:  `T_{k+1} = T_k @ exp(dt · v_{k+1})`

The rotation block is built from `cos`/`sin` of a **single scalar** θ — it
is always exactly orthogonal, regardless of step size. No drift accumulates.

## Lie-Algebraic Error State

Right perturbation (body-frame convention):

```
T = T_bar @ exp(δξ)    =>    δξ = log(T_bar⁻¹ @ T)
```

Combined 6D error state:

```
δx = [δξ;  v - v_bar]  ∈  R⁶
```

All DDP Jacobians are computed with respect to δx. Perturbing with eps ∈ R⁶:

```
T_pert = T_bar @ exp(eps[:3])
v_pert = v_bar + eps[3:]
```

## DDP Q-Function (iLQR / Gauss-Newton)

```
Q_x  = l_x  + Aᵀ V_x          (6,)
Q_u  = l_u  + Bᵀ V_x          (2,)
Q_xx = l_xx + Aᵀ V_xx A       (6,6)
Q_uu = l_uu + Bᵀ V_xx B       (2,2)   ← Tikhonov: Q_uu + ρI before invert
Q_ux = l_ux + Bᵀ V_xx A       (2,6)
```

Gains:  `k = -Q_uu⁻¹ Q_u`  (2,),   `K = -Q_uu⁻¹ Q_ux`  (2,6)

Value function recursion:

```
V_x  = Q_x  + Kᵀ(Q_uu k + Q_u) + Q_ux·k
V_xx = Q_xx + Kᵀ Q_uu K + Kᵀ Q_ux + Q_ux·K     (symmetrized)
```

## Augmented Lagrangian Penalty (inequalities g ≤ 0)

```
ψ(g, λ, ρ) = λg + ρ/2 · g²       if  g ≥ 0  or  λ + ρg ≥ 0   (active)
             -λ² / (2ρ)           otherwise                      (inactive)
```

C¹ at the switching point g = -λ/ρ (both value and derivative match).

Dual update (outer loop):  `λ ← max(0, λ + ρg)`,  `ρ ← min(β·ρ, ρ_max)`

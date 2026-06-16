"""
Pure NumPy SE(2) DDP trajectory optimizer for a Formula Student vehicle.
Phases 1-4: dynamics+tires, Lie-algebraic backward pass, forward pass with
manifold retraction, and Augmented Lagrangian constraints.
"""

import numpy as np
from dataclasses import dataclass, field


# ============================================================================
# PHASE 1
# ============================================================================

@dataclass
class VehicleParams:
    mass: float = 230.0
    Iz: float = 120.0
    lf: float = 0.8
    lr: float = 0.7
    track_f: float = 1.2
    track_r: float = 1.2

    B_lat: float = 10.0
    C_lat: float = 1.9
    D_lat: float = 1.0
    E_lat: float = 0.97

    B_long: float = 11.0
    C_long: float = 1.9
    D_long: float = 1.0
    E_long: float = 0.97

    g: float = 9.81

    Fz_f: float = field(init=False)
    Fz_r: float = field(init=False)

    def __post_init__(self):
        total_weight = self.mass * self.g
        self.Fz_f = total_weight * (self.lr / (self.lf + self.lr)) / 2.0
        self.Fz_r = total_weight * (self.lf / (self.lf + self.lr)) / 2.0


def se2_hat(xi):
    vx, vy, omega = xi
    return np.array([
        [0.0,   -omega, vx],
        [omega,  0.0,   vy],
        [0.0,    0.0,   0.0]
    ])


def se2_vee(xi_hat):
    return np.array([xi_hat[0, 2], xi_hat[1, 2], xi_hat[1, 0]])


def se2_exp(xi):
    vx, vy, omega = xi
    theta = omega

    if np.abs(theta) < 1e-8:
        A = 1.0 - (theta ** 2) / 6.0
        B = theta / 2.0 - (theta ** 3) / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta

    V = np.array([
        [A, -B],
        [B,  A]
    ])

    t = V @ np.array([vx, vy])
    c, s = np.cos(theta), np.sin(theta)

    T = np.array([
        [c, -s, t[0]],
        [s,  c, t[1]],
        [0.0, 0.0, 1.0]
    ])
    return T


def se2_log(T):
    theta = np.arctan2(T[1, 0], T[0, 0])
    t = T[:2, 2]

    if np.abs(theta) < 1e-8:
        A = 1.0 - (theta ** 2) / 6.0
        B = theta / 2.0 - (theta ** 3) / 24.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta

    V = np.array([
        [A, -B],
        [B,  A]
    ])
    v = np.linalg.solve(V, t)
    return np.array([v[0], v[1], theta])


class VehicleDynamics:
    def __init__(self, params: VehicleParams):
        self.p = params

    @staticmethod
    def _pacejka(slip, B, C, D, E, Fz):
        Bx = B * slip
        return Fz * D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))

    def lateral_force(self, alpha, Fz):
        p = self.p
        return self._pacejka(alpha, p.B_lat, p.C_lat, p.D_lat, p.E_lat, Fz)

    def longitudinal_force(self, kappa, Fz):
        p = self.p
        return self._pacejka(kappa, p.B_long, p.C_long, p.D_long, p.E_long, Fz)

    def _slip_angles(self, vx, vy, omega, delta):
        p = self.p
        vx_safe = np.sign(vx) * max(np.abs(vx), 1e-3) if vx != 0.0 else 1e-3

        vy_f = vy + p.lf * omega
        vy_r = vy - p.lr * omega

        alpha_f = delta - np.arctan2(vy_f, vx_safe)
        alpha_r = -np.arctan2(vy_r, vx_safe)
        return alpha_f, alpha_r

    def body_accelerations(self, v, u):
        p = self.p
        vx, vy, omega = v
        delta, throttle = u

        alpha_f, alpha_r = self._slip_angles(vx, vy, omega, delta)

        Fyf = 2.0 * self.lateral_force(alpha_f, p.Fz_f)
        Fyr = 2.0 * self.lateral_force(alpha_r, p.Fz_r)

        kappa_cmd = np.clip(throttle, -1.0, 1.0) * 0.15

        if throttle >= 0.0:
            Fxr = 2.0 * self.longitudinal_force(kappa_cmd, p.Fz_r)
            Fxf = 0.0
        else:
            Fxr = 2.0 * self.longitudinal_force(kappa_cmd, p.Fz_r)
            Fxf = 2.0 * self.longitudinal_force(kappa_cmd, p.Fz_f)

        Fx_total = Fxf * np.cos(delta) - Fyf * np.sin(delta) + Fxr
        Fy_total = Fyf * np.cos(delta) + Fxf * np.sin(delta) + Fyr

        ax = Fx_total / p.mass + omega * vy
        ay = Fy_total / p.mass - omega * vx

        M_yaw = p.lf * (Fyf * np.cos(delta) + Fxf * np.sin(delta)) - p.lr * Fyr
        omega_dot = M_yaw / p.Iz

        acc = np.array([ax, ay, omega_dot])
        forces = {"Fyf": Fyf, "Fyr": Fyr, "Fxf": Fxf, "Fxr": Fxr}
        return acc, forces

    def step(self, T, v, u, dt):
        acc, _ = self.body_accelerations(v, u)
        v_next = v + dt * acc

        xi = dt * v_next
        T_next = T @ se2_exp(xi)

        return T_next, v_next

    def simulate(self, T0, v0, U, dt):
        N = U.shape[0]
        Ts = [T0]
        Vs = np.zeros((N + 1, 3))
        Vs[0] = v0

        T = T0
        v = v0
        for k in range(N):
            T, v = self.step(T, v, U[k], dt)
            Ts.append(T)
            Vs[k + 1] = v

        return Ts, Vs


# ============================================================================
# PHASE 2
# ============================================================================

def retract(T, v, eps):
    eps_pose = eps[:3]
    eps_vel = eps[3:]
    T_pert = T @ se2_exp(eps_pose)
    v_pert = v + eps_vel
    return T_pert, v_pert


def state_error(T, v, T_bar, v_bar):
    pose_err = se2_log(np.linalg.inv(T_bar) @ T)
    vel_err = v - v_bar
    return np.concatenate([pose_err, vel_err])


def _dynamics_step_in_chart(dynamics, T_bar, v_bar, u_bar, dt, eps_x, eps_u):
    T, v = retract(T_bar, v_bar, eps_x)
    u = u_bar + eps_u
    T_next, v_next = dynamics.step(T, v, u, dt)

    T_next_bar, v_next_bar = dynamics.step(T_bar, v_bar, u_bar, dt)

    return state_error(T_next, v_next, T_next_bar, v_next_bar)


def _cost_in_chart(cost_fn, T_bar, v_bar, u_bar, eps_x, eps_u):
    T, v = retract(T_bar, v_bar, eps_x)
    u = u_bar + eps_u
    return cost_fn(T, v, u)


def _terminal_cost_in_chart(cost_fn, T_bar, v_bar, eps_x):
    T, v = retract(T_bar, v_bar, eps_x)
    return cost_fn(T, v)


def finite_diff_jacobian(fun, x0, h=1e-5):
    n = x0.shape[0]
    f0 = fun(x0)
    m = f0.shape[0] if hasattr(f0, "shape") and f0.shape != () else 1
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = h
        f_plus = np.atleast_1d(fun(x0 + dx))
        f_minus = np.atleast_1d(fun(x0 - dx))
        J[:, i] = (f_plus - f_minus) / (2 * h)
    return J


def finite_diff_gradient(fun, x0, h=1e-5):
    n = x0.shape[0]
    g = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = h
        g[i] = (fun(x0 + dx) - fun(x0 - dx)) / (2 * h)
    return g


def finite_diff_hessian(fun, x0, h=1e-4):
    n = x0.shape[0]
    H = np.zeros((n, n))
    f0 = fun(x0)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h
        for j in range(i, n):
            ej = np.zeros(n); ej[j] = h
            if i == j:
                fpp = fun(x0 + 2 * ei)
                fmm = fun(x0 - 2 * ei)
                H[i, i] = (fpp - 2 * f0 + fmm) / (4 * h ** 2)
            else:
                fpp = fun(x0 + ei + ej)
                fpm = fun(x0 + ei - ej)
                fmp = fun(x0 - ei + ej)
                fmm = fun(x0 - ei - ej)
                val = (fpp - fpm - fmp + fmm) / (4 * h ** 2)
                H[i, j] = val
                H[j, i] = val
    return 0.5 * (H + H.T)


class DDPSolver:
    N_X = 6
    N_U = 2

    def __init__(self, dynamics, stage_cost, terminal_cost, dt, reg_init=1e-6):
        self.dyn = dynamics
        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost
        self.dt = dt
        self.reg = reg_init

    def _linearize_dynamics(self, T_bar, v_bar, u_bar):
        zx = np.zeros(self.N_X)
        zu = np.zeros(self.N_U)

        f_of_x = lambda eps_x: _dynamics_step_in_chart(
            self.dyn, T_bar, v_bar, u_bar, self.dt, eps_x, zu)
        f_of_u = lambda eps_u: _dynamics_step_in_chart(
            self.dyn, T_bar, v_bar, u_bar, self.dt, zx, eps_u)

        A = finite_diff_jacobian(f_of_x, zx)
        B = finite_diff_jacobian(f_of_u, zu)
        return A, B

    def _expand_stage_cost(self, T_bar, v_bar, u_bar, k):
        zx = np.zeros(self.N_X)
        zu = np.zeros(self.N_U)
        z = np.zeros(self.N_X + self.N_U)

        def l_joint(z_vec):
            ex = z_vec[:self.N_X]
            eu = z_vec[self.N_X:]
            return _cost_in_chart(lambda T, v, u: self.stage_cost(T, v, u, k),
                                   T_bar, v_bar, u_bar, ex, eu)

        l_of_x = lambda ex: l_joint(np.concatenate([ex, zu]))
        l_of_u = lambda eu: l_joint(np.concatenate([zx, eu]))

        lx = finite_diff_gradient(l_of_x, zx)
        lu = finite_diff_gradient(l_of_u, zu)
        lxx = finite_diff_hessian(l_of_x, zx)
        luu = finite_diff_hessian(l_of_u, zu)

        H_joint = finite_diff_hessian(l_joint, z)
        lux = H_joint[self.N_X:, :self.N_X]

        return lx, lu, lxx, luu, lux

    def _expand_terminal_cost(self, T_bar, v_bar):
        zx = np.zeros(self.N_X)
        phi = lambda ex: _terminal_cost_in_chart(self.terminal_cost, T_bar, v_bar, ex)
        Vx = finite_diff_gradient(phi, zx)
        Vxx = finite_diff_hessian(phi, zx)
        return Vx, Vxx

    def backward_pass(self, Ts, Vs, Us):
        N = Us.shape[0]

        Vx, Vxx = self._expand_terminal_cost(Ts[N], Vs[N])

        ks = np.zeros((N, self.N_U))
        Ks = np.zeros((N, self.N_U, self.N_X))
        dV = [0.0, 0.0]

        reg = self.reg
        max_reg = 1e2

        for k in reversed(range(N)):
            T_k, v_k, u_k = Ts[k], Vs[k], Us[k]

            A, B = self._linearize_dynamics(T_k, v_k, u_k)
            lx, lu, lxx, luu, lux = self._expand_stage_cost(T_k, v_k, u_k, k)

            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            Quu_reg = Quu + reg * np.eye(self.N_U)
            success = False
            local_reg = reg
            while local_reg <= max_reg:
                try:
                    np.linalg.cholesky(Quu_reg)
                    success = True
                    break
                except np.linalg.LinAlgError:
                    local_reg *= 10.0
                    Quu_reg = Quu + local_reg * np.eye(self.N_U)

            if not success:
                return ks, Ks, dV, True

            Quu_inv = np.linalg.inv(Quu_reg)

            k_ff = -Quu_inv @ Qu
            K_fb = -Quu_inv @ Qux

            ks[k] = k_ff
            Ks[k] = K_fb

            dV[0] += float(k_ff @ Qu)
            dV[1] += float(0.5 * k_ff @ (Quu @ k_ff))

            Vx = Qx + K_fb.T @ (Quu @ k_ff) + K_fb.T @ Qu + Qux.T @ k_ff
            Vxx = Qxx + K_fb.T @ Quu @ K_fb + K_fb.T @ Qux + Qux.T @ K_fb
            Vxx = 0.5 * (Vxx + Vxx.T)

        return ks, Ks, dV, False


# ============================================================================
# PHASE 3
# ============================================================================

def trajectory_cost(solver, Ts, Vs, Us):
    N = Us.shape[0]
    total = 0.0
    for k in range(N):
        total += solver.stage_cost(Ts[k], Vs[k], Us[k], k)
    total += solver.terminal_cost(Ts[N], Vs[N])
    return total


def forward_pass(solver, Ts_bar, Vs_bar, Us_bar, ks, Ks, alpha):
    N = Us_bar.shape[0]
    dyn = solver.dyn
    dt = solver.dt

    Ts_new = [Ts_bar[0]]
    Vs_new = np.zeros((N + 1, 3))
    Vs_new[0] = Vs_bar[0]
    Us_new = np.zeros((N, 2))

    T = Ts_bar[0]
    v = Vs_bar[0]

    for k in range(N):
        delta_x = state_error(T, v, Ts_bar[k], Vs_bar[k])

        delta_u = alpha * ks[k] + Ks[k] @ delta_x
        u_new = Us_bar[k] + delta_u

        Us_new[k] = u_new

        T, v = dyn.step(T, v, u_new, dt)

        Ts_new.append(T)
        Vs_new[k + 1] = v

    return Ts_new, Vs_new, Us_new


def backtracking_line_search(solver, Ts_bar, Vs_bar, Us_bar,
                              ks, Ks, dV, J_bar,
                              alphas=None, c1=1e-4):
    if alphas is None:
        alphas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]

    for alpha in alphas:
        Ts_new, Vs_new, Us_new = forward_pass(solver, Ts_bar, Vs_bar, Us_bar,
                                               ks, Ks, alpha)
        J_new = trajectory_cost(solver, Ts_new, Vs_new, Us_new)

        expected_decrease = -(alpha * dV[0] + alpha ** 2 * dV[1])
        actual_decrease = J_bar - J_new

        if expected_decrease <= 1e-12:
            if actual_decrease > 0:
                return Ts_new, Vs_new, Us_new, J_new, alpha, True
            else:
                continue

        if actual_decrease >= c1 * expected_decrease:
            return Ts_new, Vs_new, Us_new, J_new, alpha, True

    return Ts_bar, Vs_bar, Us_bar, J_bar, 0.0, False


def ddp_solve(solver, T0, v0, U_init, max_iters=50, tol=1e-6, verbose=True):
    N = U_init.shape[0]
    dt = solver.dt

    Ts, Vs = solver.dyn.simulate(T0, v0, U_init, dt)
    Us = U_init.copy()
    J = trajectory_cost(solver, Ts, Vs, Us)
    J_history = [J]

    for it in range(max_iters):
        ks, Ks, dV, diverged = solver.backward_pass(Ts, Vs, Us)

        if diverged:
            solver.reg = min(solver.reg * 10.0, 1e2)
            if verbose:
                print(f"[iter {it}] backward pass diverged, reg={solver.reg:.2e}")
            continue

        Ts_new, Vs_new, Us_new, J_new, alpha, accepted = backtracking_line_search(
            solver, Ts, Vs, Us, ks, Ks, dV, J)

        if not accepted:
            solver.reg = min(solver.reg * 10.0, 1e2)
            if verbose:
                print(f"[iter {it}] line search failed, reg={solver.reg:.2e}")
            if solver.reg >= 1e2:
                if verbose:
                    print(f"[iter {it}] reg saturated, stopping.")
                break
            continue

        solver.reg = max(solver.reg / 5.0, 1e-8)

        if verbose:
            print(f"[iter {it}] alpha={alpha:.4g}  J: {J:.6f} -> {J_new:.6f}  "
                  f"(dJ={J - J_new:.3e}, reg={solver.reg:.1e})")

        Ts, Vs, Us = Ts_new, Vs_new, Us_new
        dJ = J - J_new
        J = J_new
        J_history.append(J)

        if abs(dJ) < tol:
            if verbose:
                print(f"Converged at iter {it}: |dJ|={abs(dJ):.2e} < tol")
            break

    return Ts, Vs, Us, J_history

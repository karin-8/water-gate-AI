import numpy as np
from scipy.optimize import minimize, fsolve

def simulate_gates(q0, h, initial_y0=12.0, Cds=0.6, gate_width=2.0, dt=1.0):
    g = 9.81  # gravity
    A_base = 1e6
    n_gates = 4
    y = [initial_y0]
    q = [q0]

    for i in range(n_gates):
        Ai = h[i] * gate_width  # effective gate opening (linear with h)
        delta_h = max(y[i] - 0.5, 0.1)  # crude estimate of drop height
        q_out = Cds[i] * Ai * np.sqrt(2 * g * delta_h)
        
        # q_next = max(q[i] - q_out, 0)
        y_next = max(y[i] - q_out * dt / (gate_width * 5), 0)  # 5m length section

        q.append(q_out)
        y.append(y_next)

    return q, y

def simulate_gates_over_time(q0, h, initial_ys=[10, 8, 7, 6], Cds=[0.6]*4, gate_width=10.0, length=5.0, dt=10, steps=360):
    import numpy as np
    from scipy.optimize import fsolve

    g = 9.81
    n_gates = len(h)
    A = gate_width * length
    A_reservoir = 1e5

    ys_over_time = [initial_ys[:]]  # record water levels
    qs_over_time = []  # record flow rates

    current_ys = initial_ys[:]
    current_q0 = q0


    for _ in range(steps):
        def equations(y):
            y = np.concatenate([y, [0]])
            eqs = []
            q_in = current_q0
            for i in range(n_gates):
                Ai = h[i] * gate_width
                delta_h = max(y[i] - y[i+1], 0.1)
                q_out = Cds[i] * Ai * np.sqrt(2 * g * delta_h)
                eq = y[i] - current_ys[i] - (q_in - q_out) * dt / A_reservoir
                eqs.append(eq)
                q_in = q_out
            return eqs

        y_sol = fsolve(equations, current_ys)
        qs = [current_q0]
        q_in = current_q0

        for i in range(n_gates):
            Ai = h[i] * gate_width
            delta_h = max(y_sol[i] - 0.5, 0.1)
            q_out = Cds[i] * Ai * np.sqrt(2 * g * delta_h)
            qs.append(q_out)
            q_in = q_out

        # Update state
        current_ys = list(y_sol)
        current_q0 = qs[-1]  # optional: assume last outflow feeds next time step

        ys_over_time.append(current_ys[:])
        qs_over_time.append(qs[:])

    return np.array(qs_over_time), np.array(ys_over_time)
    
def objective(h, q0=100):
    _, y = simulate_gates(q0, h)
    mean_y = np.mean(y)
    return sum((yi - mean_y) ** 2 for yi in y)  # variance

def optimize_gate_openings(q0=100):
    bounds = [(0.1, 2)] * 4  # h1 to h4 between 0% and 100%
    initial_guess = [0.5] * 4
    result = minimize(objective, initial_guess, args=(q0,), bounds=bounds)
    return result.x, result.fun

def hybrid_loss_fn(h, q0, initial_y0, q_target, y_target=None, initial_ys=[10, 8, 7, 6], y_min=6, y_max=12, Cds=None, dt=10, steps=360, penalty_weight=100):
    # print(q0, h, initial_ys, Cds, dt, steps)
    q, y = simulate_gates_over_time(q0, h, initial_ys=initial_ys, Cds=Cds, dt=dt, steps=steps)
    q = q[-1,1:]
    y = y[-1,1:]

    if y_target is not None:
        # 🎯 Match specific y targets
        loss = sum((yi - yti) ** 2 for yi, yti in zip(y, y_target))
    else:
        # 🛟 Keep y within safe bounds and match q_target
        penalties = sum(
            (y_min - yi) ** 2 if yi < y_min else (yi - y_max) ** 2 if yi > y_max else 0
            for yi in y
        )
        q_loss = (q[-1] - q_target) ** 2 # + sum([q[i+1]-q[i]+1 for i in range(len(q)-1) if q[i+1]-q[i]>-1])**2
        loss = q_loss + penalty_weight * penalties

    return loss

def smart_optimize_gates(
    q0=100, 
    initial_y0=12.0,
    q_target=80, 
    initial_ys=[10,8,7,6],
    y_target=None, 
    y_min=6, 
    y_max=12, 
    Cds=[0.6]*4,
    dt=10,
    steps=360
):
    bounds = [(0.1, 2)] * 4
    initial_guess = [0.5] * 4
    result = minimize(
        hybrid_loss_fn,
        initial_guess,
        args=(q0, initial_y0, q_target, y_target, initial_ys, y_min, y_max, Cds, dt, steps),
        bounds=bounds
    )
    return result.x, result.fun, simulate_gates_over_time(q0, result.x, initial_ys=initial_ys, Cds=Cds, dt=dt, steps=steps)
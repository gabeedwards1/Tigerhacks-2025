# energy_model.py
import math

class EnergyParams:
    def __init__(self,
                 mass_kg=185.0, g=3.71, Crr=0.03, eta=0.70,
                 meters_per_cell=1.0, height_scale_m=50.0,
                 k_rough_J_per_m=40.0, downhill_regen_eff=0.10,
                 min_grade=-1.0, max_grade=1.0):
        self.mass_kg = mass_kg
        self.g = g
        self.Crr = Crr
        self.eta = max(1e-6, min(1.0, eta))
        self.meters_per_cell = meters_per_cell
        self.height_scale_m = height_scale_m
        self.k_rough_J_per_m = k_rough_J_per_m
        self.downhill_regen_eff = max(0.0, min(0.5, downhill_regen_eff))
        self.min_grade = min_grade
        self.max_grade = max_grade

def _grid_step_m(dr, dc, meters_per_cell):
    base = math.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0
    return base * meters_per_cell

def move_energy_J(u, v, layers, P: EnergyParams):
    (r0,c0), (r1,c1) = u, v
    if layers.blocked[r1, c1]:
        return None
    dr, dc = r1-r0, c1-c0
    d_m = _grid_step_m(dr, dc, P.meters_per_cell)

    # Height diff: if you have real elevation_m (meters), use that;
    # otherwise convert normalized height via height_scale_m.
    if getattr(layers, "elevation_m", None) is not None:
        dh_m = float(layers.elevation_m[r1, c1] - layers.elevation_m[r0, c0])
    else:
        dh_norm = float(layers.height[r1, c1] - layers.height[r0, c0]) if layers.height is not None else 0.0
        dh_m = dh_norm * P.height_scale_m

    grade = (dh_m / d_m) if d_m > 0 else 0.0
    grade = max(P.min_grade, min(P.max_grade, grade))
    theta = math.atan(grade)

    m, g = P.mass_kg, P.g
    F_grav_up = m*g*max(0.0, math.sin(theta))
    F_roll    = P.Crr*m*g*abs(math.cos(theta))
    E_mech_no_regen = (F_grav_up + F_roll) * d_m
    E_grav_downhill = m*g*max(0.0, -math.sin(theta)) * d_m
    E_regen = P.downhill_regen_eff * E_grav_downhill
    rough = float(layers.rough[r1, c1])
    E_rough = P.k_rough_J_per_m * rough * d_m
    E_drive_in = E_mech_no_regen / P.eta
    E_batt = E_drive_in + E_rough - E_regen
    return max(0.0, E_batt)

def physical_energy_cost_fn(layers, params: EnergyParams, scale_cost=1.0):
    def cost(u, v):
        E = move_energy_J(u, v, layers, params)
        if E is None:
            return None
        return float(E * scale_cost)
    return cost

def estimate_path_energy_J(path, layers, params: EnergyParams):
    if not path or len(path) < 2:
        return 0.0
    tot = 0.0
    for u, v in zip(path[:-1], path[1:]):
        E = move_energy_J(u, v, layers, params)
        if E is not None:
            tot += E
    return tot

def joule_to_Wh(J):
    return J / 3600.0

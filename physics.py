import numpy as np
from iapws import IAPWS97
from scipy.sparse.linalg import spsolve
from stiffness import assemble_stiffness_and_rhs, build_neumann_vector


def build_water_lookup_table(T_min_K, T_max_K, n_points, P_MPa):
    """
    Précalcule les propriétés thermophysiques de l'eau sur une grille de températures.
    Pression supposée constante (REP nominal : 15.5 MPa).
    Source propriétés : IAPWS-IF97.
    """
    T_grid = np.linspace(T_min_K, T_max_K, n_points)

    rho   = np.zeros(n_points)   # densité [kg/m³]
    cp    = np.zeros(n_points)   # capacité thermique [J/(kg·K)]
    k     = np.zeros(n_points)   # conductivité thermique [W/(m·K)]
    mu    = np.zeros(n_points)   # viscosité dynamique [Pa·s]
    beta  = np.zeros(n_points)   # coeff. dilatation thermique [1/K]

    for i, T in enumerate(T_grid):
        state = IAPWS97(T=T, P=P_MPa)
        rho[i]  = state.rho
        cp[i]   = state.cp * 1e3      # IAPWS renvoie kJ/(kg·K), on convertit en J/(kg·K)
        k[i]    = state.k
        mu[i]   = state.mu
        beta[i] = state.alfav         # coeff. de dilatation isobare

    nu    = mu / rho                  # viscosité cinématique [m²/s]
    alpha = k / (rho * cp)            # diffusivité thermique [m²/s]

    return {
        "T"    : T_grid,
        "rho"  : rho,
        "cp"   : cp,
        "k"    : k,
        "mu"   : mu,
        "beta" : beta,
        "nu"   : nu,
        "alpha": alpha,
    }


def water_props_at(T_K, lut):
    """
    Interpolation rapide dans la lookup table pour une température donnée.
    Retourne un dict avec toutes les propriétés à T_K.
    """
    props = {}
    for key in ["rho", "cp", "k", "mu", "beta", "nu", "alpha"]:
        props[key] = float(np.interp(T_K, lut["T"], lut[key]))
    return props


def solve_diffusion(M, K, U0, rho, cp, k, dt, t_end,
                    theta=1.0, rhs_extra=None,
                    print_every=10, label="", plot_callback=None):
    k_is_fun = callable(k)
    M_phys = rho * cp * M

    if not k_is_fun:
        K_phys = k * K
        A = (M_phys + theta * dt * K_phys).tocsr()

    U = U0.copy()
    n_steps = int(t_end / dt)

    print(f"[{label}] Démarrage boucle temporelle...")
    for step in range(n_steps):
        t = step * dt

        if k_is_fun:
            k_n    = k(U)
            K_phys = k_n * K
            A      = (M_phys + theta * dt * K_phys).tocsr()

        B = (M_phys - (1 - theta) * dt * K_phys).dot(U)
        if rhs_extra is not None:
            B = B + dt * rhs_extra(t, U)
        U = spsolve(A, B)
        if np.min(U) < 400.0:
            print(f"[{label}] t={t:.1f}s : Attention, certains noeuds sont sous 400K (Tmin={np.min(U):.1f} K), propriétés extrapolées aux bornes de la table.")
        T_avg_global = float(np.mean(U))
        if T_avg_global < 500.0:
            print(f"[{label}] t={t:.1f}s : Tmoy={T_avg_global:.1f} K — l'eau a suffisamment refroidi, arrêt simulation.")
            break
        T_sat = 618.0
        if np.max(U) > T_sat:
            print(f"[{label}] WARN : Tmax={np.max(U):.1f} K > Tsat={T_sat:.1f} K — ébullition imminente, arrêt simulation.")
            break

        if plot_callback is not None:
            plot_callback(step, t, U)

        if step % print_every == 0:
            print(f"[{label}] t={t:.1f}s : Tmin={np.min(U):.2f} K, Tmax={np.max(U):.2f} K")

    print(f"[{label}] Boucle terminée.")
    return U


def exponential_flux(t, q0, lam):
    """
    Flux de puissance résiduelle décroissant exponentiellement après arrêt réacteur.
    q0  : flux initial [W/m²]
    lam : constante de décroissance [1/s]
    """
    return q0 * np.exp(-lam * t)


def compute_h_bar(T_eau, T_ext, H_bar, lookup):
    """Churchill-Chu (Incropera, 2002) — barre verticale en convection naturelle."""
    g     = 9.81
    props = water_props_at(T_eau, lookup)
    beta  = props["beta"]
    nu    = props["nu"]
    alpha = props["alpha"]
    k_eau = props["k"]
    Pr    = nu / alpha

    dT = T_eau - T_ext
    Ra = g * beta * abs(dT) * H_bar**3 / (nu * alpha)
    Nu = (0.825 + (0.387 * Ra**(1/6)) / ((1 + (0.492/Pr)**(9/16))**(8/27)))**2

    h     = Nu * k_eau / H_bar
    q_out = -h * dT   # négatif si T_eau > T_ext : flux sortant

    return h, q_out


def cooling_rhs_fn(t, U, num_dofs, cooling_data, cooling_dofs,
                   T_ext, H_bar, t_insert, lut, tag_to_dof):
    if t < t_insert:
        return np.zeros(num_dofs)
    T_avg = float(np.mean(U[cooling_dofs]))
    T_avg = np.clip(T_avg, 400.0, 617.0)
    h, q_out = compute_h_bar(T_avg, T_ext, H_bar, lut)
    return build_neumann_vector(num_dofs, cooling_data, q_out, tag_to_dof)
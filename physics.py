import numpy as np
from iapws import IAPWS97
from scipy.sparse.linalg import spsolve
from stiffness import build_neumann_vector


# ===============================================
# LOOKUP TABLE — PROPRIETES THERMOPHYSIQUES EAU
# ===============================================

def build_water_lookup_table(T_min_K, T_max_K, n_points, P_MPa):
    """
    Precalcule les proprietes thermophysiques de l'eau sur une grille de temperatures.
    Pression supposee constante (REP nominal : ~15.5 MPa).
    Source : IAPWS-IF97.
    """
    T_grid = np.linspace(T_min_K, T_max_K, n_points)   # grille de temperatures [K]

    # allocation des tableaux de proprietes
    rho  = np.zeros(n_points)   # densite [kg/m^3]
    cp   = np.zeros(n_points)   # capacite thermique massique [J/(kg*K)]
    k    = np.zeros(n_points)   # conductivite thermique [W/(m*K)]
    mu   = np.zeros(n_points)   # viscosite dynamique [Pa*s]
    beta = np.zeros(n_points)   # coeff. de dilatation thermique isobare [1/K]

    for i, T in enumerate(T_grid):
        state   = IAPWS97(T=T, P=P_MPa)   # etat thermodynamique a (T, P) selon IAPWS-IF97
        rho[i]  = state.rho
        cp[i]   = state.cp * 1e3           # IAPWS renvoie kJ/(kg*K) -> conversion en J/(kg*K)
        k[i]    = state.k
        mu[i]   = state.mu
        beta[i] = state.alfav              # coeff. de dilatation isobare (notation IAPWS)

    nu    = mu / rho        # viscosite cinematique [m^2/s]
    alpha = k / (rho * cp)  # diffusivite thermique [m^2/s]

    return {"T": T_grid, "rho": rho, "cp": cp, "k": k, "mu": mu, "beta": beta, "nu": nu, "alpha": alpha}


def water_props_at(T_K, lut):
    """Interpolation lineaire dans la lookup table pour une temperature donnee T_K [K]."""
    props = {}
    for key in ["rho", "cp", "k", "mu", "beta", "nu", "alpha"]:
        props[key] = float(np.interp(T_K, lut["T"], lut[key]))   # interpolation 1D dans la grille precalculee
    return props


# ===============================================
# SOLVEUR TEMPOREL — SCHEMA THETA AVEC PICARD
# ===============================================

def solve_diffusion(M, K, U0, rho, cp, k, dt, t_end, theta=1.0, rhs_extra=None, nl_tol=1e-6, nl_maxiter=20, print_every=10, label="", plot_callback=None):
    """
    Resout rho*cp * dU/dt + K_phys * U = F par schema theta.
    rho, cp, k peuvent etre des scalaires ou des fonctions de T (callables).
    theta=1.0 : schema d'Euler implicite (stable inconditionnellement)
    theta=0.5 : schema de Crank-Nicolson (ordre 2 en temps)
    """
    k_is_fun   = callable(k)
    rhocp_is_fun = callable(rho) or callable(cp)   # True si rho ou cp dependent de T

    def get_M_phys(V):
        rho_v = rho(V) if callable(rho) else rho
        cp_v  = cp(V)  if callable(cp)  else cp
        return rho_v * cp_v * M   # matrice de masse physique : rho(T)*cp(T)*M

    if not rhocp_is_fun:
        M_phys = get_M_phys(None)   # M_phys constant, calcule une fois

    if not k_is_fun and not rhocp_is_fun:
        K_phys = k * K
        A_lin = (M_phys + theta * dt * K_phys).tocsr()   # systeme entierement lineaire, precalcule une fois

    U = U0.copy()
    n_steps = int(t_end / dt)

    print(f"[{label}] Demarrage boucle temporelle...")
    for step in range(n_steps):
        t = step * dt

        # ---- second membre B au pas courant (T^n connu) ----
        if not rhocp_is_fun:
            if k_is_fun:
                k_n    = k(U)
                K_phys = k_n * K
            B = (M_phys - (1 - theta) * dt * K_phys).dot(U)
            if rhs_extra is not None:
                B = B + dt * rhs_extra(t, U)

        # ---- resolution du systeme lineaire ----
        if k_is_fun or rhocp_is_fun:
            # iterations de Picard : cherche V tel que A(V)*V = B(V)
            V = U.copy()
            for it in range(nl_maxiter):
                kv       = k(V) if k_is_fun else k
                M_phys_v = get_M_phys(V) if rhocp_is_fun else M_phys   # M_phys recalcule si rho/cp variables
                K_phys_v = kv * K

                if rhocp_is_fun:
                    # B depend de V via M_phys(V) -> recalcul a chaque iteration
                    B_v = (M_phys_v - (1 - theta) * dt * K_phys_v).dot(U)
                    if rhs_extra is not None:
                        B_v = B_v + dt * rhs_extra(t, U)
                else:
                    B_v = B   # B fixe si rho*cp constant

                A_v = (M_phys_v + theta * dt * K_phys_v).tocsr()
                R   = A_v.dot(V) - B_v       # residu de Picard
                res = np.linalg.norm(R)
                if res < nl_tol:             # convergence atteinte
                    break
                V = V + spsolve(A_v, -R)     # correction de Picard
            else:
                print(f"[{label}] WARN Picard non converge step={step} t={t:.1f}s |R|={res:.2e}")
            U = V
        else:
            U = spsolve(A_lin, B)            # resolution directe si tout est constant

        # ---- gardes thermiques (seuils physiques REP) ----
        # (warning) seuils : Tmin LUT = 400 K, Tsat eau a 15.5 MPa = 618 K
        if np.min(U) < 400.0:
            print(f"[{label}] t={t:.1f}s : Tmin={np.min(U):.1f} K < 400 K, extrapolation LUT")
        if float(np.mean(U)) < 500.0:
            print(f"[{label}] t={t:.1f}s : Tmoy={float(np.mean(U)):.1f} K — eau suffisamment refroidie, arret")
            break
        if np.max(U) > 618.0:
            print(f"[{label}] WARN : Tmax={np.max(U):.1f} K > Tsat=618 K — ebullition imminente, arret")
            break

        if plot_callback is not None:
            plot_callback(step, t, U)        # collecte du frame pour l'animation
        if step % print_every == 0:
            print(f"[{label}] t={t:.1f}s : Tmin={np.min(U):.2f} K, Tmax={np.max(U):.2f} K")

    print(f"[{label}] Boucle terminee.")
    return U


# ===============================================
# FLUX RESIDUEL POST-ARRET REACTEUR
# ===============================================

def exponential_flux(t, q0, lam):
    """
    Puissance residuelle decroissante apres arret reacteur (modele simplifie).
    q0  : flux initial [W/m^2]
    lam : constante de decroissance [1/s]  (a ajuster selon le combustible)
    """
    return q0 * np.exp(-lam * t)


# ===============================================
# CONVECTION NATURELLE — BARRE VERTICALE
# ===============================================

def compute_h_bar(T_eau, T_ext, H_bar, lookup):
    """
    Correlation de Churchill-Chu pour convection naturelle sur barre verticale (Incropera 2002).
    Retourne h [W/(m^2*K)] et q_out [W/m^2] (negatif si chaleur evacuee vers l'exterieur).
    """
    g     = 9.81   # acceleration gravitationnelle [m/s^2]
    props = water_props_at(T_eau, lookup)
    beta  = props["beta"]    # coeff. de dilatation [1/K]
    nu    = props["nu"]      # viscosite cinematique [m^2/s]
    alpha = props["alpha"]   # diffusivite thermique [m^2/s]
    k_eau = props["k"]       # conductivite de l'eau [W/(m*K)]
    Pr    = nu / alpha       # nombre de Prandtl [-]

    dT = T_eau - T_ext
    Ra = g * beta * abs(dT) * H_bar**3 / (nu * alpha)                              # nombre de Rayleigh [-]
    Nu = (0.825 + (0.387 * Ra**(1/6)) / ((1 + (0.492/Pr)**(9/16))**(8/27)))**2    # nombre de Nusselt (Churchill-Chu) [-]

    h     = Nu * k_eau / H_bar   # coeff. convectif [W/(m^2*K)]
    q_out = -h * dT              # flux convectif sortant (negatif = chaleur evacuee si T_eau > T_ext)

    return h, q_out


# ===============================================
# VECTEUR RHS NEUMANN — BARRES DE REFROIDISSEMENT
# ===============================================

def cooling_rhs_fn(t, U, num_dofs, cooling_data, cooling_dofs, T_ext, H_bar, t_insert, lut, tag_to_dof):
    """Construit le vecteur RHS Neumann du aux barres de refroidissement a l'instant t."""
    if t < t_insert:
        return np.zeros(num_dofs)          # barres pas encore inserees

    T_avg = float(np.mean(U[cooling_dofs]))
    T_avg = np.clip(T_avg, 400.0, 617.0)  # (warning) clamp dans la plage valide de la LUT

    h, q_out = compute_h_bar(T_avg, T_ext, H_bar, lut)
    return build_neumann_vector(num_dofs, cooling_data, q_out, tag_to_dof)
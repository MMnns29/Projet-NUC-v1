import numpy as np
from iapws import IAPWS97
from scipy.sparse.linalg import spsolve
from stiffness import build_robin_system


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

def solve_diffusion(M, K, U0, rho, cp, k, dt, t_end,
                    theta=1.0, rhs_extra=None, robin_extra=None,
                    nl_tol=1e-6, nl_maxiter=20,
                    print_every=10, label="", plot_callback=None):

    k_is_fun     = callable(k)
    rhocp_is_fun = callable(rho) or callable(cp)

    def get_M_phys(V):
        rho_v = rho(V) if callable(rho) else rho
        cp_v  = cp(V)  if callable(cp)  else cp
        return rho_v * cp_v * M

    if not rhocp_is_fun:
        M_phys = get_M_phys(None)

    U = U0.copy()
    n_steps = int(t_end / dt)

    print(f"[{label}] Demarrage boucle temporelle...")
    for step in range(n_steps):
        t = step * dt

        # ---- termes Robin (modifient A et B simultanement) ----
        R_robin = 0
        G_robin = np.zeros_like(U)
        if robin_extra is not None:
            R_mat, G_vec = robin_extra(t, U)
            if R_mat is not None:
                R_robin = R_mat        # s'ajoute a K dans A
                G_robin = G_vec        # s'ajoute a F dans B

        # ---- construction de K_phys (eventuel. non-lineaire) ----
        if k_is_fun:
            k_n    = k(U)
            K_phys = k_n * K
        elif not rhocp_is_fun:
            K_phys = k * K

        M_phys_cur = get_M_phys(U) if rhocp_is_fun else M_phys

        # ---- second membre B ----
        # B = [M - (1-theta)*dt*(K+R)] * U^n + dt*[theta*G + (1-theta)*G + rhs_extra]
        K_total = K_phys + R_robin    # K physique + Robin : meme traitement theta
        B = (M_phys_cur - (1 - theta) * dt * K_total).dot(U)
        B += dt * G_robin             # terme source Robin
        if rhs_extra is not None:
            B += dt * rhs_extra(t, U)

        # ---- matrice A ----
        A = (M_phys_cur + theta * dt * K_total).tocsr()

        # ---- resolution ----
        if k_is_fun or rhocp_is_fun:
            V = U.copy()
            for it in range(nl_maxiter):
                R_v = A.dot(V) - B
                res = np.linalg.norm(R_v)
                if res < nl_tol:
                    break
                V = V + spsolve(A, -R_v)
            else:
                print(f"[{label}] WARN Picard non converge step={step} |R|={res:.2e}")
            U = V
        else:
            U = spsolve(A, B)

        # ---- gardes thermiques ----
        if np.min(U) < 400.0:
            print(f"[{label}] t={t:.1f}s : Tmin={np.min(U):.1f} K < 400 K")
        if float(np.mean(U)) < 500.0:
            print(f"[{label}] t={t:.1f}s : eau suffisamment refroidie, arret")
            break
        if np.max(U) > 618.0:
            print(f"[{label}] WARN : Tmax={np.max(U):.1f} K > Tsat, arret")
            break

        if plot_callback is not None:
            plot_callback(step, t, U)
        if step % print_every == 0:
            print(f"[{label}] t={t:.1f}s : Tmin={np.min(U):.2f} K, Tmax={np.max(U):.2f} K")

    print(f"[{label}] Boucle terminee.")
    return U

# ===============================================
# FLUX RESIDUEL POST-ARRET REACTEUR
# ===============================================

def exponential_flux(t, q0, lam): #à remettre dans le main peut être 
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


def cooling_robin_terms(U, num_dofs, cooling_data, cooling_dofs,
                        T_ext, H_bar, lut, tag_to_dof):
    from stiffness import build_robin_system

    T_avg = float(np.mean(U[cooling_dofs]))
    T_avg = np.clip(T_avg, 400.0, 617.0)
    h, _  = compute_h_bar(T_avg, T_ext, H_bar, lut)

    R, G = build_robin_system(num_dofs, cooling_data, h, T_ext, tag_to_dof)
    return R, G
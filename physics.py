import numpy as np
from iapws import IAPWS97
from scipy.sparse.linalg import spsolve
from stiffness import build_robin_system
from scipy.optimize import fsolve

# ===============================================
# LOOKUP TABLE — PROPRIETES THERMOPHYSIQUES EAU
# ===============================================

def build_water_lookup_table(T_min_K, T_max_K, n_points, P_MPa):
    """
    Precalcule les proprietes thermophysiques de l'eau sur une grille de temperatures.
    Pression supposee constante (REP nominal : ~15.5 MPa).
    Source : IAPWS-IF97.
    """
    T_grid = np.linspace(T_min_K, T_max_K, n_points) # grille de temperatures [K]

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

def solve_diffusion(M, K, U0, lookup, T_ext, Lc, dt, t_end,
                    theta=1.0, rhs_extra=None, robin_extra=None,
                    nl_tol=1e-6, nl_maxiter=20,
                    print_every=10, label="", plot_callback=None):

    def get_M_phys(V):
        T = float(np.clip(np.mean(V), 400.0, 617.0))
        props = water_props_at(T, lookup)
        return props["rho"] * props["cp"] * M

    def get_k_eff(V):
        T = float(np.clip(np.mean(V), 400.0, 617.0))
        props = water_props_at(T, lookup)
        Ra = 9.81 * props["beta"] * abs(T - T_ext) * Lc**3 / (props["nu"] * props["alpha"])
        Nu = max(1.0, 0.48 * Ra**0.25)
        return props["k"] * Nu

    U = U0.copy()
    n_steps = int(t_end / dt)

    print(f"[{label}] Demarrage boucle temporelle...")
    print(f"[{label}] VRAI DÉPART (t=initial) : Tmin={np.min(U0):.2f} K, Tmax={np.max(U0):.2f} K")
    for step in range(n_steps):
        t = step * dt
        V = U.copy() # Initial guess pour le pas n+1

        for it in range(nl_maxiter):
            # --- ON RECALCULE TOUT AVEC LA DERNIÈRE ESTIMATION V ---
            K_phys   = get_k_eff(V) * K
            M_phys_V = get_M_phys(V)
            
            R_robin, G_robin = 0, np.zeros_like(U)
            if robin_extra is not None:
                R_robin, G_robin = robin_extra(t + dt, V) # Mise à jour Robin avec V

            K_total = K_phys + R_robin
            
            # Matrice A(V) et Vecteur B(U)
            # Note : B dépend de U (temps n) mais A dépend de V (itération courante)
            A = (M_phys_V + theta * dt * K_total).tocsr()
            
            # Calcul du second membre B (basé sur le pas précédent U)
            # Attention : M_phys_cur doit être celui du temps n (U)
            M_phys_n = get_M_phys(U)
            B = (M_phys_n - (1 - theta) * dt * K_total).dot(U) + dt * G_robin
            if rhs_extra is not None:
                B += dt * rhs_extra(t, U)

            # --- RÉSOLUTION ---
            R_v = A.dot(V) - B
            res = np.linalg.norm(R_v)
            
            if res < nl_tol:
                break
            
            V = spsolve(A, B) # Version simplifiée du point fixe
        
        U = V # On valide le pas de temps

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
# SOLVEUR TEMPOREL — SCHEMA THETA AVEC FSOLVE
# ===============================================

def solve_diffusion2(M, K, U0, lookup, T_ext, Lc, dt, t_end,
                    theta=1.0, rhs_extra=None, robin_extra=None,
                    nl_tol=1e-6, nl_maxiter=20, # Conservés pour la signature
                    print_every=10, label="", plot_callback=None):

    def get_M_phys(V):
        T = float(np.clip(np.mean(V), 400.0, 617.0))
        props = water_props_at(T, lookup)
        return props["rho"] * props["cp"] * M

    def get_k_eff(V):
        T = float(np.clip(np.mean(V), 400.0, 617.0))
        props = water_props_at(T, lookup)
        Ra = 9.81 * props["beta"] * abs(T - T_ext) * Lc**3 / (props["nu"] * props["alpha"])
        Nu = max(1.0, 0.48 * Ra**0.25)
        return props["k"] * Nu

    U = U0.copy()
    n_steps = int(t_end / dt)

    print(f"[{label}] Demarrage boucle temporelle avec fsolve...")
    print(f"[{label}] VRAI DÉPART (t=initial) : Tmin={np.min(U0):.2f} K, Tmax={np.max(U0):.2f} K")
    for step in range(n_steps):
        t = step * dt

        # ---- Evaluation des termes au temps t (n) pour construire le vecteur B ----
        K_phys_n = get_k_eff(U) * K
        M_phys_n = get_M_phys(U)
        
        R_robin_n = 0
        G_robin_n = np.zeros_like(U)
        if robin_extra is not None:
            R_mat, G_vec = robin_extra(t, U)
            if R_mat is not None:
                R_robin_n = R_mat
                G_robin_n = G_vec

        K_total_n = K_phys_n + R_robin_n
        
        # Construction de B(U^n)
        B = (M_phys_n - (1 - theta) * dt * K_total_n).dot(U)
        B += dt * G_robin_n
        if rhs_extra is not None:
            B += dt * rhs_extra(t, U)

        # ---- Fonction residu pour fsolve (recherche de V = U^{n+1}) ----
        def residual(V):
            K_phys_V = get_k_eff(V) * K
            M_phys_V = get_M_phys(V)
            
            R_robin_V = 0
            if robin_extra is not None:
                # La matrice Robin peut dépendre de V (T à l'étape n+1)
                R_mat, _ = robin_extra(t + dt, V)
                if R_mat is not None:
                    R_robin_V = R_mat
                    
            K_total_V = K_phys_V + R_robin_V
            
            # Calcul de A(V) * V
            A_V = (M_phys_V + theta * dt * K_total_V).dot(V)
            
            # Residu = A(V)*V - B(U^n)
            return A_V - B

        # ---- Resolution non-linéaire ----
        # U est fourni comme point de départ (guess initial)
        V_sol, info, ier, mesg = fsolve(residual, U, xtol=nl_tol, full_output=True)
        
        if ier != 1:
            print(f"[{label}] WARN fsolve non converge step={step}: {mesg}")
            
        U = V_sol

        # ---- gardes thermiques ----
        if np.min(U) < 400.0:
            print(f"[{label}] t={(t+5):.1f}s : Tmin={np.min(U):.1f} K < 400 K")
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
    Ra = g * beta * abs(dT) * H_bar**3 / (nu * alpha)    
                              # nombre de Rayleigh [-]
    Nu_CC = (0.825 + (0.387 * Ra**(1/6)) / ((1 + (0.492/Pr)**(9/16))**(8/27)))**2    # nombre de Nusselt (Churchill-Chu) [-]
    h     = Nu_CC * k_eau / H_bar   # coeff. convectif [W/(m^2*K)]

    return h


def cooling_robin_terms(U, num_dofs, cooling_data, cooling_dofs,
                        T_ext, H_bar, lut, tag_to_dof):
    from stiffness import build_robin_system

    T_avg = float(np.mean(U[cooling_dofs]))
    T_avg = np.clip(T_avg, 400.0, 617.0)
    h = compute_h_bar(T_avg, T_ext, H_bar, lut)

    R, G = build_robin_system(num_dofs, cooling_data, h, T_ext, tag_to_dof)
    return R, G
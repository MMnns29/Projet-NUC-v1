import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from gmsh_utils import (
    border_dofs_from_tags,  gmsh_init, gmsh_finalize,
    prepare_quadrature_and_basis, get_jacobians, get_boundary_segments, mesh5
)
from stiffness import assemble_stiffness_and_rhs, build_neumann_vector
from physics import build_water_lookup_table, water_props_at, solve_diffusion, compute_h_bar, cooling_rhs_fn, exponential_flux
from dirichlet import solve_dirichlet
from errors import compute_L2_H1_errors
from mass import assemble_mass
from scipy.sparse.linalg import spsolve

from plot_utils import plot_mesh_2d, plot_fe_solution_2d
import gmsh


def main(order=1):

    # ============================================================
    # PARAMÈTRES CENTRALISÉS
    # ============================================================

    # --- Géométrie ---
    m_val               = 3         # crayons par rangée par assemblage (≥2 si cooling)
    n_val               = 1         # assemblages (1, 4 ou 9)
    pitch_val           = 18.7e-3   # pas du réseau [m]
    R_rod_val           = 6.15e-3   # rayon crayon [m]
    R_cooling_val       = 2.0e-3    # rayon barre refroidissement [m]
    gap_assembly_val    = 12e-3     # espace inter-assemblages [m]
    add_cooling_rods_val = True     # activer/désactiver les barres

    # --- Maillage ---
    mesh_refinement     = 1.0       # >1 = plus fin, <1 = plus grossier
    smin_val            = 0.5e-3 / mesh_refinement  # taille min éléments [m]
    SAVE_PDF            = False      # sauvegarder le maillage en PDF

    # --- Physique ---
    T0_K    = 553.15    # température initiale [K] (~280°C, REP nominal)
    P_MPa   = 15.5      # pression [MPa] = 155 bars
    theta   = 1.0       # schéma θ : 1.0 = Euler implicite (inconditionnellement stable)
    dt      = 0.5       # pas de temps [s] (ça fait x2 je sais pas pq)
    t_end   = 60       # durée totale [s]
    q0      = 5e3       # flux initial sur les crayons [W/m²]
    lam     = 1/80.0    # constante décroissance exponentielle [1/s]
    T_ext   = 500.0     # température eau froide des barres de refroidissement [K]
    H_bar   = 0.5       # hauteur effective pour corrélation Churchill-Chu [m]
    t_insert = 20.0     # instant d'insertion des barres [s]

    # ============================================================
    # ETAPE 1 : MAILLAGE
    # ============================================================

    gmsh_init("poisson_2d")

    # Génération du maillage rectangulaire périodique avec crayons et barres
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, cooling_positions = mesh5(m=m_val, n=n_val, order=order, pitch=pitch_val, R_rod=R_rod_val, R_cooling=R_cooling_val, gap_assembly=gap_assembly_val, R_core=None, smin=smin_val, add_cooling_rods=add_cooling_rods_val)

    path = os.path.join(os.path.dirname(__file__), "mesh_plot.pdf")
    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, save_path=path if SAVE_PDF else None, cooling_rods=cooling_positions, R_cooling=R_cooling_val)
    plt.show()

    # ============================================================
    # ETAPE 2 : MAPPING TAGS GMSH → DEGRÉS DE LIBERTÉ
    # ============================================================

    # Gmsh numérote les noeuds arbitrairement ; on construit une table de correspondance
    # tag_to_dof[tag_gmsh] = indice DOF dans les vecteurs/matrices du solveur
    unique_dofs_tags = np.unique(elemNodeTags)  # noeuds effectivement utilisés
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)  # -1 = tag non utilisé
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i

    # ============================================================
    # ETAPE 3 : ASSEMBLAGE DES MATRICES K ET M
    # ============================================================

    # Quadrature de Gauss et fonctions de base sur l'élément de référence
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    # Jacobiens de la transformation élément référence → physique
    jac, det, coords = get_jacobians(elemType, xi)

    # K géométrique : kappa=1 car le vrai k(T) est appliqué dans solve_diffusion via k_of_T
    K_lil, _ = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, lambda x: 1.0, lambda x: 0.0, tag_to_dof)
    # Matrice de masse M pour le terme en dU/dt
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    K = K_lil.tocsr()  # format CSR pour spsolve
    M = M_lil.tocsr()

    print(f"[E3] K : {K.shape}, nnz={K.nnz} | M : {M.shape}, nnz={M.nnz}")
    print(f"[E3] Diag M>0 : {np.all(M.diagonal()>0)} | Diag K>0 : {np.all(K.diagonal()>0)}")

    # ============================================================
    # ETAPE 4 : LOOKUP TABLE IAPWS-IF97
    # ============================================================

    # Précalcul des propriétés de l'eau sur grille T à pression constante
    lut = build_water_lookup_table(T_min_K=400.0, T_max_K=650.0, n_points=250, P_MPa=P_MPa)
    props0 = water_props_at(T0_K, lut)  # propriétés à la température initiale
    rho_val = props0["rho"]  # densité supposée constante (hypothèse ρ ≈ cste)
    cp_val  = props0["cp"]   # capacité thermique supposée constante (idem)
    print(f"[E4] À T={T0_K}K : rho={rho_val:.1f} kg/m³, cp={cp_val:.1f} J/kgK, k={props0['k']:.4f} W/mK")

    # ============================================================
    # ETAPE 5 : TERMES SOURCE — CRAYONS ET BARRES
    # ============================================================

    # Récupération des segments 1D du bord des crayons (tag 20) pour intégration Neumann
    rod_data = get_boundary_segments(physical_tag=20, order=order)
    print(f"[E5] {len(rod_data[1])} segments sur les crayons")

    # Flux Neumann sur les crayons : q(t) = q0 * exp(-lam*t), assemblé en vecteur nodal
    def rod_rhs(t, U): return build_neumann_vector(num_dofs, rod_data, exponential_flux(t, q0, lam), tag_to_dof)

    # Conductivité non linéaire : interpolation IAPWS à T_mean clippée dans [400, 617] K
    def k_of_T(U): return water_props_at(float(np.clip(np.mean(U), 400.0, 617.0)), lut)["k"]

    if add_cooling_rods_val:
        # Récupération des segments 1D des barres de refroidissement (tag 30)
        cooling_data = get_boundary_segments(physical_tag=30, order=order)
        print(f"[E5] {len(cooling_data[1])} segments sur les barres de refroidissement")
        # DOFs des noeuds sur les barres (utilisés pour le masque colormap et Churchill-Chu)
        cooling_node_tags = np.unique(cooling_data[2])
        cooling_dofs = border_dofs_from_tags(cooling_node_tags, tag_to_dof)
        # Flux convectif Churchill-Chu : h(T_moy) * (T_moy - T_ext), actif après t_insert
        def cooling_rhs(t, U): return cooling_rhs_fn(t, U, num_dofs, cooling_data, cooling_dofs, T_ext, H_bar, t_insert, lut, tag_to_dof)
        def combined_rhs(t, U): return rod_rhs(t, U) + cooling_rhs(t, U)
    else:
        cooling_dofs = np.array([], dtype=int)
        def combined_rhs(t, U): return rod_rhs(t, U)

    # ============================================================
    # ETAPE 6 : SIMULATION TEMPORELLE (PICARD NON LINÉAIRE)
    # ============================================================

    U0 = np.full(num_dofs, T0_K)  # condition initiale : eau uniforme à T0
    frames = []
    def collect_frame(step, t, U): frames.append((t, U.copy()))  # stockage pour animation

    # Résolution du système M*dU/dt + k(U)*K*U = F par schéma θ + itérations de Picard
    U = solve_diffusion(M, K, U0, rho_val, cp_val, k_of_T, dt, t_end, theta=theta, rhs_extra=combined_rhs, print_every=10, label="SIM", plot_callback=collect_frame)

    # ============================================================
    # ETAPE 7 : ANIMATION
    # ============================================================

    # Exclure les noeuds des barres du calcul d'échelle colormap en commentant les petites T_global et décommentant les autres (et inversement)
    # (trop froids au contact direct, écraserait la dynamique du champ T)
    interior_mask = np.ones(num_dofs, dtype=bool)
    interior_mask[cooling_dofs] = False
    #T_global_min = min(np.min(U_i[interior_mask]) for _, U_i in frames)
    #T_global_max = max(np.max(U_i[interior_mask]) for _, U_i in frames)
    T_global_min = min(np.min(U_i) for _, U_i in frames)
    T_global_max = max(np.max(U_i) for _, U_i in frames)
    T_global_min_C = T_global_min - 273.15 #pour avoir en °C dans le plot 
    T_global_max_C = T_global_max - 273.15
    print(f"[E7] Echelle T (hors barres) : {T_global_min:.1f} — {T_global_max:.1f} K")

    from matplotlib.animation import FuncAnimation
    fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
    cbar_anim = [None]

    def animate(i):
        t_i, U_i = frames[i]
        if cbar_anim[0] is not None:
            cbar_anim[0].remove()
        ax_anim.clear()
        contour = plot_fe_solution_2d(elemNodeTags, nodeCoords, nodeTags, U_i - 273.15, tag_to_dof, show_mesh=False, ax=ax_anim, cooling_rods=cooling_positions, R_cooling=R_cooling_val, add_colorbar=False, vmin=T_global_min_C, vmax=T_global_max_C, cmap='jet')        
        cbar_anim[0] = fig_anim.colorbar(contour, ax=ax_anim, label="Temperature [°C]")
        ax_anim.set_title(f"t = {t_i:.1f} s  |  Tmax = {np.max(U_i) - 273.15:.1f} °C")
        ax_anim.set_aspect('equal')

    anim = FuncAnimation(fig_anim, animate, frames=len(frames), interval=100, repeat=True)
    plt.show(block=True)

    gmsh_finalize()
    return





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation thermique REP — éléments finis 2D")
    parser.add_argument("-order", type=int, default=1)
    args = parser.parse_args()
    main(order=args.order)
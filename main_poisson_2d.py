import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from gmsh_utils import (
    border_dofs_from_tags, build_2d_rectangle_mesh, gmsh_init, gmsh_finalize,
    prepare_quadrature_and_basis, get_jacobians, get_boundary_segments,
    mesh1, mesh2, mesh3, mesh4, mesh5
)
from stiffness import assemble_stiffness_and_rhs, build_neumann_vector
from physics import build_water_lookup_table, water_props_at, solve_diffusion, compute_h_bar, cooling_rhs_fn, exponential_flux
from dirichlet import solve_dirichlet
from errors import compute_L2_H1_errors
from mass import assemble_mass
from scipy.sparse.linalg import spsolve

from plot_utils import plot_mesh_2d, plot_fe_solution_2d
import gmsh


#le nouveau code stiffness encore les conditions de neumann qu'il nous faut pour NUC


def main(L=1.0, H=1.0, h=1.0, order=1):

    # ============================================================
    # FONCTIONS MATHÉMATIQUES DE TEST
    # ============================================================

    def kappa(x): return 1.0  #à regarder : kappa est 1 pour tout le code ?? 

    # ============================================================
    # PARAMÈTRES CENTRALISÉS
    # ============================================================

    # --- Géométrie des crayons et assemblages ---
    m_val = 3                # Nombre de crayons combustibles sur le côté d'un assemblage NE PEUT PAS ETRE 1 SI COOLING
    n_val = 4                # Nombre total d'assemblages dans la cuve (choix restreint à 1, 4 ou 9 pour la symétrie)
    pitch_val = 18.7e-3      # Pas du réseau : distance entre le centre de deux crayons voisins (en mètres)
    R_rod_val = 6.15e-3      # Rayon extérieur physique d'un crayon combustible (en mètres)
    
    # --- Géométrie de l'eau (espacements et cuve) ---
    gap_assembly_val = 12e-3 # Épaisseur de la lame d'eau qui sépare deux assemblages voisins (en mètres)
    gap_outer_val = 75e-3    # Épaisseur de la lame d'eau périphérique (entre les assemblages et la paroi de la cuve)

    R_cooling_val = 2.0e-3       # rayon des barres de refroidissement [m]
    
    add_cooling_rods_val = True  # True pour activer les barres, False pour les désactiver

    # --- Paramètres du maillage (Gmsh) ---
    smin_val = 1.5e-3        # Taille minimale des éléments (triangles) du maillage, contrôle la précision près des bords

    # ============================================================
    # MAILLAGE ET AFFICHAGE
    # ============================================================

    gmsh_init("poisson_2d")

    # 2. APPEL DU MAILLAGE (On récupère cooling_positions à la fin)
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, cooling_positions = mesh5(
        m=m_val, n=n_val, order=order, pitch=pitch_val, R_rod=R_rod_val,
        add_cooling_rods=add_cooling_rods_val, R_cooling=R_cooling_val,
        gap_assembly=gap_assembly_val, gap_outer=gap_outer_val, smin=smin_val
    )

    SAVE_PDF = True
    path = os.path.join(os.path.dirname(__file__), "mesh_plot.pdf")

    # 3. APPEL DE L'AFFICHAGE (On lui passe directement les positions calculées)
    plot_mesh_2d(
        elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags,
        save_path=path if SAVE_PDF else None,
        cooling_rods=cooling_positions,
        R_cooling=R_cooling_val
    )

    plt.show()

    # ============================================================
    # MAPPING TAGS GMSH -> DEGRÉS DE LIBERTÉ
    # ============================================================

    # define a mapping between the nodeTags returned from gmsh and the dofs of the system
    # ------------------------------------------
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
    # ------------------------------------------

    # ============================================================
    # ASSEMBLAGE DES MATRICES K ET M
    # ============================================================

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    K_lil, F = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, lambda x: 1.0, lambda x: 0.0, tag_to_dof)

    # --- LIGNE À AJOUTER POUR LA MASSE ---
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    # -------------------------------------

    K = K_lil.tocsr()
    M = M_lil.tocsr() #nouveau par rapport à avant pour la Masse (à vérifier)

    # ============================================================
    # ETAPE 1 : VÉRIFICATION MATRICES K ET M
    # ============================================================

    print(f"[E1] K shape : {K.shape}, nnz = {K.nnz}")
    print(f"[E1] M shape : {M.shape}, nnz = {M.nnz}")
    print(f"[E1] Diag M toujours positive : {np.all(M.diagonal() > 0)}")
    print(f"[E1] Diag K toujours positive : {np.all(K.diagonal() > 0)}")
    print(f"[E1] Somme lignes K (min, max) : {K.sum(axis=1).min():.2e}, {K.sum(axis=1).max():.2e}")
    print(f"[E1] Somme lignes M (min, max) : {M.sum(axis=1).min():.2e}, {M.sum(axis=1).max():.2e}")

    # ============================================================
    # ETAPE 2 : LOOKUP TABLE IAPWS
    # ============================================================

    T0_K    = 553.15   # température initiale de l'eau [K] (~280°C, conditions REP)
    P_MPa   = 15.5     # pression nominale [MPa] = 155 bars
    lut = build_water_lookup_table(T_min_K=400.0, T_max_K=650.0, n_points=250, P_MPa=P_MPa)
    # Vérification : propriétés à T0
    props0 = water_props_at(T0_K, lut)
    print(f"[E2] À T={T0_K}K : rho={props0['rho']:.1f} kg/m³, cp={props0['cp']:.1f} J/kgK")
    print(f"[E2] À T={T0_K}K : k={props0['k']:.4f} W/mK, nu={props0['nu']:.2e} m²/s")
    print(f"[E2] À T={T0_K}K : beta={props0['beta']:.2e} 1/K, alpha={props0['alpha']:.2e} m²/s")
    

    
    # ============================================================
    # ETAPE 3 : BOUCLE TEMPORELLE SANS SOURCE
    # ============================================================

    # Paramètres physiques constants (évalués à T0 via la lookup table)
    props0 = water_props_at(T0_K, lut)
    rho_val = props0["rho"]
    cp_val  = props0["cp"]
    k_val   = props0["k"]

    # Paramètres temporels
    theta  = 1.0          # Euler implicite (stable inconditionnellement)
    dt     = 1.0          # pas de temps [s]
    t_end  = 100.0        # durée totale [s]

    # Condition initiale : eau uniforme à T0
    U0 = np.full(num_dofs, T0_K)
    """
    # Boucle temporelle déléguée à physics.py (rhs_extra=None : pas de source à cette étape)
    U = solve_diffusion(M, K, U0, rho_val, cp_val, k_val,
                             dt, t_end, theta=theta,
                             rhs_extra=None, print_every=10, label="E3")
    
    """
    # ============================================================
    # ETAPE 4 : FLUX EXPONENTIEL SUR LES CRAYONS
    # ============================================================
    
    # Segments 1D des crayons (groupe physique 20) + quadrature associée
    rod_data = get_boundary_segments(physical_tag=20, order=order)
    print(f"[E4] {len(rod_data[1])} segments sur les crayons, quadrature prête")
    
    # Paramètres du flux de puissance résiduelle après arrêt
    q0 = 5e3   # [W/m²]
    lam = 1/80.0  # constante de décroissance [1/s]
    
    def rod_rhs(t, U):
        # Flux scalaire à l'instant t, puis assemblage du vecteur nodal
        q = exponential_flux(t, q0, lam)
        return build_neumann_vector(num_dofs, rod_data, q, tag_to_dof)
    
    def k_of_T(U):
        T_mean = float(np.clip(np.mean(U), 400.0, 617.0))
        k = water_props_at(T_mean, lut)["k"]
        return k
    

    
    # ============================================================
    # ETAPE 6 : FLUX DES BARRES DE REFROIDISSEMENT
    # ============================================================
    
    cooling_data = get_boundary_segments(physical_tag=30, order=order)
    print(f"[E6] {len(cooling_data[1])} segments sur les barres de refroidissement")
    
    # Noeuds des barres pour calculer T_avg local
    cooling_node_tags = np.unique(cooling_data[2])  # adapter selon la structure de cooling_data
    cooling_dofs = border_dofs_from_tags(cooling_node_tags, tag_to_dof)
    
    T_ext    = 500.0   # température imposée aux barres [K]
    H_bar    = 0.5     # hauteur effective des barres [m]
    t_insert = 20.0    # instant d'insertion [s]
    
    def cooling_rhs(t, U):
        return cooling_rhs_fn(t, U, num_dofs, cooling_data, cooling_dofs,
                              T_ext, H_bar, t_insert, lut, tag_to_dof)

    def combined_rhs(t, U):
        return rod_rhs(t, U) + cooling_rhs(t, U)
    
    
    frames = []

    def collect_frame(step, t, U):
        frames.append((t, U.copy()))

    U = solve_diffusion(M, K, U0, rho_val, cp_val, k_of_T,
                        dt, t_end, theta=theta,
                        rhs_extra=combined_rhs, print_every=10, label="E6",
                        plot_callback=collect_frame)
    
    # percentiles pour éviter que quelques noeuds froids au contact des barres
    # écrasent toute l'échelle — le min strict donnait 465K alors qu'on voyait
    # rien en dessous de 540K sur le plot
    
    """
    interior_mask = np.ones(num_dofs, dtype=bool)
    interior_mask[cooling_dofs] = False
    all_temps = np.concatenate([U_i[interior_mask] for _, U_i in frames])
    T_global_min = min(np.min(U_i[interior_mask]) for _, U_i in frames)
    T_global_max = max(np.max(U_i[interior_mask]) for _, U_i in frames)
    T_global_min = float(np.percentile(all_temps, 2))
    T_global_max = float(np.percentile(all_temps, 98))
    print(f"[ANIM] Echelle température (hors barres) : {T_global_min:.1f} K — {T_global_max:.1f} K")
    print(f"[ANIM] Echelle température (p2-p98, hors barres) : {T_global_min:.1f} K — {T_global_max:.1f} K")
    # --- Animation post-simulation ---
    from matplotlib.animation import FuncAnimation
    """
    #version sans les pourcentiles
    # on exclut les noeuds des barres de refroidissement du calcul de l'echelle
    # sinon les noeuds froids au contact direct des barres ecrasent tout le reste
    interior_mask = np.ones(num_dofs, dtype=bool)
    interior_mask[cooling_dofs] = False

    T_global_min = min(np.min(U_i[interior_mask]) for _, U_i in frames)
    T_global_max = max(np.max(U_i[interior_mask]) for _, U_i in frames)
    print(f"[ANIM] Echelle température (hors barres) : {T_global_min:.1f} K — {T_global_max:.1f} K")
    # --- Animation post-simulation ---
    from matplotlib.animation import FuncAnimation
    fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
    cbar_anim = [None]

    def animate(i):
        t_i, U_i = frames[i]
        if cbar_anim[0] is not None:
            cbar_anim[0].remove()
        ax_anim.clear()
        contour = plot_fe_solution_2d(
            elemNodeTags, nodeCoords, nodeTags, U_i, tag_to_dof,
            show_mesh=False, ax=ax_anim,
            cooling_rods=cooling_positions,
            R_cooling=R_cooling_val,
            add_colorbar=False,
            vmin=T_global_min,
            vmax=T_global_max
        )
        cbar_anim[0] = fig_anim.colorbar(contour, ax=ax_anim, label='T [K]')
        ax_anim.set_title(f"t = {t_i:.1f} s")
        ax_anim.set_aspect('equal')

    anim = FuncAnimation(fig_anim, animate, frames=len(frames), interval=100, repeat=True)
    plt.show(block=True)

    gmsh_finalize()
    plt.show(block=True)
    return

    """
    
    border_tags = np.concatenate(bnds_tags)

    # to apply dirichlet condition with the analytical expr, we need to evaluate nodes at right coords -> use mapping
    num_dofs = len(F) 
    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    for i, tag in enumerate(nodeTags):
        idx = tag_to_dof[int(tag)]
        if idx != -1:
            dof_coords[idx] = all_coords[i]

    # get the indices of the boundary nodes mapped to the system we solve
    border_dofs = border_dofs_from_tags(border_tags, tag_to_dof)
    # set the dirichlet values : here we evaluate the analytical function for the example
    dir_vals = [u_exact(dof_coords[d]) for d in border_dofs]

    # Solve 
    U = solve_dirichlet(K, F, border_dofs, dir_vals)

    L2, H1_semi, H1 = compute_L2_H1_errors(elemType, elemTags, elemNodeTags, U, xi, w, N, gN, jac, det, coords, u_exact, grad_exact=grad_exact, tag_to_dof=tag_to_dof)
    print("Errors: L2 = {:.6e}, H1 semi = {:.6e}, H1 = {:.6e}".format(L2, H1_semi, H1))
    gmsh_finalize()

    plot_fe_solution_2d(elemNodeTags, nodeCoords, nodeTags, U, tag_to_dof, show_mesh=False, ax=None)
    plt.show()
    
    return
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Poisson 2D with Gmsh FE")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("-L", type=float, default=1.0)
    parser.add_argument("-H", type=float, default=1.0)
    parser.add_argument("-hc", type=float, default=0.1)
    args = parser.parse_args()
    order = args.order
    L = args.L
    H = args.H
    h = args.hc

    main(L, H, h, order)

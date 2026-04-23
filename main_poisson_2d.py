import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from gmsh_utils import (border_dofs_from_tags, gmsh_init, gmsh_finalize, prepare_quadrature_and_basis, get_jacobians, get_boundary_segments, mesh5)
from stiffness import assemble_stiffness_and_rhs, build_neumann_vector
from physics import build_water_lookup_table, solve_diffusion, solve_diffusion2, cooling_robin_terms, way_wigner_flux
from mass import assemble_mass
from plot_utils import plot_mesh_2d, plot_fe_solution_2d


def main(order=1):

    # ============================================================
    # PARAMÈTRES CENTRALISÉS
    # ============================================================
    Res = 0 # if =0 Resolution Fast avec PIcard else : Resolution avec fsolve
    # --- Géométrie ---
    m_val               = 8         # crayons par rangée par assemblage (≥2 si cooling)
    n_val               = 1         # assemblages (1, 4 ou 9)
    pitch_val           = 18.7e-3   # pas du réseau [m]
    R_rod_val           = 6.15e-3   # rayon crayon [m]
    R_cooling_val       = 2e-3    # rayon barre refroidissement [m]
    gap_assembly_val    = 12e-3     # espace inter-assemblages [m]
    cooling = False     # activer/désactiver les barres

    # --- Maillage ---
    mesh_refinement     = 1       # >1 = plus fin, <1 = plus grossier
    smin_val            = 0.5e-3 / mesh_refinement  # taille min des éléments [m]
    SAVE_PDF            = False      # sauvegarder le maillage en PDF

    # --- Physique ---
    t_insert_val = 500.0  # Temps d'activation des barres [s] (0.0 pour activation immédiate)
    T0_K    = 553.15    # température initiale [K] (~280°C, REP nominal)
    P_MPa   = 15.5      # pression [MPa] = 155 bars
    theta   = 1.0       # schéma θ : 1.0 = Euler implicite (inconditionnellement stable)
    dt      = 25       # pas de temps [s] (ça fait x2 je sais pas pq)
    t_end   = 2000        # durée totale [s]
    T_ext   = 500.0     # température eau froide des barres de refroidissement [K]
    H_bar   = 0.5       # hauteur effective pour corrélation Churchill-Chu [m]

    # --- Lookup table IAPWS ---
    T_lut_min   = 400.0   # borne basse LUT [K] — aussi utilisée comme garde thermique
    T_lut_max   = 650.0   # borne haute LUT [K]
    n_lut       = 250     # points de la grille LUT (precision vs vitesse)
    
    # ============================================================
    # ETAPE 1 : MAILLAGE
    # ============================================================

    gmsh_init("poisson_2d")

    # Génération du maillage rectangulaire périodique avec crayons et barres
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, cooling_positions = mesh5(m=m_val, n=n_val, order=order, pitch=pitch_val, R_rod=R_rod_val, R_cooling=R_cooling_val, gap_assembly=gap_assembly_val, smin=smin_val, add_cooling_rods=cooling)

    path = os.path.join(os.path.dirname(__file__), "mesh_plot.pdf")
    plot_mesh_2d(nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, save_path=path if SAVE_PDF else None, cooling_rods=cooling_positions, R_cooling=R_cooling_val)
    plt.show()

    # ============================================================
    # ETAPE 2 : MAPPING TAGS GMSH → DEGRÉS DE LIBERTÉ
    # ============================================================

    unique_dofs_tags = np.unique(elemNodeTags)  # noeuds effectivement utilisés
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)  # -1 = tag non utilisé
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i

    # ============================================================
    # ETAPE 3 : ASSEMBLAGE DES MATRICES K ET M
    # ============================================================

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    K_lil = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, lambda x: 1.0, tag_to_dof)
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    K = K_lil.tocsr()  
    M = M_lil.tocsr()


    print(f"[E3] K : {K.shape}, nnz={K.nnz} | M : {M.shape}, nnz={M.nnz}")
    print(f"[E3] Diag M>0 : {np.all(M.diagonal()>0)} | Diag K>0 : {np.all(K.diagonal()>0)}")

    # ============================================================
    # ETAPE 4 : LOOKUP TABLE IAPWS-IF97
    # ============================================================

    lut = build_water_lookup_table(T_min_K=T_lut_min, T_max_K=T_lut_max, n_points=n_lut, P_MPa=P_MPa)

    # ============================================================
    # ETAPE 5 : TERMES SOURCE — CRAYONS ET BARRES
    # ============================================================

    rod_data = get_boundary_segments(physical_tag=20, order=order)
    print(f"[E5] {len(rod_data[1])} segments sur les crayons")

# Remplace rod_rhs par ceci :
    q_nom_fonctionnement = 523000.0 
    
    def rod_rhs(t, U): 
        return build_neumann_vector(num_dofs, rod_data, way_wigner_flux(t, q_nom_fonctionnement), tag_to_dof)
    if cooling:
        cooling_data = get_boundary_segments(physical_tag=30, order=order)
        print(f"[E5] {len(cooling_data[1])} segments sur les barres de refroidissement")
        cooling_node_tags = np.unique(cooling_data[2])
        cooling_dofs = border_dofs_from_tags(cooling_node_tags, tag_to_dof)
        
        def cooling_robin(t, U):
                if t < t_insert_val:
                    return 0, np.zeros(num_dofs)
                return cooling_robin_terms(U, num_dofs, cooling_data, cooling_dofs,
                                            T_ext, H_bar, lut, tag_to_dof)
    else:
            cooling_dofs = np.array([], dtype=int)
            cooling_robin = None

    def combined_rhs(t, U): return rod_rhs(t, U)

    # ============================================================
    # ETAPE 6 : SIMULATION TEMPORELLE (PICARD NON LINÉAIRE)
    # ============================================================

    U0 = np.full(num_dofs, T0_K)  

# --- AJOUT POUR LES GRAPHIQUES ---
    vec_ones = np.ones(num_dofs)
    M_ones = M.dot(vec_ones)
    total_area = np.sum(M_ones)

    frames = []
    time_list = []
    T_max_list = []
    T_min_list = [] # <--- NOUVEAU
    T_avg_list = []

    def collect_frame(step, t, U): 
        frames.append((t, U.copy()))  
        time_list.append(t)
        T_max_list.append(np.max(U))
        T_min_list.append(np.min(U)) # <--- NOUVEAU
        T_avg_list.append(np.dot(M_ones, U) / total_area)
    # ---------------------------------

    if Res == 0:
        U = solve_diffusion(M, K, U0, lut, T_ext, pitch_val, dt, t_end,
                            theta=theta, rhs_extra=combined_rhs, robin_extra=cooling_robin,
                            print_every=10, label="SIM", plot_callback=collect_frame)
    else :
        U = solve_diffusion2(M, K, U0, lut, T_ext, pitch_val, dt, t_end,
                            theta=theta, rhs_extra=combined_rhs, robin_extra=cooling_robin,
                            print_every=10, label="SIM", plot_callback=collect_frame)
        

# ============================================================
    # ETAPE 6.5 : GENERATION DU GRAPHIQUE D'EVOLUTION (STYLISÉ)
    # ============================================================
    plt.figure(figsize=(11, 7))

    # --- NOUVEAU : Conversion en Celsius et temps en Minutes ---
    time_min = np.array(time_list) / 60.0  # Conversion secondes -> minutes
    T_max_C = np.array(T_max_list) - 273.15
    T_avg_C = np.array(T_avg_list) - 273.15
    T_min_C = np.array(T_min_list) - 273.15

    # Tracé des courbes de simulation (Axe X en minutes)
    plt.plot(time_min, T_max_C, label="Température Maximale ($T_{max}$)", color='darkorange', lw=2.5)
    plt.plot(time_min, T_avg_C, label="Température Moyenne ($T_{moy}$)", color='royalblue', lw=2, linestyle='--')
    plt.plot(time_min, T_min_C, label="Température Minimale ($T_{min}$)", color='seagreen', lw=1.5, alpha=0.8)
    
    # Ligne d'ébullition "EFFRAYANTE" (Rouge très épais)
    plt.axhline(y=344.85, color='red', linestyle='-', linewidth=2, label="Limite d'ébullition ~345°", zorder=10)

    # Ombrage de la zone de danger (Basé sur les minutes)
    plt.fill_between(time_min, 344.85, 400, color='red', alpha=0.1)

    # Ligne d'activation convertie en minutes
    if cooling and t_insert_val > 0:
        plt.axvline(x=t_insert_val / 60.0, color='black', linestyle='-.', alpha=0.6, label=f"Activation Barres (t={t_insert_val/60:.1f} min)")

    plt.xlabel("Temps écoulé depuis l'arrêt [min]", fontsize=12) # Label mis à jour
    plt.ylabel("Température mesurée [°C]", fontsize=12)
    plt.title("Évolution Thermique en Défaillance de Refroidissement", fontsize=14, fontweight='bold')
    
    plt.ylim(T_min_C.min() - 5, 360) 
    plt.legend(loc='lower right', framealpha=0.9)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("accident_thermique_scary.pdf", dpi=300)
    plt.show(block=False)
    # ============================================================
    # ETAPE 7 : ANIMATION
    # ============================================================

    interior_mask = np.ones(num_dofs, dtype=bool)
    if len(cooling_dofs) > 0:
        interior_mask[cooling_dofs] = False
        
    T_global_min = min(np.min(U_i[interior_mask]) for _, U_i in frames)
    T_global_max = max(np.max(U_i[interior_mask]) for _, U_i in frames)
    T_global_min_C = np.floor((T_global_min - 273.15) / 5) * 5   
    T_global_max_C = np.ceil((T_global_max - 273.15) / 5) * 5
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
        ticks = np.arange(T_global_min_C, T_global_max_C + 5, 5)  
        cbar_anim[0] = fig_anim.colorbar(contour, ax=ax_anim, label="Temperature [°C]", ticks=ticks)
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
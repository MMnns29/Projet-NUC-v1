import numpy as np
import matplotlib.pyplot as plt
import os

from gmsh_utils import border_dofs_from_tags, gmsh_init, gmsh_finalize, prepare_quadrature_and_basis, get_jacobians, get_boundary_segments, mesh5
from stiffness import assemble_stiffness_and_rhs, build_neumann_vector
from physics import build_water_lookup_table, solve_diffusion, cooling_robin_terms, way_wigner_flux
from mass import assemble_mass

def generate_layout_plot():
    # ============================================================
    # PARAMÈTRES GLOBAUX
    # ============================================================
    m_val = 8 
    n_val = 1
    pitch_val = 18.7e-3
    R_rod_val = 6.15e-3
    R_cooling_val = 2.0e-3
    gap_assembly_val = 12e-3
    smin_val = 0.5e-3 / 0.3

    T0_K = 553.15
    P_MPa = 15.5
    theta = 22.5
    dt = 15
    t_end = 3600.0  
    T_ext = 500.0
    H_bar = 0.5
    t_ins_fixed = 0.0 # Activation immédiate pour comparer l'architecture brute

    # ============================================================
    # BOUCLE SUR LES CONFIGURATIONS GEOMETRIQUES
    # ============================================================
    layouts = ['full', 'checkerboard', 'peripheral', 'central', 'none']
    results = {}

    for layout in layouts:
        print(f"\n========================================================")
        print(f"--- Lancement Simulation : Topologie '{layout.upper()}' ---")
        print(f"========================================================")
        
        gmsh_init("poisson_2d")
        
        add_cool = layout != 'none'
        
        # Appel de mesh5 avec le nouveau paramètre cooling_layout
        elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, cooling_positions = mesh5(
            m=m_val, n=n_val, order=1, pitch=pitch_val, R_rod=R_rod_val, R_cooling=R_cooling_val, 
            gap_assembly=gap_assembly_val, smin=smin_val, add_cooling_rods=add_cool, cooling_layout=layout)

        unique_dofs_tags = np.unique(elemNodeTags)
        num_dofs = len(unique_dofs_tags)
        tag_to_dof = np.full(int(np.max(nodeTags)) + 1, -1, dtype=int)
        for i, tag in enumerate(unique_dofs_tags): tag_to_dof[int(tag)] = i

        xi, w, N, gN = prepare_quadrature_and_basis(elemType, 1)
        jac, det, coords = get_jacobians(elemType, xi)

        print(f"[{layout}] Assemblage des matrices...")
        K_lil = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, lambda x: 1.0, tag_to_dof)
        M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
        K = K_lil.tocsr()
        M = M_lil.tocsr()

        lut = build_water_lookup_table(400.0, 650.0, 250, P_MPa)
        rod_data = get_boundary_segments(20, 1)
        
        q_nom_fonctionnement = 523000.0 
        def rod_rhs(t, U): 
            return build_neumann_vector(num_dofs, rod_data, way_wigner_flux(t, q_nom_fonctionnement), tag_to_dof)

        if add_cool:
            cooling_data = get_boundary_segments(30, 1)
            cooling_dofs = border_dofs_from_tags(np.unique(cooling_data[2]), tag_to_dof)
            
            def cooling_robin(t, U):
                if t < t_ins_fixed:
                    return 0, np.zeros(num_dofs)
                return cooling_robin_terms(U, num_dofs, cooling_data, cooling_dofs, T_ext, H_bar, lut, tag_to_dof)
        else:
            cooling_robin = None

        t_list, tmax_list = [], []
        def collect_data(step, t, U):
            t_list.append(t)
            tmax_list.append(np.max(U) - 273.15)

        U0 = np.full(num_dofs, T0_K)
        solve_diffusion(M, K, U0, lut, T_ext, pitch_val, dt, t_end,
                        theta=theta, rhs_extra=rod_rhs, robin_extra=cooling_robin,
                        print_every=100, label=layout, plot_callback=collect_data)
        
        results[layout] = (t_list, tmax_list)
        gmsh_finalize()

    # ============================================================
    # GÉNÉRATION DU GRAPHIQUE COMBINÉ
    # ============================================================
    plt.figure(figsize=(11, 7))
    
    # Couleurs thématiques
    styles = {
        'full':         {'color': 'navy',      'label': 'Complète',        'ls': '-'},
        'checkerboard': {'color': 'teal',      'label': 'Damier (1 sur 2)',       'ls': '-'},
        'peripheral':   {'color': 'darkorange','label': 'Bords uniquement',       'ls': '-'},
        'central':      {'color': 'goldenrod', 'label': 'Centre uniquement',      'ls': '-'},
        'none':         {'color': 'dimgrey',   'label': 'Sans refroidissement',   'ls': '--'}
    }

    ordre_trace = ['none', 'peripheral', 'central', 'checkerboard', 'full']
    
    for l in ordre_trace: 
        t_vals, tmax_vals = results[l]
        plt.plot(t_vals, tmax_vals, color=styles[l]['color'], 
                 label=styles[l]['label'], linestyle=styles[l]['ls'], linewidth=2.5)

    plt.axhline(y=344.85, color='red', linestyle='-', linewidth=2, label="Limite d'ébullition ~345°C", zorder=10)
    plt.fill_between([0, t_end], 344.85, 400, color='red', alpha=0.1)

    plt.xlabel("Temps écoulé depuis l'arrêt [s]", fontsize=13)
    plt.ylabel("Température maximale $T_{max}$ [°C]", fontsize=13)
    plt.title("Impact de la Topologie de Refroidissement", fontsize=14, fontweight='bold')
    
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig("analyse_parametrique_layout.pdf", dpi=300)
    print("\nGraphique généré et sauvegardé sous 'analyse_parametrique_layout.pdf'")
    plt.show()

if __name__ == "__main__":
    generate_layout_plot()
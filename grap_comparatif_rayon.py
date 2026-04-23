import numpy as np
import matplotlib.pyplot as plt
import os

from gmsh_utils import border_dofs_from_tags, gmsh_init, gmsh_finalize, prepare_quadrature_and_basis, get_jacobians, get_boundary_segments, mesh5
from stiffness import assemble_stiffness_and_rhs, build_neumann_vector
from physics import build_water_lookup_table, solve_diffusion, cooling_robin_terms, way_wigner_flux
from mass import assemble_mass

def generate_comparative_radius_plot():
    # ============================================================
    # PARAMÈTRES GLOBAUX
    # ============================================================
    m_val = 8 
    n_val = 1
    pitch_val = 18.7e-3
    R_rod_val = 6.15e-3
    gap_assembly_val = 12e-3
    smin_val = 0.5e-3 / 0.3

    T0_K = 553.15
    P_MPa = 15.5
    theta = 22.5
    dt = 15
    t_end = 3600.0  
    T_ext = 500.0
    H_bar = 0.5
    
    # On fixe un temps d'insertion (ex: immédiat 0s, ou retardé)
    # 0.0 permet de voir l'efficacité pure du rayon dès le début
    t_ins_fixed = 500.0 

    # ============================================================
    # BOUCLE SUR LES DIFFÉRENTS RAYONS
    # ============================================================
    # Tailles en mètres : 1mm, 3mm, 5mm + Cas sans barres (None)
    scenarios_rayons = [0.5e-3, 1.0e-3, 2.0e-3, 3.0e-3 ,4.0e-3, 5.5e-3, None] 
    results = {}

    for R_cool in scenarios_rayons:
        label = f"Rayon = {R_cool*1000:.0f} mm" if R_cool is not None else "Sans barres"
        print(f"\n========================================================")
        print(f"--- Lancement Simulation : {label} ---")
        print(f"========================================================")
        
        # 1. INITIALISATION GMSH POUR CE RAYON SPÉCIFIQUE
        gmsh_init("poisson_2d")
        
        has_cooling = (R_cool is not None)
        R_val_mesh = R_cool if has_cooling else 2.0e-3 # Valeur dummy si pas de barres

        elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, cooling_positions = mesh5(
            m=m_val, n=n_val, order=1, pitch=pitch_val, R_rod=R_rod_val, R_cooling=R_val_mesh, 
            gap_assembly=gap_assembly_val, smin=smin_val, add_cooling_rods=has_cooling)

        unique_dofs_tags = np.unique(elemNodeTags)
        num_dofs = len(unique_dofs_tags)
        tag_to_dof = np.full(int(np.max(nodeTags)) + 1, -1, dtype=int)
        for i, tag in enumerate(unique_dofs_tags): tag_to_dof[int(tag)] = i

        xi, w, N, gN = prepare_quadrature_and_basis(elemType, 1)
        jac, det, coords = get_jacobians(elemType, xi)

        # 2. ASSEMBLAGE DES MATRICES POUR CE MAILLAGE
        print(f"[{label}] Assemblage des matrices...")
        K_lil = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, lambda x: 1.0, tag_to_dof)
        M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
        K = K_lil.tocsr()
        M = M_lil.tocsr()

        lut = build_water_lookup_table(400.0, 650.0, 250, P_MPa)
        rod_data = get_boundary_segments(20, 1)
        
        q_nom_fonctionnement = 523000.0 
        def rod_rhs(t, U): 
            return build_neumann_vector(num_dofs, rod_data, way_wigner_flux(t, q_nom_fonctionnement), tag_to_dof)

        # 3. GESTION DES CONDITIONS DE ROBIN (SELON LE RAYON)
        if has_cooling:
            cooling_data = get_boundary_segments(30, 1)
            cooling_dofs = border_dofs_from_tags(np.unique(cooling_data[2]), tag_to_dof)
            
            def cooling_robin(t, U):
                if t < t_ins_fixed:
                    return 0, np.zeros(num_dofs)
                return cooling_robin_terms(U, num_dofs, cooling_data, cooling_dofs, T_ext, H_bar, lut, tag_to_dof)
        else:
            cooling_robin = None

        # 4. RÉSOLUTION TEMPORELLE
        t_list = []
        tmax_list = []
        
        def collect_data(step, t, U):
            t_list.append(t)
            tmax_list.append(np.max(U) - 273.15)

        U0 = np.full(num_dofs, T0_K)
        solve_diffusion(M, K, U0, lut, T_ext, pitch_val, dt, t_end,
                        theta=theta, rhs_extra=rod_rhs, robin_extra=cooling_robin,
                        print_every=100, label=label, plot_callback=collect_data)
        
        results[R_cool] = (t_list, tmax_list)
        
        # FERMETURE DE GMSH (Crucial pour pouvoir re-mailler au prochain tour)
        gmsh_finalize()

    # ============================================================
    # GÉNÉRATION DU GRAPHIQUE COMBINÉ
    # ============================================================
    plt.figure(figsize=(11, 7))
    
    # Un beau dégradé "Froid -> Chaud" (sans utiliser de rouge)
    styles = {
        5.5e-3: {'color': 'navy',           'label': 'Rayon = 5.5 mm', 'ls': '-'},
        4.0e-3: {'color': 'royalblue',      'label': 'Rayon = 4.0 mm', 'ls': '-'},
        3.0e-3: {'color': 'teal',           'label': 'Rayon = 3.0 mm', 'ls': '-'},
        2.0e-3: {'color': 'mediumseagreen', 'label': 'Rayon = 2.0 mm', 'ls': '-'},
        1.0e-3: {'color': 'goldenrod',      'label': 'Rayon = 1.0 mm', 'ls': '-'},
        0.5e-3: {'color': 'darkorange',     'label': 'Rayon = 0.5 mm', 'ls': '-'},
        None:   {'color': 'dimgrey',        'label': 'Sans refroidissement', 'ls': '--'}
    }

    # On force l'ordre de tracé pour que la légende soit bien rangée (du Sans barres -> au plus gros)
    ordre_trace = [None, 0.5e-3, 1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3, 5.5e-3]
    
    for R_cool in ordre_trace: 
        if R_cool in results: # Petite sécurité
            t_vals, tmax_vals = results[R_cool]
            plt.plot(t_vals, tmax_vals, color=styles[R_cool]['color'], 
                     label=styles[R_cool]['label'], linestyle=styles[R_cool]['ls'], linewidth=2.5)

    # Ligne d'ébullition (La seule ligne rouge)
    plt.axhline(y=344.85, color='red', linestyle='-', linewidth=2, label="Limite d'ébullition ~345°C", zorder=10)
    plt.fill_between([0, t_end], 344.85, 400, color='red', alpha=0.1)

    plt.xlabel("Temps écoulé depuis l'arrêt [s]", fontsize=13)
    plt.ylabel("Température maximale $T_{max}$ [°C]", fontsize=13)
    plt.title("Évolution Thermique en Fonction du Dimensionnement des Barres (Rayon)", fontsize=14, fontweight='bold')
    
    # Légende en haut à gauche pour ne pas cacher les courbes qui descendent
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Sauvegarde
    plt.savefig("analyse_parametrique_rayon.pdf", dpi=300)
    print("\nGraphique généré et sauvegardé sous 'analyse_parametrique_rayon.pdf'")
    plt.show()

if __name__ == "__main__":
    generate_comparative_radius_plot()
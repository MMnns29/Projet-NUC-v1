import numpy as np
import matplotlib.pyplot as plt
import os

from gmsh_utils import border_dofs_from_tags, gmsh_init, gmsh_finalize, prepare_quadrature_and_basis, get_jacobians, get_boundary_segments, mesh5
from stiffness import assemble_stiffness_and_rhs, build_neumann_vector
from physics import build_water_lookup_table, solve_diffusion, cooling_robin_terms, exponential_flux
from mass import assemble_mass

def generate_comparative_plot():
    # ============================================================
    # PARAMÈTRES (Identiques à ton main)
    # ============================================================
    m_val = 8 
    n_val =  1
    pitch_val = 18.7e-3
    R_rod_val = 6.15e-3
    R_cooling_val = 2.0e-3
    gap_assembly_val = 12e-3
    smin_val = 0.5e-3 / 0.3

    T0_K = 553.15
    P_MPa = 15.5
    theta = 1.0
    dt = 0.5
    t_end = 20.0  # Mis à 120s pour bien voir le cas t_insert=90s
    q0 = 34000     
    lam = 1/80.0
    T_ext = 500.0
    H_bar = 0.5

    # ============================================================
    # INITIALISATION GÉOMÉTRIE ET MATRICES (Fait UNE SEULE fois)
    # ============================================================
    print("Initialisation du maillage et des matrices globales...")
    gmsh_init("poisson_2d")
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, cooling_positions = mesh5(
        m=m_val, n=n_val, order=1, pitch=pitch_val, R_rod=R_rod_val, R_cooling=R_cooling_val, 
        gap_assembly=gap_assembly_val, smin=smin_val, add_cooling_rods=True)

    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    tag_to_dof = np.full(int(np.max(nodeTags)) + 1, -1, dtype=int)
    for i, tag in enumerate(unique_dofs_tags): tag_to_dof[int(tag)] = i

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, 1)
    jac, det, coords = get_jacobians(elemType, xi)

    K_lil = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, lambda x: 1.0, tag_to_dof)
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    K = K_lil.tocsr()
    M = M_lil.tocsr()

    lut = build_water_lookup_table(400.0, 650.0, 250, P_MPa)

    # Récupération des bords
    rod_data = get_boundary_segments(20, 1)
    cooling_data = get_boundary_segments(30, 1)
    cooling_dofs = border_dofs_from_tags(np.unique(cooling_data[2]), tag_to_dof)

    def rod_rhs(t, U): 
        return build_neumann_vector(num_dofs, rod_data, exponential_flux(t, q0, lam), tag_to_dof)

    # ============================================================
    # BOUCLE SUR LES DIFFERENTS TEMPS D'INSERTION
    # ============================================================
    # On ajoute None pour le cas "Sans barres / False"
    scenarios = [0.0, 1.0, 5.0, 10.0, 15.0, None] 
    results = {}

    for t_ins in scenarios:
        label = f"t_ins={t_ins}s" if t_ins is not None else "Sans barres"
        print(f"\n--- Lancement Simulation : {label} ---")
        
        t_list = []
        tmax_list = []
        
        # Callback pour enregistrer la Tmax en Celsius à chaque pas
        def collect_data(step, t, U):
            t_list.append(t)
            tmax_list.append(np.max(U) - 273.15)

        # Définition de la fonction Robin spécifique à ce scénario
        if t_ins is not None:
            def cooling_robin(t, U):
                if t < t_ins:
                    return 0, np.zeros(num_dofs)
                return cooling_robin_terms(U, num_dofs, cooling_data, cooling_dofs, T_ext, H_bar, lut, tag_to_dof)
        else:
            cooling_robin = None

        U0 = np.full(num_dofs, T0_K)
        
        # Résolution
        solve_diffusion(M, K, U0, lut, T_ext, pitch_val, dt, t_end,
                        theta=theta, rhs_extra=rod_rhs, robin_extra=cooling_robin,
                        print_every=100, label=label, plot_callback=collect_data)
        
        results[t_ins] = (t_list, tmax_list)

    gmsh_finalize()

    # ============================================================
    # GÉNÉRATION DU GRAPHIQUE COMBINÉ
    # ============================================================
    plt.figure(figsize=(11, 7))
    
    # Couleurs et styles pour chaque courbe
    styles = {
        0.0: {'color': 'blue', 'label': 'Activation immédiate (0s)', 'ls': '-'},
        1.0: {'color': 'cyan', 'label': 'Retard de 10s', 'ls': '-'},
        5.0: {'color': 'green', 'label': 'Retard de 30s', 'ls': '-'},
        10.0: {'color': 'orange', 'label': 'Retard de 60s', 'ls': '-'},
        15.0: {'color': 'purple', 'label': 'Retard de 90s', 'ls': '-'},
        None: {'color': 'red', 'label': 'Sans refroidissement (Défaillance)', 'ls': '--'}
    }

    for t_ins in scenarios:
        t_vals, tmax_vals = results[t_ins]
        plt.plot(t_vals, tmax_vals, color=styles[t_ins]['color'], 
                 label=styles[t_ins]['label'], linestyle=styles[t_ins]['ls'], linewidth=2.5)

    # Ligne d'ébullition
    plt.axhline(y=344.85, color='black', linestyle=':', linewidth=2, label="Température de saturation (~345 °C)")

    plt.xlabel("Temps écoulé depuis l'arrêt [s]", fontsize=13)
    plt.ylabel("Température maximale $T_{max}$ [°C]", fontsize=13)
    plt.title("Impact du délai d'activation du refroidissement passif", fontsize=15)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Sauvegarde
    plt.savefig("analyse_parametrique_t_insert.pdf", dpi=300)
    print("\nGraphique généré et sauvegardé sous 'analyse_parametrique_t_insert.pdf'")
    plt.show()

if __name__ == "__main__":
    generate_comparative_plot()
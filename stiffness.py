# stiffness.py
import numpy as np
from scipy.sparse import lil_matrix


# ===============================================
# ASSEMBLAGE MATRICE DE RAIDEUR ET VECTEUR CHARGE
# ===============================================

def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, tag_to_dof):
    """
    Assemble K pour : -div(kappa * grad(u)) = 0
      K_ij = int kappa * grad(N_i) . grad(N_j) dx
    Les gradients gmsh sont en coord de reference, on remonte en coords physiques via inv(J).
    Le code était à la base prévu 
    """

    # ===============================================
    # ETAPE 1 : dimensions et reshape des donnees gmsh
    # ===============================================
    ne   = len(elemTags)                  # nombre d'elements
    ngp  = len(w)                         # nombre de points de Gauss par element
    nloc = int(len(conn) // ne)           # noeuds par element (3 pour P1, 6 pour P2, ...)
    nn   = int(np.max(tag_to_dof) + 1)   # nombre total de dofs

    # reshape des tableaux aplatis renvoyes par gmsh
    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)        # coords physiques des pts de Gauss
    jac   = np.asarray(jac,   dtype=np.float64).reshape(ne, ngp, 3, 3)     # jacobien de la transfo reference -> physique
    conn  = np.asarray(conn,  dtype=np.int64  ).reshape(ne, nloc)           # connectivite : tags gmsh des noeuds de chaque element
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)          # fonctions de forme evaluees aux pts de Gauss
    gN    = np.asarray(gN,    dtype=np.float64).reshape(ngp, nloc, 3)       # gradients des fonctions de forme (coords reference)

    # ===============================================
    # ETAPE 2 : initialisation K global
    # ===============================================
    K = lil_matrix((nn, nn), dtype=np.float64)   # lil_matrix : format efficace pour l'assemblage incremental

    # ===============================================
    # ETAPE 3 : boucle d'assemblage element par element
    # ===============================================
    for e in range(ne):
        dof_indices = tag_to_dof[conn[e, :]]   # mapping tags gmsh -> indices dof globaux pour cet element

        for g in range(ngp):
            xg    = xphys[e, g]               # coordonnees physiques du point de Gauss g
            wg    = w[g]                       # poids de quadrature de Gauss
            detg  = det[e, g]                  # determinant du jacobien (facteur de changement de volume)
            invJ  = np.linalg.inv(jac[e, g])   # jacobien inverse : sert a transformer les gradients reference -> physique

            kappa_g = float(kappa_fun(xg))     # conductivite thermique au pt de Gauss [W/(m*K)]

            for a in range(nloc):
                Ia     = int(dof_indices[a])
                gradNa = invJ @ gN[g, a]       # gradient physique de N_a : invJ * grad_ref(N_a)

                for b in range(nloc):
                    Ib     = int(dof_indices[b])
                    gradNb = invJ @ gN[g, b]

                    # contribution a la matrice de raideur : int kappa * grad(N_a) . grad(N_b) dx
                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

    return K


# Construit le vecteur Neumann F_i = int_Gamma q * N_i ds (si on a un cooling actif)
# indépendante (flux crayons). Peut être appelée plusieurs fois et additionnée.
def build_neumann_vector(num_dofs, boundary_data, flux_value, tag_to_dof):
    """
    Vecteur Neumann : F_i = int_{Gamma} flux_value * phi_i ds
    """
    _, elem_tags, node_tags, jac, det, xphys, w, N, gN = boundary_data
    ne   = len(elem_tags)
    ngp  = len(w)
    nloc = int(len(node_tags) // ne)

    det_r  = np.asarray(det,       dtype=np.float64).reshape(ne, ngp)
    conn_r = np.asarray(node_tags, dtype=np.int64  ).reshape(ne, nloc)
    N_r    = np.asarray(N,         dtype=np.float64).reshape(ngp, nloc)

    F = np.zeros(num_dofs, dtype=np.float64)
    for e in range(ne):
        dofs = tag_to_dof[conn_r[e, :]]
        for g in range(ngp):
            wg   = w[g]
            detg = det_r[e, g]
            for a in range(nloc):
                Ia = int(dofs[a])
                F[Ia] += wg * flux_value * N_r[g, a] * detg
    return F

# Construit la matrice Robin R_ij = int_Gamma h * N_i * N_j ds
# Et aussi le vecteur source G_i = int_Gamma h * T_ext * N_i ds. R s'ajoute à K dans le système, G au second membre.
def build_robin_system(num_dofs, boundary_data, alpha, g_R, tag_to_dof):
    """
    Condition de Robin : alpha*u + k*dn(u) = alpha*g_R sur Gamma_R
    R_ij = int_{Gamma_R} alpha * phi_i * phi_j ds
    G_i  = int_{Gamma_R} alpha * g_R * phi_i ds
    """
    _, elem_tags, node_tags, jac, det, xphys, w, N, gN = boundary_data
    ne   = len(elem_tags)
    ngp  = len(w)
    nloc = int(len(node_tags) // ne)

    det_r  = np.asarray(det,       dtype=np.float64).reshape(ne, ngp)
    conn_r = np.asarray(node_tags, dtype=np.int64  ).reshape(ne, nloc)
    N_r    = np.asarray(N,         dtype=np.float64).reshape(ngp, nloc)

    R = lil_matrix((num_dofs, num_dofs), dtype=np.float64)
    G = np.zeros(num_dofs, dtype=np.float64)

    for e in range(ne):
        dofs = tag_to_dof[conn_r[e, :]]
        for g in range(ngp):
            wg   = w[g]
            detg = det_r[e, g]
            for a in range(nloc):
                Ia = int(dofs[a])
                G[Ia] += wg * alpha * g_R * N_r[g, a] * detg
                for b in range(nloc):
                    Ib = int(dofs[b])
                    R[Ia, Ib] += wg * alpha * N_r[g, a] * N_r[g, b] * detg

    return R.tocsr(), G
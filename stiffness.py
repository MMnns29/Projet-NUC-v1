# stiffness.py
import numpy as np
from scipy.sparse import lil_matrix


# ===============================================
# ASSEMBLAGE MATRICE DE RAIDEUR ET VECTEUR CHARGE
# ===============================================

def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble K et F pour : -div(kappa * grad(u)) = f
      K_ij = int kappa * grad(N_i) . grad(N_j) dx
      F_i  = int f * N_i dx
    Les gradients gmsh sont en coords de reference, on remonte en coords physiques via inv(J).
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
    # ETAPE 2 : initialisation K et F globaux
    # ===============================================
    K = lil_matrix((nn, nn), dtype=np.float64)   # lil_matrix : format efficace pour l'assemblage incremental
    F = np.zeros(nn, dtype=np.float64)

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
            f_g     = float(rhs_fun(xg))       # terme source volumique au pt de Gauss [W/m^3]

            for a in range(nloc):
                Ia     = int(dof_indices[a])
                gradNa = invJ @ gN[g, a]       # gradient physique de N_a : invJ * grad_ref(N_a)

                # contribution au vecteur charge : int f * N_a dx
                F[Ia] += wg * f_g * N[g, a] * detg

                for b in range(nloc):
                    Ib     = int(dof_indices[b])
                    gradNb = invJ @ gN[g, b]

                    # contribution a la matrice de raideur : int kappa * grad(N_a) . grad(N_b) dx
                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

    return K, F


# ===============================================
# ASSEMBLAGE VECTEUR NEUMANN (CONDITIONS AUX LIMITES)
# ===============================================

def assemble_rhs_neumann(F, elemTags, conn, jac, det, xphys, w, N, gN, g_neu_fun, tag_to_dof):
    """
    Ajoute la contribution Neumann a F :
      F_i += int_{bord} g_neu * N_i ds
    g_neu_fun peut etre une fonction de x ou une valeur scalaire uniforme.
    jac et gN sont inutiles ici (pas de terme de diffusion sur le bord) et sont ignores.
    """
    ne   = len(elemTags)
    ngp  = len(w)
    nloc = int(len(conn) // ne)   # noeuds par element de bord (2 pour P1, 3 pour P2, ...)

    # reshape — jac et gN non utilises ici
    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    conn  = np.asarray(conn,  dtype=np.int64  ).reshape(ne, nloc)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)

    for e in range(ne):
        dof_indices = tag_to_dof[conn[e, :]]   # dofs des noeuds de l'element de bord e

        for g in range(ngp):
            xg    = xphys[e, g]
            wg    = w[g]
            detg  = det[e, g]   # det du jacobien 1D : longueur de l'arete ramenee a l'element reference

            # flux Neumann au point de Gauss (scalaire ou fonction spatiale)
            g_neu_g = float(g_neu_fun(xg)) if callable(g_neu_fun) else float(g_neu_fun)

            for a in range(nloc):
                Ia = int(dof_indices[a])
                # repartition du flux sur les noeuds via les fonctions de forme : int g_neu * N_a ds
                F[Ia] += wg * g_neu_g * N[g, a] * detg

    return F


def build_neumann_vector(num_dofs, boundary_data, flux_value, tag_to_dof):
    """
    Wrapper : construit le vecteur Neumann complet pour un bord avec flux scalaire uniforme.
    boundary_data : tuple retourne par get_boundary_segments()
    flux_value    : flux scalaire a cet instant [W/m^2]
    """
    _, elemTags, nodeTags, jac, det, xphys, w, N, gN = boundary_data   # deballage du tuple boundary_data
    F = np.zeros(num_dofs)
    assemble_rhs_neumann(F, elemTags, nodeTags, jac, det, xphys, w, N, gN, flux_value, tag_to_dof)
    return F
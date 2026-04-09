# stiffness.py
import numpy as np
from scipy.sparse import lil_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble global stiffness matrix and load vector for:
        -d/dx (kappa(x) du/dx) = f(x)

    K_ij = ∫ kappa * grad(N_i)·grad(N_j) dx
    F_i  = ∫ f * N_i dx

    Notes:
    - gmsh gives gN in reference coordinates; we map with inv(J).
    - For 1D line embedded in 3D, gmsh provides a 3x3 Jacobian; we keep the same approach.

    Returns
    -------
    K : lil_matrix (nn x nn)
    F : ndarray (nn,)
    """
    
    #fonction 
    ne = len(elemTags)#nombre d'éléments
    ngp = len(w)
    nloc = int(len(conn) // ne)  #= 3 en triangle ordre 1, et 6 en triangle ordre 3; marche aussi pour les autres formes
    #conn est le tableau aplati que rend gmsh, il a une taille nlc x ne donc ici on peut récup nloc
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            kappa_g = float(kappa_fun(xg))
            f_g = float(rhs_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                F[Ia] += wg * f_g * N[g, a] * detg

                gradNa = invjacg @ gN[g, a]
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    gradNb = invjacg @ gN[g, b]
                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

    return K, F

def assemble_rhs_neumann(F, elemTags, conn, jac, det, xphys, w, N, gN, g_neu_fun, tag_to_dof):
    #fonction qui parcourt les éléments et calcule leur jacobien pour répartir la cheleur sur les neufs avec la fonction norme Na
    #était pas dans le script du tp4 mais était dans le code d'exemple qu'ils ont passé sur moodle le 1/04 de mémoire
    #ne connait pas la valeur du flux : l'expression du flux (expo décroissante chez nous) est codée dans le main
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    #jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)      #inutile ici car pas de terme source
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    #gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)   #inutile ici car pas de terme source

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]

            if callable(g_neu_fun):
                g_neu_g = float(g_neu_fun(xg))
            else:
                g_neu_g = float(g_neu_fun)

            for a in range(nloc):
                Ia = int(dof_indices[a])
                N_a = N[g, a]
                F[Ia] += wg * g_neu_g * N_a * detg

    return F


def build_neumann_vector(num_dofs, boundary_data, flux_value, tag_to_dof):
    """
    Construit le vecteur F_neumann pour un bord donné avec un flux scalaire uniforme.
    boundary_data : tuple retourné par get_boundary_segments()
    flux_value    : valeur scalaire du flux [W/m²] à cet instant
    """
    _, elemTags, nodeTags, jac, det, xphys, w, N, gN = boundary_data
    F = np.zeros(num_dofs)
    assemble_rhs_neumann(F, elemTags, nodeTags, jac, det, xphys,
                          w, N, gN, flux_value, tag_to_dof)
    return F
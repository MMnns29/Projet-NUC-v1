# mass.py
#nouveauté par rapport au code ape4 : permet de sauter les trucs vides de gmsh
#ou juste d'acccepter un code gmsh qui s'en blc des points vides pour éviter d'intégrer
#sur plein d'éléments vides, comme expliqué au tp (pas la même chose que element wise)

#Le nouveau code des profs introduit tag_to_dof, exactement comme dans stiffness.py. 
#Ce tableau fait la conversion propre entre l'identifiant brut Gmsh 
#et un indice compact (0, 1, 2, 3...) pour ta matrice


import numpy as np
from scipy.sparse import lil_matrix


def assemble_mass(elemTags, conn, det, w, N, tag_to_dof):
    """
    Assemble global mass matrix:
        M_ij = sum_e ∫_e N_i N_j dx

    Parameters
    ----------
    elemTags : array-like, shape (ne,)
    conn     : flattened connectivity (ne*nloc)
    det      : flattened det(J) values (ne*ngp)
    w        : quadrature weights (ngp)
    N        : flattened basis values (ngp*nloc)

    Returns
    -------
    M : lil_matrix (nn x nn)
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)

    M = lil_matrix((nn, nn), dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            wg = w[g]
            detg = det[e, g]
            for a in range(nloc):
                Ia = int(dof_indices[a])
                Na = N[g, a]
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    M[Ia, Ib] += wg * Na * N[g, b] * detg

    return M

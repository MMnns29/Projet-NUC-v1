# errors.py
import numpy as np
import gmsh


def _numeric_grad_3d(u_exact, x, eps=1e-7):
    """
    Central finite-difference gradient in R^3 for a scalar function u_exact(x).
    x is a length-3 array-like.
    Returns grad as (3,) array.
    """
    x = np.asarray(x, dtype=float).reshape(3,)
    g = np.zeros(3, dtype=float)
    for i in range(3):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (float(u_exact(xp)) - float(u_exact(xm))) / (2.0 * eps)
    return g


def compute_L2_H1_errors(
    elemType,
    elemTags,
    elemNodeTags,
    U,
    xi, w, N, gN,
    jac, det, coords,
    u_exact,
    tag_to_dof,      
    grad_exact=None,
    eps_grad=1e-7
):
    import numpy as np
    import gmsh

    U = np.asarray(U, dtype=float)

    # 1. Get properties for the high-order element (e.g., nloc=10 for Order 3)
    _, dim, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    ne = len(elemTags)
    ngp = len(w)

    # 2. Reshape GMSH data
    w = np.asarray(w, dtype=float)
    det = np.asarray(det, dtype=float).reshape(ne, ngp)
    jac = np.asarray(jac, dtype=float).reshape(ne, ngp, 3, 3)
    #  'coords' are the physical locations of the GAUSS points
    coords_gp = np.asarray(coords, dtype=float).reshape(ne, ngp, 3)

    # 3. Map GMSH tags to our compact DoF indices
    raw_conn = np.asarray(elemNodeTags, dtype=np.int64).reshape(ne, nloc)
    conn = tag_to_dof[raw_conn] 

    N = np.asarray(N, dtype=float).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=float).reshape(ngp, nloc, 3)

    if grad_exact is None:
        def grad_fun(x):
            # Assumes a helper function _numeric_grad_3d exists in your scope
            return _numeric_grad_3d(u_exact, x, eps=eps_grad)
    else:
        def grad_fun(x):
            return np.asarray(grad_exact(x), dtype=float).reshape(3,)

    I_L2 = 0.0
    I_H1 = 0.0

    # 4. Integration Loop
    for e in range(ne):
        # Extract the solution values at the specific DoFs for this element
        dof_indices = conn[e, :]
        Ue = U[dof_indices] 

        for g in range(ngp):
            xg = coords_gp[e, g] 
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            # Interpolate FE value: u_h = sum(N_a * U_a)
            uh = np.dot(N[g, :], Ue)

            # Interpolate FE gradient: grad(u_h) = sum(inv(J) * gN_a * U_a)
            grad_uh = np.zeros(3)
            for a in range(nloc):
                grad_phys = invjacg @ gN[g, a]
                grad_uh += grad_phys * Ue[a]

            # Compare with exact solution at Gauss point
            uex = float(u_exact(xg))
            gex = grad_fun(xg)

            du = uh - uex
            dg = grad_uh - gex

            # Accumulate squared errors
            I_L2 += wg * (du**2) * detg
            I_H1 += wg * np.dot(dg, dg) * detg

    err_L2 = np.sqrt(max(I_L2, 0.0))
    err_H1_semi = np.sqrt(max(I_H1, 0.0))
    err_H1 = np.sqrt(max(I_L2 + I_H1, 0.0))

    return err_L2, err_H1_semi, err_H1


    

def compute_energy_gap(
    elemType,
    elemTags,
    elemNodeTags,
    U,
    kappa,
    f,
    xi, w, N, gN,
    jac, det, coords,
    u_exact,
    grad_exact=None,
    eps_grad=1e-7
):
    """
    Compute energetic error against an analytical solution.

    Parameters
    ----------
    elemType : int
        Gmsh element type (line/triangle/tetra, etc.)
    elemTags : array-like, (ne,)
        Element tags from gmsh.model.mesh.getElementsByType(elemType)[0]
    elemNodeTags : array-like, flattened length ne*nloc
        Element connectivity from gmsh.model.mesh.getElementsByType(elemType)[1]
    U : ndarray, (nn,)
        FE solution vector aligned with gmsh compact node ordering (nodeTag-1 indexing).
    kappa : callable
        kappa(xyz) -> float, diffusivity at point xyz (length-3 array-like)
    f : callable
        f(xyz) -> float, source term at point xyz (length-3 array-like)
    xi, w, N, gN :
        Outputs of prepare_quadrature_and_basis(elemType, order)
    jac, det, coords :
        Outputs of get_jacobians(elemType, xi)
        (flattened lists from gmsh.model.mesh.getJacobians)
    u_exact : callable
        u_exact(xyz) -> float, where xyz is length-3 array-like (x,y,z)
    grad_exact : callable or None
        grad_exact(xyz) -> array-like shape (3,) giving (du/dx, du/dy, du/dz).
        If None, gradient is approximated numerically from u_exact.
    eps_grad : float
        Step for numerical gradient if grad_exact is None.

    Returns
    -------
    err_energy : float
        1/2 * ||u_h - u||_{energy} = sqrt(I_Pi_h - I_Pi_ex) where I_Pi_h is the energy of the numerical solution and I_Pi_ex is the energy of the exact solution.
    """
    U = np.asarray(U, dtype=float)

    # Element properties to infer nloc
    _, dim, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    ne = len(elemTags)
    ngp = len(w)

    # reshape gmsh data
    w = np.asarray(w, dtype=float)
    det = np.asarray(det, dtype=float).reshape(ne, ngp)
    jac = np.asarray(jac, dtype=float).reshape(ne, ngp, 3, 3)
    coords = np.asarray(coords, dtype=float).reshape(ne, ngp, 3)

    conn = np.asarray(elemNodeTags, dtype=np.int64).reshape(ne, nloc) - 1  # 0-based
    N = np.asarray(N, dtype=float).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=float).reshape(ngp, nloc, 3)

    # pick gradient function
    if grad_exact is None:
        def grad_fun(x):
            return _numeric_grad_3d(u_exact, x, eps=eps_grad)
    else:
        def grad_fun(x):
            return np.asarray(grad_exact(x), dtype=float).reshape(3,)

    I_Pi_h = 0.0
    I_Pi_ex = 0.0
    I_a = 0.0
    for e in range(ne):
        nodes = conn[e, :]
        Ue = U[nodes]

        for g in range(ngp):
            xg = coords[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            uh = 0.0
            for a in range(nloc):
                uh += N[g, a] * Ue[a]

            grad_uh = np.zeros(3, dtype=float)
            for a in range(nloc):
                gradNa = invjacg @ gN[g, a]
                grad_uh += gradNa * Ue[a]

            uex = float(u_exact(xg))
            kappa_g = float(kappa(xg))
            f_g = float(f(xg))
            gex = grad_fun(xg)

            I_Pi_h += wg*(0.5 * kappa_g * float(np.dot(grad_uh, grad_uh)) - f_g * uh)*detg
            I_Pi_ex += wg*(0.5 * kappa_g * float(np.dot(gex, gex)) - f_g * uex)*detg

            I_a += wg*(kappa_g * float(np.dot(gex - grad_uh, gex - grad_uh)))*detg

    dPi = float(I_Pi_h - I_Pi_ex)

    return np.sqrt(dPi)
# gmsh_utils.py
import numpy as np
import gmsh


def gmsh_init(model_name="fem1d"):
    gmsh.initialize()
    gmsh.model.add(model_name)


def gmsh_finalize():
    gmsh.finalize()


def prepare_quadrature_and_basis(elemType, order):
    rule = f"Gauss{2 * order}"  # ordre 2k suffit pour intégrer exactement les polynômes P_k
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)  # points et poids de Gauss
    _, N, _  = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")       # fonctions de base
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")   # gradients en coords référence
    return xi, np.asarray(w, dtype=float), N, gN


def get_jacobians(elemType, xi):
    # Jacobiens de la transformation référence → physique pour tous les éléments
    jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi)
    return jacobians, dets, coords


def border_dofs_from_tags(l_tags, tag_to_dof):
    # Convertit des tags Gmsh en indices DOF compacts (0..N-1)
    l_tags = np.asarray(l_tags, dtype=int)
    valid_mask = (tag_to_dof[l_tags] != -1)  # filtre les tags de points géométriques (non DOF)
    return tag_to_dof[l_tags[valid_mask]]


def mesh5(m=3, n=4, order=1, pitch=18.7e-3, R_rod=6.15e-3, R_cooling=2.0e-3, gap_assembly=12e-3, R_core=None, smin=1.5e-3, add_cooling_rods=True):

    import sys, math
    assert n in (1, 4, 9), "n doit être 1, 4 ou 9"

    # ============================================================
    # VÉRIFICATIONS GÉOMÉTRIQUES
    # ============================================================

    gap_rod = pitch - 2.0 * R_rod  # espace libre entre deux crayons adjacents
    if gap_rod <= 0.0:
        raise ValueError("Les crayons se chevauchent : pitch=%.2f mm, 2*R_rod=%.2f mm" % (pitch*1e3, 2*R_rod*1e3))
    if gap_rod < 3 * smin:
        print("AVERTISSEMENT : gap inter-crayons = %.2f mm (~%.1f elements). Recommande smin <= %.2f mm." % (gap_rod*1e3, gap_rod/smin, gap_rod/3*1e3), file=sys.stderr)
    if gap_assembly <= 0.0:
        raise ValueError("gap_assembly doit etre > 0")

    # ============================================================
    # POSITIONS GÉOMÉTRIQUES
    # ============================================================

    L_asm   = m * pitch                                              # côté d'un assemblage [m]
    n_sqrt  = int(math.sqrt(n))
    L_cluster = n_sqrt * L_asm + (n_sqrt - 1) * gap_assembly        # largeur totale des assemblages

    # Centres des assemblages selon la configuration (1, 4 ou 9)
    if n == 1:
        centers = [(0.0, 0.0)]
    elif n == 4:
        s = L_asm / 2.0 + gap_assembly / 2.0
        centers = [(sx*s, sy*s) for sx in (-1., 1.) for sy in (-1., 1.)]
    else:
        d = L_asm + gap_assembly
        centers = [(i*d, j*d) for i in (-1, 0, 1) for j in (-1, 0, 1)]

    # Positions de tous les crayons sur la grille pitch×pitch de chaque assemblage
    offsets  = [(k - (m-1)/2.0)*pitch for k in range(m)]
    all_rods = [(cx+dx, cy+dy) for (cx, cy) in centers for dx in offsets for dy in offsets]

    # Demi-côté du domaine : bord à gap_assembly/2 au-delà des barres extérieures
    # → condition de périodicité : réplication du domaine = réseau infini uniforme
    hw   = L_cluster / 2.0 + gap_assembly / 2.0
    smax = L_cluster / 6.0  # taille max des éléments loin des bords (adaptatif)

    # Filtrage optionnel des crayons hors d'un rayon R_core
    if R_core is not None:
        kept_rods = [(rx, ry) for (rx, ry) in all_rods if math.hypot(rx, ry) <= R_core]
        n_removed = len(all_rods) - len(kept_rods)
        if n_removed:
            print("Info : %d crayons supprimes (R_core=%.1f mm)" % (n_removed, R_core*1e3))
    else:
        kept_rods = all_rods

    # ============================================================
    # GÉOMÉTRIE GMSH (OCC)
    # ============================================================

    # Domaine rectangulaire — Neumann homogène sur les 4 bords donc plan de symétrie périodique
    p1 = gmsh.model.occ.addPoint(-hw, -hw, 0)
    p2 = gmsh.model.occ.addPoint( hw, -hw, 0)
    p3 = gmsh.model.occ.addPoint( hw,  hw, 0)
    p4 = gmsh.model.occ.addPoint(-hw,  hw, 0)
    l1 = gmsh.model.occ.addLine(p1, p2)  # bas
    l2 = gmsh.model.occ.addLine(p2, p3)  # droite
    l3 = gmsh.model.occ.addLine(p3, p4)  # haut
    l4 = gmsh.model.occ.addLine(p4, p1)  # gauche
    outer_curves = [l1, l2, l3, l4]
    outer_loop   = gmsh.model.occ.addCurveLoop(outer_curves)

    # Crayons combustibles — trous dans la surface eau (Neumann : flux q(t) imposé)
    rod_circles, rod_loops = [], []
    for (rx, ry) in kept_rods:
        c  = gmsh.model.occ.addCircle(rx, ry, 0, R_rod)
        lp = gmsh.model.occ.addCurveLoop([c])
        rod_circles.append(c)
        rod_loops.append(lp)

    # Barres de refroidissement — trous dans la surface eau (Neumann : flux Churchill-Chu)
    cool_circles, cool_loops, cooling_positions = [], [], []
    if add_cooling_rods:
        outer_offset = m / 2.0 * pitch  # distance bord assemblage → centre barre extérieure

        for (cx, cy) in centers:
            # Barres intérieures : au centre de chaque carré formé par 4 crayons voisins
            for i in range(m - 1):
                for j in range(m - 1):
                    bx = cx + (i - (m - 2) / 2.0) * pitch
                    by = cy + (j - (m - 2) / 2.0) * pitch
                    c  = gmsh.model.occ.addCircle(bx, by, 0, R_cooling)
                    lp = gmsh.model.occ.addCurveLoop([c])
                    cool_circles.append(c)
                    cool_loops.append(lp)
                    cooling_positions.append((bx, by))

            # Barres extérieures : milieu de chaque paire de crayons sur les 4 bords
            # Chaque assemblage a ses propres barres → 2 rangées entre assemblages voisins
            for k in range(m - 1):
                mid = (k - (m - 2) / 2.0) * pitch
                for bx, by in [(cx - outer_offset, cy + mid), (cx + outer_offset, cy + mid), (cx + mid, cy - outer_offset), (cx + mid, cy + outer_offset)]:
                    c  = gmsh.model.occ.addCircle(bx, by, 0, R_cooling)
                    lp = gmsh.model.occ.addCurveLoop([c])
                    cool_circles.append(c)
                    cool_loops.append(lp)
                    cooling_positions.append((bx, by))

            # ==========================================
            # NOUVEAU : Barres dans les 4 coins extrêmes
            # ==========================================
            for bx, by in [
                (cx - outer_offset, cy - outer_offset), # Coin bas gauche
                (cx + outer_offset, cy - outer_offset), # Coin bas droite
                (cx - outer_offset, cy + outer_offset), # Coin haut gauche
                (cx + outer_offset, cy + outer_offset)  # Coin haut droite
            ]:
                c  = gmsh.model.occ.addCircle(bx, by, 0, R_cooling)
                lp = gmsh.model.occ.addCurveLoop([c])
                cool_circles.append(c)
                cool_loops.append(lp)
                cooling_positions.append((bx, by))
            # ==========================================

    # Surface eau = rectangle - trous crayons - trous barres
    surface = gmsh.model.occ.addPlaneSurface([outer_loop] + rod_loops + cool_loops)
    gmsh.model.occ.synchronize()

    # ============================================================
    # GROUPES PHYSIQUES (tags utilisés dans main pour get_boundary_segments)
    # ============================================================

    gmsh.model.addPhysicalGroup(2, [surface],       tag=1,  name="water")
    gmsh.model.addPhysicalGroup(1, outer_curves,    tag=10, name="outer_boundary")  # Neumann homogène
    if rod_circles:
        gmsh.model.addPhysicalGroup(1, rod_circles, tag=20, name="rod_surfaces")    # flux crayons
    if cool_circles:
        gmsh.model.addPhysicalGroup(1, cool_circles, tag=30, name="cooling_surfaces")  # flux barres

    # ============================================================
    # CHAMP DE TAILLE ADAPTATIF
    # ============================================================

    # Raffinage près des crayons et barres (Distance + Threshold)
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", rod_circles + cool_circles)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField",  dist_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin",  smin)   # taille près des bords
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax",  smax)   # taille loin des bords
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin",  0.0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax",  L_asm)  # transition sur 1 longueur d'assemblage
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    # Désactivation des autres sources de taille pour que seul le champ Threshold pilote
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # ============================================================
    # GÉNÉRATION ET EXTRACTION
    # ============================================================

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    elemType     = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    # Construction de bnds : liste (nom, dim) des groupes physiques de bord
    bnds = [("outer_boundary", 1), ("rod_surfaces", 1)]
    if cool_circles:
        bnds.append(("cooling_surfaces", 1))

    # Récupération des tags de noeuds pour chaque groupe de bord
    bnds_tags = []
    for name, dim in bnds:
        tag = next(g[1] for g in gmsh.model.getPhysicalGroups(dim) if gmsh.model.getPhysicalName(dim, g[1]) == name)
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, cooling_positions


def get_boundary_segments(physical_tag, order):
    # Collecte des éléments 1D de toutes les entités du groupe physique
    entities      = gmsh.model.getEntitiesForPhysicalGroup(1, physical_tag)
    elemTags_list = []
    nodeTags_list = []
    line_elemType = None

    for ent in entities:
        etypes, etags, enodes = gmsh.model.mesh.getElements(dim=1, tag=int(ent))
        if len(etypes) == 0:
            continue
        line_elemType = int(etypes[0])
        elemTags_list.append(etags[0])
        nodeTags_list.append(enodes[0])

    elemTags_1d = np.concatenate(elemTags_list).astype(np.int64)
    nodeTags_1d = np.concatenate(nodeTags_list).astype(np.int64)

    # Quadrature sur les segments 1D (même mécanique que pour les triangles 2D)
    xi_1d, w_1d, N_1d, gN_1d = prepare_quadrature_and_basis(line_elemType, order)

    # Jacobiens sur chaque entité du groupe (longueur du segment → det = longueur/2)
    jac_list, det_list, xphys_list = [], [], []
    for ent in entities:
        j, d, x = gmsh.model.mesh.getJacobians(line_elemType, xi_1d.flatten(), tag=int(ent))
        jac_list.append(j)
        det_list.append(d)
        xphys_list.append(x)

    return (line_elemType, elemTags_1d, nodeTags_1d, np.concatenate(jac_list), np.concatenate(det_list), np.concatenate(xphys_list), w_1d, N_1d, gN_1d)
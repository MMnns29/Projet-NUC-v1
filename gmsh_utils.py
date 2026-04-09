# gmsh_utils.py
import numpy as np
import gmsh


def gmsh_init(model_name="fem1d"):
    gmsh.initialize()
    gmsh.model.add(model_name)


def gmsh_finalize():
    gmsh.finalize()


def build_1d_mesh(L=1.0, cl1=0.02, cl2=0.10, order=1):
    """
    Build and mesh a 1D segment [0,L] with different characteristic lengths.
    Returns (line_tag, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags).
    """
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cl1)
    p1 = gmsh.model.geo.addPoint(L, 0.0, 0.0, cl2)
    line = gmsh.model.geo.addLine(p0, p1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order)

    elemType = gmsh.model.mesh.getElementType("line", order)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    return line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags


def prepare_quadrature_and_basis(elemType, order):
    """
    Returns:
      xi (flattened uvw), w (ngp), N (flattened bf), gN (flattened gbf)
    """
    rule = f"Gauss{2 * order}"
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)
    _, N, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN


def get_jacobians(elemType, xi):
    """
    Wrapper around gmsh.getJacobians.
    Returns (jacobians, dets, coords)
    """
    jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi)
    return jacobians, dets, coords


def end_dofs_from_nodes(nodeCoords):
    """
    Robustly identify first/last node dofs from coordinates (x-min, x-max).
    nodeCoords is flattened [x0,y0,z0, x1,y1,z1, ...]
    Returns (left_dof, right_dof) as 0-based indices.
    """
    X = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
    left = int(np.argmin(X))
    right = int(np.argmax(X))
    return left, right



def border_dofs_from_tags(l_tags, tag_to_dof):
    """
    Converts a list of GMSH node tags into the corresponding 
    compact matrix indices (DoFs).
    """
    # Ensure tags are integers
    l_tags = np.asarray(l_tags, dtype=int)
    
    # Filter out any tags that might not be in our DoF mapping (like geometry points)
    # then map them to our 0...N-1 indices
    valid_mask = (tag_to_dof[l_tags] != -1)
    l_dofs = tag_to_dof[l_tags[valid_mask]]
    
    return l_dofs


def build_2d_rectangle_mesh(L=1.0, H=1.0, size_field=None, order=1):
    if size_field is None:
        size_field = lambda x, y: 0.1 * L

    # --- create points
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    p1 = gmsh.model.geo.addPoint(L, 0.0, 0.0)
    p2 = gmsh.model.geo.addPoint(L, H, 0.0)
    p3 = gmsh.model.geo.addPoint(0.0, H, 0.0)

    # --- create lines
    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p0)

    l = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])

    # --- create surface
    surface = gmsh.model.geo.addPlaneSurface([l])

    # --- synchronize geometry
    gmsh.model.geo.synchronize()

    # --- set mesh size using the provided size field
    gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: size_field(x, y))

    gmsh.model.addPhysicalGroup(1, [l0, l1, l2, l3], tag=1)
    gmsh.model.setPhysicalName(1, 1, "OuterBoundary")

    # --- generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # --- element type (triangles)
    elemType = gmsh.model.mesh.getElementType("triangle", order)

    # nodes 
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    # elements 
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)
    # bnd_names and tags
    bnds = [("OuterBoundary", 1)]
    bnds_tags = []
    for name, dim in bnds:
        tag = next(g[1] for g in gmsh.model.getPhysicalGroups(dim) if gmsh.model.getPhysicalName(dim, g[1]) == name)
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags




def mesh1(L=1, H=1, order=1, smin=0.01, smax=0.1, dmin=0.1, dmax=0.3):

    # 1. Define the curve which forms the outer boundary (here a rectangle)
    p1 = gmsh.model.occ.addPoint(0, 0, 0)
    p2 = gmsh.model.occ.addPoint(L, 0, 0)
    p3 = gmsh.model.occ.addPoint(L, H, 0)
    p4 = gmsh.model.occ.addPoint(0, H, 0)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    outer_wire = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

    # 2. Define the curves which will form the holes in the mesh - a circle, an ellipsis and a rectangle
    circle_curve = gmsh.model.occ.addCircle(2*L/5, 3*H/4, 0, L/10)
    circle_wire  = gmsh.model.occ.addCurveLoop([circle_curve])

    ellipse_curve = gmsh.model.occ.addEllipse(L/4, H/4, 0, L/5, L/10)
    ellipse_wire  = gmsh.model.occ.addCurveLoop([ellipse_curve])

    pHoles = [gmsh.model.occ.addPoint(3*L/4, H/4, 0),
              gmsh.model.occ.addPoint(3*L/4 + L/8, H/4, 0), 
              gmsh.model.occ.addPoint(3*L/4 + L/8, H/4 + H/2, 0), 
              gmsh.model.occ.addPoint(3*L/4, H/4 + H/2, 0)]
    lHoles = [gmsh.model.occ.addLine(pHoles[i], pHoles[(i+1)%4]) for i in range(4)]
    square_wire = gmsh.model.occ.addCurveLoop(lHoles)    
    
    # 3. Create the surface. 
    # The first wire that is passed represent the domain, ie, the inside of the wire is meshed
    # Next wires (here circle, ellipse and square) define cuts (holes) on the surface
    surface = gmsh.model.occ.addPlaneSurface([outer_wire, circle_wire, ellipse_wire, square_wire])

    # Synchronize the OpenCASCADE CAD representation with the Gmsh model
    gmsh.model.occ.synchronize()

    # 4. Assign Physical Groups to the distinct wire
    # This can be seen as a key to later retrieve the node that lie on those curves. 
    # An easy access to the boundary node tags is important to apply boundary conditions
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], tag=1)
    gmsh.model.setPhysicalName(1, 1, "OuterBoundary")

    gmsh.model.addPhysicalGroup(1, [circle_curve], tag=2)
    gmsh.model.setPhysicalName(1, 2, "DiskHole")

    gmsh.model.addPhysicalGroup(1, [ellipse_curve], tag=3)
    gmsh.model.setPhysicalName(1, 3, "EllipseHole")

    gmsh.model.addPhysicalGroup(1, lHoles, tag=4)
    gmsh.model.setPhysicalName(1, 4, "SquareHole")

    gmsh.model.addPhysicalGroup(2, [surface], tag=5)
    gmsh.model.setPhysicalName(2, 5, "DomainSurface")

    # 5. Define a sizeField. This can be seen as an indication of the desired element size according to 
    # the position in the mesh. Some size fields are predefined in GMSH, or they can be derived from 
    # mathematical user defined functions. Multiple size fields can also be combined into one. 

    # Here, a first field that depens on the distance with the "ellipse_curve"
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [ellipse_curve])
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", smin)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", smax)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", dmin)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", dmax)

    # Here, a second field that is user defined by a mathematical expression
    # The "Step(val)" function return 0 if val < 0, else it returns 1. It is thus convenient
    # to define size fields with "if" conditions 
    backgroundField = gmsh.model.mesh.field.add("MathEval")
    condition = f"Step(y-0.5)*Step(abs(y)-x)"  
    expr = f"({condition}) * {smin*3} + (1 - ({condition})) * {smax}" 
    gmsh.model.mesh.field.setString(backgroundField, "F", expr)

    # Field 3: The final size field take the minimum of both previous sizefields and is then passed to the mesher
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field, backgroundField])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    # Set other sizefields to 0 to ensure full definition by the one we defined 
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


    # 6. Mesh generation with size callBack and desired order
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # -------------------------------
    # Getter functions
    # -------------------------------
    # element type (triangles)
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    # nodes 
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    # elements 
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)
    # bnd_names and associated node tags : this way we know which nodes are on the boundary
    bnds = [("OuterBoundary", 1), ("DiskHole", 1), ("EllipseHole", 1), ("SquareHole", 1)]
    bnds_tags = []
    for name, dim in bnds:
        tag = next(g[1] for g in gmsh.model.getPhysicalGroups(dim) if gmsh.model.getPhysicalName(dim, g[1]) == name)
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags




def mesh2(L=1, H=1, order=1, smin=0.01, smax=0.1, dmin=0.1, dmax=0.3):

    # 1. Geometry Construction using OCC Primitives
    # Control points for the top Bezier curve (Leading Edge -> Trailing Edge)
    p_le = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)           # Leading Edge (LE)
    p_top_ctrl1 = gmsh.model.occ.addPoint(0.0, 0.15, 0.0)    # Top rounded LE control
    p_top_ctrl2 = gmsh.model.occ.addPoint(0.4, 0.15, 0.0)    # Top mid-body control
    p_te = gmsh.model.occ.addPoint(1.0, 0.0, 0.0)           # Trailing Edge (TE)

    # Control points for the bottom Bezier curve (Trailing Edge -> Leading Edge)
    p_bot_ctrl1 = gmsh.model.occ.addPoint(0.4, -0.05, 0.0)   # Bottom mid-body control
    p_bot_ctrl2 = gmsh.model.occ.addPoint(0.0, -0.05, 0.0)   # Bottom rounded LE control

    # 2. Create Bezier curves
    # gmsh.model.occ.addBezier takes a list of point tags
    top_curve = gmsh.model.occ.addBezier([p_le, p_top_ctrl1, p_top_ctrl2, p_te])
    bot_curve = gmsh.model.occ.addBezier([p_te, p_bot_ctrl1, p_bot_ctrl2, p_le])

    # 3. Form a closed loop and a surface
    curve_loop = gmsh.model.occ.addCurveLoop([top_curve, bot_curve])
    surface = gmsh.model.occ.addPlaneSurface([curve_loop])

    gmsh.model.occ.synchronize()

    # 4. Global Mesh Size (Keeping it simple for the airfoil profile)
    gmsh.option.setNumber("Mesh.MeshSizeMin", smin)
    gmsh.option.setNumber("Mesh.MeshSizeMax", smax)

    # 5. Physical Groups
    # Assign physical group to the external boundary (the airfoil surface)
    gmsh.model.addPhysicalGroup(1, [top_curve, bot_curve], tag=1, name="AirfoilBoundary")
    gmsh.model.addPhysicalGroup(2, [surface], tag=2, name="AirfoilDomain")


    # 6. Mesh Generation
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # 7. Data Retrieval
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)
    
    # Adapted the retrieval logic to match the new airfoil boundary name
    bnds = [("AirfoilBoundary", 1)]
    bnds_tags = []
    for name, dim in bnds:
        p_tag = next(g[1] for g in gmsh.model.getPhysicalGroups(dim) if gmsh.model.getPhysicalName(dim, g[1]) == name)
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, p_tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags




def meshSol(L=2, H=1, order=1, smin=0.01, smax=0.1, dmin=0.1, dmax=0.3):

    # TODO :-)

    exit(0)

    return 


######################################################################
#####################################################################
#code du domaine pour les crayons nucléaires
#n le nombre d'assemblages (1, 4 ou 9) et m le nombre de crayons par coté par assemblage

def mesh3(m=3, n=4, order=1,pitch=18.7e-3, R_rod=6.15e-3,gap_assembly=12e-3, gap_outer=75e-3,
          R_core=None,smin=1.5e-3): #distance entre le bord et le coin extérieur

    import sys, math
    assert n in (1, 4, 9), "n doit être 1, 4 ou 9"

    # --- Vérifications géométriques ---
    gap_rod = pitch - 2.0 * R_rod
    if gap_rod <= 0.0:
        raise ValueError(
            "Les crayons se chevauchent : pitch=%.2f mm, 2*R_rod=%.2f mm"
            % (pitch*1e3, 2*R_rod*1e3))
    if gap_rod < 3 * smin:
        print("AVERTISSEMENT : gap inter-crayons = %.2f mm (~%.1f elements)."
              " Recommande smin <= %.2f mm."
              % (gap_rod*1e3, gap_rod/smin, gap_rod/3*1e3), file=sys.stderr)
    if gap_assembly <= 0.0:
        raise ValueError("gap_assembly doit etre > 0")
        
        
        
        

    # --- Centres des assemblages ---
    L_asm = m * pitch
    
    
    L_cluster = int(math.sqrt(n)) * L_asm + (int(math.sqrt(n)) - 1) * gap_assembly
    gap_outer = L_cluster / 3
    
    if n == 1:
        centers = [(0.0, 0.0)]
    elif n == 4:
        s = L_asm / 2.0 + gap_assembly / 2.0
        centers = [(sx*s, sy*s) for sx in (-1., 1.) for sy in (-1., 1.)]
    else:  # n == 9
        d = L_asm + gap_assembly
        centers = [(i*d, j*d) for i in (-1, 0, 1) for j in (-1, 0, 1)]

    # --- Positions de tous les crayons ---
    offsets = [(k - (m-1)/2.0)*pitch for k in range(m)]
    all_rods = [(cx+dx, cy+dy)
                for (cx, cy) in centers
                for dx in offsets for dy in offsets]

    # --- Rayon du disque ---
    R_disk = max(math.hypot(cx + sx*L_asm/2., cy + sy*L_asm/2.)
                 for (cx, cy) in centers
                 for sx in (-1., 1.) for sy in (-1., 1.)) + gap_outer
    
    
    smax = R_disk / 6   # remplace le paramètre smax passé en argument
    # --- Filtrage crayons hors R_core ---
    R_core_eff = min(R_core, R_disk - R_rod) if R_core is not None else (R_disk - R_rod)
    kept_rods  = [(rx, ry) for (rx, ry) in all_rods
                  if math.hypot(rx, ry) <= R_core_eff]
    n_removed  = len(all_rods) - len(kept_rods)
    if n_removed:
        print("Info : %d crayons supprimes (R_core=%.1f mm)"
              % (n_removed, R_core_eff*1e3))

    # --- Géométrie ---
    outer_circle = gmsh.model.occ.addCircle(0, 0, 0, R_disk)
    outer_loop   = gmsh.model.occ.addCurveLoop([outer_circle])

    rod_circles, rod_loops = [], []
    for (rx, ry) in kept_rods:
        c  = gmsh.model.occ.addCircle(rx, ry, 0, R_rod)
        lp = gmsh.model.occ.addCurveLoop([c])
        rod_circles.append(c)
        rod_loops.append(lp)

    # addPlaneSurface : 1er wire = domaine, suivants = trous (crayons)
    surface = gmsh.model.occ.addPlaneSurface([outer_loop] + rod_loops)
    gmsh.model.occ.synchronize()

    # --- Groupes physiques ---
    gmsh.model.addPhysicalGroup(2, [surface],       tag=1,  name="water")
    gmsh.model.addPhysicalGroup(1, [outer_circle],  tag=10, name="outer_boundary")
    if rod_circles:
        gmsh.model.addPhysicalGroup(1, rod_circles, tag=20, name="rod_surfaces")

    # --- Champ de taille (Distance depuis crayons + Threshold) ---
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", rod_circles)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField",  dist_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin",  smin)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax",  smax)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin",  0.0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax",   L_asm/1)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # --- Génération ---
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # --- Extraction (format identique à mesh1/mesh2) ---
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    bnds = [("outer_boundary", 1), ("rod_surfaces", 1)]
    bnds_tags = []
    for name, dim in bnds:
        tag = next(g[1] for g in gmsh.model.getPhysicalGroups(dim)
                   if gmsh.model.getPhysicalName(dim, g[1]) == name)
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags







def mesh4(H=0.02, m=3, n=4, order=1, pitch=18.7e-3, R_rod=6.15e-3, gap_assembly=12e-3, gap_outer=75e-3, R_core=None, smin=1.5e-3):
    import sys, math
    import gmsh
    import numpy as np

    assert n in (1, 4, 9), "n doit être 1, 4 ou 9"

    # --- Centres des assemblages ---
    L_asm = m * pitch
    L_cluster = int(math.sqrt(n)) * L_asm + (int(math.sqrt(n)) - 1) * gap_assembly
    
    if n == 1:
        centers = [(0.0, 0.0)]
    elif n == 4:
        s = L_asm / 2.0 + gap_assembly / 2.0
        centers = [(sx*s, sy*s) for sx in (-1., 1.) for sy in (-1., 1.)]
    else:  # n == 9
        d = L_asm + gap_assembly
        centers = [(i*d, j*d) for i in (-1, 0, 1) for j in (-1, 0, 1)]

    # --- Positions de tous les crayons ---
    offsets = [(k - (m-1)/2.0)*pitch for k in range(m)]
    all_rods = [(cx+dx, cy+dy) for (cx, cy) in centers for dx in offsets for dy in offsets]

    # --- Rayon du disque / cuve ---
    R_disk = max(math.hypot(cx + sx*L_asm/2., cy + sy*L_asm/2.)
                 for (cx, cy) in centers for sx in (-1., 1.) for sy in (-1., 1.)) + gap_outer
    smax = R_disk / 6

    # =========================================================
    # EXTRACTION DE LA COUPE SUR TOUT LE DIAMÈTRE
    # =========================================================
    # Astuce : Trouver la rangée de crayons la plus proche de y=0 (pour éviter les gaps vides)
    y_slice = min(all_rods, key=lambda p: abs(p[1]))[1]
    
    # On garde tous les crayons (de gauche à droite) situés sur cette ligne Y
    radial_rods = [rx for (rx, ry) in all_rods if abs(ry - y_slice) < 1e-5]
    radial_rods.sort()

    R_core_eff = min(R_core, R_disk - R_rod) if R_core is not None else (R_disk - R_rod)
    # On filtre pour ne garder que ceux dans le cœur
    radial_rods = [rx for rx in radial_rods if abs(rx) <= R_core_eff]

    # --- Géométrie OCC ---
    # Le domaine principal s'étend de -R_disk à +R_disk
    domain_rect = gmsh.model.occ.addRectangle(-R_disk, 0, 0, 2 * R_disk, H)
    
    rod_rects = []
    for rx in radial_rods:
        # Chaque crayon est une bande de largeur 2*R_rod
        rect = gmsh.model.occ.addRectangle(rx - R_rod, 0, 0, 2 * R_rod, H)
        rod_rects.append(rect)
            
    # Découpage booléen : Eau = Rectangle global MOINS les crayons
    if rod_rects:
        water_domain, _ = gmsh.model.occ.cut([(2, domain_rect)], [(2, r) for r in rod_rects], removeObject=True, removeTool=True)
        surface_tags = [tag for (dim, tag) in water_domain]
    else:
        surface_tags = [domain_rect]

    gmsh.model.occ.synchronize()

    # --- Groupes physiques ---
    gmsh.model.addPhysicalGroup(2, surface_tags, tag=1, name="water")
    
    bnd_entities = gmsh.model.getBoundary([(2, s) for s in surface_tags], oriented=False)
    outer_lines = []
    rod_lines = []
    
    for dim, tag in bnd_entities:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
        
        # Bords extérieurs (gauche et droite)
        if (abs(xmin - (-R_disk)) < 1e-5 and abs(xmax - (-R_disk)) < 1e-5) or \
           (abs(xmin - R_disk) < 1e-5 and abs(xmax - R_disk) < 1e-5):
            outer_lines.append(tag)
        # Parois des crayons (lignes verticales internes)
        elif abs(xmin - xmax) < 1e-5 and xmin > -R_disk + 1e-5 and xmin < R_disk - 1e-5:
            rod_lines.append(tag)
            
    gmsh.model.addPhysicalGroup(1, outer_lines, tag=10, name="outer_boundary")
    if rod_lines:
        gmsh.model.addPhysicalGroup(1, rod_lines, tag=20, name="rod_surfaces")

    # --- Champ de taille ---
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", rod_lines)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField",  dist_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin",  smin)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax",  smax)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin",  0.0)
    
    # Adoucissement proportionnel à l'espace d'eau pour que ce soit beau jusqu'au bord
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax",  gap_outer)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # --- Génération ---
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # --- Extraction ---
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    bnds = [("outer_boundary", 1), ("rod_surfaces", 1)]
    bnds_tags = []
    for name, dim in bnds:
        try:
            tag = next(g[1] for g in gmsh.model.getPhysicalGroups(dim)
                       if gmsh.model.getPhysicalName(dim, g[1]) == name)
            bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])
        except StopIteration:
            bnds_tags.append(np.array([], dtype=int))

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags




def mesh5(m=3, n=4, order=1, pitch=18.7e-3, R_rod=6.15e-3, R_cooling=2.0e-3,
          gap_assembly=12e-3, gap_outer=75e-3, R_core=None, smin=1.5e-3,
          add_cooling_rods=True):

    import sys, math
    assert n in (1, 4, 9), "n doit être 1, 4 ou 9"

    # --- Vérifications géométriques ---
    gap_rod = pitch - 2.0 * R_rod
    if gap_rod <= 0.0:
        raise ValueError(
            "Les crayons se chevauchent : pitch=%.2f mm, 2*R_rod=%.2f mm"
            % (pitch*1e3, 2*R_rod*1e3))
    if gap_rod < 3 * smin:
        print("AVERTISSEMENT : gap inter-crayons = %.2f mm (~%.1f elements)."
              " Recommande smin <= %.2f mm."
              % (gap_rod*1e3, gap_rod/smin, gap_rod/3*1e3), file=sys.stderr)
    if gap_assembly <= 0.0:
        raise ValueError("gap_assembly doit etre > 0")

    # --- Centres des assemblages ---
    L_asm = m * pitch
    
    L_cluster = int(math.sqrt(n)) * L_asm + (int(math.sqrt(n)) - 1) * gap_assembly
    gap_outer = L_cluster / 3
    
    if n == 1:
        centers = [(0.0, 0.0)]
    elif n == 4:
        s = L_asm / 2.0 + gap_assembly / 2.0
        centers = [(sx*s, sy*s) for sx in (-1., 1.) for sy in (-1., 1.)]
    else:  # n == 9
        d = L_asm + gap_assembly
        centers = [(i*d, j*d) for i in (-1, 0, 1) for j in (-1, 0, 1)]

    # --- Positions de tous les crayons ---
    offsets = [(k - (m-1)/2.0)*pitch for k in range(m)]
    all_rods = [(cx+dx, cy+dy)
                for (cx, cy) in centers
                for dx in offsets for dy in offsets]

    # --- Rayon du disque ---
    R_disk = max(math.hypot(cx + sx*L_asm/2., cy + sy*L_asm/2.)
                 for (cx, cy) in centers
                 for sx in (-1., 1.) for sy in (-1., 1.)) + gap_outer
    
    smax = R_disk / 6   # remplace le paramètre smax passé en argument
    
    # --- Positions des 4 barres de refroidissement ---
    # On les place dans l'eau périphérique pour éviter les chevauchements
    r_cool = R_disk - gap_outer / 2.0
    
    # --- Filtrage crayons hors R_core ---
    R_core_eff = min(R_core, R_disk - R_rod) if R_core is not None else (R_disk - R_rod)
    kept_rods  = [(rx, ry) for (rx, ry) in all_rods
                  if math.hypot(rx, ry) <= R_core_eff]
    n_removed  = len(all_rods) - len(kept_rods)
    if n_removed:
        print("Info : %d crayons supprimes (R_core=%.1f mm)"
              % (n_removed, R_core_eff*1e3))

    # --- Géométrie ---
    outer_circle = gmsh.model.occ.addCircle(0, 0, 0, R_disk)
    outer_loop   = gmsh.model.occ.addCurveLoop([outer_circle])

    rod_circles, rod_loops = [], []
    for (rx, ry) in kept_rods:
        c  = gmsh.model.occ.addCircle(rx, ry, 0, R_rod)
        lp = gmsh.model.occ.addCurveLoop([c])
        rod_circles.append(c)
        rod_loops.append(lp)
        
    cool_circles, cool_loops = [], []
    cooling_positions = []

    if add_cooling_rods:
        # pour chaque assemblage, on place une barre au centre de chaque carré
        # formé par 4 crayons voisins — un assemblage m×m a (m-1)² tels carrés
        for (cx, cy) in centers:
            for i in range(m - 1):
                for j in range(m - 1):
                    bx = cx + (i - (m - 2) / 2.0) * pitch
                    by = cy + (j - (m - 2) / 2.0) * pitch
                    c  = gmsh.model.occ.addCircle(bx, by, 0, R_cooling)
                    lp = gmsh.model.occ.addCurveLoop([c])
                    cool_circles.append(c)
                    cool_loops.append(lp)
                    cooling_positions.append((bx, by))

    # --- Ajout des trous de refroidissement dans la surface d'eau ---
    surface = gmsh.model.occ.addPlaneSurface([outer_loop] + rod_loops + cool_loops)
    gmsh.model.occ.synchronize()

    # --- Groupes physiques ---
    gmsh.model.addPhysicalGroup(2, [surface],       tag=1,  name="water")
    gmsh.model.addPhysicalGroup(1, [outer_circle],  tag=10, name="outer_boundary")
    if rod_circles:
        gmsh.model.addPhysicalGroup(1, rod_circles, tag=20, name="rod_surfaces")
    # --- Création du groupe physique pour appliquer la condition de Neumann dynamique ---
    if cool_circles:
        gmsh.model.addPhysicalGroup(1, cool_circles, tag=30, name="cooling_surfaces")

    # --- Champ de taille (Distance depuis crayons + Threshold) ---
    dist_field = gmsh.model.mesh.field.add("Distance")
    # --- On affine le maillage aussi autour des barres de refroidissement ---
    gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", rod_circles + cool_circles)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField",  dist_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin",  smin)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax",  smax)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin",  0.0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax",   L_asm/1)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # --- Génération ---
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # --- Extraction (format identique à mesh1/mesh2) ---
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    # --- Ajout du tag 'cooling_surfaces' pour que le main() puisse le récupérer ---
    bnds = [("outer_boundary", 1), ("rod_surfaces", 1), ("cooling_surfaces", 1)]
    bnds_tags = []
    for name, dim in bnds:
        tag = next(g[1] for g in gmsh.model.getPhysicalGroups(dim)
                   if gmsh.model.getPhysicalName(dim, g[1]) == name)
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, cooling_positions
########################################################
########################################################


def build_2d_mesh(geo_filename, mesh_size, order=1):
    """
    Load a .geo file and generate a 2D mesh with uniform element size.

    Parameters
    ----------
    geo_filename : str
        Path to the .geo file
    mesh_size : float
        Target mesh size (uniform)
    order : int
        Polynomial order of elements

    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags
    """

    import gmsh

    # --- load geometry
    gmsh.open(geo_filename)

    # --- FORCE uniform mesh size everywhere
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    # prevent boundary propagation (VERY important)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # disable curvature & point based sizing
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # --- generate 2D mesh
    gmsh.model.mesh.generate(2)

    # --- high order
    gmsh.model.mesh.setOrder(order)

    # --- element type (triangles)
    elemType = gmsh.model.mesh.getElementType("triangle", order)

    # --- nodes
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()

    # --- elements
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags



def get_boundary_segments(physical_tag, order):
    """
    Récupère les segments 1D du groupe physique donné et prépare
    leur quadrature. Retourne tout ce qu'il faut pour assembler
    un terme de Neumann sur ce bord.
    """
    entities = gmsh.model.getEntitiesForPhysicalGroup(1, physical_tag)

    elemTags_list  = []
    nodeTags_list  = []
    line_elemType  = None

    for ent in entities:
        etypes, etags, enodes = gmsh.model.mesh.getElements(dim=1, tag=int(ent))
        if len(etypes) == 0:
            continue
        line_elemType = int(etypes[0])
        elemTags_list.append(etags[0])
        nodeTags_list.append(enodes[0])

    elemTags_1d = np.concatenate(elemTags_list).astype(np.int64)
    nodeTags_1d = np.concatenate(nodeTags_list).astype(np.int64)

    # Quadrature sur les segments (même mécanique que pour les triangles)
    xi_1d, w_1d, N_1d, gN_1d = prepare_quadrature_and_basis(line_elemType, order)

    # Jacobiens uniquement sur les entités du groupe physique concerné
    jac_list, det_list, xphys_list = [], [], []
    for ent in entities:
        j, d, x = gmsh.model.mesh.getJacobians(
            line_elemType, xi_1d.flatten(), tag=int(ent))
        jac_list.append(j)
        det_list.append(d)
        xphys_list.append(x)

    return (line_elemType, elemTags_1d, nodeTags_1d,
            np.concatenate(jac_list), np.concatenate(det_list),
            np.concatenate(xphys_list), w_1d, N_1d, gN_1d)
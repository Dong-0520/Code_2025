"""
### ElementType.getFacePermutationMatrixLGL

Returns a permutation matrix that permutes SummationByParts.jl LGL face nodes
    so that they are monotonically increasing wrt their local ID.

"""
function getFacePermutationMatrixLGL(n_nodes::Integer)::Matrix{Int64}
    perm = zeros(Int64, n_nodes, n_nodes)
    mid = div(n_nodes, 2)
    if n_nodes % 2 == 1
        for i in 1:(mid + 1)
            perm[i, 2 * i - 1] = 1
        end
        for i in (mid + 2):n_nodes
            perm[i, n_nodes - 1 - 2 * (i - (mid + 2))] = 1
        end
    else
        for i in 1:mid
            perm[i, 2 * i - 1] = 1
        end
        for i in (mid + 1):n_nodes
            perm[i, n_nodes - 2 * (i - (mid + 1))] = 1
        end
    end
    return perm
end

"""
### ElementType.getFacePermutationMatrixLG

Returns a permutation matrix that permutes SummationByParts.jl LG face nodes
    so that they are monotonically increasing wrt their local ID.

NOTE: SummationByParts.jl uses opposite ordering of LGL face nodes for LG face nodes.

"""
function getFacePermutationMatrixLG(n_nodes::Integer)
    return reverse(getFacePermutationMatrixLGL(n_nodes), dims = 1)::Matrix{Int64}
end

"""
### ElementType.getFacePermutationMatrixGamma

Returns a permutation matrix that permutes SummationByParts.jl face nodes for
    TriGamma operators so that they are monotonically increasing wrt their local ID.

"""
function getFacePermutationMatrixGamma(n_nodes::Integer)::Matrix{Int64}
    if n_nodes == 2
        P = zeros(Int64, 2, 2)
        P[1, 1] = 1
        P[2, 2] = 1
        return P
    elseif n_nodes == 3
        P = zeros(Int64, 3, 3)
        P[1, 1] = 1
        P[2, 3] = 1
        P[3, 2] = 1
        return P
    elseif n_nodes == 4
        P = zeros(Int64, 4, 4)
        P[1, 1] = 1
        P[2, 4] = 1
        P[3, 3] = 1
        P[4, 2] = 1
        return P
    elseif n_nodes == 5
        P = zeros(Int64, 5, 5)
        P[1, 1] = 1
        P[2, 5] = 1
        P[3, 3] = 1
        P[4, 4] = 1
        P[5, 2] = 1
        return P
    else
        error("getFacePermutationMatrixGamma does not support n_nodes=$n_nodes")
    end
end

"""
### ElementType.meshgrid

Function that is identical to numpy's meshgrid.

"""
function meshgrid(x, y)
    lenx = length(x)
    leny = length(y)
    outx = zeros(leny, lenx)
    outy = zeros(leny, lenx)
    for i in 1:leny
        for j in 1:lenx
            outx[i, j] = x[j]
            outy[i, j] = y[i]
        end
    end
    return outx, outy
end

function meshgrid(x, y, z)
    lenx = length(x)
    leny = length(y)
    lenz = length(z)
    outx = zeros(leny, lenx, lenz)
    outy = zeros(leny, lenx, lenz)
    outz = zeros(leny, lenx, lenz)
    for i in 1:leny
        for j in 1:lenx
            for k in 1:lenz
                outx[i, j, k] = y[i]
                outy[i, j, k] = z[k]
                outz[i, j, k] = x[j]
            end
        end
    end
    return outx, outy, outz
end

"""
### ElementType.getSbpOperators1d

Returns SBP operators for the domain [-1, 1].

"""
function getSbpOperators1d(degree::Int; boundary_nodes::Bool = true)
    # Using SummationByParts.jl to create SBP operators
    operConstructor = boundary_nodes ? getLineSegSBPLobbato : getLineSegSBPLegendre
    oper = operConstructor(degree = degree)
    Q = oper.Q[:, :, 1]
    E = oper.E[:, :, 1]
    w = oper.w

    # Get nodes
    r = SymCubatures.calcnodes(oper.cub, oper.vtx)
    numnodes = length(r)

    # Get boundary projection operators
    if boundary_nodes
        # "Projection" operators just extract first and last nodes to get face nodes
        tL = zeros(Int64, 1, degree + 1)
        tR = zeros(Int64, 1, degree + 1)
        tL[1, 1] = 1
        tR[1, numnodes] = 1
        R = zeros(Int64, 1, 2) # (num nodes per face, num faces)
        R[1, 1] = 1
        R[1, 2] = numnodes
    else
        face = getLineSegFace(degree, oper.cub, oper.vtx)
        tL = zeros(1, degree + 1)
        tR = zeros(1, degree + 1)
        tL[1, :] = face.interp[face.perm[:, 1], 1]'
        tR[1, :] = face.interp[face.perm[:, 2], 1]'
    end

    # Permute operators so that nodes are monotonically increasing between -1 and 1
    P = boundary_nodes ? getFacePermutationMatrixLGL(numnodes) :
        getFacePermutationMatrixLG(numnodes)
    Q = P * Q * P'
    E = P * E * P'
    w = P * w
    r = r * P'
    if !boundary_nodes
        tL = tL * P'
        tR = tR * P'
        R = (tL, tR)
    end

    @assert E ≈ tR' * tR - tL' * tL

    return w, Q, R, r
end

"""
### ElementType.getSbpOperatorsTriDiagE

Returns SBP operators for the domain ConvexHull{(-1, -1), (1, -1), (-1, 1)},
    where the orientation is counterclockwise and faces have the numbering
    face 1 = south, face 2 = diagonal, face 3 = west.

This function is just a wrapper around SummationByParts.getTriSBPDiagE,
    but extends the latter by also returning projection/extraction matrices
    for face nodes and ordering face nodes in a better way wrt face vertices.

Quadrature degree must be twice degree or twice degree minus one.

"""
function getSbpOperatorsTriDiagE(degree::Integer, quadrature_degree::Integer;
                                 vertices::Bool = true)
    @assert 1 <= degree <= 10
    @assert quadrature_degree == 2 * degree || quadrature_degree == 2 * degree - 1

    # Get operators via SummationByParts.jl
    oper_tri = getTriSBPDiagE(degree = degree, vertices = vertices,
                              quad_degree = quadrature_degree)
    w = oper_tri.w
    Qr = oper_tri.Q[:, :, 1]
    Qs = oper_tri.Q[:, :, 2]

    # Get nodes
    rs = SymCubatures.calcnodes(oper_tri.cub, oper_tri.vtx)
    rs_quad = deepcopy(rs)

    # Create face to access projection operators and face quadrature rules
    face = TriFace{Float64}(degree, oper_tri.cub, oper_tri.vtx, vertices = vertices)

    # Get matrices holding indices of solution face nodes
    n_faces = 3
    n_nodes = size(rs)[2] # number of volume nodes
    n_face_nodes = size(face.perm)[1] # number of volume nodes on each face

    # Get permutation matrix that permutes SummationByParts.jl face node ordering
    # so that face nodes are ordered monotonically wrt face vertices
    perm = vertices ? getFacePermutationMatrixLGL(n_face_nodes) :
           getFacePermutationMatrixLG(n_face_nodes)

    # Get matrix storing indices of solution nodes on faces
    # R[:, i] contains the indices of the solution nodes on face i
    R = perm * round.(Int64, face.interp') * face.perm # shape = (n_face_nodes, n_faces)
    R_map = Tuple([SelectionMap{Float64}(R[:, i], n_nodes) for i in 1:n_faces]) #TODO: Simplify construction
    rs_face = Matrix(hcat([rs * R_map[i]' for i in 1:n_faces]...))
    f_mask = collect.([i:(i + n_face_nodes - 1) for i in 1:n_face_nodes:(n_face_nodes * n_faces)])
    # Get face quadrature weights. Same ordering as rs_face
    w_face = perm * face.wface

    return w, w_face, Qr, Qs, R_map, rs, rs_quad, rs_face, f_mask
end

"""
### ElementType.getSbpOperatorsTriOmega

Returns SBP operators for the domain ConvexHull{(-1, -1), (1, -1), (-1, 1)},
    where the orientation is counterclockwise and faces have the numbering
    face 1 = south, face 2 = diagonal, face 3 = west.

This function is just a wrapper around SummationByParts.getTriSBPOmega,
    but extends the latter by also returning projection/extraction matrices
    for face nodes and ordering face nodes in a better way wrt face vertices.

"""
function getSbpOperatorsTriOmega(degree::Integer)
    @assert 1 <= degree <= 4

    # Get operators via SummationByParts.jl
    oper_tri = getTriSBPOmega(degree = degree)
    w = oper_tri.w
    Qr = oper_tri.Q[:, :, 1]
    Qs = oper_tri.Q[:, :, 2]

    # Get nodes
    rs = SymCubatures.calcnodes(oper_tri.cub, oper_tri.vtx)
    rs_quad = deepcopy(rs)
    # Create face to access projection operators and face quadrature rules
    face = TriFace{Float64}(degree, oper_tri.cub, oper_tri.vtx, vertices = false)

    # Get matrices holding indices of solution face nodes
    n_faces = 3
    n_nodes = size(rs)[2] # number of volume nodes
    n_face_nodes = size(face.interp)[2] # number of volume nodes on each face

    # Get permutation matrix that permutes SummationByParts.jl face node ordering
    # so that face nodes are ordered monotonically wrt face vertices
    perm = getFacePermutationMatrixLG(n_face_nodes)

    # Get extrapolation matrices that project interior solution nodes to face nodes
    R = Tuple([deepcopy(face.interp)[face.perm[:, i], :] * perm' for i in [1, 3, 2]])
    rs_face = hcat([rs * R[i] for i in 1:n_faces]...)
    f_mask = collect.([i:(i + n_face_nodes - 1) for i in 1:n_face_nodes:(n_face_nodes * n_faces)])
    # Get face quadrature weights. Same ordeering as rs_face
    w_face = perm * face.wface

    return w, w_face, Qr, Qs, R, rs, rs_quad, rs_face, f_mask
end

"""
### ElementType.getSbpOperatorsTriGamma

Wrapper for SummationByParts.getTriSBPGamma that, in addition to returning SBP operators,
    returns solution nodes, face interpolation operators, and face quadrature weights.

"""
function getSbpOperatorsTriGamma(degree::Integer)
    @assert 1 <= degree <= 4

    # Get operators via SummationByParts.jl
    oper_tri = getTriSBPGamma(degree = degree)
    w = oper_tri.w
    Qr = oper_tri.Q[:, :, 1]
    Qs = oper_tri.Q[:, :, 2]

    # Get nodes
    rs = SymCubatures.calcnodes(oper_tri.cub, oper_tri.vtx)
    rs_quad = deepcopy(rs)

    # Create face to access projection operators and face quadrature rules
    face = TriFace{Float64}(degree, oper_tri.cub, oper_tri.vtx, vertices = false)

    # Get matrices holding indices of solution face nodes
    n_faces = 3
    n_nodes = size(rs)[2] # number of volume nodes
    n_face_nodes = size(face.perm)[1] # number of volume nodes on each face

    # Get permutation matrices that permutes SummationByParts.jl face node ordering
    # so that face nodes are ordered monotonically wrt face vertices
    perm_gamma = getFacePermutationMatrixGamma(n_face_nodes)
    perm_lg = getFacePermutationMatrixLG(n_face_nodes)

    # n_face_nodes-by-n_faces matrix containing indices of face nodes
    R_index = perm_gamma * face.perm

    # The 1d nodal distribution for TriSBPGamma operators is neither LG nor LGL,
    # so a second n_face_nodes-by-n_face_nodes matrix is needed that interpolates
    # from the gamma face nodes to LG nodes.
    R_interp = perm_gamma' * face.interp * perm_lg'

    # Save extrapolation matrices as a tuple of face extraction/interpolation matrices.
    R = (R_index, R_interp)
    rs_face = hcat([rs[:, R_index[:, i]] * R_interp for i in 1:n_faces]...)
    f_mask = collect.([i:(i + n_face_nodes - 1) for i in 1:n_face_nodes:(n_face_nodes * n_faces)])

    # Save face quadrature nodes (in same order as the face nodes)
    w_face = perm_lg * face.wface
    return w, w_face, Qr, Qs, R, rs, rs_quad, rs_face, f_mask
end

"""
### ElementType.getSbpOperatorsQuad2d

Returns SBP operators for a tensor product element on the domain
    ConvexHull{(-1, -1), (1, -1), (-1, 1), (1, 1)}, where
    the orientation is counterclockwise and faces have the numbering
    face 1 = south, face 2 = east, face 3 = north, face 4 = west.

"""
function getSbpOperatorsQuad2d(degree::Int; boundary_nodes::Bool = true)
    # Get 1d operators
    w, Q, R, x = getSbpOperators1d(degree, boundary_nodes = boundary_nodes)
    numnodes = length(x)

    # Nodes for quad element are given by cartesian product of 1d nodes
    xx, yy = meshgrid(x, x)
    rs = vcat(reshape(xx', 1, numnodes^2), reshape(yy', 1, numnodes^2))
    rs_quad = deepcopy(rs)

    numfaces = 4
    if boundary_nodes
        R = zeros(Int64, numnodes, numfaces)
        R[:, 1] = [i for i in 1:numnodes]
        R[:, 2] = [i * numnodes for i in 1:numnodes]
        R[:, 3] = reverse([i + numnodes * (numnodes - 1) for i in 1:numnodes])
        R[:, 4] = reverse([(i - 1) * numnodes + 1 for i in 1:numnodes])
    else
        # Project operators to faces
        R1 = kron(tL, I(numnodes))
        R2 = kron(I(numnodes), tR)
        R3 = reverse(kron(tR, I(numnodes)), dims = 1)
        R4 = reverse(kron(I(numnodes), tL), dims = 1)
        R = (R1, R2, R3, R4)
    end
    rs_face = Tuple([rs[:, R[:, i]] for i in 1:numfaces])

    return w, Q, R, rs, rs_quad, rs_face
end

function getSBPOperatorsTetDiagE(degree::Int, quad_degree::Int)
    @assert 1 <= degree <= 5

    oper_tet = getTetSBPDiagE(degree = degree, cubdegree = quad_degree)
    w = oper_tet.w
    Qr = oper_tet.Q[:, :, 1]
    Qs = oper_tet.Q[:, :, 2]
    Qt = oper_tet.Q[:, :, 3]

    rst = SymCubatures.calcnodes(oper_tet.cub, oper_tet.vtx)
    rst_quad = deepcopy(rst)
    Rs = SummationByParts.getfaceextrapolation(degree, quad_degree, 3, opertype = :DiagE,
                                               faceopertype = :DiagE)
    n_face_nodes = size(Rs[:, :, 1], 1)
    n_faces = 4
    R = Tuple([Rs[:, :, i] for i in 1:n_faces])
    rst_face = hcat([rst * R[i]' for i in 1:n_faces]...)
    f_mask = collect.([i:(i + n_face_nodes - 1) for i in 1:n_face_nodes:(n_face_nodes * n_faces)])

    face_cub, _ = getTriCubatureForTetFaceDiagE(2 * degree, Float64, faceopertype = :DiagE)
    w_face = SymCubatures.calcweights(face_cub)

    return w, w_face, Qr, Qs, Qt, R, rst, rst_quad, rst_face, f_mask
end

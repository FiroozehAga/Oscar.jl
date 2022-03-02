###
# Tropical curves in Oscar
# ========================
###



###
# 1. Definition
# -------------
# M = typeof(min) or typeof(max):
#   min or max convention, affecting initial ideals
# EMB = true or false:
#   embedded or abstract tropical curves
#   embedded tropical variety = graph embedded in euclidean space with weighted edges and vertices
#   abstract tropical variety = graph with enumerated vertices with weighted edges and vertices
###

@attributes mutable struct TropicalCurve{M,EMB} <: TropicalVarietySupertype{M,EMB}
    polyhedralComplex::PolyhedralComplex
    function TropicalCurve{M,EMB}(Sigma::PolyhedralComplex) where {M,EMB}
        if EMB
            if dim(Sigma)!=1
                error("TropicalCurve: input polyhedral complex not one-dimensional")
            end
        end
        return new{M,EMB}(Sigma)
    end
end
export TropicalCurve

function pm_object(T::TropicalCurve)
    if has_attribute(T,:polymake_bigobject)
        return get_attribute(T,:polymake_bigobject)
    end
    error("pm_object(T::TropicalCurve): no polymake bigobject attributed")
end


###
# 2. Basic constructors
# ---------------------
###

@doc Markdown.doc"""
    TropicalCurve{M, EMB}()

Construct a tropical curve from a list of edges and a vector of their lengths.
If the curve is embedded, vertices must be points in $\mathbb R^n$.
If the curve is abstract, vertices must be 1, ..., n.

# Examples
```jldoctest
julia> IM = IncidenceMatrix([[1,2],[1,3],[1,4]])
3×4 IncidenceMatrix
[1, 2]
[1, 3]
[1, 4]


julia>  VR = [0 0; 1 0; -1 0; 0 1]
4×2 Matrix{Int64}:
  0  0
  1  0
 -1  0
  0  1

julia> PC = PolyhedralComplex(IM, vr)
A polyhedral complex in ambient dimension 2

julia> TC = TropicalCurve{min}(PC)
A tropical curve in 2-dimensional Euclidean space

julia> abs_TC = TropicalCurve{min}(IM)
An abstract tropical curve
```
"""
function TropicalCurve{M}(PC::PolyhedralComplex) where {M}
   @assert dim(PC)==1 "The polyhedral complex is not of dimenion 1."
   return TropicalCurve{M, true}(PC)
end


function TropicalCurve{M}(graph::IncidenceMatrix) where {M}
    # Columns correspond to nodes
    # Rows correpons to edges
    empty = PolyhedralComplex(Polymake.fan.PolyhedralComplex())
    result = TropicalCurve{M, false}(empty)
    set_attribute!(result, :graph, graph)
    return result
end

function graph(tc::TropicalCurve{M, EMB}) where {M, EMB}
    if !has_attribute(tc, :graph)
        throw(ArgumentError("No graph attached"))
    end
    return get_attribute(tc, :graph)
end

function n_nodes(tc::TropicalCurve{M, EMB}) where {M, EMB}
    G = graph(tc)
    return Polymake.ncols(G)
end


function Base.show(io::IO, tc::TropicalCurve{M, EMB}) where {M, EMB}
    if EMB
        print(io, "A tropical curve in $(ambient_dim(tc))-dimensional Euclidean space")
    else
        print(io, "An abstract tropical curve")
    end
end


struct DivisorOnTropicalCurve{M, EMB}
    base_curve::TropicalCurve{M, EMB}
    coefficients::Vector{Int}
    function DivisorOnTropicalCurve(tc::TropicalCurve{M, EMB}, coeffs::Vector{Int}) where {M,EMB}
        if EMB
            error("Not implemented yet")
        else
            if n_nodes(tc) != length(coeffs)
                throw(ArgumentError("Wrong number coefficients"))
            end
            return new{M, EMB}(tc, coeffs)
        end
    end
end

base_curve(dtc::DivisorOnTropicalCurve{M, EMB}) where {M, EMB} = dtc.base_curve

###
# 3.Basic properties
#
coefficients(dtc::DivisorOnTropicalCurve{M, EMB}) where {M, EMB} = dtc.coefficients

degree(dtc::DivisorOnTropicalCurve{M, EMB}) where {M, EMB} = sum(coefficients(dtc))

is_effective(dtc::DivisorOnTropicalCurve{M, EMB}) where {M, EMB} = all(e -> e>=0, coefficients(dtc))

function chip_firing_move(dtc::DivisorOnTropicalCurve{M, EMB}, position::Int) where {M, EMB}
    G = graph(base_curve(dtc))
    newcoeffs = Vector{Int}(coefficients(dtc))
    for i in 1:Polymake.nrows(G)
        row = Polymake.row(G, i)
        if position in row
            newcoeffs[position] -= 1
            for i in row
                if i != position
                    newcoeffs[i] += 1
                end
            end
        end
    end
    return DivisorOnTropicalCurve(base_curve(dtc), newcoeffs)
end


export DivisorOnTropicalCurve,
    n_nodes,
    degree,
    coefficients,
    is_effective,
    graph,
    base_curve,
    chip_firing_move




###
# 4. More properties
# -------------------
###
@doc Markdown.doc"""
    ElementaryDivisors(M::MatrixElem{T}) where T <: Int

Calculates the elementary divisors of the matrix M, together with their product. 
This assumes that the matrix is square and of rank n-1. 
"""
function ElementaryDivisors(M::MatrixElem{T}) where T <: fmpz
  @assert ncols(M)==nrows(M) "Not a square matrix."
  C = snf(M)
  c0 = 1
  n = ncols(M)
  Z = Vector{T}(undef,n)
  for i in 1:n
    Z[i] = C[i,i]
    if i <= n-1
      c0 = Z[i]*c0
    end 
  end 
  return Z,c0
end

@doc Markdown.doc"""
    StructureTropicalJacobian(TC::TropicalCurve)

Computes the elementary divisors n_i of the Laplacian matrix of the tropical curve, 
together with their product $N=\prod$ n_i.The tropical Jacobian is then isomorphic to 
$\prod (Z/(n_i)Z)$ and the order of this group is N. 

    
# Examples
```jldoctest
julia> cg = Oscar.Graphs.complete_graph(5);

julia> IM1=IncidenceMatrix([[Oscar.Graphs.src(e), Oscar.Graphs.dst(e)] for e in Oscar.Graphs.edges(cg)])
10×5 IncidenceMatrix
[1, 2]
[1, 3]
[2, 3]
[1, 4]
[2, 4]
⁝

julia> TC1 = TropicalCurve{min}(IM1)
An abstract tropical curve

julia> Oscar.StructureTropicalJacobian(TC1)
(fmpz[1, 5, 5, 5, 0], 125)

julia> cg2 = Oscar.Graphs.complete_graph(3);

julia> IM2=IncidenceMatrix([[Oscar.Graphs.src(e), Oscar.Graphs.dst(e)] for e in Oscar.Graphs.edges(cg2)])
3×3 IncidenceMatrix
[1, 2]
[1, 3]
[2, 3]


julia> TC2 = TropicalCurve{min}(IM2)
An abstract tropical curve

julia> Oscar.StructureTropicalJacobian(TC2)
(fmpz[1, 3, 0], 3)

julia> IM3 = IncidenceMatrix([[1,2],[2,3],[3,4],[4,5],[1,5]])
5×5 IncidenceMatrix
[1, 2]
[2, 3]
[3, 4]
[4, 5]
[1, 5]


julia> TC3=TropicalCurve{min}(IM3)
An abstract tropical curve

julia> Oscar.StructureTropicalJacobian(TC3)
(fmpz[1, 1, 1, 5, 0], 5)
```
"""
function StructureTropicalJacobian(TC::TropicalCurve)
    gg=Graphs.Graph{Graphs.Undirected}(n_nodes(TC))
    IM = graph(TC)
    for i in 1:Polymake.nrows(IM)
        row = Vector{Int}(Polymake.row(IM,i))
        Graphs.add_edge!(gg, row[1],row[2])
    end
    lap = Polymake.graph.laplacian(Oscar.pm_object(gg))
    L = Polymake.@convert_to Matrix{Int} lap
    LL = matrix(ZZ, L)
    ED = ElementaryDivisors(LL)
    return ED
end
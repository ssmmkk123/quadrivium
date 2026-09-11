"""Sparse P1 assembly, residual estimators, and conforming local refinement."""
from __future__ import annotations
from dataclasses import dataclass
import math
import operator
from .. import numeric as np
from ..core.types import PDESolution
from ..linalg.sparse import COOMatrix
from ..linalg.iterative import preconditioned_cg

__all__ = ["TriangularMesh", "assemble_triangular", "fem_error_estimate",
           "refine_triangles", "adaptive_fem", "solve_fem_mesh"]


def _edge(a,b):
    return (min(a,b),max(a,b))


@dataclass
class TriangularMesh:
    """Planar triangular mesh with validated cells and inferred boundary nodes."""
    points: object
    triangles: object
    boundary: object = None
    def __post_init__(self):
        self.points = np.asarray(self.points,float).copy()
        raw = np.asarray(self.triangles)
        self.triangles = np.asarray(raw,int).copy()
        if (self.points.ndim!=2 or self.points.shape[1]!=2 or len(self.points)<3
                or self.triangles.ndim!=2 or self.triangles.shape[1]!=3
                or not np.all(np.isfinite(self.points))):
            raise ValueError("points must be finite (n,2) and triangles (m,3)")
        if not np.array_equal(raw,self.triangles) or np.any(self.triangles<0) or np.any(self.triangles>=len(self.points)):
            raise ValueError("invalid triangle vertex indices")
        edges = {}
        for i,tri in enumerate(self.triangles):
            ids = [int(x) for x in tri]
            a,b,c = [self.points[j] for j in ids]
            cross = float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
            if cross==0 or len(set(ids))!=3:
                raise ValueError("degenerate triangle")
            if cross<0:
                self.triangles[i] = [ids[0],ids[2],ids[1]]
            for j in range(3):
                key = _edge(ids[j],ids[(j+1)%3])
                edges[key] = edges.get(key,0)+1
                if edges[key]>2:
                    raise ValueError("nonmanifold mesh edge")
        if self.boundary is None:
            self.boundary = np.array(sorted({v for e,c in edges.items() if c==1 for v in e}),int)
        else:
            self.boundary = np.asarray(self.boundary,int).copy()
            if self.boundary.ndim!=1 or np.any(self.boundary<0) or np.any(self.boundary>=len(self.points)):
                raise ValueError("invalid boundary node indices")


def _geometry(points,tri):
    a,b,c = [points[int(j)] for j in tri]
    cross = float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
    gradients = np.array([[b[1]-c[1],c[0]-b[0]],
                          [c[1]-a[1],a[0]-c[0]],
                          [a[1]-b[1],b[0]-a[0]]])/cross
    return abs(cross)/2,gradients,(a+b+c)/3


def assemble_triangular(mesh, source=0.0, c_diff=1.0):
    """Assemble the full CSR stiffness and load in O(elements) storage.

    Diffusion is sampled at cell centroids; the load uses a degree-two,
    three-point triangle rule. Duplicate element contributions are summed.
    """
    rows,cols,data = [],[],[]
    b = np.zeros(len(mesh.points))
    barycentric = ((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3))
    for tri in mesh.triangles:
        ids = [int(j) for j in tri]
        area,gradients,center = _geometry(mesh.points,tri)
        coefficient = float(c_diff(*center) if callable(c_diff) else c_diff)
        if not math.isfinite(coefficient) or coefficient<=0:
            raise ValueError("diffusion coefficient must be finite and positive")
        element = coefficient*area*(gradients@gradients.T)
        for i in range(3):
            for j in range(3):
                rows.append(ids[i]); cols.append(ids[j]); data.append(float(element[i,j]))
        for lam in barycentric:
            q = sum(lam[i]*mesh.points[ids[i]] for i in range(3))
            value = float(source(*q) if callable(source) else source)
            if not math.isfinite(value):
                raise ValueError("source must be finite")
            for i in range(3):
                b[ids[i]] += area/3*lam[i]*value
    A = COOMatrix(rows,cols,data,(len(mesh.points),)*2).tocsr().sum_duplicates()
    return A,b


def solve_fem_mesh(mesh, source, bc=0.0, c_diff=1.0, tol=1e-10, max_iter=None):
    """Solve a scalar Dirichlet elliptic problem using sparse assembly and PCG."""
    A,b = assemble_triangular(mesh,source,c_diff)
    boundary = {int(i) for i in mesh.boundary}
    interior = [i for i in range(len(mesh.points)) if i not in boundary]
    mapping = {j:i for i,j in enumerate(interior)}
    u = np.zeros(len(mesh.points))
    for j in boundary:
        u[j] = bc(*mesh.points[j]) if callable(bc) else bc
    rows,cols,data = [],[],[]
    rhs = b[np.array(interior,int)].copy()
    for local,row in enumerate(interior):
        for k in range(int(A.indptr[row]),int(A.indptr[row+1])):
            col,value = int(A.indices[k]),float(A.data[k])
            if col in boundary:
                rhs[local] -= value*u[col]
            else:
                rows.append(local); cols.append(mapping[col]); data.append(value)
    iterations,converged = 0,True
    if interior:
        restricted = COOMatrix(rows,cols,data,(len(interior),)*2).tocsr()
        diagonal = restricted.diagonal()
        result = preconditioned_cg(restricted,rhs,tol=tol,max_iter=max_iter,
                                    M=lambda v:v/diagonal)
        u[np.array(interior,int)] = result.x
        iterations,converged = result.iterations,result.converged
    result = PDESolution(u,(mesh.points[:,0],mesh.points[:,1]),method="sparse_fem_2d",
                         iterations=iterations,converged=converged)
    result.mesh,result.stiffness = mesh,A
    return result


def fem_error_estimate(mesh, values, source, c_diff=1.0):
    """Cellwise residual/normal-flux-jump indicators for conforming linear FEM.

    Returns nonnegative energy-error indicators (one per triangle), including
    coefficient derivatives in the volume residual for callable diffusion.
    These are estimators, not guaranteed upper error bounds.
    """
    values = np.asarray(values,float)
    if values.shape!=(len(mesh.points),):
        raise ValueError("one solution value per mesh point is required")
    eta2 = np.zeros(len(mesh.triangles))
    edges = {}
    for element,tri in enumerate(mesh.triangles):
        ids = [int(j) for j in tri]
        area,gradients,center = _geometry(mesh.points,tri)
        grad_u = values[np.array(ids,int)]@gradients
        coefficient = float(c_diff(*center) if callable(c_diff) else c_diff)
        residual = float(source(*center) if callable(source) else source)
        if callable(c_diff):
            for axis in range(2):
                h = 1e-5*max(1,abs(float(center[axis])))
                plus,minus = center.copy(),center.copy()
                plus[axis] += h; minus[axis] -= h
                residual += (c_diff(*plus)-c_diff(*minus))/(2*h)*grad_u[axis]
        diameter2 = max(float(np.sum((mesh.points[ids[j]]-mesh.points[ids[(j+1)%3]])**2)) for j in range(3))
        eta2[element] = diameter2*area*residual**2/coefficient
        for j in range(3):
            key = _edge(ids[j],ids[(j+1)%3])
            tangent = mesh.points[key[1]]-mesh.points[key[0]]
            length = float(np.linalg.norm(tangent))
            normal = np.array([tangent[1],-tangent[0]])/length
            flux = coefficient*float(grad_u@normal)
            edges.setdefault(key,[]).append((element,flux,length,coefficient))
    for entries in edges.values():
        if len(entries)==2:
            a,b = entries
            jump = (a[1]-b[1])**2*a[2]**2/(2*min(a[3],b[3]))
            eta2[a[0]] += jump
            eta2[b[0]] += jump
    return np.sqrt(eta2)


def refine_triangles(mesh, marked):
    """Conforming local red/green refinement; no hanging nodes are introduced.

    Marked triangles split all three edges. Adjacent triangles receive one- or
    two-edge green subdivisions, using globally shared edge midpoint indices.
    """
    marked = {operator.index(i) for i in marked}
    if any(i<0 or i>=len(mesh.triangles) for i in marked):
        raise ValueError("marked element index out of range")
    points = mesh.points.tolist()
    midpoint = {}
    for i in sorted(marked):
        ids = [int(j) for j in mesh.triangles[i]]
        for j in range(3):
            key = _edge(ids[j],ids[(j+1)%3])
            if key not in midpoint:
                midpoint[key] = len(points)
                points.append(((mesh.points[key[0]]+mesh.points[key[1]])/2).tolist())
    triangles = []
    for tri in mesh.triangles:
        a,b,c = map(int,tri)
        mids = [midpoint.get(_edge(a,b)),midpoint.get(_edge(b,c)),midpoint.get(_edge(c,a))]
        count = sum(m is not None for m in mids)
        if count==0:
            triangles.append([a,b,c])
        elif count==3:
            ab,bc,ca = mids
            triangles.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
        elif count==1:
            vertices = [a,b,c]
            j = next(j for j,m in enumerate(mids) if m is not None)
            a,b,c = vertices[j],vertices[(j+1)%3],vertices[(j+2)%3]
            m = mids[j]
            triangles.extend([[a,m,c],[m,b,c]])
        else:
            # Rotate so the two split edges are a-b and b-c.
            vertices = [a,b,c]
            j = next(j for j in range(3) if mids[j] is not None and mids[(j+1)%3] is not None)
            a,b,c = vertices[j],vertices[(j+1)%3],vertices[(j+2)%3]
            ab,bc = mids[j],mids[(j+1)%3]
            triangles.extend([[b,bc,ab],[a,ab,c],[ab,bc,c]])
    boundary = {int(j) for j in mesh.boundary}
    incidence = {}
    for tri in mesh.triangles:
        vertices = list(map(int,tri))
        for j in range(3):
            edge = _edge(vertices[j],vertices[(j+1)%3])
            incidence[edge] = incidence.get(edge,0)+1
    refined_boundary = boundary | {mid for edge,mid in midpoint.items()
                                   if incidence[edge]==1 and all(v in boundary for v in edge)}
    return TriangularMesh(points,triangles,sorted(refined_boundary))


def adaptive_fem(source, mesh=None, *, bc=0.0, c_diff=1.0, tol=1e-3,
                 marking_fraction=0.5, max_refinements=8, max_elements=100000,
                 linear_tol=1e-10, callback=None):
    """Solve-estimate-mark-refine loop with Dörfler bulk marking and a mesh cap."""
    if not 0<marking_fraction<=1 or tol<=0 or max_refinements<0 or max_elements<1:
        raise ValueError("invalid adaptive FEM controls")
    if mesh is None:
        from .fem import unit_square_mesh
        mesh = TriangularMesh(*unit_square_mesh(2))
    history = []
    for level in range(max_refinements+1):
        result = solve_fem_mesh(mesh,source,bc,c_diff,tol=linear_tol)
        indicators = fem_error_estimate(mesh,result.u,source,c_diff)
        error = float(np.linalg.norm(indicators))
        history.append((len(mesh.triangles),error))
        result.error_estimate,result.element_errors = error,indicators
        result.refinement_history = history.copy()
        result.converged = result.converged and error<=tol
        if callback is not None and callback(mesh,result) is True:
            return result
        if error<=tol or level==max_refinements:
            return result
        order = sorted(range(len(indicators)),key=lambda j:float(indicators[j]),reverse=True)
        marked,total = [],0.0
        for j in order:
            marked.append(j)
            total += float(indicators[j])**2
            if total>=marking_fraction*error**2:
                break
        refined = refine_triangles(mesh,marked)
        if len(refined.triangles)>max_elements:
            return result
        mesh = refined
    return result

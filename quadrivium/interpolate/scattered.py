"""Automatic planar triangulation and memory-bounded local RBF interpolation."""
from __future__ import annotations
from collections import OrderedDict
import heapq
import math
import operator
from .. import numeric as np

__all__ = ["KDTree", "Delaunay", "LinearNDInterpolator", "RBFInterpolator",
           "scattered_interpolator"]


class KDTree:
    """Balanced median-split k-d tree with exact Euclidean nearest-neighbor queries."""
    def __init__(self, points):
        self.points = np.asarray(points,float).copy()
        if (self.points.ndim!=2 or not len(self.points) or self.points.shape[1]==0
                or not np.all(np.isfinite(self.points))):
            raise ValueError("points must be a finite nonempty (n,dimension) array")
        self.dimension = self.points.shape[1]
        coords = self.points.tolist()
        def build(indices,depth):
            if not indices:
                return None
            axis = depth%self.dimension
            indices.sort(key=lambda i:(coords[i][axis],i))
            mid = len(indices)//2
            return (indices[mid],axis,build(indices[:mid],depth+1),build(indices[mid+1:],depth+1))
        self._coords = coords
        self._root = build(list(range(len(coords))),0)

    def query(self, points, k=1):
        k = operator.index(k)
        if not 1<=k<=len(self.points):
            raise ValueError("k must be between 1 and the number of points")
        q = np.asarray(points,float)
        scalar = q.ndim==1
        q = np.atleast_2d(q)
        if q.shape[1]!=self.dimension or not np.all(np.isfinite(q)):
            raise ValueError("query must have the tree dimension and finite coordinates")
        if len(q)==0:
            shape = (0,) if k==1 else (0,k)
            return np.empty(shape),np.empty(shape,dtype=int)
        all_dist,all_indices = [],[]
        for query in q.tolist():
            heap = []
            def visit(node):
                if node is None:
                    return
                index,axis,left,right = node
                point = self._coords[index]
                distance = sum((a-b)**2 for a,b in zip(point,query))
                entry = (-distance,-index)
                if len(heap)<k:
                    heapq.heappush(heap,entry)
                elif entry>heap[0]:
                    heapq.heapreplace(heap,entry)
                delta = query[axis]-point[axis]
                near,far = (left,right) if delta<=0 else (right,left)
                visit(near)
                if len(heap)<k or delta*delta<=-heap[0][0]:
                    visit(far)
            visit(self._root)
            ordered = sorted((-d,-i) for d,i in heap)
            all_dist.append([math.sqrt(d) for d,_ in ordered])
            all_indices.append([i for _,i in ordered])
        distances,indices = np.array(all_dist),np.array(all_indices,int)
        if k==1:
            distances,indices = distances[:,0],indices[:,0]
        return (distances[0],indices[0]) if scalar else (distances,indices)


def _orientation(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _incircle(a,b,c,p):
    ax,ay = a[0]-p[0],a[1]-p[1]
    bx,by = b[0]-p[0],b[1]-p[1]
    cx,cy = c[0]-p[0],c[1]-p[1]
    return ((ax*ax+ay*ay)*(bx*cy-by*cx)
            -(bx*bx+by*by)*(ax*cy-ay*cx)
            +(cx*cx+cy*cy)*(ax*by-ay*bx))


class Delaunay:
    """Incremental 2-D Delaunay triangulation with bounding-box point location.

    Duplicate/collinear point sets are rejected. Coordinates are normalized
    before geometric predicates. Near-degenerate configurations still use
    floating-point predicates; this is not an exact computational geometry API.
    Construction uses O(n²) work in the worst case and O(n) mesh storage.
    """
    def __init__(self, points):
        self.points = np.asarray(points,float).copy()
        if self.points.ndim!=2 or self.points.shape[1]!=2 or len(self.points)<3 or not np.all(np.isfinite(self.points)):
            raise ValueError("at least three finite planar points are required")
        original = self.points.tolist()
        if len({tuple(p) for p in original})!=len(original):
            raise ValueError("duplicate points are not permitted")
        low = [min(p[j] for p in original) for j in range(2)]
        high = [max(p[j] for p in original) for j in range(2)]
        scale = max(high[j]-low[j] for j in range(2))
        if not math.isfinite(scale) or scale==0:
            raise ValueError("invalid point coordinate range")
        self._origin,self._scale = low,scale
        points = [[(p[j]-low[j])/scale for j in range(2)] for p in original]
        n = len(points)
        points.extend([[-32,-16],[32,-16],[0,32]])
        triangles = [(n,n+1,n+2)]
        for index in range(n):
            bad = [tri for tri in triangles if _incircle(*(points[j] for j in tri),points[index])>0]
            edges = {}
            for a,b,c in bad:
                for u,v in ((a,b),(b,c),(c,a)):
                    key = (min(u,v),max(u,v))
                    if key in edges:
                        del edges[key]
                    else:
                        edges[key] = (u,v)
            removed = set(bad)
            triangles = [tri for tri in triangles if tri not in removed]
            for a,b in edges.values():
                cross = _orientation(points[a],points[b],points[index])
                if cross>0:
                    triangles.append((a,b,index))
                elif cross<0:
                    triangles.append((b,a,index))
        triangles = [tri for tri in triangles if max(tri)<n]
        if not triangles:
            raise ValueError("points are collinear or numerically degenerate")
        self.simplices = np.array(triangles,int)
        self.triangles = self.simplices
        self._coords = points[:n]
        boxes = []
        for tri in triangles:
            vertices = [points[i] for i in tri]
            boxes.append((min(p[0] for p in vertices),max(p[0] for p in vertices),
                          min(p[1] for p in vertices),max(p[1] for p in vertices)))
        def build(indices):
            box = (min(boxes[i][0] for i in indices),max(boxes[i][1] for i in indices),
                   min(boxes[i][2] for i in indices),max(boxes[i][3] for i in indices))
            if len(indices)<=8:
                return box,indices,None,None
            axis = 0 if box[1]-box[0]>=box[3]-box[2] else 2
            indices.sort(key=lambda i:boxes[i][axis]+boxes[i][axis+1])
            mid = len(indices)//2
            return box,None,build(indices[:mid]),build(indices[mid:])
        self._root = build(list(range(len(triangles))))

    def _locate(self, query, tol):
        point = [(float(query[j])-self._origin[j])/self._scale for j in range(2)]
        stack = [self._root]
        while stack:
            box,indices,left,right = stack.pop()
            if not(box[0]-tol<=point[0]<=box[1]+tol and box[2]-tol<=point[1]<=box[3]+tol):
                continue
            if indices is None:
                stack.extend((right,left))
                continue
            for index in indices:
                a,b,c = [self._coords[int(j)] for j in self.simplices[index]]
                area = _orientation(a,b,c)
                lam = [_orientation(point,b,c)/area,_orientation(a,point,c)/area,
                       _orientation(a,b,point)/area]
                if min(lam)>=-tol:
                    return index,lam
        return -1,None

    def find_simplex(self, points, tol=1e-12):
        q = np.asarray(points,float)
        scalar = q.ndim==1
        q = np.atleast_2d(q)
        if q.shape[1]!=2 or not np.all(np.isfinite(q)) or tol<0:
            raise ValueError("finite planar queries and nonnegative tolerance required")
        result = np.array([self._locate(point,tol)[0] for point in q],int)
        return int(result[0]) if scalar else result


class LinearNDInterpolator:
    """Piecewise-linear scattered 2-D interpolation with explicit hull behavior.

    ``outside`` selects ``'fill'`` (default NaN), ``'nearest'``, or ``'raise'``.
    Reuse an existing Delaunay triangulation by supplying it as ``points``.
    Values may have trailing component dimensions.
    """
    def __init__(self, points, values, *, outside="fill", fill_value=np.nan):
        self.triangulation = points if isinstance(points,Delaunay) else Delaunay(points)
        self.values = np.asarray(values,float).copy()
        if self.values.ndim<1 or len(self.values)!=len(self.triangulation.points):
            raise ValueError("one value per input point is required")
        if outside not in {"fill","nearest","raise"}:
            raise ValueError("outside must be 'fill', 'nearest', or 'raise'")
        self.outside,self.fill_value = outside,float(fill_value)
        self.tree = KDTree(self.triangulation.points) if outside=="nearest" else None
    def __call__(self, points):
        q = np.asarray(points,float)
        scalar = q.ndim==1
        q = np.atleast_2d(q)
        if q.shape[1]!=2 or not np.all(np.isfinite(q)):
            raise ValueError("finite planar queries required")
        out = np.empty((len(q),)+self.values.shape[1:])
        for i,query in enumerate(q):
            index,lam = self.triangulation._locate(query,1e-12)
            if index<0:
                if self.outside=="raise":
                    raise ValueError("query lies outside the convex hull")
                out[i] = self.values[int(self.tree.query(query)[1])] if self.tree is not None else self.fill_value
            else:
                vertices = self.triangulation.simplices[index]
                out[i] = sum(lam[j]*self.values[int(vertices[j])] for j in range(3))
        return out[0] if scalar else out


def scattered_interpolator(points, values, **kwargs):
    """Construct a :class:`LinearNDInterpolator` from scattered planar samples."""
    return LinearNDInterpolator(points,values,**kwargs)


def _powers(dimension,degree):
    if degree<0:
        return []
    result = []
    def generate(prefix,remaining):
        if len(prefix)==dimension:
            result.append(tuple(prefix))
            return
        for exponent in range(remaining+1):
            prefix.append(exponent)
            generate(prefix,remaining-exponent)
            prefix.pop()
    generate([],degree)
    return result


class RBFInterpolator:
    """Local radial basis interpolation with polynomial reproduction and bounded cache.

    ``neighbors`` bounds each solve to k samples (O(k²) matrix storage), and a
    k-d tree selects them without a dense all-pairs distance matrix. Cache size
    bounds retained coefficient vectors. Use ``neighbors=None`` for global RBF.
    Neighbor changes can introduce derivative discontinuities between patches.
    """
    def __init__(self, points, values, *, kernel="thin_plate", epsilon=1.0,
                 smooth=0.0, neighbors=None, degree=None, cache_size=64):
        from .multivariate import _RBF_KERNELS
        self.tree = KDTree(points)
        self.points = self.tree.points
        self.values = np.asarray(values,float).copy()
        if self.values.ndim!=1 or len(self.values)!=len(self.points) or not np.all(np.isfinite(self.values)):
            raise ValueError("one finite scalar value per point is required")
        if kernel not in _RBF_KERNELS:
            raise ValueError("unknown RBF kernel")
        if not math.isfinite(epsilon) or epsilon<=0 or not math.isfinite(smooth) or smooth<0:
            raise ValueError("epsilon must be positive and smooth nonnegative")
        self.kernel = _RBF_KERNELS[kernel]
        self.epsilon,self.smooth = float(epsilon),float(smooth)
        self.neighbors = len(self.points) if neighbors is None else operator.index(neighbors)
        self.cache_size = operator.index(cache_size)
        if not 1<=self.neighbors<=len(self.points) or self.cache_size<0:
            raise ValueError("invalid neighbors or cache_size")
        if degree is None:
            degree = {"linear":0,"cubic":1,"quintic":2,"thin_plate":1}.get(kernel,0)
        degree = operator.index(degree)
        if degree < -1 or degree>3:
            raise ValueError("polynomial degree must be -1 through 3")
        self.powers = _powers(self.points.shape[1],degree)
        if len(self.powers)>self.neighbors:
            raise ValueError("not enough neighbors for the polynomial basis")
        self.cache = OrderedDict()
        if neighbors is None:
            self._global = self._fit(tuple(range(len(self.points))))
        else:
            self._global = None

    def _polynomial(self, points):
        return np.array([[math.prod(float(x[j])**power[j] for j in range(len(power)))
                          for power in self.powers] for x in points]).reshape((len(points),len(self.powers)))

    def _fit(self, indices):
        key = tuple(sorted(indices))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        indices = np.array(key,int)
        points = self.points[indices]
        center = points.mean(axis=0)
        scale = max(float(np.max(np.abs(points-center))),1e-100)
        normalized = (points-center)/scale
        count,polynomials = len(points),len(self.powers)
        # Pairwise scratch is bounded by the chosen neighborhood size.
        distance = np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
        matrix = np.zeros((count+polynomials,count+polynomials))
        matrix[:count,:count] = self.kernel(distance,self.epsilon)+self.smooth*np.eye(count)
        if polynomials:
            P = self._polynomial(normalized)
            matrix[:count,count:],matrix[count:,:count] = P,P.T
        rhs = np.concatenate([self.values[indices],np.zeros(polynomials)])
        coefficients = np.linalg.solve(matrix,rhs)
        result = (points,center,scale,coefficients)
        if self.cache_size:
            self.cache[key] = result
            if len(self.cache)>self.cache_size:
                self.cache.popitem(last=False)
        return result

    def __call__(self, points):
        q = np.asarray(points,float)
        scalar = q.ndim==1
        q = np.atleast_2d(q)
        if q.shape[1]!=self.points.shape[1] or not np.all(np.isfinite(q)):
            raise ValueError("invalid RBF query shape or coordinates")
        out = np.empty(len(q))
        for i,query in enumerate(q):
            if self._global is None:
                _,indices = self.tree.query(query,self.neighbors)
                fit = self._fit(tuple(int(j) for j in np.atleast_1d(indices)))
            else:
                fit = self._global
            selected,center,scale,coef = fit
            r = np.linalg.norm(selected-query,axis=1)
            value = float(self.kernel(r,self.epsilon)@coef[:len(selected)])
            if self.powers:
                value += float(self._polynomial(np.atleast_2d((query-center)/scale))[0]@coef[len(selected):])
            out[i] = value
        return float(out[0]) if scalar else out

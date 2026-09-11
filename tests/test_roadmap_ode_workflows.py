"""Independent identities, convergence, and bounded-storage checks for workflows."""
import json
import math
import tracemalloc

import numpy as oracle
import pytest
from quadrivium import numeric as np
from quadrivium.core import SolverCheckpoint, resume_ode, resume_pde
from quadrivium.ode import (
    rk4, dormand_prince, backward_euler, adams_bashforth, bdf_adaptive,
    radau_adaptive, BandedJacobian, velocity_verlet, gragg_bulirsch_stoer,
    rk_nystrom, stormer_cowell, dae_index1_bdf, dde_method_of_steps,
    solve_ivp_sensitivities, adjoint_sensitivity, solve_ivp,
)
from quadrivium.pde import (
    heat_btcs, heat_ftcs, heat_theta, heat_2d_adi, fourier_heat, advection_upwind,
    fvm_muscl, weno_burgers, fem_2d_triangular, TriangularMesh, unit_square_mesh,
    refine_triangles, adaptive_fem, fem_error_estimate, solve_fem_mesh,
)
from quadrivium.interpolate import Delaunay, LinearNDInterpolator, RBFInterpolator, KDTree
from quadrivium.linalg import LinearOperator, diags


@pytest.mark.parametrize("solver",[rk4,dormand_prince,backward_euler,adams_bashforth,
                                    bdf_adaptive,radau_adaptive,gragg_bulirsch_stoer])
def test_final_only_matches_full_trajectory(solver):
    full=solver(lambda t,y:-y,(0,1),[1])
    final=solver(lambda t,y:-y,(0,1),[1],final_only=True)
    assert final.t.shape==(1,)
    assert final.y.shape==(1,1)
    oracle.testing.assert_allclose(final.y_final,full.y_final,rtol=3e-7,atol=1e-12)
    assert final.checkpoint.t==1


def test_selected_dense_samples_keep_actual_endpoint():
    samples=[0.1,0.3,0.7]
    result=dormand_prince(lambda t,y:-y,(0,1),[1],save_at=samples,rtol=1e-10)
    oracle.testing.assert_array_equal(result.t,samples)
    oracle.testing.assert_allclose(result.y[:,0],oracle.exp(-oracle.array(samples)),rtol=1e-7)
    assert result.success
    assert float(result.y_final[0])==pytest.approx(math.exp(-1),rel=1e-9)
    assert result.checkpoint.t==1


def test_reverse_time_and_stride_output():
    result=rk4(lambda t,y:y,(1,0),[math.e],n=100,save_at=[.8,.4,0])
    oracle.testing.assert_allclose(result.y[:,0],oracle.exp([.8,.4,0]),rtol=1e-9)
    stride=rk4(lambda t,y:np.ones(1),(0,1),[0],n=10,save_every=4)
    oracle.testing.assert_allclose(stride.t,[0,.4,.8,1])
    oracle.testing.assert_allclose(stride.y[:,0],stride.t)


def test_callback_copies_and_early_stop():
    seen=[]
    def callback(t,y):
        seen.append((t,float(y[0])))
        y[:]=1234
        if t>=.5:
            return True
    result=rk4(lambda t,y:-y,(0,1),[1],n=100,callback=callback,final_only=True)
    assert len(seen)==51
    assert result.checkpoint.t==.5
    assert result.y_final[0]==pytest.approx(math.exp(-.5),rel=1e-8)
    assert not result.success


@pytest.mark.parametrize("options",[{"save_every":0},{"save_at":[]},
                                    {"save_at":[.5,.1]},
                                    {"save_at":[2]},
                                    {"final_only":True,"save_at":[0]}])
def test_invalid_output_policy(options):
    with pytest.raises(ValueError):
        rk4(lambda t,y:y,(0,1),[1],**options)


def test_final_only_memory_does_not_grow_with_steps():
    def peak(steps):
        tracemalloc.start()
        result=rk4(lambda t,y:-y,(0,1),[1,2],n=steps,final_only=True)
        _,value=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert result.y.shape==(1,2)
        return value
    small=peak(300)
    large=peak(3000)
    assert large<small*3+40000


def test_json_checkpoint_restart_explicit_and_stiff():
    for solver in [dormand_prince,bdf_adaptive,radau_adaptive]:
        partial=solver(lambda t,y:-2*y,(0,.4),[1],final_only=True,rtol=1e-8)
        encoded=partial.checkpoint.to_json()
        assert json.loads(encoded)["version"]==1
        restored=SolverCheckpoint.from_json(encoded)
        resumed=resume_ode(lambda t,y:-2*y,restored,1,rtol=1e-8,final_only=True)
        assert resumed.y_final[0]==pytest.approx(math.exp(-2),rel=8e-6)


@pytest.mark.parametrize("solver",[bdf_adaptive,radau_adaptive])
def test_stiff_tracking_converges_with_tolerance(solver):
    errors=[]
    for tolerance in [1e-4,1e-7]:
        result=solver(lambda t,y:-1000*(y-np.cos(t))-np.sin(t),(0,3),[1],
                      jac=np.array([[-1000.]]),rtol=tolerance,atol=tolerance*.01,
                      final_only=True)
        assert result.success and result.n_steps<300
        assert result.n_jac_evals==1
        assert result.n_linear_solves>=result.n_factorizations
        errors.append(abs(float(result.y_final[0])-math.cos(3)))
    assert errors[1]<errors[0]/20


def test_bdf_changes_order_and_retries_failed_newton():
    result=bdf_adaptive(lambda t,y:-y,(0,2),[1],rtol=1e-7,final_only=True)
    assert result.max_order_used>=3
    hard=bdf_adaptive(lambda t,y:-y**3,(0,1),[10],h0=1,newton_max_iter=3,
                      rtol=1e-6,atol=1e-9,final_only=True)
    assert hard.success and hard.n_newton_failures>0
    assert hard.y_final[0]==pytest.approx(10/math.sqrt(201),rel=2e-4)


@pytest.mark.parametrize("solver",[bdf_adaptive,radau_adaptive])
@pytest.mark.parametrize("kind",["sparse","banded","operator"])
def test_structured_jacobian_and_component_tolerances(solver,kind):
    diagonal=np.array([-1.,-20.,-100.])
    if kind=="sparse":
        jac=diags([diagonal],[0],shape=(3,3))
    elif kind=="banded":
        jac=BandedJacobian(np.atleast_2d(diagonal),0,0)
    else:
        jac=LinearOperator((3,3),lambda v:diagonal*v)
    result=solver(lambda t,y:diagonal*y,(0,.1),[1,1e-5,1e4],jac=jac,
                  atol=[1e-10,1e-13,1e-6],rtol=[1e-7,1e-7,1e-7],final_only=True)
    expected=oracle.array([1,1e-5,1e4])*oracle.exp(oracle.array([-1,-20,-100])*.1)
    oracle.testing.assert_allclose(result.y_final,expected,rtol=2e-4,atol=3e-7)


def test_dispatch_stiff_default_and_legacy_n():
    assert solve_ivp(lambda t,y:-y,(0,1),[1],method="bdf").method=="bdf_adaptive"
    assert solve_ivp(lambda t,y:-y,(0,1),[1],method="bdf",n=20).method=="bdf2"


def test_forward_sensitivity_parameter_initial_and_dependent_initial():
    p=.3
    result=solve_ivp_sensitivities(lambda t,y,p:p[0]*y,(0,2),[2],[p],initial=True,
        jac_y=lambda t,y,p:np.array([[p[0]]]),jac_p=lambda t,y,p:np.atleast_2d(y),
        initial_sensitivity=[[3]],final_only=True,rtol=1e-10)
    assert result.y_final[0]==pytest.approx(2*math.exp(2*p),rel=1e-9)
    assert result.sensitivities[0,0,0]==pytest.approx(7*math.exp(2*p),rel=1e-9)
    assert result.initial_sensitivities[0,0,0]==pytest.approx(math.exp(2*p),rel=1e-9)


def test_checkpointed_adjoint_includes_direct_parameter_dependence():
    result=adjoint_sensitivity(lambda t,y,p:p[0]*y,(0,1),[2],[.3],
        lambda y,p:.5*y[0]**2+p[0]**2,
        terminal_y=lambda y,p:y,terminal_p=lambda y,p:2*p,
        jac_y=lambda t,y,p:np.array([[p[0]]]),jac_p=lambda t,y,p:np.atleast_2d(y),
        checkpoints=4,rtol=1e-10)
    assert result.gradient[0]==pytest.approx(4*math.exp(.6)+.6,rel=2e-7)
    assert result.initial_gradient[0]==pytest.approx(2*math.exp(.6),rel=1e-8)
    assert result.checkpoints==5 and result.recomputed_steps>0


@pytest.mark.parametrize("solver",[rk_nystrom,stormer_cowell])
def test_second_order_bounded_output(solver):
    f=(lambda t,y,dy:-y) if solver is rk_nystrom else (lambda t,y:-y)
    result=solver(f,(0,1),[1],[0],n=200,final_only=True)
    assert result.y.shape==(1,1) and result.velocity.shape==(1,1)
    assert result.y_final[0]==pytest.approx(math.cos(1),abs=2e-8)


def test_dae_and_delay_bounded_output():
    dae=dae_index1_bdf(lambda t,y,z:-y,lambda t,y,z:z-y*y,
                       (0,1),[1],[1],n=100,final_only=True)
    assert dae.y.shape==dae.z.shape==(1,1)
    assert dae.z[0,0]==pytest.approx(dae.y[0,0]**2,abs=1e-10)
    delay=dde_method_of_steps(lambda t,y,lags:lags[0],lambda t:[1],
                              [1],(0,2),final_only=True,rtol=1e-9)
    assert delay.y.shape==(1,1)
    assert delay.y_final[0]==pytest.approx(3.5,abs=2e-7)


@pytest.mark.parametrize("solver",[heat_ftcs,heat_btcs,heat_theta])
def test_heat_streaming_and_restart(solver):
    options=dict(alpha=.1,x_span=(0,1),nx=10,nt=100)
    full=solver(lambda x:np.sin(np.pi*x),t_span=(0,.2),**options)
    selected=solver(lambda x:np.sin(np.pi*x),t_span=(0,.2),save_every=25,**options)
    oracle.testing.assert_allclose(selected.final,full.final,atol=1e-14)
    assert selected.u.shape==(5,11)
    partial=solver(lambda x:np.sin(np.pi*x),t_span=(0,.1),final_only=True,
                    **dict(options,nt=50))
    resumed=resume_pde(solver,partial.checkpoint,.2,alpha=.1,x_span=(0,1),nx=10,
                       final_only=True)
    oracle.testing.assert_allclose(resumed.final,full.final,atol=1e-12)


def test_pde_output_uses_lazy_time_grid(monkeypatch):
    original=np.linspace
    def bounded_linspace(start,stop,num=50,**kwargs):
        assert num<1000,"time grid should not allocate one value per step"
        return original(start,stop,num,**kwargs)
    monkeypatch.setattr(np,"linspace",bounded_linspace)
    result=heat_btcs(lambda x:0,1,(0,1),(0,1),nx=4,nt=1000,final_only=True)
    assert result.u.shape==(1,5)


def test_weno_and_adi_save_at():
    weno=weno_burgers(lambda x:1,(0,1),(0,.02),nx=12,save_at=[.01,.02])
    oracle.testing.assert_allclose(weno.u,1,atol=1e-14)
    adi=heat_2d_adi(lambda x,y:np.sin(np.pi*x)*np.sin(np.pi*y),.1,
                    (0,1),(0,1),(0,.1),nx=5,ny=5,nt=10,final_only=True)
    assert adi.u.shape==(1,6,6)


def test_sparse_fem_manufactured_solution_refines():
    errors=[]
    for n in [4,8]:
        result=fem_2d_triangular(lambda x,y:2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y),n=n)
        exact=oracle.sin(math.pi*oracle.asarray(result.x))*oracle.sin(math.pi*oracle.asarray(result.y))
        errors.append(oracle.max(oracle.abs(oracle.asarray(result.u)-exact)))
        assert result.stiffness.nnz<8*len(result.u)
        assert result.converged
    assert errors[1]<errors[0]/3


def test_local_mesh_refinement_conforming_and_affine_exact():
    mesh=TriangularMesh(*unit_square_mesh(3))
    refined=refine_triangles(mesh,[0])
    assert len(mesh.triangles)<len(refined.triangles)<4*len(mesh.triangles)
    # A conforming triangulation has exactly two incident cells per interior edge.
    edges={}
    for tri in refined.triangles.tolist():
        for i in range(3):
            key=tuple(sorted((tri[i],tri[(i+1)%3])))
            edges[key]=edges.get(key,0)+1
    assert max(edges.values())==2
    result=solve_fem_mesh(refined,0,bc=lambda x,y:2*x-3*y+1)
    expected=2*refined.points[:,0]-3*refined.points[:,1]+1
    oracle.testing.assert_allclose(result.u,expected,atol=1e-9)
    indicators=fem_error_estimate(refined,result.u,0)
    assert np.linalg.norm(indicators)<1e-8


def test_adaptive_fem_estimator_reduces_and_mesh_cap():
    result=adaptive_fem(1,max_refinements=3,max_elements=1000)
    assert len(result.refinement_history)==4
    assert result.refinement_history[-1][1]<result.refinement_history[0][1]*.65
    capped=adaptive_fem(1,max_elements=8)
    assert len(capped.mesh.triangles)==8 and not capped.converged


def test_delaunay_affine_and_outside_hull_modes():
    points=np.array([[0,0],[1,0],[1,1],[0,1],[.3,.2],[.6,.8]])
    values=2*points[:,0]-3*points[:,1]+4
    tri=Delaunay(points)
    query=np.array([[.1,.2],[.5,.7],[1,1]])
    interpolation=LinearNDInterpolator(tri,values)
    oracle.testing.assert_allclose(interpolation(query),2*query[:,0]-3*query[:,1]+4,atol=1e-13)
    assert tri.find_simplex([2,2])==-1 and np.isnan(interpolation([2,2]))
    with pytest.raises(ValueError):
        LinearNDInterpolator(tri,values,outside="raise")([2,2])
    assert LinearNDInterpolator(tri,values,outside="nearest")([2,2])==3


def test_kdtree_matches_independent_distance_order():
    rng=oracle.random.default_rng(521)
    points=rng.normal(size=(50,4))
    queries=rng.normal(size=(4,4))
    distance,index=KDTree(points.tolist()).query(queries.tolist(),k=5)
    expected=oracle.sum((queries[:,None,:]-points[None,:,:])**2,axis=2)
    oracle.testing.assert_array_equal(index,oracle.argsort(expected,axis=1)[:,:5])
    oracle.testing.assert_allclose(distance,oracle.sqrt(oracle.take_along_axis(expected,oracle.asarray(index),axis=1)))


def test_local_rbf_affine_reproduction_and_bounded_cache():
    points=np.array([[i/5,j/5] for i in range(6) for j in range(6)])
    values=3*points[:,0]-2*points[:,1]+7
    interpolation=RBFInterpolator(points,values,neighbors=8,cache_size=3)
    queries=np.array([[i/17,.31] for i in range(18)])
    oracle.testing.assert_allclose(interpolation(queries),3*queries[:,0]-2*queries[:,1]+7,atol=1e-12)
    assert len(interpolation.cache)<=3


@pytest.mark.parametrize("points",[[[0,0],[1,0],[2,0]],[[0,0],[0,0],[1,1]]])
def test_degenerate_triangulation_rejected(points):
    with pytest.raises(ValueError):
        Delaunay(points)


def test_sensitivity_endpoint_when_samples_omit_endpoint():
    result=solve_ivp_sensitivities(lambda t,y,p:p[0]*y,(0,2),[2],[.3],
                                  save_at=[.25,.5],rtol=1e-10)
    assert result.sensitivities.shape==(2,1,1)
    assert result.sensitivity_final[0,0]==pytest.approx(4*math.exp(.6),rel=1e-8)
    assert result.solution.checkpoint is None
    assert result.augmented_checkpoint.y.size==2


@pytest.mark.parametrize("name",["wave_explicit","wave_implicit","leapfrog_advection"])
def test_recurrence_checkpoint_restart_is_exact(name):
    from quadrivium import pde
    solver=getattr(pde,name)
    parameters=dict(c=.3,x_span=(0,1),nx=20)
    if name.startswith("wave"):
        parameters["v0"]=lambda x:0
    initial=lambda x:np.sin(2*np.pi*x)
    full=solver(initial,t_span=(0,.2),nt=40,**parameters)
    partial=solver(initial,t_span=(0,.1),nt=20,final_only=True,**parameters)
    parameters.pop("v0",None)
    resumed=resume_pde(solver,partial.checkpoint,.2,final_only=True,**parameters)
    oracle.testing.assert_allclose(resumed.final,full.final,atol=1e-14,rtol=0)


def test_fixed_ode_restart_adapter_and_delay_window():
    partial=rk4(lambda t,y:-y,(0,.5),[1],n=50,final_only=True)
    resumed=resume_ode(lambda t,y:-y,partial.checkpoint,1,final_only=True)
    assert resumed.y_final[0]==pytest.approx(math.exp(-1),rel=1e-9)
    oscillator=velocity_verlet(lambda q:-q,(0,.5),[1],[0],n=500,final_only=True)
    adapter=lambda f,t_span,y0,**kw:velocity_verlet(f,t_span,y0[:1],y0[1:],n=500,**kw)
    final=resume_ode(lambda q:-q,oscillator.checkpoint,1,solver=adapter,final_only=True)
    assert final.y_final[0]==pytest.approx(math.cos(1),abs=1e-7)
    rhs=lambda t,y,lags:lags[0]
    delay=dde_method_of_steps(rhs,lambda t:[1],[1],(0,2),final_only=True)
    resumed=resume_ode(rhs,delay.checkpoint,3,final_only=True)
    assert resumed.y_final[0]==pytest.approx(37/6,abs=2e-6)


def test_method_of_lines_adaptive_stiff_and_time_boundaries():
    from quadrivium.pde import method_of_lines
    def rhs(t,u,x,dx):
        value=np.zeros_like(u)
        value[1:-1]=(u[2:]-2*u[1:-1]+u[:-2])/dx**2
        return value
    result=method_of_lines(lambda x:x,rhs,(0,1),(0,.1),nx=10,solver="bdf",
                           bc=(0,1),rtol=1e-7,final_only=True)
    oracle.testing.assert_allclose(result.final,result.x,atol=1e-8)


def test_pde_callback_stops_without_retaining_history():
    samples=[]
    def callback(t,u):
        samples.append(t)
        return t>=.03
    result=heat_btcs(lambda x:0,1,(0,1),(0,.1),nx=5,nt=10,
                     callback=callback,final_only=True)
    assert len(samples)==4 and result.checkpoint.t==.03
    assert not result.converged and result.u.shape==(1,6)


def test_bdf_variable_history_convergence_and_reverse_time():
    errors=[]
    for tolerance in [1e-6,1e-8,1e-10]:
        result=bdf_adaptive(lambda t,y:-y,(0,3),[1],jac=np.array([[-1.]]),
                            rtol=tolerance,atol=tolerance*1e-3,h0=.03)
        errors.append(float(np.max(np.abs(result.y[:,0]-np.exp(-result.t)))))
        assert result.max_order_used>=4
        assert result.n_steps<250
    assert errors[1]<errors[0]/20 and errors[2]<errors[1]/20
    reverse=bdf_adaptive(lambda t,y:-y,(3,0),[math.exp(-3)],rtol=1e-8,
                          atol=1e-11,final_only=True)
    assert reverse.y_final[0]==pytest.approx(1,rel=2e-6)


def test_tampered_bdf_checkpoint_is_rejected():
    partial=bdf_adaptive(lambda t,y:-y,(0,1),[1],final_only=True)
    checkpoint=SolverCheckpoint.from_json(partial.checkpoint.to_json())
    checkpoint.y[0]=99
    with pytest.raises(ValueError,match="history"):
        resume_ode(lambda t,y:-y,checkpoint,2)


def test_refinement_preserves_partial_dirichlet_boundary():
    mesh=TriangularMesh([[0,0],[1,0],[1,1],[0,1]],[[0,1,2],[0,2,3]],[0,3])
    refined=refine_triangles(mesh,[0,1])
    assert len(refined.boundary)==3
    oracle.testing.assert_array_equal(refined.points[refined.boundary][:,0],0)


def test_empty_kdtree_queries_and_high_dimensional_rbf():
    tree=KDTree([[0,0],[1,1]])
    distances,indices=tree.query(np.empty((0,2)),k=2)
    assert distances.shape==indices.shape==(0,2)
    rng=oracle.random.default_rng(941)
    points=np.array(rng.normal(size=(32,25)).tolist())
    values=points.sum(axis=1)+4
    interpolation=RBFInterpolator(points,values,kernel="linear",degree=1,neighbors=30)
    query=np.zeros(25)
    assert interpolation(query)==pytest.approx(4,abs=1e-10)

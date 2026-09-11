"""Independent identities and numerical references for the workflow additions."""
import json
import math
import numpy as np
import pytest
from quadrivium import numeric as qnp
from quadrivium import stochastic as st, transforms as tr, diff, integrate, approx, rootfind


@pytest.mark.parametrize('dist,cdf', [
    (st.Normal(), lambda x: .5*math.erfc(-x/math.sqrt(2))),
    (st.Exponential(2), lambda x: -math.expm1(-max(x,0)/2)),
    (st.Gamma(1,2), lambda x: -math.expm1(-max(x,0)/2)),
    (st.Beta(1,1), lambda x: min(1,max(x,0))),
    (st.StudentT(1), lambda x: .5+math.atan(x)/math.pi),
    (st.ChiSquare(2), lambda x: -math.expm1(-max(x,0)/2)),
])
def test_distribution_cdf_identities(dist,cdf):
    x=np.array([-.5,.1,.5,1.,2.,8.])
    np.testing.assert_allclose(dist.cdf(x), [cdf(v) for v in x], atol=2e-14)
    np.testing.assert_allclose(qnp.asarray(dist.cdf(x))+dist.sf(x),1,atol=2e-14)


@pytest.mark.parametrize('dist', [st.Normal(2,3),st.Uniform(-2,5),st.Exponential(2),
                                  st.Gamma(.7,2),st.Beta(.3,2.5),st.StudentT(7)])
def test_distribution_quantile_tail_roundtrip(dist):
    p=np.array([1e-10,.001,.2,.5,.9,1-1e-10])
    np.testing.assert_allclose(dist.cdf(dist.ppf(p)),p,rtol=3e-9,atol=2e-14)
    np.testing.assert_allclose(dist.sf(dist.isf(p)),p,rtol=3e-9,atol=2e-14)
    np.testing.assert_array_equal(dist.rvs(size=10,rng=73),dist.rvs(size=10,rng=73))
    with pytest.raises(ValueError): dist.ppf(1.2)


def test_log_tail_survives_underflow():
    d=st.Normal()
    assert d.sf(40)==0
    assert math.isfinite(d.logsf(40))
    assert abs(d.logsf(d.isf(1e-250))-math.log(1e-250))<1e-10
    assert st.Gamma(1).logsf(1000)==pytest.approx(-1000)


def test_discrete_probabilities_and_quantiles():
    for d, exact in [(st.Poisson(3),[math.exp(-3)*3**k/math.factorial(k) for k in range(10)]),
                     (st.Binomial(9,.3),[math.comb(9,k)*.3**k*.7**(9-k) for k in range(10)])]:
        np.testing.assert_allclose(d.pmf(np.arange(10)),exact,rtol=2e-13)
        np.testing.assert_allclose(d.cdf(np.arange(10)),np.cumsum(exact),atol=2e-14)
        for p in (.01,.3,.9,.999):
            k=d.ppf(p)
            assert d.cdf(k)>=p-2e-14
            assert d.cdf(k-1)<p+2e-14


def test_sobol_balance_chunk_restart_and_high_dimension():
    e=st.Sobol(40,scramble=True,seed=15,bits=12)
    all_points=np.asarray(e.random_base2(8))
    assert np.all((all_points>=0)&(all_points<1))
    for column in all_points.T:
        np.testing.assert_array_equal(np.sort(np.floor(column*256)),np.arange(256))
    e.reset()
    first=e.random(31)
    saved=json.loads(json.dumps(e.state))
    rest=st.Sobol.from_state(saved).random(225)
    np.testing.assert_array_equal(np.vstack((first,rest)),all_points)
    np.testing.assert_array_equal(e.reset().fast_forward(31).random(5),all_points[31:36])
    assert not np.array_equal(all_points[:,5],all_points[:,30])
    invalid=st.Sobol(2)
    invalid.random(3)
    with pytest.raises(ValueError): invalid.random_base2(2)


def test_sobol_unscrambled_first_coordinate_and_rng_children():
    expected=[0,.5,.75,.25,.375,.875,.625,.125]
    np.testing.assert_array_equal(np.asarray(st.Sobol(1,scramble=False).random(8))[:,0],expected)
    streams=st.spawn_rngs(31,3)
    values=[np.asarray(r.random(10)) for r in streams]
    assert not np.array_equal(values[0],values[1])
    np.testing.assert_array_equal(values[2],st.spawn_rngs(31,3)[2].random(10))


def test_randomized_qmc_estimate_and_reproducibility():
    fn=lambda x: x[0]*x[1]
    a=st.randomized_qmc(fn,[0,0],[1,1],m=9,replicates=6,seed=7,batch_size=73)
    b=st.randomized_qmc(fn,[0,0],[1,1],m=9,replicates=6,seed=7,batch_size=256)
    assert a.function_calls==512*6
    assert a.error_estimate>0
    assert abs(a.value-.25)<5e-4
    assert a.value==pytest.approx(b.value,abs=1e-15)


def test_mcmc_diagnostics_detect_shift_and_autocorrelation():
    rng=np.random.default_rng(1)
    iid=rng.normal(size=(4,700))
    assert st.split_rhat(iid)<1.02
    assert st.bulk_ess(iid)>1400
    shifted=iid+np.arange(4)[:,None]*2
    assert st.split_rhat(shifted)>1.3
    correlated=iid.copy()
    for k in range(1,700): correlated[:,k]=.95*correlated[:,k-1]+iid[:,k]
    assert st.bulk_ess(correlated)<st.bulk_ess(iid)/5
    assert st.mcse(iid)==pytest.approx(1/math.sqrt(iid.size),rel=.2)
    assert st.tail_ess(iid)>1000


@pytest.mark.parametrize('complex_input',[False,True])
def test_streaming_filters_match_independent_convolution(complex_input):
    rng=np.random.default_rng(31)
    x=rng.normal(size=103)+(1j*rng.normal(size=103) if complex_input else 0)
    taps=np.array([.2,.3,.4,.1])
    filt=tr.FIRFilter(taps)
    chunks=[filt.process(x[:17]),filt.process(x[17:60]),filt.process(x[60:])]
    np.testing.assert_allclose(np.concatenate(chunks),np.convolve(x,taps)[:len(x)],atol=1e-14)
    filt=tr.IIRFilter([.3],[1,-.7])
    first=filt.process(x[:41])
    state=filt.state
    second=tr.IIRFilter([.3],[1,-.7],state=state).process(x[41:])
    expected=np.zeros(x.shape,dtype=x.dtype)
    for k in range(x.size): expected[k]=.3*x[k]+(.7*expected[k-1] if k else 0)
    np.testing.assert_allclose(np.concatenate((first,second)),expected,atol=1e-14)


def test_filter_design_frequency_response():
    sos=np.asarray(tr.butterworth_sos(5,12,fs=100))
    def response(freq):
        z=np.exp(-2j*np.pi*freq/100)
        return np.prod((sos[:,0]+sos[:,1]*z+sos[:,2]*z*z)/(sos[:,3]+sos[:,4]*z+sos[:,5]*z*z))
    assert abs(response(0))==pytest.approx(1,abs=2e-14)
    assert abs(response(12))==pytest.approx(1/math.sqrt(2),rel=2e-13)
    impulse=np.zeros(300);impulse[0]=1
    one=tr.SOSFilter(sos).process(impulse)
    filt=tr.SOSFilter(sos)
    np.testing.assert_allclose(np.concatenate((filt.process(impulse[:7]),filt.process(impulse[7:]))),one)
    assert np.sum(tr.firwin(31,.2))==pytest.approx(1)


@pytest.mark.parametrize('up,down',[(1,1),(2,3),(3,2),(4,6)])
def test_polyphase_matches_zero_insertion_convolution(up,down):
    x=np.random.default_rng(4).normal(size=31)
    taps=np.array([.1,.2,.4,.2,.1])
    g=math.gcd(up,down);u,d=up//g,down//g
    expanded=np.zeros(len(x)*u);expanded[::u]=x
    full=np.convolve(expanded,taps*u)
    expected=full[(len(taps)-1)//2::d][:math.ceil(len(x)*u/d)]
    np.testing.assert_allclose(tr.resample_poly(x,up,down,taps=taps),expected,atol=1e-14)
    r=tr.PolyphaseResampler(up,down,taps=taps)
    a=r.process(x[:9]);state=json.loads(json.dumps(r.state))
    r2=tr.PolyphaseResampler(up,down,taps=taps);r2.state=state
    b=r2.process(x[9:])
    np.testing.assert_allclose(np.concatenate((a,b)),full[::d][:math.ceil(len(x)*u/d)],atol=1e-14)
    assert len(r2.state['tail_real'])<=math.ceil(len(taps)/u)


@pytest.mark.parametrize('length,segment,hop',[(1,8,4),(71,16,4),(103,17,6),(128,32,16)])
@pytest.mark.parametrize('complex_input',[False,True])
def test_stft_reconstructs_boundary_and_partial_frames(length,segment,hop,complex_input):
    rng=np.random.default_rng(87)
    x=rng.normal(size=length)+(1j*rng.normal(size=length) if complex_input else 0)
    result=tr.stft(x,segment=segment,hop=hop,nfft=segment+3)
    np.testing.assert_allclose(tr.istft(result),x,atol=2e-13)
    np.testing.assert_allclose(tr.istft(result,coefficients=result.coefficients*2),2*x,atol=4e-13)


def test_tensor_array_chain_rule_broadcast_and_index_accumulation():
    x=np.arange(12,dtype=float).reshape(3,4)/10
    weights=np.arange(4)+1
    value,grad=diff.array_value_and_grad(lambda t: (((t+weights).sin())**2).sum(),x)
    np.testing.assert_allclose(value,np.sum(np.sin(x+weights)**2),atol=2e-14)
    np.testing.assert_allclose(grad,np.sin(2*(x+weights)),atol=2e-14)
    np.testing.assert_array_equal(diff.array_gradient(lambda t:t[[0,0,2]].sum(),[1,2,3]),[2,0,1])


def test_tensor_jvp_vjp_matmul_and_public_dispatch():
    A=np.array([[1.,2.,3.],[-1,4,2]])
    x=np.array([.2,.3,.4]);v=np.array([1.,-2,.3]);w=np.array([.3,-.7])
    f=lambda t:qnp.sin(qnp.matmul(qnp.asarray(A),t))
    y,jv=diff.jvp(f,x,v)
    value,pullback=diff.vjp(f,x)
    np.testing.assert_allclose(y,np.sin(A@x),atol=1e-14)
    np.testing.assert_allclose(jv,np.cos(A@x)*(A@v),atol=1e-14)
    np.testing.assert_allclose(pullback(w),A.T@(np.cos(A@x)*w),atol=1e-14)
    assert float(qnp.dot(jv,w))==pytest.approx(float(qnp.dot(v,pullback(w))))
    np.testing.assert_allclose(diff.array_gradient(lambda t:qnp.mean(qnp.exp(t)),x),np.exp(x)/3)


def test_vector_quad_shared_nodes_and_infinite_limits():
    calls=[]
    def f(t):
        calls.append(t)
        return qnp.array([[t,t*t],[math.sin(t),complex(math.cos(t),math.sin(t))]])
    r=integrate.quad_vec(f,0,1)
    expected=np.array([[.5,1/3],[1-math.cos(1),math.sin(1)+1j*(1-math.cos(1))]])
    assert r.converged and r.function_calls==len(calls)
    np.testing.assert_allclose(r.value,expected,atol=2e-12)
    r=integrate.quad_vec(lambda t:qnp.array([math.exp(-t),math.exp(-2*t)]),0,math.inf)
    np.testing.assert_allclose(r.value,[1,.5],rtol=1e-8)
    with pytest.raises(ValueError): integrate.quad_vec(lambda t:[t],0,1,epsabs=-1)


def test_adaptive_chebyshev_calculus_and_roots():
    c=approx.chebfun(lambda x:x**4-5*x*x+4,(-3,3))
    assert c.converged
    x=np.linspace(-3,3,71)
    np.testing.assert_allclose(c.derivative()(x),4*x**3-10*x,atol=1e-10)
    assert c.integrate()==pytest.approx(2*(3**5/5-5*3**3/3+12),abs=1e-10)
    np.testing.assert_allclose(c.roots(),[-2,-1,1,2],atol=1e-8)
    cusp=approx.chebfun(abs,(-1,1),max_degree=32,max_pieces=32)
    assert cusp.converged and len(cusp.pieces)>1
    np.testing.assert_allclose(cusp(x/3),abs(x/3),atol=1e-9)


def test_pseudo_arclength_crosses_fold():
    result=rootfind.pseudo_arclength(lambda x,p:qnp.array([x[0]**2-p]),[1.],1.,
        direction=-1,max_steps=25,ds=.1,stability=True)
    assert result.converged and result.x[-1,0]<-.5
    np.testing.assert_allclose(qnp.asarray(result.x[:,0])**2,result.parameters,atol=2e-9)
    assert any(point['type']=='fold' for point in result.bifurcations)
    assert not result.stable[0] and result.stable[-1]
    assert np.max(result.residuals)<1e-9


def test_distribution_extreme_shapes_and_locations_remain_defined():
    for shape in [.001,1e-300]:
        samples=np.asarray(st.Beta(shape,shape).rvs(1000,rng=4))
        assert np.all(np.isfinite(samples)) and np.all((samples>=0)&(samples<=1))
        assert .4<samples.mean()<.6
    cauchy=st.StudentT(1)
    assert cauchy.sf(1e200)==pytest.approx(1/(math.pi*1e200),rel=1e-12,abs=0)
    assert cauchy.logpdf(1e200)==pytest.approx(-math.log(math.pi)-400*math.log(10),rel=1e-14)
    huge=st.Normal(1e308,1e308)
    assert huge.cdf(1e308)==.5
    assert huge.logcdf(1e308)==pytest.approx(-math.log(2))
    assert huge.ppf(.5)==pytest.approx(1e308,rel=1e-14)


def test_nonfinite_qmc_and_unresolved_approximation_are_explicit():
    with pytest.raises(ValueError): st.randomized_qmc(lambda x:math.nan,[0],[1],m=2,replicates=2)
    with pytest.raises(ValueError): st.randomized_qmc(lambda x:x,[0],[math.inf],m=2,replicates=2)
    unresolved=approx.chebfun(lambda x:abs(x-.12345),max_degree=16,max_pieces=1,atol=1e-14,rtol=1e-14)
    assert not unresolved.converged and not unresolved.derivative().converged


def test_tensor_preserves_forward_values_and_index_keys():
    x=diff.Tensor([2.,3.]);index=[0]
    y=x[index].sum();index[0]=1
    y.backward()
    np.testing.assert_array_equal(x.grad,[1,0])
    with pytest.raises(ValueError): x.data[:]=9
    with pytest.raises(ValueError): x.data.flags.writeable=True
    a=diff.Tensor([[1.,2.],[3.,4.]])
    np.testing.assert_array_equal(a.transpose(-1,-2).data,[[1,3],[2,4]])


@pytest.mark.parametrize('kind',['optimize','rootfind'])
def test_results_do_not_retain_callback_and_captured_data(kind):
    import gc
    import weakref
    from quadrivium import optimize
    class Callback:
        def __init__(self): self.buffer=np.zeros(1000)
        def __call__(self,x): return False
    callback=Callback();reference=weakref.ref(callback)
    if kind=='optimize':
        result=optimize.bfgs(lambda x:float(x@x),[2.],callback=callback,store_history=False)
    else:
        result=rootfind.newton(lambda x:x*x-2,1.5,callback=callback,store_history=False)
    del callback
    gc.collect()
    assert reference() is None
    assert result.converged and result.history==[]

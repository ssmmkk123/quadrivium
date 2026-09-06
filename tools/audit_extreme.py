#!/usr/bin/env python3
"""Independent, process-isolated extreme-input audit (NumPy required).

Exit nonzero for mismatches, unexpected exceptions, native crashes or deadlines.
Normal workers have a 1 GiB virtual-memory ceiling; ASan workers must disable it.
The test bodies are preserved in JSON so every finding is directly reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ.get('QUADRIVIUM_AUDIT_ROOT', Path(__file__).resolve().parents[1]))
CASES = []


def add(name, body, rejects=()):
    CASES.append(dict(name=name, body=body, rejects=rejects))


for value in ('float("nan")', 'float("inf")', '1e30', '-1'):
    add('poisson invalid rate ' + value,
        f'a.random.default_rng(7).poisson({value}, size=10)', ('ValueError', 'OverflowError'))
for probs in ('[0.,0.]', '[-1.,2.]', '[float("nan"),1.]', '[float("inf"),1.]', '[.2,.2]'):
    add('choice invalid weights ' + probs,
        f'a.random.default_rng(7).choice(2, size=100, p={probs})', ('ValueError',))
for lo, hi in [('3', '1'), ('float("nan")', '1'), ('0', 'float("inf")'), ('-1e308', '1e308')]:
    add(f'uniform invalid range {lo} {hi}',
        f'a.random.default_rng(7).uniform({lo},{hi},size=10)', ('ValueError', 'OverflowError'))
add('choice sparse probabilities valid', '''
x = a.random.default_rng(7).choice(4,size=10000,p=[0.,0.,1.,0.])
np.testing.assert_array_equal(np.asarray(x), np.full(10000,2))
''')
add('choice no replacement million population', '''
x=np.asarray(a.random.default_rng(7).choice(1000000,size=10000,replace=False))
assert len(np.unique(x))==10000 and x.min()>=0 and x.max()<1000000
''')
add('full signed integer RNG domain', '''
x=np.asarray(a.random.default_rng(7).integers(-2**63,2**63-1,size=10000))
y=np.random.default_rng(7).integers(-2**63,2**63-1,size=10000)
np.testing.assert_array_equal(x,y)
''')
add('RNG zero sized broadcasting', '''
x=a.random.default_rng(7).normal(loc=a.zeros((0,3)),scale=a.ones(3))
assert x.shape==(0,3)
''')

for scale in ('1e-300', '1e-200', '1e200', '1e300'):
    add('norm scale ' + scale, f'''
x=a.array([3.,4.])*{scale}
got=float(a.linalg.norm(x)); expected=5*{scale}
assert math.isfinite(got) and abs(got/expected-1)<1e-14, (got,expected)
''')
    for layer in ('a', 'q'):
        add(f'solve {layer} scale {scale}', f'''
mat=a.array([[3.,1.],[1.,2.]])*{scale}
b=a.array([5.,5.])*{scale}
x={layer}.linalg.solve(mat,b)
np.testing.assert_allclose(np.asarray(x),[1.,2.],rtol=1e-12,atol=0)
''')
    add('slogdet scale ' + scale, f'''
sgn, val=a.linalg.slogdet(a.eye(3)*{scale})
assert sgn==1 and abs(val-3*math.log({scale}))<1e-10, (sgn,val)
''')

for layer in ('a', 'q'):
    add('cholesky extreme dynamic range ' + layer, f'''
mat=a.diag(a.array([1e-200,1.,1e200]))
L={layer}.linalg.cholesky(mat)
np.testing.assert_allclose(np.asarray(L),np.diag([1e-100,1.,1e100]),rtol=1e-12,atol=0)
''')

for dtype in ('float', 'complex', 'int', 'bool'):
    add('foreign buffer layouts ' + dtype, f'''
raw=np.arange(120).reshape(4,5,6).astype({dtype})
for view in (raw.T,raw[::-1,::2,::-1],np.asfortranarray(raw)):
    np.testing.assert_array_equal(np.asarray(a.asarray(view)),view)
''')
add('foreign nonnative endian buffer', '''
x=np.array([1.,2.,-3.,1e100],dtype='>f8')
np.testing.assert_array_equal(np.asarray(a.asarray(x)),x)
''')
add('foreign unaligned buffer', '''
x=np.ndarray((100,),dtype=np.float64,buffer=bytearray(801),offset=1)
x[:]=np.arange(100)
np.testing.assert_array_equal(np.asarray(a.asarray(x)),x)
''')
add('view survives source collection', '''
x=a.arange(10000,dtype=float).reshape(100,100)
view=x[::-3,::7].T
expected=np.arange(10000,dtype=float).reshape(100,100)[::-3,::7].T.copy()
del x; gc.collect()
np.testing.assert_array_equal(np.asarray(view),expected)
view[:]=-1
assert np.all(np.asarray(view)==-1)
''')
add('readonly alias assignment', '''
x=a.arange(10); x.flags.writeable=False
x[:]=1
''', ('ValueError',))
add('huge slice bounds', '''
x=a.arange(8)
for key in (slice(-10**100,10**100),slice(None,None,10**100),slice(None,None,-10**100)):
    np.testing.assert_array_equal(np.asarray(x[key]),np.arange(8)[key])
''')
add('maximum dimensions broadcast transpose', '''
x=a.ones((1,)*16)
assert (x+x.transpose(tuple(reversed(range(16))))).shape==(1,)*16
''')
add('empty zero-stride broadcast', '''
x=a.broadcast_to(a.array(3.),(0,3,2))
assert x.shape==(0,3,2) and a.sum(x)==0
''')

for expr in ('a.zeros((2**32,2**32))', 'a.empty(2**61)',
             'a.ones((1,)*10000)', 'a.arange(3)[(None,)*10000]',
             'a.repeat(a.ones(4),2**62)', 'a.tile(a.ones(4),(2**62,))'):
    add('allocation/shape guard ' + expr, expr, ('MemoryError','ValueError','OverflowError','IndexError'))
add('allocation byte overflow write', '''
x=a.empty(2**61,dtype=float)
x[1024]=7.
''', ('MemoryError','ValueError','OverflowError'))
add('recover after allocation failure', '''
try:
    a.empty(2**40)
except MemoryError:
    pass
else:
    raise AssertionError('allocation unexpectedly succeeded under memory cap')
np.testing.assert_array_equal(np.asarray(a.arange(10)+1),np.arange(10)+1)
''')

for ptr, idx, data in [('[0,2,1]', '[0]', '[1.]'), ('[1,1,1]', '[]','[]'),
                       ('[0,1,1]', '[2]', '[1.]'), ('[0,1,1]', '[-1]', '[1.]'),
                       ('[0,1,1]', '[0.5]', '[1.]'), ('[0,1,1]', '[0]', '[]')]:
    add(f'CSR malformed {ptr} {idx} {data}',
        f'q.linalg.CSRMatrix({ptr},{idx},{data},(2,2)).matvec(a.ones(2))',
        ('ValueError','DimensionError','IndexError'))
add('CSR mutated indices', '''
mat=q.linalg.identity_sparse(3)
mat.indices[0]=2**62
mat.matvec(a.ones(3))
''', ('ValueError','IndexError'))
add('CSR mutated row pointers', '''
mat=q.linalg.identity_sparse(3)
mat.indptr[1]=2**62
mat.matvec(a.ones(3))
''', ('ValueError','IndexError'))
add('CSR duplicate unsorted entries and empty rows', '''
mat=q.linalg.CSRMatrix([0,3,3,5],[2,0,2,1,1],[2.,3.,-1.,4.,-4.],(3,3))
np.testing.assert_array_equal(np.asarray(mat.matvec(a.array([1.,2.,3.]))),[6.,0.,0.])
''')

for size in (1,2,31,257,10007):
    add('FFT large amplitude round trip ' + str(size), f'''
x=np.cos(np.arange({size}))*1e290
y=np.asarray(q.transforms.ifft(q.transforms.fft(a.array(x))))
np.testing.assert_allclose(y/1e290,x/1e290,rtol=1e-9,atol=1e-10)
''')
for name in ('brent','bisection'):
    add('root tiny bracket scale ' + name, f'''
r=q.rootfind.{name}(lambda x:x-1e-200,0.,3e-200,tol=1e-210)
assert r.converged and abs(r.root/1e-200-1)<1e-8, repr(r)
''')
    add('root callback NaN ' + name, f'''
r=q.rootfind.{name}(lambda x:float('nan'),-1.,1.,max_iter=10)
assert not r.converged, repr(r)
''', ('ValueError','ConvergenceError'))
add('ODE callback exception propagation', '''
def rhs(t,y):
    raise LookupError('callback sentinel')
try:
    q.solve_ivp(rhs,(0,1),[1.])
except LookupError as exc:
    assert str(exc)=='callback sentinel'
else:
    raise AssertionError('callback exception swallowed')
''')
add('ODE finite time blowup', '''
try:
    r=q.solve_ivp(lambda t,y:y*y,(0,2),[1.],max_steps=10000)
except (RuntimeError,ArithmeticError) as exc:
    pass
else:
    assert not r.success, repr(r)
''', ('StepSizeError','ConvergenceError'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=8)
    parser.add_argument('--backend',choices=('rust','python'),default='rust')
    parser.add_argument('--no-memory-limit',action='store_true')
    parser.add_argument('--filter',default='')
    args=parser.parse_args()
    env=dict(os.environ, OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             RAYON_NUM_THREADS='1',QUADRIVIUM_NO_ACCEL='1' if args.backend=='python' else '0')
    report=dict(root=str(ROOT),backend=args.backend,timeout=args.timeout,
                memory_limit_bytes=None if args.no_memory_limit else 1024**3,results=[])
    for case in CASES:
        if args.filter not in case['name']:
            continue
        if args.no_memory_limit and case['name']=='recover after allocation failure':
            continue
        pre="import resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
        if not args.no_memory_limit:
            pre+="resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))\n"
        pre+=f'import sys; sys.path.insert(0,{str(ROOT)!r})\n'
        pre+='import json,math,gc,numpy as np\nimport quadrivium as q\nfrom quadrivium import numeric as a\n'
        pre+='try:\n    exec('+repr(case['body'])+')\n'
        pre+='    print(json.dumps(dict(outcome="return")))\n'
        pre+='except Exception as exc:\n    print(json.dumps(dict(outcome="exception",exception=type(exc).__name__,message=str(exc))))\n'
        start=time.monotonic()
        try:
            p=subprocess.run([sys.executable,'-X','faulthandler','-c',pre],env=env,
                             capture_output=True,text=True,timeout=args.timeout)
            r=json.loads(p.stdout) if p.returncode==0 else dict(outcome='crash',returncode=p.returncode)
            r['stderr']=p.stderr[-12000:]
        except subprocess.TimeoutExpired:
            r=dict(outcome='timeout')
        except Exception as exc:
            r=dict(outcome='harness_error',message=repr(exc))
        # Bodies that assert a property may explicitly allow rejection too.
        rejection_only=bool(case['rejects']) and 'assert ' not in case['body']
        r['passed']=(r['outcome']=='return' and not rejection_only) or (r.get('exception') in case['rejects'])
        r.update(case,seconds=time.monotonic()-start)
        report['results'].append(r)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(case['name'],r['outcome'],r['passed'],flush=True)
    return int(any(not r['passed'] for r in report['results']))


if __name__=='__main__':
    raise SystemExit(main())

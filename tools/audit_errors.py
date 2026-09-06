#!/usr/bin/env python3
"""Process-isolated edge/error probes; crashes and hangs cannot stop the sweep.

Every child has a 1 GiB address-space limit and a five-second deadline on Unix.
Use QUADRIVIUM_AUDIT_ROOT to point at a separately instrumented package build.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get("QUADRIVIUM_AUDIT_ROOT", Path(__file__).resolve().parents[1]))
CASES = [
    ("integer floor division overflow", "a.array([-2**63],dtype=int)//a.array([-1],dtype=int)", "return"),
    ("integer remainder overflow", "a.array([-2**63],dtype=int)%a.array([-1],dtype=int)", "return"),
    ("integer absolute minimum", "a.absolute(a.array([-2**63],dtype=int))", "return"),
    ("integer addition wrap", "a.array([2**63-1],dtype=int)+1", "return"),
    ("integer product wrap", "a.array([2**62],dtype=int)*4", "return"),
    ("empty mean", "a.mean(a.array([],dtype=float))", "return"),
    ("empty max", "a.max(a.array([],dtype=float))", "reject"),
    ("empty argmax", "a.argmax(a.array([],dtype=float))", "reject"),
    ("negative allocation", "a.zeros((-1,2))", "reject"),
    ("overflow allocation", "a.zeros((2**62,8))", "reject"),
    ("excessive dimensions", "a.ones((1,)*17)", "reject"),
    ("zero with huge dimensions", "a.zeros((0,2**62,8))", "either"),
    ("reshape two inferred axes", "a.ones(6).reshape(-1,-1)", "reject"),
    ("reshape invalid count", "a.ones(6).reshape(4,4)", "reject"),
    ("reshape invalid negative", "a.ones(6).reshape(-2,3)", "either"),
    ("transpose duplicate axis", "a.ones((2,2)).transpose(0,0)", "reject"),
    ("transpose out of bounds", "a.ones((2,2)).transpose(0,3)", "reject"),
    ("reduction bad axis", "a.sum(a.ones((2,2)),axis=9)", "reject"),
    ("reduction repeated axis", "a.sum(a.ones((2,2)),axis=(0,0))", "reject"),
    ("slice zero step", "a.ones(3)[::0]", "reject"),
    ("scalar index out of bounds", "a.ones(3)[3]", "reject"),
    ("negative index out of bounds", "a.ones(3)[-4]", "reject"),
    ("advanced index out of bounds", "a.ones(3)[[0,3]]", "reject"),
    ("floating index", "a.ones(3)[a.array([0.5])]", "reject"),
    ("mismatched boolean mask", "a.ones(3)[a.array([True,False])]", "reject"),
    ("too many new axes", "a.ones(1)[(None,)*20]", "reject"),
    ("empty boolean scalar index", "a.array(1)[False]", "return"),
    ("advanced scalar true", "a.ones(3)[a.array(True)]", "return"),
    ("advanced scalar false", "a.ones(3)[a.array(False)]", "return"),
    ("broadcast incompatible", "a.ones((2,3))+a.ones((3,2))", "reject"),
    ("matmul incompatible", "a.ones((2,3))@a.ones((2,3))", "reject"),
    ("empty matmul", "a.ones((2,0))@a.ones((0,3))", "return"),
    ("matmul scalar", "a.array(3)@a.array(4)", "reject"),
    ("singular core solve", "a.linalg.solve(a.zeros((2,2)),a.ones(2))", "reject"),
    ("nonsquare core solve", "a.linalg.solve(a.ones((2,3)),a.ones(2))", "reject"),
    ("mismatched core rhs", "a.linalg.solve(a.eye(2),a.ones(3))", "reject"),
    ("singular public solve", "q.linalg.solve(a.zeros((2,2)),a.ones(2))", "reject"),
    ("nonpositive cholesky", "q.linalg.cholesky(-a.eye(3))", "reject"),
    ("mismatched public rhs", "q.linalg.solve(a.eye(2),a.ones(3))", "reject"),
    ("empty core fft", "a.fft.fft(a.array([]))", "reject"),
    ("negative core fft length", "a.fft.fft(a.ones(4),n=-1)", "reject"),
    ("empty public fft", "q.transforms.fft([])", "either"),
    ("invalid DCT type", "q.transforms.dct(a.ones(4),kind=9)", "reject"),
    ("negative RNG scale", "a.random.default_rng(1).normal(scale=-1,size=2)", "reject"),
    ("invalid RNG range", "a.random.default_rng(1).integers(3,1,size=2)", "reject"),
    ("invalid RNG probability", "a.random.default_rng(1).binomial(3,2,size=2)", "reject"),
    ("negative sparse dimension", "q.linalg.identity_sparse(-1)", "reject"),
    ("unbracketed root", "q.rootfind.brent(lambda x:x*x+1,-1,1)", "reject"),
    ("negative ODE tolerance", "q.ode.solve_ivp(lambda t,y:-y,(0,1),[1],rtol=-1)", "reject"),
    ("nan ODE tolerance", "q.ode.solve_ivp(lambda t,y:-y,(0,1),[1],rtol=float('nan'))", "reject"),
    ("nonfinite ODE span", "q.ode.solve_ivp(lambda t,y:-y,(0,float('inf')),[1])", "reject"),
    ("mismatched ODE callback", "q.ode.solve_ivp(lambda t,y:a.ones(3),(0,1),[1,2])", "reject"),
    ("gamma pole documented infinity", "q.special.gamma(-2.)", "return"),
    ("invalid erf inverse", "q.special.erfinv(2.)", "reject"),
    ("invalid elliptic domain", "q.special.elliptic_k(2.)", "reject"),
    ("invalid logarithmic integral", "q.special.exponential_integral(0.)", "reject"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.)
    parser.add_argument("--no-memory-limit", action="store_true", help="ASan reserves a huge virtual address range")
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    results = []
    for name, expression, expected in CASES:
        preamble = "import resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
        if not args.no_memory_limit:
            preamble += "resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))\n"
        program = preamble + f"import sys,json;sys.path.insert(0,{str(ROOT)!r})\nimport quadrivium as q\nfrom quadrivium import numeric as a\ntry:\n result=eval({expression!r})\n print(json.dumps({{'outcome':'return','value':repr(result)}}))\nexcept Exception as exc:\n print(json.dumps({{'outcome':'reject','exception':type(exc).__name__,'message':str(exc)}}))\n"
        try:
            proc = subprocess.run([sys.executable,"-X","faulthandler","-c",program], env=env, capture_output=True,text=True,timeout=args.timeout)
            result = json.loads(proc.stdout) if proc.returncode == 0 else {"outcome":"crash", "returncode":proc.returncode}
            result["stderr"] = proc.stderr[-12000:]
        except subprocess.TimeoutExpired:
            result = {"outcome":"timeout"}
        result.update(case=name, expression=expression, expected=expected)
        result["passed"] = result["outcome"] == expected or (expected == "either" and result["outcome"] in ("return","reject"))
        results.append(result)
        print(name,result["outcome"],flush=True)
        args.output.write_text(json.dumps({"root":str(ROOT),"timeout":args.timeout,"results":results},indent=2)+"\n")
    return int(any(not r["passed"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())

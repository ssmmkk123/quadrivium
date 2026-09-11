/* Reusable native factorizations and complex Hermitian kernels. */
static PyObject *py_lu_factor(PyObject *self, PyObject *arg) {
    (void)self; int n; QArray *a0=require_square(arg,"lu_factor",&n);
    if(!a0)return NULL;
    QArray *a=qnp_astype(a0,a0->dtype,1);Py_DECREF(a0);if(!a)return NULL;
    qintp nn=n; QArray *p=qnp_new(1,&nn,QNP_INT64);
    int *piv=malloc((size_t)(n?n:1)*sizeof(int));
    if(!p||!piv){Py_XDECREF(p);Py_DECREF(a);free(piv);return PyErr_NoMemory();}
    int rc,sign;
    Py_BEGIN_ALLOW_THREADS
    rc=a->dtype==QNP_COMPLEX128?lu_decomp_c((qcomplex *)a->data,n,piv,&sign):lu_decomp_d((double *)a->data,n,piv,&sign);
    for(int i=0;i<n;i++)((int64_t *)p->data)[i]=piv[i];
    Py_END_ALLOW_THREADS
    free(piv);
    if(rc<0){Py_DECREF(a);Py_DECREF(p);PyErr_SetString(QNP_LinAlgError,"Singular matrix");return NULL;}
    return Py_BuildValue("NN",a,p);
}
static PyObject *py_lu_solve(PyObject *self, PyObject *args) {
    (void)self;PyObject *fac,*rhs,*ao,*po;
    if(!PyArg_ParseTuple(args,"OO:lu_solve",&fac,&rhs)||!PyArg_ParseTuple(fac,"OO:factor",&ao,&po))return NULL;
    int n;QArray *a=require_square(ao,"lu_solve",&n);if(!a)return NULL;
    QArray *p=qnp_from_any(po,QNP_INT64,1),*b0=qnp_from_any(rhs,-1,0),*b=NULL;
    int *piv=NULL;
    if(!p||!b0)goto fail;
    if(p->nd!=1||p->shape[0]!=n|| (b0->nd!=1&&b0->nd!=2)||b0->shape[0]!=n){PyErr_SetString(PyExc_ValueError,"incompatible packed LU or right-hand side dimensions");goto fail;}
    int dt=qnp_promote(a->dtype,b0->dtype);
    QArray *wide=qnp_astype(a,dt,0);Py_DECREF(a);a=wide;if(!a)goto fail;
    b=qnp_astype(b0,dt,1);if(!b)goto fail;
    piv=malloc((size_t)(n?n:1)*sizeof(int));if(!piv){PyErr_NoMemory();goto fail;}
    for(int i=0;i<n;i++){
        int64_t v=*(int64_t *)(p->data+i*p->strides[0]);
        if(v<i||v>=n){PyErr_SetString(PyExc_ValueError,"invalid LU pivot index");goto fail;}piv[i]=(int)v;
        if(qc_abs(qnp_read_number(a->data+(i*n+i)*QNP_ITEMSIZE(dt),dt))==0){PyErr_SetString(QNP_LinAlgError,"Singular matrix");goto fail;}
    }
    int nrhs=b->nd==1?1:(int)b->shape[1];
    Py_BEGIN_ALLOW_THREADS
    if(dt==QNP_COMPLEX128)lu_solve_c((qcomplex *)a->data,n,piv,(qcomplex *)b->data,nrhs);
    else lu_solve_d((double *)a->data,n,piv,(double *)b->data,nrhs);
    Py_END_ALLOW_THREADS
    free(piv);Py_DECREF(a);Py_DECREF(p);Py_DECREF(b0);return (PyObject *)b;
fail:free(piv);Py_XDECREF(a);Py_XDECREF(p);Py_XDECREF(b0);Py_XDECREF(b);return NULL;
}
static PyObject *complex_cholesky(QArray *a0) {
    int n=(int)a0->shape[0];QArray *out=qnp_astype(a0,QNP_COMPLEX128,1);if(!out)return NULL;
    qcomplex *L=(qcomplex *)out->data;int failed=0;
    Py_BEGIN_ALLOW_THREADS
    for(int i=0;i<n&&!failed;i++){
        for(int j=0;j<=i;j++){
            qcomplex s=L[i*n+j];if(i==j)s.im=0;
            for(int k=0;k<j;k++)s=qc_sub(s,qc_mul(L[i*n+k],qc_conj(L[j*n+k])));
            if(i==j){if(!(s.re>0)||!isfinite(s.re)){failed=1;break;}L[i*n+j]=qc(sqrt(s.re),0);}
            else L[i*n+j]=qc(s.re/L[j*n+j].re,s.im/L[j*n+j].re);
        }
        for(int j=i+1;j<n;j++)L[i*n+j]=qc(0,0);
    }
    Py_END_ALLOW_THREADS
    if(failed){Py_DECREF(out);PyErr_SetString(QNP_LinAlgError,"Matrix is not positive definite");return NULL;}
    return (PyObject *)out;
}
static PyObject *complex_eigh(QArray *a0,int vectors) {
    int n=(int)a0->shape[0];qintp nn=n,sh[2]={n,n};
    QArray *a=qnp_astype(a0,QNP_COMPLEX128,1),*v=qnp_new(2,sh,QNP_COMPLEX128),*w=qnp_new(1,&nn,QNP_FLOAT64);
    if(!a||!v||!w){Py_XDECREF(a);Py_XDECREF(v);Py_XDECREF(w);return NULL;}
    qcomplex *h=(qcomplex *)a->data,*z=(qcomplex *)v->data;int converged=n<2;
    double scale=0;
    for(int i=0;i<n;i++)for(int j=0;j<=i;j++)scale=fmax(scale,qc_abs(h[i*n+j]));
    if(!isfinite(scale)){Py_DECREF(a);Py_DECREF(v);Py_DECREF(w);PyErr_SetString(QNP_LinAlgError,"eigh input must be finite");return NULL;}
    Py_BEGIN_ALLOW_THREADS
    memset(z,0,(size_t)n*n*sizeof(qcomplex));
    for(int i=0;i<n;i++){
        z[i*n+i]=qc(1,0);
        for(int j=0;j<=i;j++){h[i*n+j]=scale?qc(h[i*n+j].re/scale,h[i*n+j].im/scale):qc(0,0);h[j*n+i]=qc_conj(h[i*n+j]);}
        h[i*n+i].im=0;
    }
    for(int sweep=0;sweep<100&&!converged;sweep++){
        double off=0;
        for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){
            double r=qc_abs(h[p*n+q]);off=fmax(off,r);
            if(r<2e-16)continue;
            double tau=(h[q*n+q].re-h[p*n+p].re)/(2*r);
            double t=copysign(1.0,tau)/(fabs(tau)+hypot(1,tau)),c=1/sqrt(1+t*t),s=t*c;
            qcomplex phase=qc(h[p*n+q].re/r,h[p*n+q].im/r);
            h[p*n+p].re-=t*r;h[q*n+q].re+=t*r;h[p*n+q]=h[q*n+p]=qc(0,0);
            for(int k=0;k<n;k++){
                if(k!=p&&k!=q){
                    qcomplex x=h[k*n+p],y=h[k*n+q];
                    h[k*n+p]=qc_sub(qc(c*x.re,c*x.im),qc_mul(qc(s*phase.re,-s*phase.im),y));
                    h[k*n+q]=qc_add(qc_mul(qc(s*phase.re,s*phase.im),x),qc(c*y.re,c*y.im));
                    h[p*n+k]=qc_conj(h[k*n+p]);h[q*n+k]=qc_conj(h[k*n+q]);
                }
                qcomplex x=z[k*n+p],y=z[k*n+q];
                z[k*n+p]=qc_sub(qc(c*x.re,c*x.im),qc_mul(qc(s*phase.re,-s*phase.im),y));
                z[k*n+q]=qc_add(qc_mul(qc(s*phase.re,s*phase.im),x),qc(c*y.re,c*y.im));
            }
        }
        converged=off<2e-15;
    }
    for(int i=0;i<n;i++)((double *)w->data)[i]=h[i*n+i].re*scale;
    for(int i=0;i<n;i++){
        int at=i;for(int j=i+1;j<n;j++)if(((double *)w->data)[j]<((double *)w->data)[at])at=j;
        if(at!=i){double t=((double *)w->data)[i];((double *)w->data)[i]=((double *)w->data)[at];((double *)w->data)[at]=t;
            for(int k=0;k<n;k++){qcomplex t=z[k*n+i];z[k*n+i]=z[k*n+at];z[k*n+at]=t;}}
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(a);
    if(!converged){Py_DECREF(w);Py_DECREF(v);PyErr_SetString(QNP_LinAlgError,"Hermitian eigensolver did not converge");return NULL;}
    if(!vectors){Py_DECREF(v);return (PyObject *)w;}
    return Py_BuildValue("NN",w,v);
}
/* Orthonormal completion only for zero singular vectors and full factors. */
static void complete_complex_columns(qcomplex *u,int m,int cols,int first) {
    for(int j=first;j<cols;j++){
        double norm=0;
        for(int seed=0;seed<m;seed++){
            for(int i=0;i<m;i++)u[i*cols+j]=qc(i==seed,0);
            for(int pass=0;pass<2;pass++)for(int k=0;k<j;k++){
                qcomplex d=qc(0,0);for(int i=0;i<m;i++)d=qc_add(d,qc_mul(qc_conj(u[i*cols+k]),u[i*cols+j]));
                for(int i=0;i<m;i++)u[i*cols+j]=qc_sub(u[i*cols+j],qc_mul(u[i*cols+k],d));
            }
            norm=0;for(int i=0;i<m;i++)norm=hypot(norm,qc_abs(u[i*cols+j]));
            if(norm>1e-12)break;
        }
        if(norm)for(int i=0;i<m;i++)u[i*cols+j]=qc(u[i*cols+j].re/norm,u[i*cols+j].im/norm);
    }
}
static PyObject *complex_svd(QArray *a,int full,int uv) {
    int am=(int)a->shape[0],an=(int)a->shape[1],wide=am<an,m=wide?an:am,n=wide?am:an;
    qintp sh[2]={m,n},vhsh[2]={n,n},sn=n;
    QArray *work=qnp_new(2,sh,QNP_COMPLEX128),*v=qnp_new(2,vhsh,QNP_COMPLEX128),*vals=qnp_new(1,&sn,QNP_FLOAT64);
    if(!work||!v||!vals){Py_XDECREF(work);Py_XDECREF(v);Py_XDECREF(vals);return NULL;}
    qcomplex *b=(qcomplex *)work->data,*z=(qcomplex *)v->data,*src=(qcomplex *)a->data;
    double scale=0;for(int i=0;i<am;i++)for(int j=0;j<an;j++)scale=fmax(scale,qc_abs(src[i*an+j]));
    if(!isfinite(scale)){Py_DECREF(work);Py_DECREF(v);Py_DECREF(vals);PyErr_SetString(QNP_LinAlgError,"svd input must be finite");return NULL;}
    int converged=n<2; double *s=(double *)vals->data;
    Py_BEGIN_ALLOW_THREADS
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){qcomplex t=wide?qc_conj(src[j*an+i]):src[i*an+j];b[i*n+j]=scale?qc(t.re/scale,t.im/scale):qc(0,0);}
    memset(z,0,(size_t)n*n*sizeof(qcomplex));for(int i=0;i<n;i++)z[i*n+i]=qc(1,0);
    for(int sweep=0;sweep<100&&!converged;sweep++){
        converged=1;
        for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){
            double aa=0,bb=0;qcomplex ab=qc(0,0);
            for(int i=0;i<m;i++){qcomplex x=b[i*n+p],y=b[i*n+q];aa+=x.re*x.re+x.im*x.im;bb+=y.re*y.re+y.im*y.im;ab=qc_add(ab,qc_mul(qc_conj(x),y));}
            double r=qc_abs(ab);if(r<=2e-15*sqrt(aa)*sqrt(bb)||r==0)continue;
            converged=0;double tau=(bb-aa)/(2*r),t=copysign(1.,tau)/(fabs(tau)+hypot(1,tau)),c=1/sqrt(1+t*t),s=t*c;
            qcomplex phase=qc(ab.re/r,ab.im/r);
            for(int i=0;i<m;i++){qcomplex x=b[i*n+p],y=b[i*n+q];b[i*n+p]=qc_sub(qc(c*x.re,c*x.im),qc_mul(qc(s*phase.re,-s*phase.im),y));b[i*n+q]=qc_add(qc_mul(qc(s*phase.re,s*phase.im),x),qc(c*y.re,c*y.im));}
            for(int i=0;i<n;i++){qcomplex x=z[i*n+p],y=z[i*n+q];z[i*n+p]=qc_sub(qc(c*x.re,c*x.im),qc_mul(qc(s*phase.re,-s*phase.im),y));z[i*n+q]=qc_add(qc_mul(qc(s*phase.re,s*phase.im),x),qc(c*y.re,c*y.im));}
        }
    }
    for(int j=0;j<n;j++){s[j]=0;for(int i=0;i<m;i++)s[j]=hypot(s[j],qc_abs(b[i*n+j]));}
    for(int j=0;j<n;j++){
        int at=j;for(int k=j+1;k<n;k++)if(s[k]>s[at])at=k;
        if(at!=j){double t=s[j];s[j]=s[at];s[at]=t;for(int i=0;i<m;i++){qcomplex t=b[i*n+j];b[i*n+j]=b[i*n+at];b[i*n+at]=t;}for(int i=0;i<n;i++){qcomplex t=z[i*n+j];z[i*n+j]=z[i*n+at];z[i*n+at]=t;}}
    }
    Py_END_ALLOW_THREADS
    if(!converged){Py_DECREF(work);Py_DECREF(v);Py_DECREF(vals);PyErr_SetString(QNP_LinAlgError,"Complex SVD did not converge");return NULL;}
    if(!uv){for(int j=0;j<n;j++)s[j]*=scale;Py_DECREF(work);Py_DECREF(v);return (PyObject *)vals;}
    int uc=full?m:n;qintp ush[2]={m,uc};QArray *u=qnp_new(2,ush,QNP_COMPLEX128);
    if(!u){Py_DECREF(work);Py_DECREF(v);Py_DECREF(vals);return NULL;}
    qcomplex *up=(qcomplex *)u->data;int rank=0;
    Py_BEGIN_ALLOW_THREADS
    memset(up,0,(size_t)m*uc*sizeof(qcomplex));
    for(int j=0;j<n;j++){if(s[j]>0){for(int i=0;i<m;i++)up[i*uc+j]=qc(b[i*n+j].re/s[j],b[i*n+j].im/s[j]);rank++;}s[j]*=scale;}
    complete_complex_columns(up,m,uc,rank);
    Py_END_ALLOW_THREADS
    Py_DECREF(work);
    /* Adjoint of the right factor. For a wide input the factors exchange. */
    QArray *right=wide?u:v,*left=wide?v:u;
    qintp rsh[2]={right->shape[1],right->shape[0]};QArray *vh=qnp_new(2,rsh,QNP_COMPLEX128);
    if(!vh){Py_DECREF(u);Py_DECREF(v);Py_DECREF(vals);return NULL;}
    qcomplex *rp=(qcomplex *)right->data,*hp=(qcomplex *)vh->data;
    for(qintp i=0;i<rsh[0];i++)for(qintp j=0;j<rsh[1];j++)hp[i*rsh[1]+j]=qc_conj(rp[j*rsh[0]+i]);
    Py_DECREF(right);return Py_BuildValue("NNN",left,vals,vh);
}

"""
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# DESCRIPTION # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

	LSRN computes the min-length solution of linear least squares via LSQR with randomized preconditioning, currently for systems with
	m >> n (highly overdetermined). 
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# AUTHOR(S) # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	Original Author: 
	  Xiangrui Meng <mengxr@stanford.edu>
	  iCME, Stanford University

	Port to this file (and further edits): 
	  W. Ross Morrow <morrowwr@gmail.com>
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# TODO  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	1. make a spec/split for the lsqr routine here vs. scipy.sparse.linalg.lsqr (and do a test comparison); use cgls?
	2. replace time calls with timeit 
	3. include a "biggish" test, say with the medicare part D prescriber data
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

"""

# from exceptions import NotImplementedError # not needed, "exceptions" not clear

import unittest

from time import time, clock
from math import log, sqrt

import numpy as np
from numpy.linalg import norm, svd, lstsq
from numpy.random import randn

from scipy.sparse.linalg import aslinearoperator, LinearOperator, lsqr

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 
# routines
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def _gen_prob( m, n, cond, r = None ):

	""" docstring TBD """

	if r is None: r = np.min([m,n])

	U, = svd( randn( np.int64( m ) , np.int64( r ) ) , False )[:1]
	s  = np.linspace(1,cond,r)
	V, = svd( randn( np.int64( n ) , np.int64( r ) ) , False )[:1]

	A   = (U*s).dot(V.T)
	b   = A.dot( randn( np.int64( n ) ) )
	res = randn( np.int64( m ) )
	theta = 0.25;
	b    += theta*norm(b) + res/norm(res)

	x_opt = V.dot(U.T.dot(b)/s)

	return A, b, x_opt
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	
def cgls( A, b, tol = 1e-6, iter_lim = None ):
	
	"""
	CGLS without shift

	Parameters
	----------
	
	A : m-by-n {ndarray, matrix, sparse, LinearOperator}

	b : (m,) ndarray

	tol      : tolerance

	iter_lim : max number of iterations

	Returns
	-------

	x : (n,) ndarray, solution

	flag : int, 0 means cgls converged, 1 means cgls didn't converge

	itn : int, number of iterations
	"""
	
	A         = aslinearoperator(A)
	m, n      = A.shape

	x         = np.zeros(n)
	r         = b.squeeze().copy()
	nrm_r_0   = norm(r)
	s         = A.rmatvec(r)
	p         = s.copy()
	sq_s_0    = np.dot(s,s)
	nrm_s_0   = sqrt(sq_s_0)
	gamma     = sq_s_0

	x_best      = np.zeros( np.int64( n ) )
	relres_best = 1.0
	itn_best    = 0

	itn       = 0
	stag      = 0
	stag_lim  = 10
	converged = False

	if iter_lim is None: iter_lim = 2*np.min([m,n])
	
	while (not converged) and (itn < iter_lim):

		itn    += 1

		q       = A.matvec(p)
		sq_q    = np.dot(q,q)
		alpha   = gamma / sq_q
		x      += alpha*p
		r      -= alpha*q

		# nrm_r   = norm(r)
		# if (nrm_r_0-nrm_r)/nrm_r_0 < np.finfo(float).eps:
		#     break
		# nrm_r_0  = nrm_r

		s       = A.rmatvec(r)
		sq_s    = np.dot(s,s)
		nrm_s   = sqrt(sq_s)
		gamma_0 = gamma
		gamma   = sq_s
		beta    = gamma / gamma_0
		if beta > 1: break
		p       = s + beta*p

		relres  = nrm_s/nrm_s_0

		if relres < relres_best:
			relres_best = relres
			x_best      = x.copy()
			itn_best    = itn
			stag        = 0
		else: stag += 1

		if stag > stag_lim: break
		
		if relres < tol: converged = True

	flag = 1 - converged

	return x_best, flag, itn

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def ls_chebyshev( A, b, s_max, s_min, tol = 1e-8, iter_lim = None ):
    """
    Chebyshev iteration for linear least squares problems
    """

    A     = aslinearoperator(A)
    m, n  = A.shape
    
    d     = (s_max*s_max+s_min*s_min)/2.0
    c     = (s_max*s_max-s_min*s_min)/2.0

    theta   = (1.0-s_min/s_max)/(1.0+s_min/s_max) # convergence rate
    itn_est = np.ceil((log(tol)-log(2))/log(theta))
    if (iter_lim is None) or (iter_lim < itn_est) : iter_lim = itn_est

    alpha = 0.0
    beta  = 0.0

    r     = b.copy()
    x     = np.zeros( np.int64( n ) )
    v     = np.zeros( np.int64( n ) )

    # print( iter_lim )
    for k in range(np.int64(iter_lim)):

        if k == 0:
            beta  = 0.0
            alpha = 1.0/d
        elif k == 1:
            beta  = -1.0/2.0*(c*c)/(d*d)
            alpha =  1.0*(d-c*c/(2.0*d))
        else:
            beta  = -(c*c)/4.0*(alpha*alpha)
            alpha = 1.0/(d-(c*c)/4.0*alpha)

        v  = A.rmatvec(r) - beta*v
        x += alpha*v
        r -= alpha*A.matvec(v)

    return x

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def lsne( A, b, try_chol = True ):
	"""
	LSNE computes the min-length solution of linear least squares via the normal
	equation.

	Parameters 
	----------
	A        : {matrix, sparse matrix, ndarray, LinearOperator} of size m-by-n
	b        : (m,) ndarray
	try_chol : if True, LSNE will try Cholesky factorization on the normal equation, if
			   the system is rank deficient, an eigen-decomposition will be used instead.

	Returns
	-------
	x        : (m,) ndarray, the min-length solution
	r        : int, rank of A
	"""

	m , n = A.shape

	if m >= n:
		
		AtA = A.T.dot(A)
		if issparse(AtA):
			AtA = AtA.todense()

		Atb = A.T.dot(b)
		
		try:
			if try_chol:
				C  = cho_factor(AtA)
				x  = cho_solve(C,Atb)
				r  = n
			else: raise LinAlgError
		except LinAlgError:
			s2, V = eigh(AtA)
			eps   = np.finfo(float).eps
			tol   = m*s2[-1]*eps
			ii,   = np.where(s2>tol)
			r     = ii.size
			s2    = s2[ii]
			V     = V[:,ii]
			x     = V.dot((1.0/s2)*V.T.dot(Atb))

	else:

		AAt = A.dot(A.T)
		if issparse(AAt): AAt = AAt.todense() # it would be really nice to avoid this...
		
		try:
			if try_chol:
				C = cho_factor(AAt)
				y = cho_solve(C,b)
				x = A.T.dot(y)
				r = m
			else:
				raise LinAlgError
		except LinAlgError:
			s2, V = eigh(AAt)
			eps   = np.finfo(float).eps
			tol   = n*s2[-1]*eps
			ii,   = np.where(s2>tol)
			r     = ii.size
			s2    = s2[ii]
			V     = V[:,ii]
			y     = V.dot((1.0/s2)*V.T.dot(b))
			x     = A.T.dot(y)
	
	return x, r

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def simple_lsqr( A, b, tol=1e-14, iter_lim=None ):
	"""
	A simple version of LSQR
	"""

	A    = aslinearoperator(A)
	m, n = A.shape

	eps  = 32*np.finfo(float).eps;      # slightly larger than eps

	if tol < eps: tol = eps
	elif tol >= 1: tol = 1-eps

	u    = b.squeeze().copy()
	beta = norm(u)
	if beta != 0: u /= beta

	v     = A.rmatvec(u)
	alpha = norm(v)
	if alpha != 0: v /= alpha

	w     = v.copy()

	x     = np.zeros( np.int64( n ) )

	phibar = beta
	rhobar = alpha

	nrm_a    = 0.0
	cnd_a    = 0.0
	sq_d     = 0.0
	nrm_r    = beta
	nrm_ar_0 = alpha*beta

	if nrm_ar_0 == 0: return x, 0, 0

	nrm_x  = 0
	sq_x   = 0
	z      = 0
	cs2    = -1
	sn2    = 0

	max_n_stag = 3
	stag       = 0

	flag = -1
	if iter_lim is None : iter_lim = np.max( [20, 2*np.min([m,n])] )

	for itn in range( np.int64(iter_lim) ) :

		u    = A.matvec(v) - alpha*u
		beta = norm(u)
		u   /= beta

		# estimate of norm(A)
		nrm_a = sqrt(nrm_a**2 + alpha**2 + beta**2)

		v     = A.rmatvec(u) - beta*v
		alpha = norm(v)
		v    /= alpha

		rho    =  sqrt(rhobar**2+beta**2)
		cs     =  rhobar/rho
		sn     =  beta/rho
		theta  =  sn*alpha
		rhobar = -cs*alpha
		phi    =  cs*phibar
		phibar =  sn*phibar

		x     += (phi/rho)*w
		w      = v-(theta/rho)*w

		# estimate of norm(r)
		nrm_r   = phibar

		# estimate of norm(A'*r)
		nrm_ar  = phibar*alpha*np.abs(cs)

		# check convergence
		if nrm_ar < tol*nrm_ar_0:
			flag = 0
			break

		if nrm_ar < eps*nrm_a*nrm_r:
			flag = 0
			break

		# estimate of cond(A)
		sq_w    = np.dot(w,w)
		nrm_w   = sqrt(sq_w)
		sq_d   += sq_w/(rho**2)
		cnd_a   = nrm_a*sqrt(sq_d)

		# check condition number
		if cnd_a > 1/eps:
			flag = 1
			break

		# check stagnation
		if abs(phi/rho)*nrm_w < eps*nrm_x: stag += 1
		else: stag  = 0
		
		if stag >= max_n_stag:
			flag = 1
			break

		# estimate of norm(x)
		delta   =  sn2*rho
		gambar  = -cs2*rho
		rhs     =  phi - delta*z
		zbar    =  rhs/gambar
		nrm_x   =  sqrt(sq_x + zbar**2)
		gamma   =  sqrt(gambar**2 + theta**2)
		cs2     =  gambar/gamma
		sn2     =  theta /gamma
		z       =  rhs   /gamma
		sq_x   +=  z**2

	return x, flag, itn

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def nesterov( A, b, s_max, s_min, tol = 1e-8, iter_lim = None ):
	"""
	Nesterov's method for linear least squares problems
	
	INPUTS:
	-------

	A : (m,n) ndarray, matrix, or LinearOperator

	b : (m,) ndarray
	
	s_max : float, an upper bound of the largest singular value of A

	s_min : float, a lower bound of the smallest non-zero singular value of A

	tol : float, tolerance on relative error

	iter_lim : int, iteration limit

	OUTPUTS:
	--------

	x : (n,) ndarray, the approximate solution

	flag : int, 0 : approximately solved, 1 : [s_min, s_max] doesn't bound A's singular values
	
	itn : int, number of iterations
	"""

	A    = aslinearoperator(A)
	m, n = A.shape

	# def AtA_op_matvec(v): return A.rmatvec(A.matvec(v))
	AtA = LinearOperator( (np.int64(n),np.int64(n)), matvec = lambda v : A.rmatvec(A.matvec(v)) )
	Atb = A.rmatvec(b)

	x, flag, itn = quad_nesterov( AtA, Atb, s_max**2, s_min**2, tol, iter_lim )

	return x, flag, itn

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def quad_nesterov( A, b, L, mu, tol=1e-8, iter_lim = None ):
	"""
	Nesterov's method for symmetric and positive semi-definite linear systems or linear
	least squares problems.

	INPUTS:
	-------
	
	A : (n,n) ndarray, matrix, LinearOperator

	b : (n,) ndarray
	
	L : float, an upper bound of the largest eigenvalue of A

	mu : float, a lower bound of the smallest non-zero eigenvalue of A

	tol : float, tolerance on relative error

	iter_lim : int, iteration limit

	OUTPUTS:
	--------

	x : (n,) ndarray, the approximate solution

	flag : int, 0 : approximately solved, 1 : [mu, L] doesn't bound A's eigenvalues
	
	itn : int, number of iterations
	"""

	A    = aslinearoperator(A)
	n    = A.shape[0]

	alpha = 4.0/(3.0*L+mu)
	beta  = (1.0-sqrt(alpha*mu))/(1.0+sqrt(alpha*mu))
	rate  = 1.0-sqrt(alpha*mu)

	maxit = np.ceil(log(tol)/log(rate))
	if (iter_lim is None) or (iter_lim > maxit) : iter_lim = maxit

	x    = np.zeros( np.int64(n) )
	y    = np.zeros( np.int64(n) )

	for itn in range( np.int64(iter_lim) ) :
		x_p = x
		x   = y - alpha*(A.matvec(y)-b)
		y   = (1+beta)*x - beta*x_p

	if norm(x) <= norm(b)/mu: flag = 0
	else: flag = 1

	return y, flag, iter_lim

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def lsrn( A, b, gamma=2.0, tol=np.finfo(float).eps, rcond=-1, ls_solver=lsqr ):
	"""
	LSRN computes the min-length solution of linear least squares via LSQR with
	randomized preconditioning

	Parameters
	----------
	
	A       : {matrix, sparse matrix, ndarray, LinearOperator} of size m-by-n

	b       : (m,) ndarray

	gamma : float (>1), oversampling factor

	tol : float, tolerance such that norm(A*x-A*x_opt)<tol*norm(A*x_opt)

	rcond : float, reciprocal condition number

	Returns
	-------
	x      : (n,) ndarray, the min-length solution
	
	r      : int, the rank of A

	flag : int,

	itn : int, iteration number

	timing : dict, 
	"""

	m , n = A.shape

	if rcond < 0: rcond = np.min([m,n])*np.finfo(float).eps
	
	timing = { 'rand': 0.0, 'mult': 0.0, 'svd': 0.0, 'cg': 0.0 }

	if m > n:                           # over-determined

		s      = np.ceil(gamma*n)
		As     = np.zeros( (np.int64(s),np.int64(n)))
		blk_sz = 128
		for i in range( np.int64(np.ceil(1.0*s/blk_sz)) ):
			blk_begin = np.int64( i * blk_sz )
			blk_end   = np.int64( np.min([(i+1)*blk_sz,s]) )
			blk_len   = blk_end - blk_begin
			t = time()
			G = randn( blk_len , np.int64(m) )
			timing['rand'] += time() - t
			t = time()
			As[ blk_begin : blk_end , : ] = G.dot(A)
			timing['mult'] += time() - t

		t = time()
		U, S, V = svd(As, False)
		# determine the rank
		r_tol = S[0]*rcond
		r     = np.sum(S>r_tol)
		timing['svd'] += time() - t
		
		t = time()
		N = V[:r,:].T/S[:r]
		# def AN_op_matvec(v) : return A.dot(N.dot(v))
		# def AN_op_rmatvec(v): return N.T.dot(A.T.dot(v))
		AN_op = LinearOperator( (np.int64(m),np.int64(r)) , matvec = lambda v : A.dot(N.dot(v)) , rmatvec = lambda v : N.T.dot(A.T.dot(v)) )
		gamma    = 1.0*s/r                             # re-estimate gamma
		condest  = (sqrt(gamma)+1.0)/(sqrt(gamma)-1.0) # condition number of AN
		iter_lim = np.ceil(-2*(log(tol)-log(2))/log(gamma))
		if (ls_solver == lsqr) or (ls_solver == cgls):
			y, flag, itn = simple_lsqr( AN_op, b, tol=tol/condest, iter_lim=iter_lim )
			# NOTE: update to use scipy.sparse.linalg.lsqr, which should perform better (I'de assume)
			# NOTE: should we actually use cgls? Why was this blocked out? 
		elif ls_solver == nesterov:
			y, flag, itn = nesterov( AN_op, b, 1.0/(sqrt(s)-sqrt(r)), 1.0/(sqrt(s)+sqrt(r)), tol=tol )
		x = N.dot(y)
		timing['cg'] += time() - t
		
	else:

		raise NotImplementedError( 'Sorry, LSRN(A,...) for m x n A with m << n is not yet built.' )
		
	return x, r, flag, itn, timing

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# tests
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
# interesting trick I pulled from stack exchange for generic function name
# printing inside functions without using those function's names explicitly
def passmein( func ) :
	def wrapper( *args , **kwargs ) : return func( func , *args , **kwargs )
	return wrapper
	
class LSRNTests( unittest.TestCase ) :
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	@passmein
	def test_genprob( me , self ):

		""" test genprob """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		m     = 1e4
		n     = 1e2
		r     = np.ceil(n/2)
		cond  = 10.0
		rcond = 1.0/(10.0*cond);

		A, b, x_opt = _gen_prob(m,n,cond,r)

		x_lstsq, res, r_lstsq = lstsq(A,b,rcond)[:3]
		if not np.allclose(x_lstsq, x_opt):
			sys.exit( "Over-determined test failed." )

		A, b, x_opt = _gen_prob(n,m,cond,r) 

		x_lstsq, res, r_lstsq = lstsq(A,b,rcond)[:3]
		if not np.allclose(x_lstsq, x_opt): sys.exit( "Under-determined test failed." )

		print( "All tests passed." )
		print( '\n' )
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

	@passmein
	def test_cgls( me , self ):
	
		""" testing CGLS """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		m = np.int64( 3e2 )
		n = np.int64( 1e2 )

		A = randn(m,n)
		b = randn(m)

		x_opt, = lstsq(A,b)[:1]

		tol      = 1e-14
		iter_lim = 2*np.ceil(log(tol)/log(n/m));

		x, flag, itn = cgls(A,b,0,10*iter_lim)
		relerr       = norm(x-x_opt)/norm(x_opt)

		if flag == 0: print( "CGLS converged in %d/%d iterations." % (itn,iter_lim) )
		else: print( "CGLS didn't converge in %d/%d iterations." % (itn,iter_lim) )

		if relerr < tol: print( "CGLS test passed with relerr %G." % (relerr,) )
		else: print( "CGLS test failed with relerr %G." % (relerr,) )
		
		print( '\n' )

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	@passmein
	def test_chebyshev( me , self ):
	
		""" testing chebyshev """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )
		
		m      = np.int64( 100 )
		n      = np.int64( 400 )
		A      = np.random.randn(m,n)
		s_max  = 1.02*(sqrt(n)+sqrt(m))
		s_min  = 0.98*(sqrt(n)-sqrt(m))
		b      = np.random.randn(m)
		x_opt, = lstsq(A,b)[:1]

		tol    = 1e-14
		x      = ls_chebyshev(A,b,s_max,s_min,tol)

		relerr = norm(x-x_opt)/norm(x_opt)

		print( relerr )
		
		print( '\n' )

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	@passmein		
	def test_lsqr( me , self ) : 
	
		""" testing LSQR """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )

		m = 1e2
		n = 1e4
		r = 80
		c = 1e3                            # well-conditioned

		A, b, x_opt = _gen_prob( m, n, c, r )
		
		tol      = 1e-14
		iter_lim = 400 # np.ceil( (log(tol)-log(2.0))/log((c-1.0)/(c+1.0)) )

		x, flag, itn = simple_lsqr(A,b,tol/c,iter_lim)
		relerr       = norm(x-x_opt)/norm(x_opt)

		if flag == 0: print( "LSQR converged in %d iterations." % (itn,) )
		else: print( "LSQR didn't converge in %d iterations." % (itn,) )

		if relerr < 1e-10: print( "LSQR test passed with relerr %G." % (relerr,) )
		else: print( "LSQR test failed with relerr %G." % (relerr,) ) 
		
		print( '\n' ) 

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

	@passmein
	def test_nesterov( me , self ):
	
		""" testing nesterov """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )

		m      = 100
		n      = 200
		A      = np.random.randn(m,n)
		s_max  = sqrt(n)+sqrt(m)
		s_min  = sqrt(n)-sqrt(m)
		b      = np.random.randn(m)
		x_opt, = lstsq(A,b)[:1]

		tol          = 1e-12
		x, flag, itn = nesterov(A,b,s_max,s_min,tol)[:3]

		relerr = norm(x-x_opt)/norm(x_opt)

		print( relerr, flag, itn )
		
		print( '\n' )

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

	@passmein
	def test_lsrn( me , self ):
	
		""" testing LSRN """
		print( "%s: %s" % ( me.__name__ , me.__doc__ ) )

		m     = 1e4
		n     = 1e2
		r     = 50
		c     = 1e6
		gamma = 3
		tol   = 1e-14

		A, b, x_opt = _gen_prob( m, n, c, r )

		x, r, flag, itn, timing = lsrn( A, b, gamma=gamma, tol=tol, ls_solver=lsqr )

		print( 'rank: %d' % (r,) )
		print( 'iter: %d' % (itn,) )
		print( 'flag: %d' % (flag,) )

		relerr = norm(x-x_opt)/norm(x_opt)
		print( 'relerr: %G' % (relerr,) )
		relerr_AtA = norm(A.dot(x-x_opt))/norm(A.dot(x_opt))
		print( 'relerr_AtA: %G' % (relerr_AtA,) )
		
		print( '\n' )

	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == '__main__':
	unittest.main()

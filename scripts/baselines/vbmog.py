import numpy as np
from scipy.special import logsumexp, gammaln, digamma
log = lambda x: np.log(x + 1e-10)

class VBMOG(object):
    def __init__(self, K,
            a=1.0,
            b=1.0,
            tau=1e-2,
            alpha=1.0):
        self.K = K
        self.a = a
        self.b = b
        self.tau = tau
        self.alpha = alpha

    def init(self, X):
        N, D = X.shape
        K = self.K
        self.m = X.mean(0)
        self.R = np.random.rand(N, K)
        self.R /= self.R.sum(-1, keepdims=True)

    def iterate(self, X):
        N, D = X.shape
        K = self.K

        hN = self.R.sum(0)
        self.valpha = self.alpha + hN
        self.va = self.a + 0.5*hN
        self.vtau = self.tau + hN
        self.vm = ((self.tau*self.m)[None] + np.dot(self.R.T, X)) / self.vtau[:,None]
        self.vb = self.b + 0.5*np.dot(self.R.T, X*X) \
                + 0.5*(self.tau*self.m**2)[None] - 0.5*self.vtau[:,None]*self.vm**2

        log_R = np.zeros((N, K))
        log_R += digamma(self.valpha) - digamma(self.valpha.sum())
        log_R += -0.5*D/self.vtau
        va, vb = self.va[None,:,None], self.vb[None]
        log_R += 0.5*(digamma(va) - log(vb) - (va/vb)*(X[:,None]-self.vm[None])**2).sum(-1)
        log_R -= logsumexp(log_R, -1)[:,None]
        self.R = np.exp(log_R)

    def compute_elbo(self, X):
        N, D = X.shape
        K = self.K
        elbo = -0.5*N*D*log(2*np.pi) + 0.5*D*K
        elbo += gammaln(K*self.alpha) - gammaln(self.valpha.sum())
        elbo += (gammaln(self.valpha) - gammaln(self.alpha)).sum()
        elbo += -(self.R*log(self.R)).sum()
        elbo += 0.5*D*(log(self.tau) - log(self.vtau)).sum()

        va, vb = self.va[:,None], self.vb
        elbo -= (va*log(vb) - va - gammaln(va) \
                - self.a*log(self.b) + gammaln(self.a)).sum()
        return elbo

    def run(self, X, verbose=True, num_iters=500):
        self.init(X)
        elbo_prev = 0
        for i in range(num_iters):
            self.iterate(X)
            elbo = self.compute_elbo(X)
            if verbose and (i+1)%100 == 0:
                print ('step %d, elbo %f' % (i+1, elbo))
            if abs((elbo - elbo_prev)/elbo) < 1e-8:
                if verbose:
                    print ('step %d converged, elbo %f' % (i+1, elbo))
                break
            else:
                elbo_prev = elbo

    def labels(self):
        return self.R.argmax(-1)

    def loglikel(self, X):
        pi = self.valpha / self.valpha.sum()
        mu = self.vm
        sigma2 = self.vb / (self.va[:,None]-1 + 1e-10)
        ll = -0.5*(X[:,None] - mu)**2/sigma2 - 0.5*np.log(2*np.pi) - 0.5*np.log(sigma2)
        ll = ll.sum(-1) + np.log(pi + 1e-10)
        ll = logsumexp(ll, -1).mean()
        return ll

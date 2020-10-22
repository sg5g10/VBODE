import numpy as np
import scipy.stats

class Priors(object):

        def __init__(self, priors):

            self.dimension = len(priors)

            self.prior_dist = []
            self.hyperparams = []

            for p in priors:

                self.prior_dist.append(p[0])
                self.hyperparams.append(p[1])

        def sample(self):

            samples = np.zeros(self.dimension)
            for i,d in enumerate(self.prior_dist):

                if d == 'uniform':
                    samples[i] = float(np.random.uniform(low=self.hyperparams[i][0],high=self.hyperparams[i][1],size=1))

                elif d == 'gamma':
                    alpha = self.hyperparams[i][0]
                    beta = 1./self.hyperparams[i][1]
                    samples[i] =  float(np.random.gamma(alpha,beta))

                elif d == 'normal':
                    samples[i] = float(np.random.normal(self.hyperparams[i][0],self.hyperparams[i][1],size=1))

                elif d == 'beta':
                    samples[i] = float(np.random.beta(self.hyperparams[i][0],self.hyperparams[i][1],size=1))
                
                elif d == 'halfnormal':
                    samples[i] = float(scipy.stats.halfnorm(self.hyperparams[i][0]).rvs())                


            return samples

        def priorproduct(self, theta):

            prob = np.zeros(self.dimension)

            for i,d in enumerate(self.prior_dist):

                if d == 'uniform':
                    support = self.hyperparams[i][1] - self.hyperparams[i][0]
                    prob[i] = scipy.stats.uniform.pdf(theta[i], loc=self.hyperparams[i][0], scale=support)

                elif d == 'gamma':
                    alpha = self.hyperparams[i][0]
                    beta = 1./self.hyperparams[i][1]
                    prob[i] = scipy.stats.gamma.pdf(theta[i], alpha, scale=beta)

                elif d == 'normal':
                    prob[i] = scipy.stats.norm.pdf(theta[i], loc = self.hyperparams[i][0],scale=self.hyperparams[i][1])

                elif d == 'beta':
                    prob[i] = scipy.stats.beta.pdf(theta[i], a = self.hyperparams[i][0],b=self.hyperparams[i][1])

                elif d == 'halfnormal':
                    prob[i] = scipy.stats.halfnorm.pdf(theta[i], scale=self.hyperparams[i][0])
            return np.prod(prob)

import numpy as np
import scipy
#from sklearn.covariance import ledoit_wolf
import sys
from .priors import *
from .kernels import *

class ABCSMC(object):
	# The ABC-SMC algorithm essentially follows the pseudocode in
	# Algorithm 1 of "On optimality of kernels for approximate Bayesian computation using sequential Monte Carlo. S. Filippi"
	# Right now only incorporates an Optimal Global Covariance kernel, equation 13 in the paper above.

	# Inputs to the algorithm:

	# nparam -- Number of model parameters to be inferred
	# npart -- Number of particles
	# data -- Data to be used for inference
	# niters -- Number of SMC steps
	# priors -- A list of prior distributions for the parameters. See `testabc.py`
	# simulator -- A function that generates simulated data
	# dfunc -- A distance function which receives the data and simulation and calulates a distance
	# schedule -- A list/array of tolerances. Alternatively just the endpoints can be specified
	# quantile -- If a quantile is specified an adaptive schedule will be used where the tolerance
	#			  for the 't'-th SMC step will be calculated as the supplied percentile (quantile) 
	#             of the distances of all particles from last step 't-1'

	def __init__(self, nparam, npart, data, niter, priors, simulator, dfunc, schedule, quantile=None):

		self.data = data

		if npart < nparam + 1:
			raise ValueError('Not enough particles')
		else:
			self.npart = npart

		self.nparam = nparam
		
		if not(quantile == None) and niter < 2:
			raise ValueError('Requires more iterations')

		elif not(quantile == None):
			quantile = float(quantile)
			self.adapt = True
			self.quantile = quantile
			self.niter = niter
			self.epsilon = np.tile(schedule[0],niter)
			if len(schedule) > 1:
				self.tmin = schedule[-1]#give some other name
			else:
				self.tmin = 0.0

		if (quantile == None) and len(schedule) < 2:
			raise ValueError('Atleast two tolerances required')

		elif (quantile == None) and len(schedule) > 2:
			self.epsilon = np.asarray(schedule)
			self.adapt = False
			self.niter = len(schedule)

		elif (quantile == None):
			if schedule[1] >= schedule[0]:
				raise ValueError('Schedule endpoint must be smaller than start')
			
			if niter < 2:
				raise ValueError('Requires more iterations')

			self.epsilon = np.linspace(schedule[0], schedule[1], num=niter)
			self.adapt = False
			self.niter = niter

		if not(self.adapt):
				self.tmin = 0.0
		self.simulator = simulator
		self.dfunc = dfunc


		self.theta=np.zeros([self.niter,self.npart,self.nparam])
		self.wt=np.zeros([self.niter,self.npart])
		self.delta=np.zeros([self.niter,self.npart])
		self.bt = np.zeros([self.niter,self.npart])

		self.pert_kernel = FilippiOCM(nparam,npart)

		self.verbose = True

		if not(len(priors)==nparam):
			raise ValueError('Priors must be specified for all parameters')
		else:
			self.priors = Priors(priors)


		self.end_sampling = False


	def dist(self, x):
		return self.dfunc(self.data,x)


	def next_epsilon(self,t):
		new_epsilon = np.percentile(self.delta[t], self.quantile)

		if new_epsilon < self.tmin:
			new_epsilon = self.tmin
		return new_epsilon


	def calculate_weight(self, t, Pid, covariance):
		kernelPdf = [scipy.stats.multivariate_normal(mean=self.theta[t-1,p],cov=covariance).pdf( \
			self.theta[t,Pid]) for p in range(self.npart)]

		if  np.any(self.wt[t-1]) ==0 or np.any(kernelPdf)==0:
			print ("Kernel or weights error", kernelPdf, self.wt[t-1])
			sys.exit(1)

		priorproduct = self.priors.priorproduct(self.theta[t,Pid])
		return (priorproduct*self.bt[t,Pid])/(np.sum(self.wt[t-1]*kernelPdf))


	def calculate_covariance(self, t):
		covariance = self.pert_kernel.covariance(t, self.theta[t-1], self.delta[t-1], self.epsilon[t], self.wt[t-1])
		
		if np.linalg.det(covariance) <1.E-15:
			covariance  =  ledoit_wolf(self.theta[t-1])[0]
		
		return covariance


	def sample(self):
		t = 0
		while self.end_sampling == False:

			if  t == self.niter or self.epsilon[t] == self.tmin:
				self.end_sampling = True
				return self.theta[t-1]

			if t==0:
				for p in range(self.npart):
					self.theta[t,p], self.delta[t,p], self.bt[t,p] = self.stepper(t, p)

				self.wt[t,:] = self.bt[t,:]/np.sum(self.bt[t,:])
				if self.verbose:
						print ('Stage: ',t,'tol: ',self.epsilon[t],'Params: ',[np.mean(self.theta[t,:,i]) for i in range(self.nparam)])

				if self.adapt:
					self.epsilon[t+1] = self.next_epsilon(t)
				t += 1

			else:
				### TODO: For OLCM, don't calculate covariance here do it inside stepper
				covariance = self.calculate_covariance(t)

				for p in range(self.npart):
					# TODO: For OLCM return the OLCM covariance for particle p, and use this covariance to calculate wieght of theta[t,p]
					self.theta[t,p], self.delta[t,p], self.bt[t,p] = self.stepper(t, p, covariance) # TODO: For OLCM don't pass covariance here
					self.wt[t,p] = self.calculate_weight(t, p, covariance)

				self.wt[t,:] = self.wt[t,:]/np.sum(self.wt[t,:])

				if self.verbose:
					print( 'Step: ',t,'epsilon-t: ',self.epsilon[t],'Params: ',[np.mean(self.theta[t,:,i]) for i in range(self.nparam)])

				if self.adapt and t <self.niter-1:
						self.epsilon[t+1] = self.next_epsilon(t)
				t += 1

	def stepper(self, t, Pid, covariance=None):
		while True:
			if t ==0: 
				
				theta_star = self.priors.sample()
				x = self.simulator(theta_star)
				rho = self.dist(x)

			else:
	            
				ispart = int(np.random.choice(self.npart,size=1,p=self.wt[t-1]))
				theta_old = self.theta[t-1,ispart]
				# TODO: For OLCM calculate covariance here and return back this covariance after perturbation
				theta_star = scipy.stats.multivariate_normal.rvs(mean=theta_old,cov=covariance,size=1)
				x = self.simulator(theta_star)
				rho = self.dist(x)

			rho_lt_epsilon = np.where(rho<=self.epsilon[t])[0]
			bt = len(rho_lt_epsilon)/len(rho)
			if bt>0:
				break
		return theta_star, rho.min(), bt

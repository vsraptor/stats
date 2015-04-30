import numpy as np

class stats():

	"""
		Calculate the deviations in a sample.
	"""
	@staticmethod
	def dev(xs): return xs - np.mean(xs)

	"""
		Calculate the covariance between two data sets.
			sample=True : sample covarince
			sample=False: population covariance
	"""
	@staticmethod
	def cov(xs,ys,sample=True):
		dec = 1 if sample else 0 #if sample-cov decrement len by 1
		#sum of products of deviations
		return np.dot( stats.dev(xs), stats.dev(ys) ) / (len(xs) - dec)

	@staticmethod
	def var(xs,sample=True):
		dec = 1 if sample else 0 #if sample-cov decrement len by 1
		return np.sum(stats.dev(xs)**2) / (len(xs) - dec)

	@staticmethod
	def std_dev(xs,sample=True):
		return np.sqrt( stats.var(xs, sample) )

	"""
		Calculate Pearson correlation.
	"""
	@staticmethod
	def corr(xs,ys,sample=True) :
		varx = stats.var(xs)
		vary = stats.var(ys)

		corr = stats.cov(xs,ys,sample)/ np.sqrt(varx * vary)
		return corr

	@staticmethod
	def rank(xs): return np.argsort(np.argsort(xs))
	@staticmethod
	def spearman_corr(xs,ys,sample=True):
		xranks = stats.rank(xs)
		yranks = stats.rank(ys)
		return stats.corr(xranks,yranks,sample)

	@staticmethod
	def r2(ys,residuals): #coef of determination
		return ( 1 - stats.var(residuals) ) / stats.var(ys)

	"""
		Calculate auto correlation coefficients for a time series or list of values.
			lag : specifies up to how many lagging coef will be calculated.
				!! The lag should be at most the lenght of the data minus 2, we skip lag-zero.
	"""
	@staticmethod
	def auto_corr(xs,lag=1,sample=True):
		if lag > len(xs) - 2 : raise Exception("Lag(%s) is bigger than the data-len(%s) - 2" % (lag,len(xs)))
		ac = np.zeros(lag)
		for i in range(1, lag+1) :
			ac[i-1] = stats.corr(xs[:-i], xs[i:], sample)
		return ac



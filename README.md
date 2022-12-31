# HOC: Spike trains with higher order correlations                                                                         

In this project, Python programs are developed to implement algorithms of: 
1. Generating correlated Poissonian spike trains with higher-order correlations. 
2. Creating non-Poissonian spike trains via a renewal process with associated hazard function. Spikes are rejected properly so that the ISI distribution follows the desired lognormal distribution. 
3. Decompound correlation structures from population activities via two methods: EDP [3] and fitting log amplitude characteristic functions which avoids computing with complex numbers.


The whole workflow is displayed as following graph:

<img src="https://user-images.githubusercontent.com/56949661/210152278-47327730-d228-43f1-92ed-0240d9b18e35.PNG" width="700">


It is noticed that correlation structures can be inferred from population activity in a very high accuracy and bin size of empirical sampling is critical to get an approximate estimation. Fitting amplitude characteristic functions can be slightly better than EDP by using Moore-Penrose inverse. Applying fitting with constraints elminates negative rates but not necessarily improves estimation accuracy compare with Moore-Penrose inverse.
*References:*

[1] A. Kuhn, A. Aertsen, and S. Rotter, “Higher-order statistics of input ensembles
and the response of simple model neurons,” Neural Comput, vol. 15, pp. 67–101,
Jan 2003.

[2] S. G. Benjamin Staude and S. Rotter, “Higher-order correlations and cumulants
in book analysis of parallel spike trains,” in Analysis of Parallel Spike Trains
(S. Grün and S. Rotter, eds.), pp. 253–280, Springer US, 2010.

[3] I. C. G. Reimer, B. Staude, W. Ehm, and S. Rotter, “Modeling and analyzing
higher-order correlations in non-poissonian spike trains,” J Neurosci Methods,
vol. 208, pp. 18–33, Jun 2012.

[4] W. Ehm, B. Staude, and S. Rotter, “Decomposition of neuronal assembly activity
via empirical de-poissonization,” Electron. J. Stat., vol. 1, pp. 473–495, 2007.

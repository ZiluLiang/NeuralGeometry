"""
This module holds the class to generate stimuli set and the class to generate neuron and ensemble for simulating synthetic dataset

"""
import numpy
import scipy
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

class StimuliSet():
  def __init__(self)->None:
    pass

  def __len__(self)->int:
    return self.n_stimuli
  
  def __str__(self)->None:
    print (f'{self.__class__.__name__}: n_feature={self.n_feature}, n_sample={self.n_stimuli}\n')
    for fvec,fname in zip(self.features,self.featurenames):
      print(f'{fname}: {fvec}\n')
    
  def shuffle(self, randomseed=None, ordering:numpy.ndarray=None):
    rng = numpy.random.default_rng(seed=randomseed)
    ordering = rng.permutation(self.n_stimuli) if ordering is None else ordering
    return self.stimuli[ordering,:]

class TreeStimuli(StimuliSet):
  """generate tree stimuli set from factorial combinations of leafiness and branchiness features

  Parameters
  ----------
  leafiness : array_like
      list of leafiness features. by default is 20 evenly spaced values from .2 to 1.4
  branchiness : array_like
      list of branchiness features. by default is [0.02,1]
  """
  def __init__(self,leafiness=None,branchiness=None) -> None:
    super().__init__()
    leafiness   = numpy.linspace(0.2,1.4,20,False) if leafiness is None else leafiness
    branchiness = [0.02,1] if branchiness is None else branchiness
    
    self.features     = [leafiness,branchiness]#
    self.featurenames = ["leafiness","branchiness"]
    self.stimuli      = numpy.array(list(itertools.product(*self.features)))
    self.n_stimuli    = self.stimuli.shape[0]
    self.n_features   = len(self.featurenames)

class Neuron():
  """
  noise_mu : int, optional
      mean of gaussian noise, by default 0
  noise_sig : int, optional
      variance of gaussian noise, by default 1
  """
  def __init__(self,noise_mu:int=0,noise_sig:int=1) -> None:
    self.noise     = lambda n_samples: numpy.random.normal(noise_mu,noise_sig,size=(n_samples,1))
    self.firing_description = "base class neuron with no firing function"
  
  def fire(self,stimuli:numpy.ndarray) -> numpy.ndarray:
    """output neuronal response for different stimuli

    Parameters
    ----------
    stimuli : numpy.ndarray
        stimuli matrix of shape (n_sample, n_feature)

    Returns
    -------
    numpy.ndarray
        a 2D numpy array of shape (n_sampe,1), where n_sample = stimuli.shape[0]
    """
    return self.activate(stimuli)+self.noise(stimuli.shape[0])
  
  def __str__(self):
    print(f'{self.__class__.__name__}:\n {self.firing_description}')
  
  def plot_firingrate(self,stimuli):
    n_cols = stimuli.shape[1]
    fig,axes = plt.subplots(nrows=1,ncols=n_cols,figsize=(n_cols*3.5,4))
    for j,ax in enumerate(axes):
      sns.scatterplot(x=stimuli[:,j],y=self.fire(stimuli).squeeze(),ax=ax)
      ax.set_xlabel(f'feature {j}')
      ax.set_ylabel(f'firing rate')
    fig.tight_layout()
    fig.suptitle(self.__class__.__name__,fontweight="bold",position=(0.5,1.02),fontsize="x-large")
    return fig

class Ensemble():
  # assuming that neurons in an ensemble fires individually, no inhibition/recurrence, no correlated noise
  def __init__(self,NeuronClasses,NeuronParams) -> None:
    self.neurons = [NC(**params) for NC,params in zip(NeuronClasses,NeuronParams)]
    self.n_neuron = len(self.neurons)

  def fire(self,stimuli):
    return numpy.concatenate([n.fire(stimuli) for n in self.neurons],axis=1)
  

class LinearNeuron(Neuron):
  """
  neuron that fire as a linear function of the linear combination of stimuli parameters. 
  Pure linear selectivity is achieved by setting only one element in the weight vector to one and others to zero. 
  Otherwise, linear mixed selectivity is achieved.
  Stochasticity of neuron firing is determined by gaussian noise parameters.

  Parameters
  ----------
  w : 2D numpy array
      weight vector of shape (n_feature,1)
  noise_mu : int, optional
      mean of gaussian noise, by default 0
  noise_sig : int, optional
      variance of gaussian noise, by default 1
  """
  def __init__(self,w:numpy.ndarray,noise_mu:int=0,noise_sig:int=0.001) -> None:
    
    super().__init__(noise_mu,noise_sig)
    self.activate = lambda stimuli: stimuli@w

class InteractionNeuron(Neuron):
  """
  neuron that fire as a linear function of the product of stimuli parameters. 
  This neuron will demonstrate non-linear mixed selectivity.
  Stochasticity of neuron firing is determined by gaussian noise parameters.

  Parameters
  ----------
  w : int
      slopes of the firing rate as a function of the product of stimuli parmeters, by default 1
  noise_mu : int, optional
      mean of gaussian noise, by default 0
  noise_sig : int, optional
      variance of gaussian noise, by default 1
  """
  def __init__(self,w:int=1,noise_mu:int=0,noise_sig:int=0.001) -> None:
    super().__init__(noise_mu,noise_sig)
    self.activate = lambda stimuli: numpy.atleast_2d(w*numpy.prod(stimuli, axis=1)).T

class MultivariateGaussianNeuron(Neuron):
  """
  neuron that fire as a multivariate gaussian function of the stimuli parameters. 
  This neuron will demonstrate non-linear mixed selectivity.
  Stochasticity of neuron firing is determined by gaussian noise parameters.

  Parameters
  ----------
  mus : 2D numpy array
      mean of multivariate gaussian distrubution
  cov: array_like or Covariance
      Symmetric positive (semi)definite covariance matrix of the distribution.
  noise_mu : int, optional
      mean of gaussian noise, by default 0
  noise_sig : int, optional
      variance of gaussian noise, by default 1
  """
  def __init__(self,mus:numpy.ndarray,cov=1,noise_mu:int=0,noise_sig:int=0.001) -> None:
    super().__init__(noise_mu,noise_sig)
    self.activate = lambda stimuli: numpy.atleast_2d(scipy.stats.multivariate_normal(mus,cov).pdf(stimuli)).T
    self.firing_description = f"multivariate gaussian centered around {mus},\n covariance={cov} \n"

class UnivariateGaussianNeuron(Neuron):
  """
  neuron that fire as a univariate gaussian function of the linear combination of stimuli parameters. 
  Pure nonlinear selectivity is achieved by setting only one element in the weight vector to one and others to zero. 
  Otherwise, nonlinear mixed selectivity is achieved.
  Stochasticity of neuron firing is determined by gaussian noise parameters.

  Parameters
  ----------
  w : 2D numpy array
      weight vector of shape (n_feature,1)
  mu : int
      mean of univariate gaussian function
  sig: int
      variance of univariate gaussian function
  noise_mu : int, optional
      mean of gaussian noise, by default 0
  noise_sig : int, optional
      variance of gaussian noise, by default 1
  """
  def __init__(self,w:numpy.ndarray,mu:int,sig:int,noise_mu:int=0,noise_sig:int=0.001) -> None:
    super().__init__(noise_mu,noise_sig)
    self.activate = lambda stimuli: scipy.stats.norm(mu,sig).pdf(stimuli@w)
    self.firing_description = f"feature weights = {w} \nmultivariate gaussian centered around {mu},\n sigma={sig} \n"

class LogisticNeuron(Neuron):
  """
  neuron that fire as a logistic function of the linear combination of stimuli parameters. 
  Pure selectivity is achieved by setting only one element in the weight vector to one and others to zero. 
  Otherwise, mixed selectivity is achieved.
  Stochasticity of neuron firing is determined by gaussian noise parameters.

  Parameters
  ----------
  w : 2D numpy array
      weight vector of shape (n_feature,1)
  k : int
      steepness of the curve
  x0: int
      midpoint of the logistic function
  noise_mu : int, optional
      mean of gaussian noise, by default 0
  noise_sig : int, optional
      variance of gaussian noise, by default 1
  """
  def __init__(self,w,k:int=1,x0:int=0,noise_mu:int=0,noise_sig:int=0.001) -> None:
    super().__init__(noise_mu,noise_sig)
    self.activate = lambda stimuli:1/(1+numpy.exp(-k*(stimuli@w-x0)))
    self.firing_description = f"feature weights = {w} \n logistic function k={k},\n x0={x0} \n"
"""
This module holds the class to generate stimuli set used for simulation

"""
import numpy
import itertools

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
  def __init__(self) -> None:
    super().__init__()
    self.features     = [numpy.linspace(0.2,1.4,6,False),numpy.linspace(0.2,1.4,6,False)]
    self.featurenames = ["leafiness","branchiness"]
    self.stimuli      = numpy.array(list(itertools.product(*self.features)))
    self.n_stimuli    = self.stimuli.shape[0]
    self.n_features   = len(self.featurenames)

class Neuron():
  def __init__(self,activation_function) -> None:
    self.activate=activation_function

  def fire(self,stimuli):
    return self.activate(stimuli)  

class Ensemble():
  def __init__(self,activation_functions:list) -> None:
    self.neurons = [Neuron(func) for func in activation_functions]
    self.n_neuron = len(self.neurons)

  def fire(self,stimuli):
    return numpy.array([n.fire(stimuli) for n in self.neurons])
#Support Vector Machine Deep Belief Network Code(SVM-DBN) hybrid (svdbc1.0.0)
#This SVM-DBN is designed to pick SVM classified data
#the classified dat is then subjected to multiple layers of RBM
#input includes:
#SVM output/ Classified data
#the sumary collumn will be the prediction
#NB: If you edit this code, change the editor section
#@Author: Karari
#Address: ephantus.karari@ibearesearch.org
#website: ibearesearch.org
#Date Created: 10/7/2020
#Last Edited: 10/7/2020
#Editor: Karari
#version: 1.0.0
#Appreciation: All creators of packages used in this code

#import predefined codes
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import re, math
import sklearn as sk
import easygui


class RBM:
  
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    # and sqrt(6. / (num_hidden + num_visible)). One could vary the 
    # standard deviation by multiplying the interval with appropriate value.
    # Here we initialize the weights with mean 0 and standard deviation 0.1. 
    # Reference: Understanding the difficulty of training deep feedforward 
    # neural networks by Xavier Glorot and Yoshua Bengio
    np_rng = np.random.RandomState(1234)

    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000, learning_rate = 0.1):
    """
    Train the machine.

    Parameters
    ----------
    data: A matrix where each row is a training example consisting of the states of visible units.    
    """
    #num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)
    num_examples = data.shape[0]


    for epoch in range(max_epochs):      
      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
      # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
      # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
      # themselves.
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
        if epoch % 2000 == 0:
          print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.
    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    
    ##num_examples = data.shape[0]
    num_examples = data.shape[0]    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)
    num_examples = data.shape[0]
    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1
  
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  # TODO: Remove the code duplication between this method and `run_visible`?
  def run_hidden(self, data):
    

    #num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)
    num_examples = data.shape[0]
    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
  def daydream(self, num_samples):
    """
    Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
    (where each step consists of updating all the hidden units, and then updating all of the visible units),
    taking a sample of the visible units at each step.
    Note that we only initialize the network *once*, so these samples are correlated.

    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    # Note that we keep the hidden units binary states, but leave the
    # visible units as real probabilities. See section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines"
    # for more on why.
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]        
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
  r = RBM(num_visible = 4, num_hidden = 2)
  #obtain the path to the matrix file that makes up the training data
  #data consists of all senteces that made the user summary/all feature rows with sumarry value f 1
  #source is 'C:\pfiles\Matrix'
  mf_path=easygui.fileopenbox()

  #training_data = np.array([[0,1,0,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,0,0,1,1],[0,1,1,0,1,1],[0,1,0,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,1,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,0,0,1,1],[0,1,1,0,1,1],[0,1,0,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,1,0,0,1],[0,1,1,0,0,1],[0,1,1,0,0,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,0,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,0,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[0,1,1,0,1,1],[1,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,0,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,1,1,0],[0,1,1,1,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,1,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,1,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,0,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,0,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,0,0,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,0,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,0,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,0,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0],[0,1,1,0,1,0]])
  #Derive RBM data
  
  df = pd.read_csv(mf_path,delimiter=',', header=None, names=['tf','ss','td','sp','ds'])
  #np.set_printoptions(linewidth=np.inf)
  training_data=np.array(df.drop(['ds'], 1))
  
  #print(training_data)
  r.train(training_data, max_epochs = 10000)
  #print(r.weights)

  #visible_data = np.array([[0,0,0,1,1,0]])
  #print(visible_data)
  #hidden_state=(r.run_visible(visible_data))
  #print(hidden_state)
  #hidden_data=np.array([[0.1]])
  #visible_state=r.run_hidden(hidden_data)
  #print(visible_state)
  #input_str = 'C:\\PFiles\\Matrix -Training\\test_matrix_fullx.txt'
  #input_str = 'C:\\PFiles\\Matrix -Training\\test_matrix_full5.txt'
  input_str = 'C:\\pfiles\\2007\\Matrix\\svm_dbn_matrix.csv'
  Xrbm=np.empty((0,4))
  yUser=np.empty((0,2),np.int)
with open(input_str) as fp2:  
   line = fp2.readline()
   cnt = 1
   while line:
       test_matrix = format(line.strip())
       x=tuple(test_matrix.split(','))
       y=str(x[4:5])
       u=x[0:4]
       z=y[2:3]
       #print(y)
       if(test_matrix != ''):
         if(y != "('0',)"):
           #print(y)
           test_matrix = tuple(map(float, u))
           user = np.array([test_matrix])
           Xrbm=np.append(Xrbm,user,axis=0)
           #print(user,end=" ")
           #print(r.run_visible(user))
           yUser=np.append(yUser,r.run_visible(user),axis=0)
         else:
           #print(u)
           #test_matrix = tuple(map(float, test_matrix.split(',')))
           test_matrix = tuple(map(float, u))
           user = np.array([test_matrix])
           Xrbm=np.append(Xrbm,user,axis=0)
           #print(user,end=" ")
           #print(r.run_visible(user))
           ##print(Xrbm)
           #The code below ensures that if th esentence was rejected by SVM
           #it does not get picked by RBM
           #because RBM reads the entire output from SVM in the svm_dbn hybrid
           tm=tuple(map(float,('0.0','0.0')))
           w = np.array([tm])
           yUser=np.append(yUser,w,axis=0)
                          
           
       line = fp2.readline()
       cnt += 1
#print(yUser)
yUser=np.reshape(yUser,(3921,2))
Xrbm=np.concatenate((Xrbm,yUser),axis=1)
#print(Xrbm)
pd.DataFrame(Xrbm).to_csv("C:\\pfiles\\2007\\results\\svm_rbm_predicted.csv", header=None, index=None)

       



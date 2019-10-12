import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  num_train=X.shape[0]
  for i in xrange(X.shape[0]):
        score=X[i].dot(W)
        logC=-score[np.argmax(score)]
        temp=np.log(np.sum(np.exp(score)))
        loss+=temp-score[y[i]]
        f=np.exp(score+logC)/np.sum(np.exp(score+logC))
        f[y[i]]=-1
        dW+=(f[:,np.newaxis]*X[i]).T
  loss/=num_train
  dW/=num_train
  loss+=reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  num_train=X.shape[0]
  score=X.dot(W) # shape (N,C)
  score_transpose=score.T # shape (C,N) Pay attention: broadcast from the last dim, so in order
    # to compute score+logC, we must first transpose score to score_transpose to make the last dim equal to N
  logC=-np.nanmax(score_transpose, axis=0) # shape (N,) seems to use np.max is OK

  col=np.array(xrange(num_train))
  exp_output=np.exp(score_transpose)
  
  denom_sum=np.sum(exp_output,axis=0)
  f=(np.exp(score_transpose+logC))/(np.sum(np.exp(score_transpose+logC),axis=0)) # used for gradient computation considering numerically stabality
  loss=-np.sum(score_transpose[y,col])+np.sum(np.log(denom_sum))

  f[y,col]=-1
  dW=np.sum(X.T[:,np.newaxis]*f,axis=2)
        
  loss/=num_train
  dW/=num_train
  loss+=reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


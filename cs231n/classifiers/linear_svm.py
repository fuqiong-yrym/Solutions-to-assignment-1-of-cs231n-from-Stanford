import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count=0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        count+=1
        dW[:,j]+=X[i]
    dW[:,y[i]]+=(-1)*count*X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_tr  ain.
  loss /= num_train
  dW/=num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  import numpy as np
  loss = 0.0
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero
  dW_transpose=dW.T
  #correct_class_list=[]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #pass
  S=X.dot(W)
  score=S.T
  col=np.array(xrange(num_train))
  #for i in xrange(num_train):
  #  correct_class_list.append(score[y[i],i])
  #correct_class_label=np.array(correct_class_list)
  correct_class_label=score[y,col]
  margin=score-correct_class_label+1
  
  margin[y,col]=0
  
  loss=np.sum(np.where(margin>0,margin,0))
  loss/=num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  
  map=np.where(margin>0,1,0)
  
  #for j in xrange(num_train):
  #  map[y[j],j]=(-1)*np.count_nonzero(map[:,j])
  
  map[y,col]=(-1)*np.count_nonzero(map,axis=0)
  
  for j in xrange(num_train):
    dW_transpose+=map[:,j,np.newaxis]*X[j]
  dW=dW_transpose.T
  #dW=np.sum(X.T[:,np.newaxis]*map,axis=2)
  
  dW/=num_train
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

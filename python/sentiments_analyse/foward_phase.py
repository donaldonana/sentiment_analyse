import numpy as np
from numpy.random import randn
from data_set import train_data, test_data
import random
import pickle


class RNN:
	"""docstring for RNN"""
	def __init__(self, input_size, output_size, hidden_size=64):
		
		self.Whh = randn(hidden_size, hidden_size) / 1000
		self.Wxh = randn(hidden_size, input_size) / 1000
		self.Why = randn(output_size, hidden_size) / 1000

		# Biases
		self.bh = np.zeros((hidden_size, 1))
		self.by = np.zeros((output_size, 1))

	def forward(self, inputs):
		'''
		Perform a forward pass of the RNN using the given inputs.
		Returns the final output and hidden state.
		- inputs is an array of one hot vectors with shape (input_size, 1).
		'''
		h = np.zeros((self.Whh.shape[0], 1))

		self.last_inputs = inputs
		self.last_hs = { 0: h }

		# Perform each step of the RNN
		for i, x in enumerate(inputs):
			h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
			self.last_hs[i + 1] = h

		# Compute the output
		y = self.Why @ h + self.by

		return y, h

	def backprop(self, d_y, learn_rate=2e-2):

	    '''
	    Perform a backward pass of the RNN.
	    - d_y (dL/dy) has shape (output_size, 1).
	    - learn_rate is a float.
	    '''
	    n = len(self.last_inputs)

	    # Calculate dL/dWhy and dL/dby.
	    d_Why = d_y @ self.last_hs[n].T
	    d_by = d_y

	    # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
	    d_Whh = np.zeros(self.Whh.shape)
	    d_Wxh = np.zeros(self.Wxh.shape)
	    d_bh = np.zeros(self.bh.shape)

	    # Calculate dL/dh for the last h.
	    d_h = self.Why.T @ d_y

	    # Backpropagate through time.
	    for t in reversed(range(n)):
	      # An intermediate value: dL/dh * (1 - h^2)
	      temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

	      # dL/db = dL/dh * (1 - h^2)
	      d_bh += temp

	      # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
	      d_Whh += temp @ self.last_hs[t].T

	      # dL/dWxh = dL/dh * (1 - h^2) * x
	      d_Wxh += temp @ self.last_inputs[t].T

	      # Next dL/dh = dL/dh * (1 - h^2) * Whh
	      d_h = self.Whh @ temp

	    # Clip to prevent exploding gradients.
	    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
	      np.clip(d, -1, 1, out=d)

	    # Update weights and biases using gradient descent.
	    self.Whh -= learn_rate * d_Whh
	    self.Wxh -= learn_rate * d_Wxh
	    self.Why -= learn_rate * d_Why
	    self.bh -= learn_rate * d_bh
	    self.by -= learn_rate * d_by


# Create the vocabulary.
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab.sort()
vocab_size = len(vocab)
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }


def createInputs(text):
  '''
  Returns an array of one-hot vectors representing the words
  in the input text string.
  - text is a string
  - Each one-hot vector has shape (vocab_size, 1)
  '''
  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs


def softmax(xs):
	# Applies the Softmax Function to the input array.
	return np.exp(xs) / sum(np.exp(xs))



def processData(data, backprop=True):
  '''
  Returns the RNN's loss and accuracy for the given data.
  - data is a dictionary mapping text to True or False.
  - backprop determines if the backward phase should be run.
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0
  

  for x, y in items:
    inputs = createInputs(x)
    target = int(y)

    # Forward
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Calculate loss / accuracy
    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    if backprop:
      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Backward
      rnn.backprop(d_L_d_y)

  return loss / len(data), num_correct / len(data)







global rnn
rnn = RNN(vocab_size, 2)

# Training loop








print("-----------------------------------------end-----------------------------")

loaded_model = pickle.load(open("rnn_sentiment_analyse.pkl" , "rb"))
inputs = createInputs("i am happy")
print(inputs)
 	


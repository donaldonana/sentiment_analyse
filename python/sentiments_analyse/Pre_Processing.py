"""  We’ll have to do some pre-processing to get the data into a usable format. To start, 
we’ll construct a vocabulary of all words that exist in our data"""


from data_set import train_data, train_data2, test_data
import numpy as np


vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab.sort()
print(vocab)
global vocab_size , word_to_idx , idx_to_word
vocab_size= len(vocab)
# print('%d unique words found' % vocab_size) # 18 unique words found
# print(vocab)

word_to_idx = { w: i for i, w in enumerate(vocab) }
# print(word_to_idx)
idx_to_word = { i: w for i, w in enumerate(vocab) }
# print(word_to_idx['good']) # 16 (this may change)
# print(idx_to_word[0]) # sad (this may change)

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
    # print(word_to_idx[w])
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs


# inputs = createInputs("this is good")

# print(inputs)
import pickle


def softmax(xs):
	# Applies the Softmax Function to the input array.
	return np.exp(xs) / sum(np.exp(xs))


loaded_model = pickle.load(open("rnn_sentiment_analyse.pkl" , "rb"))

# inputs = createInputs("yes is good")

# out, _ = loaded_model.forward(inputs)
# probs = softmax(out)

# print("hello")
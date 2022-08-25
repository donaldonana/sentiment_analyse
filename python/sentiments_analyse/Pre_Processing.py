"""  We’ll have to do some pre-processing to get the data into a usable format. To start, 
we’ll construct a vocabulary of all words that exist in our data"""


from data_set import train_data, test_data

items = list(train_data.items())

with open('../../sentiments.csv', 'w') as mon_fichier:

	mon_fichier.write("text,label\n")
	for x, y in items:
		target = int(y)
		# print("\n",target,"\n")
		# mon_fichier.write(str(target)+"\n")
		print("\n---")
		mon_fichier.write(str(x)+","+str(target)+"\n")
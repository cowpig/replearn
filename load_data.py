import cPickle
import numpy as np
from os import listdir

DATA_PATH = "./timit/readable/"

DATA_FILES = listdir(DATA_PATH)

def load_data(dataset="train"):
	data = {}

	datasets = set(["train", "test", "valid"])

	# pass in None to grab everything
	if dataset != None:
		datasets.remove(dataset)
	else:
		datasets = set()

	for f in DATA_FILES:
		# exclude the train/test if we want valid, e.g.
		if not any(word in f for word in datasets):
			if "pkl" in f:
				with open(DATA_PATH + f, "rb") as open_file:
					data[f[:f.find(".")]] = cPickle.load(open_file)
			elif "npy" in f:
				data[f[:f.find(".")]] = np.load(DATA_PATH + f)

	data['spkrinfo'] = data['spkrinfo'].tolist().toarray()

	return data

def generate_play_set(d, phn=0):
	raw_data = []
	for i, row in enumerate(d['train_phn']):
		# start by modelling just phoneme 'iy'
		if row[2] == phn:
			seq = d['train_phn_to_seq'][i]
			for i in xrange(row[0], row[1]-301):
				data = (d['spkrinfo'][seq], # vector that describes speaker
						d['train_x_raw'][seq][i:i+300], # raw sound data
						d['train_x_raw'][seq][i+300]) # label
				raw_data.append(data)
import cPickle
import numpy as np
from os import listdir

DATA_PATH = "./timit/readable/"
DATA_FILES = listdir(DATA_PATH)
NUM_SPEAKERS = 630
ACOUSTIC_SAMPLE_SIZE = 300

def load_dataset(dataset="train"):
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

def generate_play_set(d=load_dataset(), phn=0):
	raw_data = []
	one_hots = np.identity(NUM_SPEAKERS)

	# I've pre-calculated these values
	max_acoustic = 9899
	min_acoustic = -6884

	# maybe try normalizing later
	# d['train_x_raw'] = (raws - (min_acoustic)) / (max_acoustic - (min_acoustic))

	for i, row in enumerate(d['train_phn']):
		# start by modelling just phoneme 'iy'
		if row[2] == phn:
			seq = d['train_phn_to_seq'][i]
			for i in xrange(row[0], row[1]-ACOUSTIC_SAMPLE_SIZE-1):
				spkr_id = d['train_spkr'][seq]
				x_raw = d['train_x_raw'][seq][i:i+ACOUSTIC_SAMPLE_SIZE]

				# max_acoustic = max(np.max(x_raw), max_acoustic)
				# min_acoustic = min(np.min(x_raw), min_acoustic)

				data = ((one_hots[spkr_id,:], #one-hot vector for speaker ID
						d['spkrinfo'][spkr_id], # vector that describes speaker
						x_raw), # raw sound data
						d['train_x_raw'][seq][i+ACOUSTIC_SAMPLE_SIZE]) # label
				raw_data.append(data)

	return raw_data



# def generate_play_set(d=load_dataset(), phn=0):
# 	# normalization
# 	max_acoustic = -99999
# 	min_acoustic = 99999
# 	d['spkrinfo'][:,0] /= 76.
# 	d['spkrinfo'][:,15] /= 203.2

# 	spkrs = []
# 	spkr_infos = []
# 	raws = []
# 	labels = []

# 	for i, row in enumerate(d['train_phn']):
# 		# start by modelling just phoneme 'iy'
# 		if row[2] == phn:
# 			seq = d['train_phn_to_seq'][i]
# 			for j in xrange(row[0], row[1]-301):
# 				speaker_id = d['train_spkr'][seq]
# 				one_hot_spkr = np.zeros(630)
# 				one_hot_spkr[speaker_id] = 1
# 				spkrs.append(one_hot_spkr)

# 				spkr_infos.append(d['spkrinfo'][speaker_id])

# 				raw = d['train_x_raw'][seq][j:j+300]
# 				max_acoustic = max(np.max(raw), max_acoustic)
# 				min_acoustic = min(np.min(raw), min_acoustic)
# 				raws.append(raw)

# 				labels.append(d['train_x_raw'][seq][j+300])

# 	raw = np.array(raw, dtype=np.float64)
# 	raw = (raw - min_acoustic) / (max_acoustic - min_acoustic)

# 	ins = np.array(spkrs, dtype=float64)
# 	ins = np.concatenate(ins, np.array(spkr_infos, dtype=float64))
# 	ins = np.concatenate(ins, raws)

# 	return (ins, np.array(labels, dtype=np.bool))
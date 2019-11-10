import os, csv
import pickle
from collections import defaultdict
from util import *
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def read(FILE = "./images_labelling.csv"):
    DATA = "./images/"
    labels = []
    classes = defaultdict(str)
    features = []
    with open(FILE, "r") as fl:
        for row in csv.reader(fl):
            path = DATA + "%s.png" % row[0]
            if os.path.isfile(path):
                image = cv2.imread(path)
                features.append(get_feature(image))
                labels.append(int(row[1]))
                classes[int(row[1])] = row[2]
    return features, labels, classes


features, labels, classes = read()
targetNames = np.unique(labels)
le = LabelEncoder()
Y = le.fit_transform(labels)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(features)
#X = features

with open(os.path.join(DATA_DIR, "data.dat"), "wb") as fl:
    pickle.dump([np.array(X), np.array(Y)], fl)

with open(os.path.join(DATA_DIR, "encoder.dat"), "wb") as fl:
    pickle.dump(le, fl)
    
with open(os.path.join(DATA_DIR, "classes.dat"), "wb") as fl:
    pickle.dump(classes, fl)

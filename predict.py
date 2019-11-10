import pickle, os
import cv2
from util import *

class ModelPredict:
    def __init__(self):
        with open(os.path.join(DATA_DIR, "model.pkl"), "rb") as fl:
            self.cls = pickle.load(fl)
        with open(os.path.join(DATA_DIR, "encoder.dat"), "rb") as fl:
            self.le = pickle.load(fl)
        with open(os.path.join(DATA_DIR, "classes.dat"), "rb") as fl:
            self.classes = pickle.load(fl)
    
    def predict(self, img):
        cr = self.cls.predict([get_feature(img)])
        label = self.le.inverse_transform(cr)[0]
        return label, self.classes[label]

if __name__ == "__main__":
    p = ModelPredict()
    img = cv2.imread("/home/mint/Workspace/UMA/ML/Bot/images/476.png")
    print (p.predict(img))

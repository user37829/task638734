import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.cluster import *
from sklearn.neighbors import *
from sklearn.feature_selection import *
from util import *

with open(os.path.join(DATA_DIR, "data.dat"), "rb") as fl:
    X, Y = pickle.load(fl)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=seed)
print ("Training...")

cls = RandomForestClassifier(n_estimators=170)
#cls = LogisticRegression(C=65, solver="lbfgs", max_iter=1000)
#cls = SVC(50, gamma="scale", kernel="poly", degree=5)
#cls = DecisionTreeClassifier(criterion="entropy", max_features=4)

cls.fit(X_train, Y_train)

print (cls.score(X_test, Y_test))

with open(os.path.join(DATA_DIR, "model.pkl"), "wb") as fl:
    pickle.dump(cls, fl)

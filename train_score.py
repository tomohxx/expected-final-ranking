import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

class Ranking:
  def __init__(self, clfs):
    self.clfs = clfs
    self.coeff = 10000

  def predict(self, kyoku, score):
    feature = np.array([[(score[0]-score[1])/self.coeff, (score[0]-score[2])/self.coeff, (score[0]-score[3])/self.coeff]])
    proba = self.clfs[kyoku].predict_proba(feature)[0]

    return np.array([ \
      [ \
        proba[0]+proba[1]+proba[2]+proba[3]+proba[4]+proba[5], \
        proba[6]+proba[7]+proba[8]+proba[9]+proba[10]+proba[11], \
        proba[12]+proba[13]+proba[14]+proba[15]+proba[16]+proba[17], \
        proba[18]+proba[19]+proba[20]+proba[21]+proba[22]+proba[23], \
      ], \
      [ \
        proba[6]+proba[7]+proba[12]+proba[13]+proba[18]+proba[19], \
        proba[0]+proba[1]+proba[14]+proba[15]+proba[20]+proba[21], \
        proba[2]+proba[3]+proba[8]+proba[9]+proba[22]+proba[23], \
        proba[4]+proba[5]+proba[10]+proba[11]+proba[16]+proba[17], \
      ], \
      [ \
        proba[8]+proba[10]+proba[14]+proba[16]+proba[20]+proba[22], \
        proba[2]+proba[4]+proba[12]+proba[17]+proba[18]+proba[23], \
        proba[0]+proba[5]+proba[6]+proba[11]+proba[19]+proba[21], \
        proba[1]+proba[3]+proba[7]+proba[9]+proba[13]+proba[15], \
      ], \
      [ \
        proba[9]+proba[11]+proba[15]+proba[17]+proba[21]+proba[23], \
        proba[3]+proba[5]+proba[13]+proba[16]+proba[19]+proba[22], \
        proba[1]+proba[4]+proba[7]+proba[10]+proba[18]+proba[20], \
        proba[0]+proba[2]+proba[6]+proba[8]+proba[12]+proba[14], \
      ] \
    ])

if __name__ == '__main__':
  df = pd.read_csv('score.csv', header=None)
  clfs = []

  for i in range(4):
    clfs.append(LogisticRegression(tol=1e-6, random_state=0, verbose=0))
    clf = clfs[-1]
    x = df[df[13] == i].iloc[:, 9:12]
    y = df[df[13] == i].iloc[:, 12]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf.fit(x_train, y_train)

    print('{} Accuracy (train) :'.format(i), clf.score(x_train, y_train))
    print('{} Accuracy (test)  :'.format(i), clf.score(x_test, y_test))

  with open('score_predictor.bin', mode='wb') as f:
    pickle.dump(clfs, f)

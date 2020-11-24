# imports
import sys
import os
import urllib.request
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# preparing output folder
if (not(os.path.isdir("report"))):
    os.mkdir("report")

# collecting data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
urllib.request.urlretrieve(url, 'winequality-white.csv')

# reading data
df = pd.read_csv('winequality-white.csv', sep=';')
df_X, df_y0 = np.split(df, [11], axis=1)

# preprocessing data
df_X["fixed acidity"] = df_X["fixed acidity"] / df_X["fixed acidity"].max()
df_X["volatile acidity"] = df_X["volatile acidity"] / df_X["volatile acidity"].max()
df_X["citric acid"] = df_X["citric acid"] / df_X["citric acid"].max()
df_X["residual sugar"] = df_X["residual sugar"] / df_X["residual sugar"].max()
df_X["chlorides"] = df_X["chlorides"] / df_X["chlorides"].max()
df_X["free sulfur dioxide"] = df_X["free sulfur dioxide"] / df_X["free sulfur dioxide"].max()
df_X["total sulfur dioxide"] = df_X["total sulfur dioxide"] / df_X["total sulfur dioxide"].max()
df_X["density"] = df_X["density"] / df_X["density"].max()
df_X["pH"] = df_X["pH"] / df_X["pH"].max()
df_X["sulphates"] = df_X["sulphates"] / df_X["sulphates"].max()
df_X["alcohol"] = df_X["alcohol"] / df_X["alcohol"].max()
def reformat(v):
  arr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  arr[int(v)] = 1
  return np.asarray(arr)
df_y0.quality = df_y0.quality.apply(reformat)
df_y0[['1','2', '3', '4', '5', '6', '7', '8', '9', '10']] = pd.DataFrame(df_y0.quality.tolist())
df_y = df_y0[['1','2', '3', '4', '5', '6', '7', '8', '9', '10']]

# generating batches
X_train, X_test, y_train, y_test = train_test_split(df_X,df_y, test_size=0.1)

# defining loss function
def tril_indices(n, k=0):
  """Return the indices for the lower-triangle of an (n, m) array.
  Works similarly to `np.tril_indices`
  Args:
    n: the row dimension of the arrays for which the returned indices will
      be valid.
    k: optional diagonal offset (see `np.tril` for details).
  Returns:
    inds: The indices for the triangle. The returned tuple contains two arrays,
      each with the indices along one dimension of the array.
  """
  m1 = tensorflow.tile(tensorflow.expand_dims(tensorflow.range(n), axis=0), [n, 1])
  m2 = tensorflow.tile(tensorflow.expand_dims(tensorflow.range(n), axis=1), [1, n])
  mask = (m1 - m2) >= -k
  ix1 = tensorflow.boolean_mask(m2, tensorflow.transpose(mask))
  ix2 = tensorflow.boolean_mask(m1, tensorflow.transpose(mask))
  return ix1, ix2

def ecdf(p):
  """Estimate the cumulative distribution function.
  The e.c.d.f. (empirical cumulative distribution function) F_n is a step
  function with jump 1/n at each observation (possibly with multiple jumps
  at one place if there are ties).
  For observations x= (x_1, x_2, ... x_n), F_n is the fraction of
  observations less or equal to t, i.e.,
  F_n(t) = #{x_i <= t} / n = 1/n \sum^{N}_{i=1} Indicator(x_i <= t).
  Args:
    p: a 2-D `Tensor` of observations of shape [batch_size, num_classes].
      Classes are assumed to be ordered.
  Returns:
    A 2-D `Tensor` of estimated ECDFs.
  """
  # if the following line produces a weird bug, replace it with `n = 10`
  n = 10
  indices = tril_indices(n)
  indices = tensorflow.transpose(tensorflow.stack([indices[1], indices[0]]))
  ones = tensorflow.ones([int(n * (n + 1) / 2)])
  triang = tensorflow.scatter_nd(indices, ones, [n, n])
  return tensorflow.linalg.matmul(tensorflow.cast(p, tensorflow.float32), 
                                  tensorflow.cast(triang, tensorflow.float32))

def emd_loss(p, p_hat, r=2, scope=None):
  """Compute the Earth Mover's Distance loss.
  Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
  Distance-based Loss for Training Deep Neural Networks." arXiv preprint
  arXiv:1611.05916 (2016).
  Args:
    p: a 2-D `Tensor` of the ground truth probability mass functions.
    p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
    r: a constant for the r-norm.
    scope: optional name scope.
  `p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
  \sum^{N}_{i=1} p_hat_i
  Returns:
    A 0-D `Tensor` of r-normed EMD loss.
  """
  with tensorflow.keras.backend.name_scope('EmdLoss'):
    ecdf_p = ecdf(p)
    ecdf_p_hat = ecdf(p_hat)
    emd = tensorflow.reduce_mean(tensorflow.pow(tensorflow.abs(ecdf_p - ecdf_p_hat), r), axis=-1)
    emd = tensorflow.pow(emd, 1 / r)
    return emd
    return tensorflow.reduce_mean(emd)

# model structuring
model = keras.Sequential()
model.add(keras.layers.Dense(units=31, input_shape=[11]))
model.add(keras.layers.Dense(units=11))
model.add(keras.layers.Dense(units=10))
model.add(keras.layers.Softmax())

# hyper parameters
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=keras.optimizers.get(opt), loss=emd_loss)

# printing summary
fstructure = open("report/structure.txt", "w")
model.summary(print_fn=lambda x: fstructure.write(x + '\n'))
fstructure.close()

# training
history = model.fit(X_train, y_train, epochs=500, verbose=2)

# evaluation
losses = history.history['loss']
epochs = history.epoch

plt.plot(epochs[50:], losses[50:])
plt.xlabel("epochs")
plt.ylabel('losses')
plt.title('Loss per Epoch')
plt.show()
plt.savefig('report/history.png')
fmetrics = open("report/metrics.txt", "w")
print('Initial loss value : ', losses[0], file=fmetrics)
print('Final loss value : ', losses[-1], file=fmetrics)
fmetrics.close()

# saving
model.save('report/weights.h5')

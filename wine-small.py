import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
import numpy as np
tf.enable_eager_execution()
VERBOSE = 2

#Datasets
train_dataset_location = "wine_train_small.csv"
test_dataset_location = "wine.csv"

#Read training data and convert target to one-hot as needed by categorical cross_entropy per documentation
dataset = np.loadtxt(open(train_dataset_location), delimiter=',')
X_train = dataset[:,0:13].astype(np.float64)
Y_train = dataset[:,13].astype(int)
Y_train = tf.keras.utils.to_categorical(Y_train)

#Custom loss function
def my_cat_crossentropy(target,output,from_logits=False,axis=-1):
	return tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,logits=output)

my_batch_size = 20

#Model definition
my_model = Sequential()
my_model.add(Dense(15, input_dim=13, kernel_initializer='glorot_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
my_model.add(Dropout(0.001))
my_model.add(Dense(3, kernel_initializer='glorot_uniform'))

#Train model using multi-stage optimiser
#Stage 1
my_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.8,beta2=0.999), loss=my_cat_crossentropy, metrics=['accuracy'])
my_model.fit(x=X_train, y=Y_train, batch_size=my_batch_size, epochs=50, verbose=VERBOSE, shuffle=True)

#Stage 2
my_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.005,beta1=0.9,beta2=0.99), loss=my_cat_crossentropy, metrics=['accuracy'])
my_model.fit(x=X_train, y=Y_train, batch_size=my_batch_size, epochs=100, verbose=VERBOSE, shuffle=True)

#Stage 3
my_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0005,beta1=0.9,beta2=0.99), loss=my_cat_crossentropy, metrics=['accuracy'])
my_model.fit(x=X_train, y=Y_train, batch_size=my_batch_size, epochs=100, verbose=VERBOSE, shuffle=True)

#Stage 4
my_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.00005,beta1=0.9,beta2=0.99), loss=my_cat_crossentropy, metrics=['accuracy'])
my_model.fit(x=X_train, y=Y_train, batch_size=my_batch_size, epochs=100, verbose=VERBOSE, shuffle=True)

#Evaluate model on training data
print("evaluation on training data", my_model.evaluate(x=X_train, y=Y_train, batch_size=my_batch_size))

#Read testing data and convert target to one-hot as needed by categorical_cross_entropy per documentation
ds = np.loadtxt(open(test_dataset_location), delimiter=',')
X_test = ds[:,0:13].astype(np.float64)
Y_test = ds[:,13].astype(int)
Y_test = tf.keras.utils.to_categorical(Y_test)

#Evaluate on testing data
print("evaluation on test data", my_model.evaluate(x=X_test, y=Y_test, batch_size=my_batch_size))

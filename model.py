import pandas as pd
import tensorflow as tf
import keras
from keras import models, layers
keras.__version__

df = pd.read_csv("data.csv")
df.head()

train = df[50:]
test = df[:50]

x = train.drop(['Result'], axis=1)
y = train['Result']

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(3,)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc
    
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[auc])


history = model.fit(x,
                    y,
                    epochs=100,
                    batch_size=100,
                    validation_split = .2,
                    verbose=0)

x_test = test.drop(['Result'], axis=1)
y_test = test['Result']

results = model.evaluate(x_test, y_test, verbose = 0)
print(results)

model.save("mod.h5")

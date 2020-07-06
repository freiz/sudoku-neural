from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape


def get_data(file):
    data = pd.read_csv(file)

    feat_raw = data['quizzes']
    label_raw = data['solutions']

    feat = []
    label = []

    for i in feat_raw:
        x = np.array([int(j) for j in i]).reshape((9, 9, 1))
        feat.append(x)

    feat = np.array(feat)
    #     feat = feat / 9
    #     feat -= .5

    for i in label_raw:
        x = np.array([int(j) for j in i]).reshape((81, 1)) - 1
        label.append(x)

    label = np.array(label)

    del (feat_raw)
    del (label_raw)

    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.05, random_state=42)

    return x_train, x_test, y_train, y_test


def get_model():
    model = keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(81 * 9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data('data/sudoku.csv')
    model = get_model()

    opt = keras.optimizers.Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=30, shuffle=True, validation_data=(x_test, y_test))

    model.save('model')

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout
import time
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight

from utils.config import config as cf

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def initialize_nn_model(input_dim, output_dim):
    input_dim = input_dim  # X.shape[1]
    output_dim = output_dim  # y.shape[1]
    model = Sequential()
    model.add(layers.Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate_cv_model(X, y, y_label, label_encoder, cv=5, epochs=50, batch_size=32, class_weights=None):
    print('===== Cross validation : CV=', cv)
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=98)
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    fit_time_list = []
    for train, test in kfold.split(X, y_label):
        model = initialize_nn_model(X.shape[1], y.shape[1])
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_label[train]), y_label[train])
        time_callback = TimeHistory()
        model.fit(X[train], y[train], epochs=epochs, verbose=0, validation_data=(X[test], y[test]),
                  batch_size=batch_size, class_weight=class_weights)
        y_pred = model.predict(X[test])
        y_pred_decode = pd.DataFrame(y_pred).idxmax(axis=1)
        y_pred_decode = label_encoder.inverse_transform(y_pred_decode)

        accuracy = accuracy_score(y_label[test], y_pred_decode)
        precision = precision_score(y_label[test], y_pred_decode, average='macro')
        recall = recall_score(y_label[test], y_pred_decode, average='macro')
        f1 = f1_score(y_label[test], y_pred_decode, average='macro')

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        fit_time_list.append(np.sum(time_callback))

        print('accuracy:', accuracy)
        print('precision:', precision)
        print('recall:   ', recall)
        print('f1 score: ', f1)
        print('-' * 50)

    print('===== Avg F1-Score : {:.3f}'.format(np.mean(f1_list)))
    return pd.DataFrame({'Model': 'NeuralNetwork', 'Accuracy': accuracy_list,
                         'Precision': precision_list, 'Recall': recall_list, 'F1-score': f1_list,
                         'fit_time': fit_time_list})


def benchmark_nn_train_size(X, y, label_encoder, train_size):
    train_scores = []
    test_scores = []
    if train_size < len(y):
        test_size = 1 - (train_size / float(len(y)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=cf.RANDOM_ST)

        model = initialize_nn_model(X_train.shape[1], y_train.shape[1])
        history = model.fit(X_train, y_train, epochs=40,  verbose=0,  validation_data=(X_test, y_test), batch_size=10)

        y_train_pred = model.predict(X_train)
        y_train_pred_decode = pd.DataFrame(y_train_pred).idxmax(axis=1)
        y_train_pred_decode = label_encoder.inverse_transform(y_train_pred_decode)

        y_train_decode = np.argmax(y_train, axis=1)
        y_train_decode = label_encoder.inverse_transform(y_train_decode)

        y_test_pred = model.predict(X_test)
        y_test_pred_decode = pd.DataFrame(y_test_pred).idxmax(axis=1)
        y_test_pred_decode = label_encoder.inverse_transform(y_test_pred_decode)

        y_test_decode = np.argmax(y_test, axis=1)
        y_test_decode = label_encoder.inverse_transform(y_test_decode)
        train_scores.append(f1_score(y_train_pred_decode, y_train_decode, average='macro'))
        test_scores.append(f1_score(y_test_pred_decode, y_test_decode, average='macro'))
        plot_history(history, figsize=(15,5),export_filename=('neural_network_epoch_'+str(train_size)+'.png'))
        val_acc = history.history['val_acc']
    return np.mean(train_scores), np.mean(test_scores), val_acc


def plot_history(history, figsize=(12,5), export_filename = None):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    if export_filename is not None :
        plt.savefig(cf.EXPORT_PATH + export_filename , bbox_inches='tight')
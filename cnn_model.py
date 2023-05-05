import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, plot_roc_curve, roc_curve 

from PIL import Image
import matplotlib.pyplot as plt


def load_data(train_dir, test_dir, validation_dir, IMAGE_SIZE = (300, 300)):

    # Load datasets for training, testing, and validation
    train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir, 
                                                                    image_size=IMAGE_SIZE,
                                                                    label_mode='binary', 
                                                                    batch_size=64,
                                                                    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                    image_size=IMAGE_SIZE,
                                                                    batch_size=64,
                                                                    label_mode="binary",shuffle=False)


    val_data = tf.keras.preprocessing.image_dataset_from_directory(directory=validation_dir,
                                                                    image_size=IMAGE_SIZE,
                                                                    batch_size=64,
                                                                    label_mode="binary", shuffle=False)
    
    
    # Preprocess the data
    AUTOTUNE = tf.data.AUTOTUNE

    train_data = train_data.prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.prefetch(buffer_size=AUTOTUNE)
    val_data = val_data.prefetch(buffer_size=AUTOTUNE)

    return train_data, test_data, val_data


def compute_class_weights(train_dir):
    labels = ['NORMAL', 'PNEUMONIA']
    train_counts = {}

    for l in labels:
        # train counts
        path = os.path.join(train_dir,l)
        fileList=os.listdir(path)
        train_counts[l] = len(fileList)

    class_weights = {}
    class_weights[0] = (1/train_counts['NORMAL']) * ((train_counts['NORMAL']+train_counts['PNEUMONIA'])/ 2)
    class_weights[1] = (1/train_counts['PNEUMONIA']) * ((train_counts['NORMAL']+train_counts['PNEUMONIA'])/ 2)

    return class_weights


def build_model(IMAGE_SIZE=(300,300)):

    # Construct the Model - 4 Conv layers w 2x2 Max Pooling layers after each
    model = models.Sequential()

    model.add(layers.Conv2D(16, (3, 3), activation='relu',
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compiling the model
    model.compile(loss='binary_crossentropy',
             optimizer=tf.keras.optimizers.Adam(),
             metrics=["acc", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                       tf.keras.metrics.SpecificityAtSensitivity(0.5), tf.keras.metrics.SensitivityAtSpecificity(0.5)])
    
    return model


def plot_training_acc_loss(history):
    acc = history.history['acc']
    loss = history.history['loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.title('Training accuracy')
    plt.legend()
    plt.savefig('images/train_acc.png')

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig('images/train_loss.png')


def evaluate_model(model, test_data):
    class_names = ['NORMAL', 'PNEUMONIA']

    x_test = np.concatenate([x for x, y in test_data], axis=0)
    y_test = np.concatenate([y for x, y in test_data], axis=0)

    predictions = (model.predict(test_data) > 0.5).astype("int32")

    print(classification_report(y_test, predictions))
    
    # Helper function for plotting confusion matrix from tf model
    class estimator:
        _estimator_type = ''
        classes_=[]
        def __init__(self, model, classes):
            self.model = model
            self._estimator_type = 'classifier'
            self.classes_ = classes
        def predict(self, X):
            y_prob= self.model.predict(X)
            y_pred = y_prob.argmax(axis=1)
            return y_pred

    # Plot confusin matrix    
    classifier = estimator(model, class_names)
    fig=plot_confusion_matrix(estimator=classifier, X=x_test, y_true=y_test)
    plt.savefig('images/cnn_conf_mat.png')


    # Plot ROC curve
    fig=plot_roc_curve(estimator=classifier, X=x_test, y_true=y_test, color='r')
    plt.legend()
    plt.title("ROC CURVE")
    plt.savefig('images/cnn_conf_mat.pngROC.png')





def main():
    # Get the image file directories - could convert this to use argparse so that it works well on multiple systems
    train_dir = os.path.join('/Users/luctheduke/Desktop/UVM Grad School/Year 2/STAT 287 - DS 1/chest_xray/train')
    test_dir = os.path.join('/Users/luctheduke/Desktop/UVM Grad School/Year 2/STAT 287 - DS 1/chest_xray/test')
    validation_dir = os.path.join('/Users/luctheduke/Desktop/UVM Grad School/Year 2/STAT 287 - DS 1/chest_xray/val')

    # load in the data
    train_data, test_data, val_data = load_data(train_dir, test_dir, validation_dir)

    # compute class weights
    class_weights = compute_class_weights(train_dir)

    # create the model
    model = build_model()
    print(model.summary())

    # Fit the model
    history = model.fit(train_data,
                        epochs=10,
                        steps_per_epoch=len(train_data),
                        class_weight=class_weights
                        )
    
    # run model evaluations
    plot_training_acc_loss(history)

    # get and print training metrics
    all_metrics = model.evaluate(test_data, steps=10, batch_size=32)
    print(all_metrics)

    # Evaluate model based off test data
    evaluate_model(model, test_data)









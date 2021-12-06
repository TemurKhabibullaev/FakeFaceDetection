import numpy as np
import os
import pathlib
import skimage.io as io
import skimage.transform as tf
import torch
from haroun import Data, Model
from haroun.augmentation import augmentation
from haroun.losses import rmse
from NeuralNetworks import Network

# Process of loading and labelling the data
# Load up the pictures


def loaderOfPics():
    path = pathlib.Path.cwd().parent / "FakeFaceDetection" / "real_and_fake_face_detection"
    path = path / "real_and_fake_face"
    images, labels = [], []

    for directory in os.listdir(path):
        dataPath = path / directory

        for im in os.listdir(dataPath)[:]:
            image = io.imread(f"{dataPath}/{im}")
            image = tf.resize(image, (64, 64))
            images.append(image)
            if directory == "training_fake":
                labels.append("fake")
            elif directory == "training_real":
                labels.append("real")

    images = np.array(images)
    labels = np.array(labels)

    images, labels = augmentation(images, labels, flip_y=True, flip_x=True, brightness=True)

    return images, labels


# Labelling
classes = {'real': 0, 'fake': 1}
data = Data(loader=loaderOfPics(), classes=classes)
data.stat()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data.dataset(split_size=0.05, shuffle=True, random_state=42, images_format=torch.float32, labels_format=torch.float32,
             permute=True, one_hot=True, device=device)
net = Network()
catfishClassifier = Model(net, "adam", rmse, device)
catfishClassifier.train(train_data=(data.train_inputs, data.train_outputs),
                        val_data=(data.val_inputs, data.val_outputs),
                        epochs=1, patience=20, batch_size=100, learning_rate=1.0E-3)

catfishClassifier.evaluate(test_data=(data.test_inputs, data.test_outputs))
catfishClassifier.plot()

# Keras CNN Classifier

This repository provides code to set up and train your own image classifiers based on pre-trained Convolutional Neural Networks (CNNs) as provided by the Keras API in Tensorflow 2.0. You can make use of transfer learning to train on your own collected datasets, or train networks from scratch to compare performance on large datasets.

## How to use

This repository's main code is contained in the `ClassifierCNN` class in `Classifier.py`. All you really need to do is import this class,
set up a proper instance specifying the desired dataset and models, and you're good to go. Some examples are provided here below:

```
# Import pre-trained Keras CNN Model
from keras.applications.densenet import DenseNet169
# Import ClassifierCNN class
from Classifier import ClassifierCNN

# Instantiate class with CNN model, dataset directory, and experiment name
cls = ClassifierCNN(DenseNet169, 'Pets', 'd169_pets_finetuned')
# Set number of layers to fine-tune
cls.finetuning_layers = 224
# Run Training
cls.train()
```
The code above imports the DenseNet169 model and the ClassifierCNN class, and then creates a class instance with the desired model to train, a dataset name, and an experiment name. You can also change many of the class variables to suit your need.

The `finetuning_layers` variable is the number of layers to fine-tune, into the model. The default is set to `20`. For example, at the default value, all model layers will be frozen during training except for the last 20 layers.

Another example, if you want to train the network from scratch (no pretraining), set `from_scratch = True`, as shown above.

The `run_experiments.py` file provides the same example, and you can modify the file to run your own series of experiments.

You can change many other variables, such as:

* Image Dimensions: `height` and `width` (make sure `height = weight`), default is `224`.
* Batch Size: `batch_size`, default set to `8`.
* Loss: `loss`, set to `'default'` for categorical_crossentropy, or to `'focal'` for focal loss.
* Initial Learning Rate: `lrate` with default value of `0.0001`.
* Early Stopping Patience: `es_patience` with default set to `10`.

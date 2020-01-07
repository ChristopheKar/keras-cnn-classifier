# Import pre-trained Keras CNN Model
from keras.applications.densenet import DenseNet169
# Import ClassifierCNN class
from Classifier import ClassifierCNN

# Instantiate class with CNN model, dataset directory, and experiment name
cls = ClassifierCNN(DenseNet169, 'Pets', 'd169_pets_finetuned')
# Set number of layers to fine-tune
cls.finetuning_layers = 224
# Launch Training
cls.train()

# Instantiate new class
cls = ClassifierCNN(DenseNet169, 'Pets', 'd169_pets_scratch')
# Set to training from scratch
cls.finetuning_layers = 0
cls.from_scratch = True
# Launch training
cls.train()

# -*- coding: utf-8 -*-

# Import standard libraries
import os
import math
import time
from shutil import copyfile

# Import main libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Import scikit-learn metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# Import keras utilities
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

# Import keras model layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization

# Import keras CNN models
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169


class ClassifierCNN:

    def __init__(self, backbone, experiment_name, dataset=None, class_mode='binary'):

        # Clear session variables
        if K.backend() == 'tensorflow':
            K.clear_session()

        # Set up root paths
        self.models_root_path = 'models'
        self.data_root = 'datasets'
        self.logs_root_path = 'logs'
        # Set up model path
        self.model_name = experiment_name + '.h5'
        self.model_path = os.path.join(self.models_root_path, self.model_name)
        # Set up log path
        self.logs_name = experiment_name
        self.logs_path = os.path.join(self.logs_root_path, self.logs_name)
        # Set up model backbone
        self.backbone = backbone

        # Set image and data variables
        self.height = 224
        self.width = 224
        self.batch_size = 8

        # Set default training variables
        self.init_lrate=0.0001
        self.finetuning_layers = 20
        self.from_scratch = False
        self.augment_data = True
        self.class_weights = 'balanced'
        self.warmup_epochs = 5
        self.training_epochs = 25

        # Set default callbacks
        self.earlystopping = False
        self.earlystop_patience = 10
        self.log_to_tensorboard = False
        self.log_to_csv = True
        self.use_lr_decay = False
        self.use_lr_plateau = True

        # Create metrics log directory if not available
        self.metrics_path = os.path.join(self.logs_path, 'metrics')
        if not os.path.exists(self.metrics_path):
            os.makedirs(self.metrics_path)

        # Set up dataset with default paths
        if dataset is not None:
            self.setup_dataset(class_mode, dataset)


    def setup_dataset(self, class_mode, dataset_path, train_path='train', valid_path='valid', mode='directory', use_default_root=True, x_col=None, y_col=None):

        flag_path = 0

        if mode == 'directory':
            # Set dataset paths
            if use_default_root is True:
                self.dataset_root = os.path.join(self.data_root, dataset_path)
                self.train_path = os.path.join(self.dataset_root, train_path)
                self.valid_path = os.path.join(self.dataset_root, valid_path)
            else:
                self.train_path = train_path
                self.valid_path = valid_path
            # Check for validity of dataset paths
            flag_path += not(os.path.isdir(self.train_path))
            flag_path += not(os.path.isdir(self.valid_path))

        elif mode == 'dataframe':
            # Set dataset paths
            if use_default_root is True:
                self.dataset_root = os.path.join(self.data_root, dataset_path)
                self.train_path = os.path.join(self.dataset_root, train_path)
                self.valid_path = os.path.join(self.dataset_root, valid_path)
            else:
                self.dataset_root = dataset_path
                self.train_path = train_path
                self.valid_path = valid_path
            # Set datframe columns
            self.xcol_name = x_col
            self.ycol_name = y_col
            # Check for validity of dataset paths
            flag_path += not(os.path.isfile(self.train_path))
            flag_path += not(os.path.isfile(self.valid_path))

        assert (flag_path == 0), "Dataset path error"

        self.dataset_mode = mode
        self.set_class_mode(class_mode)


    def set_class_mode(self, class_mode):

        self.class_mode = class_mode
        # Set training loss and activation functions
        if self.class_mode == 'binary':             # Binary classification
            self.activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
        elif self.class_mode == 'categorical':      # Multi-class classification
            self.activation = 'softmax'
            self.loss = 'categorical_crossentropy'

    def load_dataset_generators(self):

        # Create training generator and augment training data
        if self.augment_data is True:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=90,
                width_shift_range=0.4,
                height_shift_range=0.4,
                shear_range=0.4,
                zoom_range=0.4,
                horizontal_flip=True,
                fill_mode='nearest')
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)

        # Don't augment data in the validation generator
        validation_datagen = ImageDataGenerator(rescale=1./255)

        if self.dataset_mode == 'directory':
            # Training generator
            self.train_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size = (self.height, self.width),
                batch_size = self.batch_size,
                class_mode = self.class_mode,
                shuffle = True)
            # Validation generator
            self.validation_generator = validation_datagen.flow_from_directory(
                self.valid_path,
                target_size = (self.height, self.width),
                batch_size = self.batch_size,
                class_mode = self.class_mode,
                shuffle = True)
            # Set number of classes
            num_classes_train = self.train_generator.num_classes
            num_classes_valid = self.validation_generator.num_classes

        elif self.dataset_mode == 'dataframe':
            # Training generator
            self.train_generator = train_datagen.flow_from_dataframe(
                dataframe = pd.read_csv(self.train_path).astype('str'),
                directory = self.dataset_root,
                x_col = self.xcol_name,
                y_col = self.ycol_name,
                target_size = (self.height, self.width),
                batch_size = self.batch_size,
                class_mode = self.class_mode,
                shuffle = True)
            # Validation generator
            self.validation_generator = validation_datagen.flow_from_dataframe(
                dataframe = pd.read_csv(self.valid_path).astype('str'),
                directory = self.dataset_root,
                x_col = self.xcol_name,
                y_col = self.ycol_name,
                target_size = (self.height, self.width),
                batch_size = self.batch_size,
                class_mode = self.class_mode,
                shuffle = True)
            # Set number of classes
            num_classes_train = len(self.train_generator.class_indices)
            num_classes_valid = len(self.validation_generator.class_indices)

        # Set number of samples
        self.num_train_samples = self.train_generator.samples
        self.num_valid_samples = self.validation_generator.samples
        # Check if number of training classes == number of validation classes
        assert num_classes_train == num_classes_valid, "number of classes in training and validation sets do not match"

        # Set class-level number of classes
        self.num_classes = num_classes_train

        if self.class_weights == 'balanced':
            self.class_weights = class_weight.compute_class_weight(
                                        'balanced',
                                        np.unique(self.train_generator.classes),
                                        self.train_generator.classes)
        else:
            self.class_weights = None

    def draw_plots(self):

        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.savefig(os.path.join(self.metrics_path, 'acc.png'))
        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig(os.path.join(self.metrics_path, 'loss.png'))

    def evaluate(self):

        img_datagen = ImageDataGenerator(rescale=1./255)

        img_generator = img_datagen.flow_from_directory(
            self.val_dir,
            target_size = (self.height, self.width),
            batch_size = 1,
            shuffle = False,
            class_mode = self.class_mode)

        img_generator.reset()
        classes = img_generator.classes[img_generator.index_array][0]
        nb_samples = len(classes)

        img_generator.reset()
        Y_pred = self.model.predict_generator(img_generator, steps=nb_samples)
        pred_prob = np.array([a[0] for a in Y_pred])
        pred_classes = pred_prob.round().astype('int32')
        self.pred_classes = pred_classes
        self.pred_prob = pred_prob

        metrics = ''
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(classes, pred_classes)
        metrics = metrics + 'Accuracy: {:f}\n'.format(accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(classes, pred_classes)
        metrics = metrics + 'Precision: {:f}\n'.format(precision)
        # recall: tp / (tp + fn)
        recall = recall_score(classes, pred_classes)
        metrics = metrics + 'Recall: {:f}\n'.format(recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(classes, pred_classes)
        metrics = metrics + 'F1 score: {:f}\n'.format(f1)
        # ROC AUC
        auc = roc_auc_score(classes, pred_prob)
        metrics = metrics + 'ROC AUC: {:f}\n'.format(auc)
        # confusion matrix
        matrix = confusion_matrix(classes, pred_classes)
        metrics = metrics + 'Confusion Matrix:\n' + str(matrix) + '\n'

        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        metrics = metrics + 'Max Training Accuracy: {:f}\n'.format(max(acc))
        metrics = metrics + 'Max Validation Accuracy: {:f}\n'.format(max(val_acc))
        metrics = metrics + 'Min Training Loss: {:f}\n'.format(max(loss))
        metrics = metrics + 'Min Validation Loss: {:f}\n'.format(min(val_loss))
        metrics = metrics + 'Training Time: {:f} hours\n'.format(self.total_time)

        self.metrics_path = os.path.join(self.logs_path, 'metrics')
        if not os.path.exists(self.metrics_path):
            os.makedirs(self.metrics_path)

        f = open(os.path.join(self.metrics_path, 'metrics.txt'), 'w')
        f.write(metrics)
        f.close()

        copyfile(os.path.realpath(__file__),
                 os.path.join(self.metrics_path, 'train.py'))

        print('Accuracy: %f' % accuracy)
        print('Precision: %f' % precision)
        print('Recall: %f' % recall)
        print('F1 score: %f' % f1)
        print('ROC AUC: %f' % auc)
        print(matrix)

        self.draw_plots()

    def create_fclayer(self, conv_base, pre=False):

        conv_base.trainable = False

        self.model = Sequential()
        if pre is False:
            self.model.add(conv_base)
        else:
            self.model.add(conv_base.layers[0])
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.num_classes, activation=self.activation))

    def fine_tune(self, conv_base):

        conv_base.trainable = True

        if self.from_scratch is False:
            for layer in conv_base.layers[:-self.finetuning_layers]:
                layer.trainable = False

    def step_decay(self, epoch):

        min_lrate = 0.00000001
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate*math.pow(drop, math.floor((1+epoch)/epochs_drop))
        if lrate < min_lrate:
            lrate = min_lrate
        return lrate

    def compile_model(self):

        adam = Adam(lr=self.init_lrate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=None,
                    decay=0.0,
                    amsgrad=True)

        self.model.compile(loss=self.loss,
                           optimizer=adam,
                           metrics=['accuracy'])

    def fit_model(self, steps='warmup'):

        callbacks = []
        lr_callback = []

        # Save best models
        checkpoint = ModelCheckpoint(self.model_path,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)
        callbacks.append(checkpoint)

        # Reduce learning rate on plateau
        if self.use_lr_plateau is True:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.3,
                                          patience=5,
                                          min_delta=0.0001,
                                          verbose=1,
                                          min_lr=0.00000001)
            callbacks.append(reduce_lr)
            lr_callback.append(reduce_lr)

        # Set up learning rate schedule
        elif self.use_lr_decay is True:
            schedule_lr = LearningRateScheduler(self.step_decay)
            callbacks.append(schedule_lr)
            lr_callback.append(schedule_lr)


        # Early Stopping on Plateau
        if self.earlystopping is True:
            earlystop = EarlyStopping(monitor='val_acc',
                                      mode='max',
                                      verbose=1,
                                      patience=self.earlystop_patience)
            callbacks.append(earlystop)

        # Log to tensorboard
        if self.log_to_tensorboard is True:
            tensorboard = TensorBoard(log_dir=self.logs_path,
                                      histogram_freq=0,
                                      write_graph=True,
                                      write_images=False)
            callbacks.append(tensorboard)

        # Log to CSV
        if self.log_to_csv is True:
            csv_logger = CSVLogger(
                            os.path.join(self.metrics_path, 'training.log'),
                            separator=',',
                            append=False)
            callbacks.append(csv_logger)

        # fit model
        if steps == 'warmup':
            self.history_ = self.model.fit_generator(
                    self.train_generator,
                    steps_per_epoch = self.num_train_samples//self.batch_size,
                    epochs = self.warmup_epochs,
                    validation_data = self.validation_generator,
                    validation_steps = self.num_valid_samples//self.batch_size,
                    callbacks = lr_callback,
                    class_weight = self.class_weights)

        elif steps == 'finetune':
            self.history = self.model.fit_generator(
                    self.train_generator,
                    steps_per_epoch = self.num_train_samples//self.batch_size,
                    epochs = self.training_epochs,
                    validation_data = self.validation_generator,
                    validation_steps = self.num_valid_samples//self.batch_size,
                    callbacks = callbacks,
                    class_weight = self.class_weights)

    def train(self):

        self.load_dataset_generators()

        # Creating model
        if isinstance(self.backbone, str):
            model_path = os.path.join(self.models_root, self.backbone)
            base_model = load_model(model_path)
            for i in range(6):
                base_model._layers.pop()
            base_model.summary()
            self.create_fclayer(base_model, True)
        else:
            if self.from_scratch is True:
                weights = None
            else:
                weights = 'imagenet'

            base_model = self.backbone(
                                include_top=False,
                                input_shape = (self.height, self.width, 3),
                                weights=weights)
            self.create_fclayer(base_model)

        self.compile_model()
        start_time = time.time()
        self.fit_model('warmup')
        self.fine_tune(base_model)
        self.compile_model()
        self.fit_model('finetune')
        end_time = time.time()
        self.total_time = (time.time() - start_time) / 3600
        self.evaluate()

# import standard libraries
import os
import math
import time
from shutil import copyfile

# import main library
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# import scikit-learn metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# import keras utilities
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

# import keras model layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization

# import keras CNN models
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169


class ClassifierCNN:

    def __init__(self, backbone, dataset, model_name):

        if K.backend() == 'tensorflow':
            K.clear_session()

        self.home = os.environ['HOME']

        self.global_root = os.path.join(self.home, 'wrist/classification')
        self.models_root = os.path.join(self.global_root, 'models')
        self.logs_root = os.path.join(self.global_root, 'logs')
        self.datasets_root = os.path.join(self.home, 'wrist/datasets')

        self.model_name = model_name + '.h5'
        self.logs_name = model_name

        self.model_path = os.path.join(self.models_root, self.model_name)
        self.logs_path = os.path.join(self.logs_root, self.logs_name)

        self.backbone = backbone
        self.define_dataset(dataset)

        if self.classes == 1:
            self.class_mode = 'binary'
            self.activation = 'sigmoid'
        elif self.classes > 1:
            self.class_mode = 'categorical'
            self.activation = 'softmax'


        self.height = 224
        self.width = 224
        self.batch_size = 8
        self.loss = 'default'
        self.lrate=0.0001
        self.layers = 19
        self.scratch = False
        self.total_time = 0
        self.es_patience = 10

        self.metrics_path = os.path.join(self.logs_path, 'metrics')
        if not os.path.exists(self.metrics_path):
            os.makedirs(self.metrics_path)

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
        # kappa
        kappa = cohen_kappa_score(classes, pred_classes)
        metrics = metrics + 'Cohens Kappa: {:f}\n'.format(kappa)
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

    def define_dataset(self, dataset):

        if dataset == 'Pets':
            dataset_base = os.path.join(self.datasets_root, 'aub_disp')
            self.num_train = 3419
            self.num_val = 380
            self.classes = 1

        self.train_dir = os.path.join(dataset_base, 'train')
        self.val_dir = os.path.join(dataset_base, 'valid')
        self.data_classes = [f for f in os.listdir(self.train_dir) if os.path.isdir(f)]

    def load_dataset_generators(self):

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=90,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.4,
            zoom_range=0.4,
            horizontal_flip=True,
            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(
            rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size = (self.height, self.width),
            batch_size = self.batch_size,
            class_mode = self.class_mode,
            classes = self.data_classes)

        self.validation_generator = validation_datagen.flow_from_directory(
            self.val_dir,
            target_size = (self.height, self.width),
            batch_size = self.batch_size,
            class_mode = self.class_mode,
            classes = self.data_classes)

        self.class_weights = class_weight.compute_class_weight(
                                        'balanced',
                                        np.unique(self.train_generator.classes),
                                        self.train_generator.classes)

    def create_fclayer(self, conv_base, pre=False):

        conv_base.trainable = False

        self.model = Sequential()
        if pre is False:
            self.model.add(conv_base)
        else:
            self.model.add(conv_base.layers[0])
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.classes, activation=self.activation))

    def fine_tune(self, conv_base):

        conv_base.trainable = True

        for layer in conv_base.layers[:-self.layers]:
            layer.trainable = False

    def step_decay(self, epoch):

        min_lrate = 0.00000001
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        if lrate < min_lrate:
            lrate = min_lrate
        return lrate

    def compile_model(self):

        adam = Adam(lr=self.lrate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=None,
                    decay=0.0,
                    amsgrad=True)

        if self.loss == 'default' and self.classes == 1:
            loss_f = 'binary_crossentropy'
        if self.loss == 'default' and self.classes > 1:
            loss_f = 'categorical_crossentropy'
        if self.loss == 'focal' and self.classes == 1:
            loss_f = binary_focal_loss(alpha=.25, gamma=2)
        if self.loss == 'focal' and self.classes > 1:
            loss_f = categorical_focal_loss(alpha=.25, gamma=2)

        self.model.compile(loss=loss_f,
                           optimizer=adam,
                           metrics=['accuracy'])

    def fit_model(self, steps='init'):

        # reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.1,
                                      patience=3,
                                      min_delta=0.0001,
                                      verbose=1,
                                      min_lr=0.0000001)

        # save best models
        checkpoint = ModelCheckpoint(self.model_path,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)
        # log to tensorboard
        tensorboard = TensorBoard(log_dir=self.logs_path,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)

        # set up learning rate schedule
        lrate = LearningRateScheduler(self.step_decay)

        # Early Stopping on Plateau
        es = EarlyStopping(monitor='val_acc',
                           mode='max',
                           verbose=1,
                           patience=self.es_patience)

        csv_logger = CSVLogger(os.path.join(self.metrics_path, 'training.log'),
                               separator=',',
                               append=False)

        # fit model
        if steps == 'init':
            self.history = self.model.fit_generator(
                                self.train_generator,
                                steps_per_epoch=150,
                                epochs=25,
                                validation_data=self.validation_generator,
                                validation_steps=self.num_val//self.batch_size,
                                callbacks=[checkpoint, reduce_lr],
                                class_weight=self.class_weights)

        elif steps == 'fine':

            self.history = self.model.fit_generator(
                                self.train_generator,
                                steps_per_epoch=150,
                                epochs=100,
                                validation_data=self.validation_generator,
                                validation_steps=self.num_val//self.batch_size,
                                callbacks=[checkpoint, reduce_lr, es, csv_logger],
                                class_weight=self.class_weights)

    def train(self):

        # creating model
        if isinstance(self.backbone, str):
            model_path = os.path.join(self.models_root, self.backbone)
            base_model = load_model(model_path)
            for i in range(6):
                base_model._layers.pop()
            base_model.summary()
            self.create_fclayer(base_model, True)
        else:
            if self.scratch is True:
                w = None
            else:
                w = 'imagenet'

            base_model = self.backbone(include_top=False,
                                       input_shape = (self.height,self.width,3),
                                       weights=w)
            self.create_fclayer(base_model)

        self.load_dataset_generators()
        self.compile_model()
        start_time = time.time()
        self.fit_model('init')
        checkpoint_1 = time.time()
        self.fine_tune(base_model)
        self.compile_model()
        checkpoint_2 = time.time()
        self.fit_model('fine')
        end_time = time.time()
        total_time = (checkpoint_1 - start_time) + (end_time - checkpoint_2)
        self.total_time = total_time / 3600
        self.evaluate()

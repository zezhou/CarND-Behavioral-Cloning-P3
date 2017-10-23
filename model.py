
import sklearn
import keras

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Lambda,\
        Convolution2D, MaxPooling2D , Cropping2D,\
        Dropout, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.backend import stop_gradient
import pandas as pd
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

def generator(samples, data_dir = "data/", batch_size=32):
    """
    The real batch_size is batch_size * 2, because of image flip
    """
    if data_dir is None:
        data_dir = "data/"
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for index, row in batch_samples.iterrows():
                center_image, left_image, right_image = row['center'], row['left'], row['right']
                steering = row['steering']
                file_path = data_dir + center_image
                orig_image = cv2.imread(file_path)
                
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                measurement = float(steering)
                images.append(image)
                angles.append(measurement)

                # use left and right image for data augmenting
                left_image_path = data_dir + left_image.strip()
                orig_left_image = cv2.imread(left_image_path)
                left_image_content = cv2.cvtColor(orig_left_image, cv2.COLOR_BGR2RGB)
                images.append(left_image_content)
                angles.append(measurement + 0.2)
                right_image_path = data_dir + right_image.strip()
                orig_right_image = cv2.imread(right_image_path)
                #from project guid, I found a comments says that drive.py sends RGB images to the model;
                #cv2.imread() reads images in BGR format
                #using following code to fix this problem
                right_image_content = cv2.cvtColor(orig_right_image, cv2.COLOR_BGR2RGB)
                images.append(right_image_content)
                angles.append(measurement - 0.2)

                # Flipping for data augmenting
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield shuffle(inputs, outputs)

def load_example_datasets(training_dir = None, batch_size = 32):
    if training_dir is None:
        logs = pd.read_csv("data/driving_log.csv")
        # collected training data for special cases
        augment_logs = pd.read_csv("data/driving_log_augment.csv", header = 0, names = ["center","left","right",
                "steering","throttle","brake","speed"])
        # anothor collected training data for special cases
        augment_logs3 = pd.read_csv("data/driving_log_augment3.csv", header = 0, names = ["center","left","right",
                "steering","throttle","brake","speed"])
        total_logs = pd.concat([logs, augment_logs, augment_logs3], ignore_index=True)
    elif type(training_dir) is str:
        logs = pd.read_csv(training_dir + "driving_log.csv", header = 0, names = ["center","left","right",
                "steering","throttle","brake","speed"])
        total_logs = logs
    else:
        raise ValueError("drive_log_path is wrong")
    train_examples, valid_examples =  train_test_split(total_logs)
    len_train = len(train_examples) 
    len_valid = len(valid_examples)
    train_gen = generator(train_examples, data_dir = training_dir, batch_size = batch_size)
    valid_gen = generator(valid_examples, data_dir = training_dir, batch_size = batch_size)
    print ("origin train exmples num: %d" % len_train)
    print ("origin valid exmples num: %d" % len_valid)
    return train_gen, valid_gen, len_train, len_valid

def create_model(input_shape = (160, 320, 3)):
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 -0.5, input_shape = input_shape))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(12, 3, 3, activation='relu', border_mode='same', name='block1_conv1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    model.add(Convolution2D(12, 3, 3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(50,  activation="relu", name = "fc2"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, name = "fc3"))
    model.compile(loss='mse', optimizer = 'adam')
    model.summary()
    return model

def train(train_gen, valid_gen, model, len_train, len_valid, dump_file, pretrain = None, epochs = 5):
    if pretrain is not None:
        model.load_weights(pretrain)
    model.fit_generator(generator= train_gen, samples_per_epoch= len_train,
            nb_epoch=epochs, validation_data=valid_gen, nb_val_samples = len_valid)
    model.save(dump_file)
    print ('model saved to %s.' % dump_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Behavioral cloning.')
    parser.add_argument('--pretrain', default=None, help='pretrain parameters file')
    parser.add_argument('--training_dir', default=None, help='training features dir')
    parser.add_argument('--dump_file', default="model2.h5", help='model parameters dumping file')
    parser.add_argument('--epochs', default=5, type=int, help="training number")

    args = parser.parse_args()
    from keras import backend as K
    print ( "keras version:" + keras.__version__)
    train_gen,valid_gen, len_train, len_valid = load_example_datasets(training_dir = args.training_dir)
    model = create_model()
    train(train_gen, valid_gen, model, len_train * 4, len_valid *4,\
            dump_file = args.dump_file, pretrain = args.pretrain, epochs = args.epochs)
    K.clear_session() # fix Keras 1.2.1 exits issue
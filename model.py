
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

def generator(samples, batch_size=32):
    """
    The real batch_size is batch_size * 2, because of image flip
    """
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
                file_path = "data/" + center_image
                orig_image = cv2.imread(file_path)
                
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                measurement = float(steering)
                images.append(image)
                angles.append(measurement)
                # use left and right image
                left_image_path = "data/" + left_image.strip()
                orig_left_image = cv2.imread(left_image_path)
                left_image_content = cv2.cvtColor(orig_left_image, cv2.COLOR_BGR2RGB)
                images.append(left_image_content)
                angles.append(measurement + 0.2)
                right_image_path = "data/" + right_image.strip()
                orig_right_image = cv2.imread(right_image_path)
                right_image_content = cv2.cvtColor(orig_right_image, cv2.COLOR_BGR2RGB)
                images.append(right_image_content)
                angles.append(measurement - 0.2)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)
            inputs = np.array(images)
            outputs = np.array(angles)
            yield shuffle(inputs, outputs)

def load_example_datasets(drive_log_path = "data/driving_log.csv", batch_size = 32):
    logs = pd.read_csv(drive_log_path)
    # collected training data for special case
    augment_logs = pd.read_csv("data/driving_log_augment.csv", header = 0, names = ["center","left","right",
            "steering","throttle","brake","speed"])
    # anothor collected training data for special case
    augment_logs3 = pd.read_csv("data/driving_log_augment3.csv", header = 0, names = ["center","left","right",
            "steering","throttle","brake","speed"])
    merge_logs = pd.concat([logs, augment_logs, augment_logs3], ignore_index=True)
    train_examples, valid_examples =  train_test_split(merge_logs)
    len_train = len(train_examples) 
    len_valid = len(valid_examples)
    train_gen = generator(train_examples, batch_size = batch_size)
    valid_gen = generator(valid_examples, batch_size = batch_size)
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

def train(train_gen, valid_gen, model, len_train, len_valid):
    model.fit_generator(generator= train_gen, samples_per_epoch= len_train,
            nb_epoch=5, validation_data=valid_gen, nb_val_samples = len_valid)
    model.save("model.h5")
    print ('model saved.')

if __name__ == "__main__":
    from keras import backend as K
    print ( "keras version:" + keras.__version__)
    train_gen,valid_gen, len_train, len_valid = load_example_datasets()
    model = create_model()
    train(train_gen, valid_gen, model, len_train * 4, len_valid *4)
    K.clear_session() # fix Keras 1.2.1 exits issue
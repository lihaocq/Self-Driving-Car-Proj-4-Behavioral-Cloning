import csv
import cv2
import numpy as np
from scipy import ndimage
import os
import matplotlib.pyplot as plt
import random
random.seed(2)

from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

import sklearn
from sklearn.model_selection import train_test_split
import math


##### set perameters

validation_data_ratio = 0.2   # ratio of original data that used for validation
ch, row, col = 3, 160, 320    # orginal image size: channel, row, col
correction = 0.2              # for left and right camera image, correct the steering angle.
ratio_straight_keep = 0.4     # only keep a part of straight road images

learnning_rate = 0.001        # learning rate of model
epochs_num = 5
batch_size = 32
dropout_ratio = 0.5

####################### Functions for use later ########################

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # collect images of left camera, central camera, right camera.
                for i in range(3):
                    name = 'data/IMG/'+batch_sample[i+0].split('/')[-1]
                    #image = cv2.imread(current_path)
                    # cv2.imread will get images in BGR format, while drive.py uses RGB. So ndimage is used here.
                    image = ndimage.imread(name)

                    # add a correction for left camera image and right camera image
                    if i == 0:
                        angle = float(batch_sample[3])
                    elif i == 1:
                        angle = float(batch_sample[3]) + correction
                    elif i==2:
                        angle = float(batch_sample[3]) - correction

                    images.append(image)
                    angles.append(angle)

            ##### augment the data by flipping images and take opposite sign of steering angle
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1)


            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)



####################### main body #################################



############ Get the image data and measurements
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    # skip the first line
    firstline = True
    for line in reader:
        if firstline == True:
            firstline = False
            continue

        # if steering angle is 0, the possibility of keep is ratio_straight_keep
        if float(line[3]) - 0 <0.000001:
            a = random.random()
            if a < ratio_straight_keep:
                continue

        samples.append(line)

# split data for training and validation
train_samples, validation_samples = train_test_split(samples, test_size=validation_data_ratio)

# use generator function to get batch data for traninig
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



########### set neural netwrok model
model = Sequential()
# normalize the input data
# pre-process the input data: normalize it and shift it by 0.5.
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))

# cropping the image to get rid of distractive informaiton. remove 60 pixels at top, remove 25 pixels at buttom.
model.add(Cropping2D(cropping=((60,25),(0,0))))

# 1st #: number of convolution filters to use
# 2nd #: the number of rows in each convolution kernel
# 3rd #: number of columns in each convolution kernel
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(dropout_ratio))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
print(model.summary())
# for the loss function, we use mean squre erro, MSE, which is different with the cross-entropy function.
# it is beacause that this is a regression rather than classification network.
adam = optimizers.Adam(lr=learnning_rate)
model.compile(loss='mse', optimizer = adam)

history_object = model.fit_generator(train_generator,
            steps_per_epoch = math.ceil(len(train_samples)/batch_size),
            validation_data = validation_generator,
            validation_steps = math.ceil(len(validation_samples)/batch_size),
            epochs=epochs_num, verbose=1)

# save the history data for loss function
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('hisotry.png')

model.save('model.h5')

exit()

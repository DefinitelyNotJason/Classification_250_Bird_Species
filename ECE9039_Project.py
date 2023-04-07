import os
import shutil
import glob as gb
import random
import tensorflow as tf
import keras_tuner as KT
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications.vgg16 import preprocess_input as v16_preprocess_input
from keras.applications.vgg19 import preprocess_input as v19_preprocess_input
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, Nadam
from tensorflow import keras

# Enable GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ------Dataset------
# 1) Data loading
trainpath = './train/'
testpath = './test/'
valpath = './valid/'
# visualize some images in some train folders
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 10))
folders = ['ABBOTTS BABBLER', 'BARN OWL', 'CANARY', 'GOLDEN EAGLE']
random_samples = [random.sample(os.listdir(trainpath + folders[0]), 4),
                  random.sample(os.listdir(trainpath + folders[1]), 4),
                  random.sample(os.listdir(trainpath + folders[2]), 4),
                  random.sample(os.listdir(trainpath + folders[3]), 4)]
for a in range(4):
    for b in range(4):
        ax[a, b].imshow(plt.imread(trainpath + folders[a] + '/' + random_samples[a][b]))
        ax[a, b].set_title(folders[a])
plt.tight_layout()
plt.show()


# 2) Training dataset count up
def dataset_info(path, which):
    print(f'Info of {which} data: ')
    total_count = 0
    total_folder = len(os.listdir(path))
    min_count = float('inf')
    max_count = float('-inf')
    for folder in os.listdir(path):
        files = gb.glob(pathname=str(path + folder + '/*.jpg'))
        total_count += len(files)
        min_count = min(min_count, len(files))
        max_count = max(max_count, len(files))
        # print(f'{len(files)} images in {folder}')
    print(f'There are total %d images split into %d folder in the {which} subset.' % (total_count, total_folder))
    print(f'Minimum %d images in a class and maximum %d images in a class.' % (min_count, max_count))
    return [total_count, min_count, max_count]


[train_total_count, train_min, train_max] = dataset_info(trainpath, 'training')


# 3) Data Balancing
# there are a total of 500 categories in each subset,
# however, due to the computational resources' constraint,
# we decided to use only the first 250 species to train our model.
def train_balance(path):
    folder_count = 0
    for folder in os.listdir(path):
        if folder_count < 250:
            files = gb.glob(pathname=str(path + folder + '/*.jpg'))
            # keep each class contains no more than 200 images
            if len(files) > 200:
                ran_del = random.sample(os.listdir(path + folder), len(files) - 200)
                for file in ran_del:
                    os.remove(path + folder + '/' + file)
        # drop the last 250 class
        else:
            shutil.rmtree(path + folder)
        folder_count += 1


train_balance(trainpath)
dataset_info(trainpath, 'train folder')
train_balance(valpath)
dataset_info(valpath, 'valid folder')


# ------Data Preprocessing------
# 1) Remove Corrupt Image
def remove_corrupt(path, which):
    corrupt = 0
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            fpath = os.path.join(folder_path, file)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                corrupt += 1
                # drop corrupted image
                os.remove(fpath)
    print(f'Total %d corrupt images in {which} dataset.' % corrupt)
    print('All corrupt images were dropped.' + '\n')


remove_corrupt(trainpath, 'train')

# 2) Data preprocessing & Data transformation
# using vgg19 mode to preprocess the input observation
train_gen = ImageDataGenerator(preprocessing_function=v19_preprocess_input, rotation_range=10,
                               width_shift_range=0.2, height_shift_range=0.2,
                               shear_range=0.1, zoom_range=0.1, validation_split=0.2,
                               featurewise_std_normalization=False)
test_gen = ImageDataGenerator(preprocessing_function=v19_preprocess_input)
# train valid split directly processed on train folder subset
train_set = train_gen.flow_from_directory(trainpath, subset='training',
                                          color_mode='rgb', target_size=(224, 224),
                                          batch_size=64, class_mode='categorical')
val_set = train_gen.flow_from_directory(trainpath, subset='validation',
                                        color_mode='rgb', target_size=(224, 224),
                                        batch_size=64, class_mode='categorical')
test_set = test_gen.flow_from_directory(valpath,
                                        color_mode='rgb', target_size=(224, 224),
                                        batch_size=64, class_mode='categorical')
labels = {value: key for key, value in train_set.class_indices.items()}
print(labels)
# plot random images from image flow
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
for a in range(3):
    for b in range(3):
        img, label = train_set.next()
        ax[a, b].imshow(img[0])
plt.tight_layout()
plt.show()

# ------Model Learning------
# VGG16 modeling
# base_model = VGG16(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
# base_model.trainable = True
# # enable trainable for last set of convolutional layers
# for layer in base_model.layers:
#     if layer.name not in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
#         layer.trainable = False
# base_model.summary()
#
# def build_model(hp):
#     model = keras.Sequential()
#     model.add(base_model)
#     # add a flatten layer
#     model.add(keras.layers.Flatten())
#     # add FC layer(s)
#     for i in range(hp.Int('Num_FC_Layers', 1, 3)):
#         model.add(keras.layers.Dense(units=hp.Choice('FC_Layer_Units', values=[2048, 4096]),
#                                      activation='relu'))
#     # add dropout layer to prevent overfitting
#     for i in range(hp.Int('Num_Dropout_Layers', 0, 1)):
#         model.add(keras.layers.Dropout(hp.Choice('Dropout_Rate', values=[0.1, 0.2])))
#     # add output layer
#     model.add(keras.layers.Dense(units=250, activation='softmax', name='Output_Layer'))
# 
#     # Adam and Nadam are two candidate optimizers
#     opt = [Adam(learning_rate=hp.Choice('Learning_Rate', values=[1e-4, 5e-5, 1e-5])),
#            SGD(learning_rate=hp.Choice('Learning_Rate', values=[1e-4, 5e-5, 1e-5]))]
#     # here 0 represents the Adam and 1 represents the SGD
#     model.compile(optimizer=opt[hp.Choice('Optimizer', values=[0, 1])],
#                   loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
# 
# tuner = KT.RandomSearch(hypermodel=build_model, objective="val_accuracy", max_trials=30,
#                         directory='./tuner', project_name='ece9039_project_tuner',
#                         executions_per_trial=2)
# tuner.search_space_summary()
# tuner.search(train_set, epochs=1, validation_data=val_set)
# # save the best model
# best_model = tuner.get_best_models()[0]
# best_model.summary()
# best_model.save('Best_VGG16_Unfit')

# 1) Base model (transfer learning)
base_model = VGG19(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base_model.trainable = True
# enable trainable for last set of convolutional layers
for layer in base_model.layers:
    if layer.name not in ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4']:
        layer.trainable = False
base_model.summary()

# 2) Construct customized output layer and Hyperparameters tuning
# Hyperparameters for tuning:
# keras tuner:
    # number of fc layers
    # fc layer units
    # number of dropout layer
    # dropout rate
    # optimizer
    # learning rate
# epoch

def build_model(hp):
    model = keras.Sequential()
    model.add(base_model)
    # add a flatten layer
    model.add(keras.layers.Flatten())
    # add FC layer(s)
    for i in range(hp.Int('Num_FC_Layers', 1, 3)):
        model.add(keras.layers.Dense(units=hp.Choice('FC_Layer_Units', values=[1024, 2048]),
                                     activation='relu'))
    # add dropout layer to prevent overfitting
    for i in range(hp.Int('Num_Dropout_Layers', 0, 1)):
        model.add(keras.layers.Dropout(hp.Choice('Dropout_Rate', values=[0.1, 0.5])))
    # add output layer
    model.add(keras.layers.Dense(units=250, activation='softmax', name='Output_Layer'))

    # Adam and Nadam are two candidate optimizers
    opt = [Adam(learning_rate=hp.Choice('Learning_Rate', values=[1e-4, 5e-5, 1e-5])),
           Nadam(beta_1=0.9, beta_2=0.999, epsilon=1e-07, learning_rate=hp.Choice('Learning_Rate',
                                                                                  values=[1e-4, 5e-5, 1e-5]))]
    # here 0 represents the Adam and 1 represents the Nadam
    model.compile(optimizer=opt[hp.Choice('Optimizer', values=[0, 1])],
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 3) Fine tuning
tuner = KT.RandomSearch(hypermodel=build_model, objective="val_accuracy", max_trials=30,
                        directory='./tuner', project_name='ece9039_project_tuner_VGG19',
                        executions_per_trial=1)
tuner.search_space_summary()
tuner.search(train_set, epochs=5, validation_data=val_set)
# save the best model
best_model = tuner.get_best_models()[0]
best_model.summary()
best_model.save('Best_VGG19_Unfit')

# 4) Find the best epoch value
# set up early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
best_model = keras.models.load_model('Best_VGG19_Unfit')
history = best_model.fit(train_set, epochs=50, validation_data=val_set, callbacks=[callback])
best_model.save('Best_VGG19_Fitted')

# 5) Plot epoch vs validation loss and epoch vs accuracy
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
print('Training Loss:', train_loss)
print('Training Accuracy:', train_accuracy)
print('Validation Loss:', val_loss)
print('Validation Accuracy:', val_accuracy)
# epoch vs loss
fig, ax = plt.subplots(nrows=2, figsize=(12, 10))
ax[0].set_title('Epoch vs Loss')
ax[0].plot(train_loss)
ax[0].plot(val_loss)
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend(['Training Loss', 'Validation Loss'], loc='upper left')
# epoch vs accuracy
ax[1].set_title('Epoch vs Accuracy')
ax[1].plot(train_accuracy)
ax[1].plot(val_accuracy)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
plt.tight_layout()
plt.show()

# ------Result------
# 1) Fitted model predict on test set
model = keras.models.load_model('Best_VGG19_Fitted')
model.evaluate(test_set)

# 2) Plot samples from prediction
predictions = model.predict(test_set).argmax(axis=-1)
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
index = 0
for i in range(4):
    for j in range(4):
        img, label = test_set.next()
        predicted_label = labels[predictions[index]]
        ax[i, j].set_title('Prediction: '+predicted_label)
        ax[i, j].imshow(img[0])
        ax[i, j].axis("off")
        index += 1
plt.tight_layout()
plt.suptitle("Prediction")
plt.show()

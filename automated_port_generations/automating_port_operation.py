import os 
import shutil
import splitfolders
import tensorflow as tf
from os import listdir
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.metrics import Recall, Precision
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
import warnings
warnings.simplefilter(action='ignore')

#Listing input images
def get_file_ext(file_name):
    split_tup = os.path.splitext(file_name)
    return split_tup[1]

def is_image_file(file_ext):
    if file_ext == '.jpeg' or file_ext == '.jpg' or \
        file_ext == '.gif' or file_ext == '.png':
        return True
    else:
        return False
    
def list_image_files(rootdir):
    images_dirs = listdir(path=rootdir)
    image_dir_dict = {}
    i = 1
    print('-------------------------------------------------------------')
    print(f'Following sub-directories found in "{rootdir}" directory')
    for dirs in images_dirs:
        print(f'{i}: {dirs}')
        i = i+1
    print('-------------------------------------------------------------')
    total_image_count = 0
    for dirs in images_dirs:
        file_list = listdir(path=rootdir + '/' + dirs) 
        image_files = [ file_name for file_name in file_list \
                       if is_image_file(get_file_ext(file_name)) ]
        image_count = len(image_files)
        total_image_count = total_image_count + image_count
        image_dir_dict[dirs]=image_count
    print(f'Total images found = {total_image_count}')
    for key in image_dir_dict.keys():
        dir_image_count = image_dir_dict[key]
        print(f'Directory "{key}" contains {dir_image_count} image files' + \
              f' = {round(dir_image_count/total_image_count,3):.3%}')
        
        
list_image_files('Automating_Port_Operations_dataset')

def create_output_dir(output_folder_name):
    output_folder = './' + output_folder_name
    # Check if the directory exists
    if os.path.exists(output_folder):
        # If it exists, remove it (and all its contents)
        shutil.rmtree(output_folder)
        print(f"Directory {output_folder} has been removed")
    else:
        # If the directory does not exist, do nothing
        print(f"Directory {output_folder} does not exist")
        
create_output_dir('output')

splitfolders.ratio('Automating_Port_Operations_dataset', output='output', seed=1, \
                   ratio=(.8,.0,.2), group_prefix=None)

list_image_files('output/train')
# list_image_files('output/val')
list_image_files('output/test')

train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values to [0, 1]
    validation_split=0.2    # Reserve 20% of data for validation
)
# Directory containing the training images
train_dir = 'output/train'

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),\
    batch_size=32,\
    classes=['ferry_boat', 'gondola', 'sailboat', 'cruise_ship', 'kayak', \
              'inflatable_boat', 'paper_boat', 'buoy', 'freight_boat'], \
    class_mode='categorical',\
    shuffle=True,        # Use 'categorical' for multi-class classification
    subset='training'             # Set as training data
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    train_dir,
   target_size=(224, 224),\
    batch_size=32,\
    classes=['ferry_boat', 'gondola', 'sailboat', 'cruise_ship', 'kayak', \
              'inflatable_boat', 'paper_boat', 'buoy', 'freight_boat'], \
    class_mode='categorical',\
    shuffle=True,        # Use 'categorical' for multi-class classification
    subset='validation'           # Set as validation data
)
# Create a separate ImageDataGenerator for test data with rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Directory containing the test images
test_dir = 'output/test'

# Test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),\
    batch_size=32,\
    classes=['ferry_boat', 'gondola', 'sailboat', 'cruise_ship', 'kayak', \
              'inflatable_boat', 'paper_boat', 'buoy', 'freight_boat'], \
    class_mode='categorical',\
    shuffle=True,        # Use 'categorical' for multi-class classification
)

model = Sequential()
#adding first layer
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

#adding 2 layers
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(9, activation='softmax'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), \
                loss='categorical_crossentropy', \
                metrics=['accuracy',Precision(),Recall()])
print(len(train_generator), len(test_generator))
print(test_generator.classes)
model.fit(train_generator, epochs=20)

# Evaluate the model on the test data
test_loss, test_accuracy,test_precision,test_recall  = model.evaluate(test_generator, batch_size=32)

# Print the test loss and accuracy
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
y_pred_prob = model.predict(test_generator)

# Predicted classes
y_pred = np.argmax(y_pred_prob, axis=1)
print(y_pred)

f, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(test_generator.classes, \
                                        y_pred, \
                                        ax = ax, \
                                        normalize='true', \
                                        values_format='.0%')
plt.title(f'Confusion matrix')
plt.show()
# print(confusion_matrix(test_generator.classes, y_pred))

print(classification_report(test_generator.classes, y_pred))


#Using pretrained model mobilenetv2 and do the same steps again
splitfolders.ratio('Automating_Port_Operations_dataset', output='output', seed=1, \
                   ratio=(.7,.0,.3), group_prefix=None)

list_image_files('output/train')
list_image_files('output/test')

train_datagen1= ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
train_generator1 = train_datagen1.flow_from_directory(
  train_dir,
  target_size=(224,224),
  batch_size=32,
  classes = ['ferry_boat', 'gondola', 'sailboat', 'cruise_ship', 'kayak', \
              'inflatable_boat', 'paper_boat', 'buoy', 'freight_boat'], \
  class_mode ='categorical',
  shuffle=True,
  subset ='training'
)

#Validation generator
validation_generator1 = train_datagen1.flow_from_directory(
  train_dir,
  target_size=(224,224),
  batch_size=32,
  classes=['ferry_boat', 'gondola', 'sailboat', 'cruise_ship', 'kayak', \
              'inflatable_boat', 'paper_boat', 'buoy', 'freight_boat'], \
  class_mode= 'categorical',
  shuffle = True,
  subset ='validation'
)
test_data_gen1 = ImageDataGenerator(rescale=1/255.0)
test_generator1 = test_data_gen1.flow_from_directory(
  test_dir,
  target_size=(224,224),
  batch_size=32,
  classes=['ferry_boat', 'gondola', 'sailboat', 'cruise_ship', 'kayak', \
              'inflatable_boat', 'paper_boat', 'buoy', 'freight_boat'], \
  class_mode='categorical',
  shuffle=True
)

model = Sequential()
base_model = MobileNet(weights='imagenet', include_top=False)
average_pooling_layer = GlobalAveragePooling2D()(base_model.output)
dropout_layer_1 = Dropout(rate=0.2)(average_pooling_layer)
hidden_layer_1 = Dense(256, activation='relu')(dropout_layer_1)
normalization_layer_1 = BatchNormalization()(hidden_layer_1)
dropout_layer_2 = Dropout(rate=0.1)(normalization_layer_1)
hidden_layer_2 = Dense(128, activation='relu')(dropout_layer_2)
normalization_layer_2 = BatchNormalization()(hidden_layer_2)
dropout_layer_3 = Dropout(rate=0.1)(normalization_layer_2)
predictions = Dense(9, activation='softmax')(dropout_layer_3)

model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), \
              loss='categorical_crossentropy', \
              metrics=['accuracy',Precision(),Recall()])

early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(train_generator1,callbacks=[early_stopping_cb], epochs = 50)

history_df = pd.DataFrame(history.history)

history_df.plot(
    figsize=(8, 5), xlim=[0, early_stopping_cb.stopped_epoch], ylim=[0, 1], \
    grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower right")
plt.title('Training/Validation loss & accuracy measured over each epoch')
plt.show()

test_loss, test_acc, test_precision, test_recall = \
    model.evaluate(test_generator1, steps=test_generator1.samples // 32)
print('Metrics obtained on test images')
print(f'Accuracy: {test_acc}, Loss: {test_loss}, ' +\
      f'Precision: {test_precision}, Recall: {test_recall}')

y_test_proba = model.predict(test_generator1)
y_test_pred = np.argmax(y_test_proba, axis=1)

y_test = test_generator1.classes
print(classification_report(y_test, y_test_pred))

f, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_test, \
                                        y_test_pred, \
                                        ax = ax, \
                                        normalize='true', \
                                        values_format='.0%')
plt.title(f'Confusion matrix')
plt.show()


# We created two models.

# A full custom model
# A model baased on MobileNet with top layers replaced

# </ol> The accuracy obtained with a full custom model is much lower than that obtained with MobileNet pre-trained model. For automating port operations, the MobileNet custom model can be used and produces much better results.
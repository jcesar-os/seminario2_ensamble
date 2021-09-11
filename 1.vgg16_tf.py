import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from sklearn.metrics import confusion_matrix

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

skin_df = pd.read_csv('D:/Universidad/Tesis/HAM10000/HAM10000_metadata.csv')
#skin_df = skin_df[skin_df.dx.isin(['mel','nv'])]

df_0 = skin_df[skin_df['dx'] == 'nv'] #Lunar
df_1 = skin_df[skin_df['dx'] == 'mel'] #Melanoma

df_0['label'] = 0
df_1['label'] = 1

SIZE = 256

# Distribution of data into various classes 
from sklearn.utils import resample
print(skin_df['dx'].value_counts())


#Balance data.
# Many ways to balance data... you can also try assigning weights during model.fit
#Separate each classes, resample, and combine back into single dataframe


n_samples=1000
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 

#Combined back to a single dataframe
skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced])

#Check the distribution. All classes should be balanced now.
print(skin_df_balanced['dx'].value_counts())

#Now time to read images based on image ID from the CSV file
#This is the safest way to read images as it ensures the right image is read for the right ID
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('D:/Universidad/Tesis/HAM10000/', '*', '*.jpg'))}

#Define the path and add as a new column
skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)
#Use the path to read images.
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))


n_samples = 5  # number of samples for plotting
# Plotting
fig, m_axs = plt.subplots(2, n_samples, figsize = (4*n_samples, 3*2))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_df_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')

#Convert dataframe column of images into numpy array
X = np.asarray(skin_df_balanced['image'].tolist())
X = X/255.  # Scale values to 0-1. You can also used standardscaler or other scaling methods.
Y = skin_df_balanced['label']  #Assign label values to Y
#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

#Load model wothout classifier/fully connected layers


num_classes = 1

tl = Sequential()

tl.add(VGG16(weights='imagenet', 
             include_top=False,
             pooling='avg',
             input_shape=(SIZE, SIZE, 3)))
tl.add(Dense(256, activation='relu'))
tl.add(Dropout(0.5))
tl.add(Dense(256, activation='relu'))
tl.add(Dropout(0.5))
tl.add(Dense(num_classes, activation='sigmoid'))
#tl.add(Dense(num_classes, activation='softmax'))

tl.layers[0].trainable = False

tl.summary()

tl.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


tl.fit(
    x_train,
    y_train,
    epochs = 10,
    batch_size = 32)

scores = tl.evaluate(x_test, y_test)

print("Test Loss:{}".format(scores[0]))
print("Test Accuracy:{}".format(scores[1]))

#80.19%
SAVE_MODEL = 'vgg16_trainedmodel.hd5'
#tl.save(SAVE_MODEL)



















# deep-learning-model
Familiariazation of Machine Learning for image processing
->Here the project is implemented to classify the image and predict the animals(Dog and cat)
->The project is create using Google colab
->The accuracy is about 87%
Procedure to create a colab notebook:
Pre-requisites:
Step0: Go to runtime and change runtime->Hardware accelerator -> GPU -> save
#Basically the runtime is changed to process a huge datasets.
Step0: We are importing the dataset from kaggle website, hence create a account here.
# Dataset - https://www.kaggle.com/datasets/salader/dogs-vs-cat
(Link for dataset)
Execution steps in (Google colab)
Step 1:
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
•	In kaggle (Go to profile -> account -> API -> create a new API token)
•	Upload the json file and run the code

Step 2:
!kaggle datasets download -d salader/dogs-vs-cats
•	In kaggle(main page(hamburger menu)) copy the API command

Output: Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /root/.kaggle/kaggle.json'
Downloading dogs-vs-cats.zip to /content
 99% 1.06G/1.06G [00:06<00:00, 228MB/s]
100% 1.06G/1.06G [00:06<00:00, 185MB/s]
#Content from kaggle server will be created
#dogs-vs-cats.zip folder will be created

Step 3:
import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
#To unzip the folder downloaded in step 2, test and train data folder will be added each with cat and dog file

Step 4:
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,
Dropout
#to import necessary libraries

Step 5:
# generators
train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/train',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(256,256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/test',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(256,256)
) 
Output: Found 20000 files belonging to 2 classes.
Found 5000 files belonging to 2 classes.
#Basically here the keras generator is used to process the large amount of data, it divides data in batches
#two generators are 1.For train data
                                 2.Validating data
#reshaping of image of any size to (256,256) is done here

Step 6:
# Normalize
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

Step 7:
#create CNN model

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))

Step 8:
model.summary()

Step 9:
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

Step 10:
history = model.fit(train_ds,epochs=10,validation_data=validation_ds)

Step 11:
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

step 12:
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

step 13:
plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

step 14:
plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

step 15:
import cv2

step 16:
test_img = cv2.imread('/content/dog.10010.jpg')

step 17:
plt.imshow(test_img)

step 18:
test_img.shape

step 19:
test_img = cv2.resize(test_img,(256,256))

step 20:
test_input = test_img.reshape((1,256,256,3))

step 21:
model.predict(test_input)


https://colab.research.google.com/drive/1uXeUtLpYukekgSEQyi2CZz4-vy46ju23#scrollTo=syGMbjmU9grR
https://colab.research.google.com/drive/1rS3oF7ZDW65xoPW9XZ0kmxX0JxYc3-nM#scrollTo=syGMbjmU9grR



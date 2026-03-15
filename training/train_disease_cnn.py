import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

IMG_SIZE=224
BATCH_SIZE=32
EPOCHS=10
DATASET_PATH="../dataset/plant_disease"

train_datagen=ImageDataGenerator(rescale=1.0/255,validation_split=0.2,rotation_range=20,
                                 zoom_range=0.2,horizontal_flip=True)

train_data=train_datagen.flow_from_directory(DATASET_PATH,target_size=(IMG_SIZE,IMG_SIZE),
                                        batch_size=BATCH_SIZE,class_mode='categorical',
                                        subset='training')
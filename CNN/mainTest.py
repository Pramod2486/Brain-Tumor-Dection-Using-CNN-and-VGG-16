import cv2
from tensorflow import keras
from keras.models import load_model

from PIL import Image
import numpy as np


model=load_model('BrainTumor10epochs.h5')

image=cv2.imread('C:\\Users\\Sudha\\Desktop\\CSE\\brain_tumor_dataset\\pred\\Y2.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img,axis=0)

# result=model.predict(input_img)
result=(model.predict(input_img) > 0.5).astype("int32")
# result=np.argmax(model.predict_classes(input_img))
print(result)
print(img)
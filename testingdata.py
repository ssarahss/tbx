import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten

# Inisialisasi ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
# Path ke direktori gambar
image_dir = r'C:\TBX11K\val'
# Menggunakan ImageDataGenerator untuk memuat dan memproses gambar
image_generator = datagen.flow_from_directory(
    image_dir,
    target_size=(224, 224), # Resize resolusi gambar menjadi 224x224
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Menggunakan VGG16 sebagai base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Membuat model baru dengan output dari layer terakhir sebelum FCL
output = base_model.layers[-1].output
output = Flatten()(output)
model = Model(inputs=base_model.input, outputs=output)

with tf.device('/device:GPU:0'):
    # Menghasilkan output dari model untuk data gambar
    features = model.predict(image_generator)

print("Output dimensi:", features.shape)

arr = np.hstack((features, image_generator.classes.reshape(-1, 1)))
np.save("val.npy", arr)
loaded_data = np.load("val.npy")
label = loaded_data[:, 25088]

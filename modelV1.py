import numpy as np
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
Model = keras.Model
import os
import cv2
from sklearn.model_selection import train_test_split

def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Decoder
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    up1 = layers.UpSampling2D((2, 2))(conv4)
    
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up2 = layers.UpSampling2D((2, 2))(conv5)
    
    conv6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    up3 = layers.UpSampling2D((2, 2))(conv6)
    
    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(up3)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def load_data(data_dir, img_folder, mask_folder, test_size=0.2, random_state=42):
    img_path = os.path.join(data_dir, img_folder)
    mask_path = os.path.join(data_dir, mask_folder)
    
    img_names = os.listdir(img_path)
    mask_names = os.listdir(mask_path)
    
    X = []
    y = []
    
    for img_name in img_names:
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img = cv2.imread(os.path.join(img_path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))  
            X.append(img)
            
            mask_name = img_name.split('.')[0] + '.png'  
            if mask_name in mask_names:
                mask = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (256, 256))  
                mask = np.expand_dims(mask, axis=-1)
                y.append(mask)
                
    X = np.array(X) / 255.0  
    y = np.array(y) / 255.0  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

data_dir = 'F:/medhavi/EDU/Final_Project/'
img_folder = 'Images'
mask_folder = 'Masks'

X_train, X_test, y_train, y_test = load_data(data_dir, img_folder, mask_folder)

input_shape = X_train.shape[1:]
model = build_model(input_shape)

model.compile(optimizer='adam', loss='binary_crossentropy')


model.fit(X_train, y_train, batch_size=10, epochs=100, validation_data=(X_test, y_test))


model.save('crowd_detection_model_V2.h5')

loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)


predictions = model.predict(X_test)


def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

accuracies = []
for i in range(len(X_test)):
    dice = dice_coefficient(y_test[i], predictions[i])
    accuracies.append(dice)

average_accuracy = np.mean(accuracies)
print("Average Dice Coefficient (Accuracy):", average_accuracy)

import matplotlib.pyplot as plt


num_samples = 5
indices = np.random.choice(len(X_test), num_samples, replace=False)

for idx in indices:
    # Original image
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[idx])
    plt.title('Original Image')
    plt.axis('off')

    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(y_test[idx]), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(predictions[idx]), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()

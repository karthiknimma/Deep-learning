#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install TensorFlow

import tensorflow as tf
print(tf.__version__)

# additional imports
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential


# In[35]:


# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Split data into training and testing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalising the data between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# Printing out the shapes 
print("x_train.shape:", x_train.shape)
print("y_train.shape", y_train.shape)

# number of classes
K = len(set(y_train))
print("number of classes:", K)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


# Instantiating our Sequential model.
model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# We are using earlystop to prevent model acc from deviating from val_acc above 3 epochs
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', 
                           restore_best_weights=True )


# In[5]:


# Fit without augmentation
model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[early_stop], epochs=30)


# In[6]:


# Plot loss per iteration
# Model shows clear signs of overfitting
plt.plot(model.history.history['loss'], label='loss')
plt.plot(model.history.history['val_loss'], label='val_loss')
plt.legend()


# In[7]:


# Plot accuracy per iteration
plt.plot(model.history.history['accuracy'], label='acc')
plt.plot(model.history.history['val_accuracy'], label='val_acc')
plt.legend()


# In[8]:


# Using ReducedLrOnPlateau to change learning rate if val_loss doesnt decrease after 5 epochs
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)


# In[9]:


# Fit with data augmentation
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size
model.fit(train_generator, validation_data=(x_test, y_test),callbacks=[reduce_lr], steps_per_epoch=steps_per_epoch, epochs=30,)


# In[ ]:





# In[10]:


# Plot loss per iteration
# Although the val_loss is unstable, it oscillates between a small interval.
plt.plot(model.history.history['loss'], label='loss')
plt.plot(model.history.history['val_loss'], label='val_loss')
plt.legend()


# In[ ]:





# In[11]:


# Plot accuracy per iteration
# Val_acc follows a similiar trend.
plt.plot(model.history.history['accuracy'], label='acc')
plt.plot(model.history.history['val_accuracy'], label='val_acc')
plt.legend()


# In[12]:


# Plot confusion matrix
from sklearn.metrics import confusion_matrix,classification_report


# In[13]:


labels_data = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

p_test = model.predict(x_test).argmax(axis=1)
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,p_test),annot=True)


# In[14]:


print(classification_report(y_test,p_test))


# In[15]:


# label mapping
labels = '''airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck'''.split()


# In[30]:


# Show some misclassified examples
misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]));


# In[32]:


# Correctly classified examples
classified_idx = np.where(p_test == y_test)[0]
i = np.random.choice(classified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]));


# In[30]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# predicting images
path=r'C:\Users\karth\Documents\Deep Learning\airplane.jpg'
img=image.load_img(path, target_size=(32,32))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict_classes(images)
classes
# Correctly classified examples
i = classes[0]
plt.imshow(img, cmap='gray')
plt.title("Predicted: %s" % (labels[i]));


# In[ ]:





# In[25]:


# predicting images
path=r'C:\Users\karth\Documents\Deep Learning\Dog.jpg'
img=image.load_img(path, target_size=(32,32))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict_classes(images)
classes
# Correctly classified examples
i = classes[0]
plt.imshow(img, cmap='gray')
plt.title("Predicted: %s" % (labels[i]));


# In[32]:


# Now that the model is so large, it's useful to summarize it
model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:





# In[18]:


# model.save(r'C:\Users\karth\Documents\Deep Learning\project_ReducedLr')


# In[24]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





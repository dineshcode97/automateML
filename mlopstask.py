#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential


# In[6]:


model = Sequential()


# In[7]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))


# In[8]:


model.summary()


# In[9]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[10]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[11]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[12]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[13]:


model.summary()


# In[14]:


model.add(Flatten())


# In[15]:


model.summary()


# In[16]:


model.add(Dense(units=128, activation='relu'))


# In[17]:


model.summary()


# In[18]:


model.add(Dense(units=1, activation='sigmoid'))


# In[19]:


model.summary()


# In[20]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[21]:


from keras_preprocessing.image import ImageDataGenerator


# In[22]:


f=open("val.txt","r")
value=f.readline()
if value == "":
    value='5'
print(value)
f.close()


# In[23]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/mlopsdata/birdspecies/train/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/mlopsdata/birdspecies/test/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=100,
        epochs=int(value),
        validation_data=test_set,
        validation_steps=800)


# In[39]:


history=model.history
acc=history.history['accuracy']
max_acc=max(acc)
print(max_acc)
reader=open("accuracy.txt",'r')
old=reader.read()
print(old)
if old<str(max_acc):
    print("new")
    writer=open("accuracy.txt",'w+')
    writer.write(str(acc))


# In[41]:


model.save('duck-and-owl.h5')


# In[26]:


from keras.models import load_model


# In[27]:


m = load_model('duck-and-owl.h5')


# 

# In[28]:


from keras.preprocessing import image


# In[29]:


test_image = image.load_img('/mlopsdata/birdspecies/test/Duck/5569c5da43094d5b954c042002bb188f.jpg', 
               target_size=(64,64))


# In[30]:


type(test_image)


# In[31]:


test_image


# In[32]:


test_image = image.img_to_array(test_image)


# In[33]:


type(test_image)


# In[34]:


test_image.shape


# In[35]:


import numpy as np 


# In[36]:


test_image = np.expand_dims(test_image, axis=0)


# In[37]:


test_image.shape


# In[ ]:





# In[39]:


result = m.predict(test_image)


# In[40]:


result


# In[41]:


if result[0][0] == 1.0:
    print('duck')
else:
    print('owl')


# In[ ]:





# In[87]:





# In[ ]:





# In[ ]:





# In[ ]:





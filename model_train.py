import numpy as np
import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt

#%%

IMAGE_SIZE=250;
BATCH_SIZE=32;
CHANNEL=3
EPOCH=50

#%%

# MODULE 1
# Data Cleaning and Preprocess 

dataset=tf.keras.preprocessing.image_dataset_from_directory("dataset", shuffle=True,
                                                    image_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                    batch_size=BATCH_SIZE);


#%%


class_name=dataset.class_names;
print(class_name);

#%%

for img_data,lab_data in dataset.take(1):
    plt.imshow(img_data[0].numpy().astype('uint8'))
    
#%%

#spiting the dataset to train =80% and test=20==> validation = 10% test=10%


train_data=dataset.take(54);
test_data=dataset.skip(54);

#%%
#split test into  dataset into 10% each as validation and test

val_data=test_data.take(6);
test_data=test_data.skip(6);

#%%


def get_dataset_partition_tf( dataset,train_split=0.8,
                             test_split=0.1,val_split=0.1,
                             shuffle=True,shuffle_size=10000):
    
    ds_size=len(dataset)
    
    
    if shuffle:
        dataset=dataset.shuffle(shuffle_size,seed=12)
    train_size=int(train_split*ds_size);
    
    val_size=int(val_split*ds_size);
    
    train_data=dataset.take(train_size)
    val_data=dataset.skip(train_size).take(val_size)
    test_data=dataset.skip(train_size).skip(val_size)
    
    
    return train_data,test_data,val_data;
#%%

train_ds,test_ds,val_ds=get_dataset_partition_tf(dataset)

#%%

print(len(train_ds))
print(len(test_ds))
print(len(val_ds))

#%%

resize_and_rescale=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255) 
    ])

#%%

data_augmentation=tf.keras.Sequential([
     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      layers.experimental.preprocessing.RandomRotation(0.2)
    
    ])


#%%



# MODULE 2 BUILD CNN NETWORK 

input_shape=(BATCH_SIZE,IMAGE_SIZE, IMAGE_SIZE,CHANNEL)
n_classes=3

model=models.Sequential([
    resize_and_rescale,
   
    layers.Conv2D(32,(3,3), activation='relu',input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax')
       
    ])


model.build(input_shape=input_shape)

#%%
print(model.summary())


#%%


model.compile(
    
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
  
    )
#%%

history=model.fit(
    
        train_ds,
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds    
        )
    

    


#%%


score=model.evaluate(test_ds)
print(score)    

#%%

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCH), acc, label='Training Accuracy')
plt.plot(range(EPOCH), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCH), loss, label='Training Loss')
plt.plot(range(EPOCH), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
    
#%%

for img,lab in test_ds.take(1):
    
    aimg=img[0].numpy().astype('uint8')
    alab=lab[0].numpy()

    print("Image")
    plt.imshow(aimg)
    
    print("actual label",class_name[alab])
    
    plab=model.predict(img)
    print("Predicted Label :",np.argmax(plab[0]))
    
    
#%%

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence
   
        
#%%

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_name[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")
        
#%%


model_ver=1;
model.save("C:\data science\Potato  Work Internship\model")



#%%

model.save("C:\data science\Potato  Work Internship\model\potatoes.h5")








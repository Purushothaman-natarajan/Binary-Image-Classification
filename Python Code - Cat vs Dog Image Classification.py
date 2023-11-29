#!/usr/bin/env python
# coding: utf-8

# ## Transfer Learning for Image Classification

# #### Unzipping the dataset

# In[1]:


#Necessary libraries for modelling
get_ipython().system('pip install --upgrade tensorflow')
get_ipython().system('pip install --upgrade keras')
get_ipython().system('pip install Pillow')
get_ipython().system('pip install seaborn')


# In[2]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[3]:


import pandas as pd


# In[4]:


import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current directory
print("Current Directory:", current_directory)


# In[6]:


import os
import zipfile

def unzip_nested_zip(zip_file_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    
    print(f"Unzipped '{zip_file_path}' to '{output_folder}'.")
    
    # Check for nested zip files and unzip them recursively
    extracted_files = os.listdir(output_folder)
    for extracted_file in extracted_files:
        extracted_file_path = os.path.join(output_folder, extracted_file)
        if extracted_file_path.endswith('.zip') and zipfile.is_zipfile(extracted_file_path):
            # Recursive call to handle nested zip files
            unzip_nested_zip(extracted_file_path, os.path.splitext(extracted_file_path)[0])


# In[7]:


# Specify the path to the initial zip folder
initial_zip_folder_path = "/tf/trailanderror/CatvsDog/Cat vs Dog.zip"

# Specify the output folder where contents will be extracted
output_folder = "/tf/trailanderror/CatvsDog"

# Call the function to recursively unzip folders within folders and nested zip files
unzip_nested_zip(initial_zip_folder_path, output_folder)


# In[8]:


from PIL import Image


# In[9]:


images = []
labels = []

master_data_path="/tf/trailanderror/CatvsDog"

def load_images_from_folder(folder_path, label):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            # If the item is a directory, recursively load images from it
            load_images_from_folder(file_path, filename)
        elif filename.lower().endswith(('png', 'jpg', 'jpeg')):
            # If the item is an image file, load it and add it to the images list
            try:
                img = Image.open(file_path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f'Error loading image {file_path}: {str(e)}')
        else:
            print(f'Skipping non-image file: {file_path}')

# Iterate through folders in the master folder
for folder_name in os.listdir(master_data_path):
    folder_path = os.path.join(master_data_path, folder_name)
    if os.path.isdir(folder_path):
        # Load images and labels from the current folder
        load_images_from_folder(folder_path, folder_name)

print(f'Images loaded: {len(images)}')
print(f'Labels loaded: {len(labels)}')


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


# Iterate through the first two elements of labels and images
for label, image in zip(labels[0:2], images[0:2]):
    print(label)
    plt.imshow(image)
    plt.show()


# In[12]:


unique_labels = list(set(labels))
print(unique_labels)


# ### Data Processing

# In[17]:


Dimensions = []

for idx, img in enumerate(images):
    width, height = img.size
    current_dimension = (width, height)
    Dimensions.append(current_dimension)

unique_dimension_count = len(list(set(Dimensions)))

print(f'we have images with {unique_dimension_count} various dimensions')


# #### Image :- we have to convert all the images to have same dimesion and normalize them

# In[18]:


import numpy as np


# In[19]:


# Convert PIL images to numpy arrays
numpy_images = [np.array(image) for image in images]

# Resize and convert images to RGB format if necessary
target_size = (224, 224)
reshaped_images = []
for idx, image in enumerate(numpy_images):
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize(target_size)
    # Convert to RGB if image is grayscale
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    reshaped_images.append(np.array(pil_image))

# Shape of reshaped images
for idx, img in enumerate(reshaped_images):
    print(f'Image {idx+1} shape after resizing {img.shape}')


# In[20]:


Dimensions = []

for idx, img in enumerate(reshaped_images):
    width, height, channel = img.shape
    current_dimension = width, height, channel
    Dimensions.append(current_dimension)
    
unique_dimension_count_after_reshaping = len(list(set(Dimensions)))

print(f'Number of unique dimension: {unique_dimension_count_after_reshaping}')

print("Dimension of reshaped_images:", reshaped_images[0].shape)

print("Data type of reshaped_images:", reshaped_images[0].dtype)
                    


# In[21]:


# Convert images to float32 and normalize to [0, 1]
normalized_images = np.array(reshaped_images, dtype=np.float32) / 255.0

# Verify the shape and data type of processed_images
print("Shape of normalized_images:", normalized_images.shape)
print("Data type of processed_images:", normalized_images.dtype)


# In[22]:


print(labels[0])
plt.imshow(normalized_images[0])
plt.show()


# ## Handling Data imbalance :- 
# 
# The data is significantly imbalanced, so data balancing is essential to prevent the model from biassing towards ships.

# In[23]:


#counting the number of labels in each classes

count_of_classes = {}

for label in labels:
    if label in count_of_classes:
        count_of_classes[label] +=1
    else:
        count_of_classes[label] = 1
        
        
for key, value in count_of_classes.items():
    print(f'{key}:{value}')


# #### The data is already balanced so doesn't require balancing.

# In[24]:


import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[25]:


# Initialize empty lists to store processed images and corresponding labels
processed_images = []
processed_labels = []

# Load, preprocess, and align images and labels
for image, label in zip(normalized_images, labels):
    try:
        processed_images.append(image)
        processed_labels.append(label)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(processed_labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded_labels = integer_encoded_labels.reshape(len(integer_encoded_labels), 1)
onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded_labels)

# Convert processed_images and onehot_encoded_labels to numpy arrays
processed_images = np.array(processed_images)
onehot_encoded_labels = np.array(onehot_encoded_labels)

# Save processed_images and onehot_encoded_labels in the current directory
np.save("processed_images_fr.npy", processed_images)
np.save("onehot_encoded_labels_fr.npy", onehot_encoded_labels)

# Verify the shapes of processed_images and onehot_encoded_labels
print("Shape of processed_images:", processed_images.shape)
print("Shape of onehot_encoded_labels:", onehot_encoded_labels.shape)


# # Model Selection and Transfer Learning

# ## Base Models

# In[26]:


# Import ResNeXt50 from tensorflow.keras.applications
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, DenseNet121, MobileNetV2


# In[27]:


# ResNeXt50, SEResNet50 ; we have to manually import as they not available directly through tf.keras.applications 
# or import from torch #

#importing the model & dense layer for customizing the neural network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


# In[28]:


#input shape & dimension for the pre-trained models
shape=(224, 224, 3)


# In[29]:


# Load pre-trained models
base_model_1 = VGG16(weights='imagenet', include_top=False, input_shape=shape)
base_model_2 = VGG19(weights='imagenet', include_top=False, input_shape=shape)
base_model_3 = ResNet50(weights='imagenet', include_top=False, input_shape=shape)
base_model_4 = InceptionV3(weights='imagenet', include_top=False, input_shape=shape)
base_model_5 = DenseNet121(weights='imagenet', include_top=False, input_shape=shape)
base_model_6 = MobileNetV2(weights='imagenet', include_top=False, input_shape=shape)


# In[30]:


base_models = [base_model_1, base_model_2, base_model_3, base_model_4, base_model_5, base_model_6]


# In[31]:


# Looping through the Base models and printing the summaries
for idx, model in enumerate(base_models):
    print(f'Summary of Base Model {idx +1}:')
    model.summary()


# In[32]:


#Freezing the pre-trained model's last layer for transfer learning
for model in base_models:
    for layer in model.layers:
        layer.trainable=False


# ## Customize & Compile the Base Models

# In[33]:


base_models = [base_model_1, base_model_2, base_model_3, base_model_4, base_model_5, base_model_6]

custom_models = []

for idx, model in enumerate(base_models):
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    custom_model = Model(inputs=model.input, outputs=predictions)
    custom_models.append(custom_model)
    print(f"Customized the model - {idx+1}")


# In[34]:


# Looping through the Base models and printing the summaries
for idx, model in enumerate(custom_models):
    print(f'Summary of Custom Model {idx +1}:')
    model.summary()


# In[35]:


compiled_models = []
for idx, custom_model in enumerate(custom_models):
    custom_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    compiled_models.append(custom_model)
    print(f"Compiled Custom Model {idx + 1}")


# ## Train the model using the Augmented_FR Data

# In[36]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from tensorflow.keras.optimizers import Adam, SGD, RMSprop


# In[38]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_images, onehot_encoded_labels, test_size=0.2, random_state=42)

# Verify the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)


# In[39]:


model_names = ["VGG16", "VGG19", "ResNet50", "InceptionV3", "DenseNet121", "MobileNetV2"]
class_labels = ['Cat', 'Dog']


# In[42]:


def train_and_evaluate_models(X_train, y_train, X_test, y_test, compiled_models, model_names, epochs=100):
    results = []
    for model, model_name in zip(compiled_models, model_names):
        # Initialize variables for tracking maximum accuracy
        max_accuracy = 0
        max_accuracy_epoch = 0

        # Define a checkpoint to save the model when target accuracy is reached
        checkpoint = ModelCheckpoint(f'{model_name}_model_FR.h5', monitor='val_accuracy', 
                                     save_best_only=True, save_weights_only=False, mode='max', verbose=1)

        # Define early stopping to stop training if accuracy doesn't improve for 10 epochs
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, verbose=1)

        # Train the current model 
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32, callbacks=[checkpoint, early_stopping])
        
        
        current_epoch_predictions = []
        current_epoch_labels = []
        
        for epoch, val_accuracy in enumerate(history.history['val_accuracy'], 1):
            # Check if the current epoch achieves higher accuracy than the previous maximum
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                max_accuracy_epoch = epoch
                current_epoch_predictions = model.predict(X_test)
                current_epoch_labels = y_test.argmax(axis=1)

        # Get the maximum accuracy and the corresponding epoch
        max_accuracy_epoch = np.argmax(history.history['val_accuracy'])
        max_accuracy = history.history['val_accuracy'][max_accuracy_epoch]

        # Evaluate the model on the test data
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test Accuracy for {model_name}: {test_accuracy}')

        # Store results
        results.append({'Model': model_name, 'Accuracy': max_accuracy})

        print("-" * 40)  # Print a line of dashes 
        print(f"Maximum accuracy of {max_accuracy:.2f} achieved at epoch {max_accuracy_epoch+1}")
        print("Model with high accuracy is saved using the keras ModelCheckpoint")
        print("-" * 40)  # Print a line of dashes 
        print("\n")

        # Assuming model.predict returns probabilities for each class
        y_pred_probs = model.predict(X_test)

        # Convert probabilities to class labels
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Convert true labels to class labels if y_test is one-hot encoded
        y_true = np.argmax(y_test, axis=1)

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(model_name)
        plt.show()
    return results


# In[ ]:


#Train and Evaluate the model
results = train_and_evaluate_models(X_train, y_train, X_test, y_test, compiled_models, model_names, epochs=100)


# In[49]:


results_df = pd.DataFrame(results)
# Print results in tabular form
print(results_df)


# In[51]:


from tabulate import tabulate


# Define the headers for the table
headers = ["Model", "Test Accuracy"]

# Print the table
print('Transfer Learning - Balanced Data :- 2 Classes')
print(tabulate(results_df, headers, tablefmt="grid"))


# #### THE END OF CLASSIFICATION MODEL

import tensorflow as tsf
import zipfile
import os
import shutil

class preprocess():
    
    def unzip_data():
        print("Unzipping_data")
        #Sets paths to archive and new directory PlantVillage
        path_to_zip_file = os.path.join(os.getcwd(), "archive.zip")
        extracted_dir = os.path.join(os.getcwd(), "PlantVillage")

        #Unzipping
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())

        if os.path.exists(os.path.join(extracted_dir, "PlantVillage")):
           shutil.rmtree(os.path.join(extracted_dir, "PlantVillage"))

        return extracted_dir

    def fetch_data(path_to_train_directory):
       datagen = tsf.keras.preprocessing.image.ImageDataGenerator(
       rescale=1./255,
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
        horizontal_flip=True)   

       train_generator = datagen.flow_from_directory(
       path_to_train_directory,
       target_size=(224, 224),
       batch_size=32,
       class_mode='categorical')

       return train_generator

import tensorflow as tsf
import zipfile
import os
import shutil
import cv2
import glob

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

        image_files = glob.glob(os.path.join(extracted_dir, "**/*.jpg"), recursive=True)

        for image_file in image_files:
            preprocess.resize_image(image_file)

        return extracted_dir

    def fetch_data(path_to_train_directory, validation_split = 0.3):
       datagen = tsf.keras.preprocessing.image.ImageDataGenerator(
       rescale=1./255,
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split,
        )   

       train_generator = datagen.flow_from_directory(
       path_to_train_directory,
       target_size=(224, 224),
       batch_size=32,
       class_mode='categorical',
       subset="training")
       

       validation_generator = datagen.flow_from_directory(
        path_to_train_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

       return train_generator, validation_generator
    
    @staticmethod
    def resize_image(image_path, target_size=(244, 244)):
        print(image_path)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, target_size)
        cv2.imwrite(image_path, resized_image)

from preprocessing import preprocess as prep
import datetime
import tensorflow as tsf
from AlexNet import AlexNet

data = prep.unzip_data()
train_generator, validation_generator = prep.fetch_data(data)

train_num = train_generator.samples
valid_num = validation_generator.samples

EPOCHS = 100
BATCH_SIZE = 32

train_dir = "./content/train"
valid_dir = "./content/validation"
model_dir = "./my_model.h5"

log_dir="./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tsf.keras.callbacks.TensorBoard(log_dir=log_dir)
callback_list = [tensorboard_callback]
model = AlexNet((244,244,3), 15)

model.summary()
model.fit(train_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=train_num // BATCH_SIZE,
                    validation_data=validation_generator,
                    validation_steps=valid_num // BATCH_SIZE,
                    callbacks=callback_list,
                    verbose=0)

model.save(model_dir)
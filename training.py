from preprocessing import preprocess as prep

data = prep.unzip_data()
train_generator = prep.fetch_data(data)
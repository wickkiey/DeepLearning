import tensorflow as tf

# Reading the TF file
data_path = "../data/pets.tfrecord"
data = tf.data.TFRecordDataset(data_path)

raw_data = data.take(1)
example = tf.train.Example()
example.ParseFromString(raw_data.numpy())

print(example)


for raw_record in data.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)
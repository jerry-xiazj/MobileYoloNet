import tensorflow as tf
from core.model import MobileYolo_small
from config import CFG


model_input = tf.keras.layers.Input([CFG.input_shape[0], CFG.input_shape[1], 3])
model_output = MobileYolo_small(model_input, training=True)
model = tf.keras.Model(model_input, model_output)
tf.keras.utils.plot_model(model, to_file=CFG.log_dir+"model.png", show_shapes=True, show_layer_names=True)
model.summary()

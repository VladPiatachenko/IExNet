import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Створюємо кастомний клас для шару Cast
class Cast(tf.keras.layers.Layer):
    def __init__(self, dtype="float32", **kwargs):
        super(Cast, self).__init__(**kwargs)
        self.dtype_cast = dtype

    def call(self, inputs):
        return tf.cast(inputs, self.dtype_cast)

    def get_config(self):
        config = super().get_config()
        config.update({"dtype": self.dtype_cast})
        return config

# Вказуємо шлях до файлу моделі
model_path = "/content/model.h5"

# Завантажуємо модель з кастомним шаром
with tf.keras.utils.custom_object_scope({"Cast": Cast}):
    model = tf.keras.models.load_model(model_path)

# Виводимо архітектуру
model.summary()

# Зберігаємо схему моделі у PNG
plot_model(model, to_file="model_structure.png", show_shapes=True, show_layer_names=True)

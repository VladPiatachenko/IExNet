import os
import numpy as np
from sklearn.metrics import pairwise_distances
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Завантаження даних
def load_images_from_directory(base_path, target_size=(64, 64), max_images_per_class=2000):
    data, labels, class_map = [], [], {}
    for idx, class_name in enumerate(sorted(os.listdir(base_path))):
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        class_map[idx] = class_name
        print(f"Loading images for class: {class_name}")
        class_images = []
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = load_img(file_path, target_size=target_size, color_mode='grayscale')
                img_array = img_to_array(img) / 255.0
                class_images.append((img_array, idx))
                if len(class_images) >= max_images_per_class:
                    break
        for img_array, label in class_images:
            data.append(img_array)
            labels.append(label)
    return np.array(data), np.array(labels), class_map

# Модель CNN
def trainable_cnn(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Функція для бінаризації ознак
def binarize_features(features, thresholds):
    """
    Бінаризація ознак із використанням заданих порогів.

    features: np.array
        Матриця ознак (n_samples, n_features).
    thresholds: np.array
        Пороги для кожної ознаки (n_features,).

    Returns:
        np.array
        Бінаризована матриця ознак (0 або 1).
    """
    return (features > thresholds).astype(np.float32)

# Клас k-NN класифікатора
class KNNClassifier:
    def __init__(self, k=5, metric="hamming"):
        self.k = k
        self.metric = metric
        self.train_features = None
        self.train_labels = None

    def fit(self, train_features, train_labels):
        """
        Збереження навчальних даних (формування правил класифікатора).

        Parameters:
        train_features: np.array
            Ознаки для навчання (n_samples, n_features).
        train_labels: np.array
            Мітки класів для навчання (n_samples,).
        """
        self.train_features = train_features
        self.train_labels = train_labels

    def predict(self, test_features):
        """
        Розпізнавання (класифікація) тестових даних.

        Parameters:
        test_features: np.array
            Ознаки для тестування (n_samples, n_features).

        Returns:
        np.array
            Передбачені мітки класів для тестових даних.
        """
        if self.train_features is None or self.train_labels is None:
            raise ValueError("Класифікатор не було навчено. Використайте метод fit перед predict.")

        # Обчислення відстаней між тестовими та навчальними ознаками
        distances = pairwise_distances(test_features, self.train_features, metric=self.metric)
        predictions = []
        for dist in distances:
            # Знаходимо k найближчих сусідів
            k_indices = np.argsort(dist)[:self.k]
            k_labels = self.train_labels[k_indices]
            # Вибираємо найпоширеніший клас серед сусідів
            pred_label = np.argmax(np.bincount(k_labels))
            predictions.append(pred_label)
        return np.array(predictions)

if __name__ == "__main__":
    # Завантаження даних
    dataset_path = "/content/dataset/img"  # Змініть на ваш шлях до даних
    image_size = (64, 64)
    data, labels, class_map = load_images_from_directory(dataset_path, target_size=image_size)

    # Розподіл на навчальні та тестові набори
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Додаємо канал (видаляємо зайвий вимір, якщо він існує)
    train_data = np.squeeze(train_data)
    test_data = np.squeeze(test_data)
    train_data = train_data[..., np.newaxis] if train_data.ndim == 3 else train_data
    test_data = test_data[..., np.newaxis] if test_data.ndim == 3 else test_data

    # Перевіряємо форми
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Навчання CNN
    cnn = trainable_cnn(image_size + (1,), len(class_map))
    cnn.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    # Витягування ознак
    feature_extractor = Sequential(cnn.layers[:-2])  # Виключаємо останній шар класифікації
    feature_extractor.compile(optimizer='adam')  # Додамо, щоб уникнути помилок при використанні моделі

    # Передбачення ознак
    train_features = feature_extractor.predict(train_data, verbose=1)
    test_features = feature_extractor.predict(test_data, verbose=1)

    # Виведемо частину даних до бінаризації
    print("\n--- Дані до бінаризації (перші 2 рядки) ---")
    print(f"Train features (до бінаризації):\n{train_features[:2]}")
    print(f"Test features (до бінаризації):\n{test_features[:2]}")

    # Розрахунок порогів на основі навчальних даних
    thresholds = np.mean(train_features, axis=0)  # Або np.median(train_features, axis=0)
    print(f"\nАдаптивні пороги для ознак (на основі навчальних даних):\n{thresholds}")

    # Бінаризація ознак
    train_features = binarize_features(train_features, thresholds)
    test_features = binarize_features(test_features, thresholds)

    # Виведемо частину даних після бінаризації
    print("\n--- Дані після бінаризації (перші 2 рядки) ---")
    print(f"Train features (після бінаризації):\n{train_features[:2]}")
    print(f"Test features (після бінаризації):\n{test_features[:2]}")

    # 1. Формування правил класифікатора (навчання)
    knn = KNNClassifier(k=5, metric="hamming")
    knn.fit(train_features, train_labels)

    # 2. Розпізнавання тестових даних
    predicted_labels = knn.predict(test_features)

    # Оцінка точності
    accuracy = np.mean(predicted_labels == test_labels)
    print(f"\nAccuracy: {accuracy:.2f}")

    # Показуємо деякі результати
    print("\nSample predictions:")
    for i in range(10):
        true_label = class_map[test_labels[i]]
        predicted_label = class_map[predicted_labels[i]]
        print(f"Test Image {i}: True Label = {true_label}, Predicted Label = {predicted_label}")

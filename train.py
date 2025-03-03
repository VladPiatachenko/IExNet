import os
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Add, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.activations import mish
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Input,
    GlobalAveragePooling2D, Add
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,
    Dense, Dropout, SpatialDropout2D, Input, Add, Flatten, DepthwiseConv2D
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import LearningRateScheduler
import wandb
from wandb.integration.keras import WandbMetricsLogger
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model

set_global_policy("mixed_float16")
# Ініціалізація W&B
wandb.init(project="CNN_IEIT_training", config={"epochs": 50, "batch_size": 32})

def clr(epoch):
    min_lr = 1e-5
    max_lr = 1e-3
    cycle = 10  # Reset кожні 10 епох
    return min_lr + (max_lr - min_lr) * (1 + np.cos(epoch * np.pi / cycle)) / 2

# Завантаження даних
def load_images_from_directory(base_path, target_size=(64, 64), max_images_per_class=2000):
    data, labels, class_map = [], [], {}
    class_counts = {}  # Для підрахунку зображень у кожному класі

    for idx, class_name in enumerate(sorted(os.listdir(base_path))):
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith('.'):
            continue
        class_map[idx] = class_name
        class_counts[class_name] = 0  # Лічильник для кожного класу
        print(f"Loading images for class: {class_name}")

        class_images = []
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = load_img(file_path, target_size=target_size, color_mode='grayscale')
                img_array = img_to_array(img) / 255.0
                class_images.append((img_array, idx))
                class_counts[class_name] += 1  # Збільшуємо лічильник
                if len(class_images) >= max_images_per_class:
                    break

        for img_array, label in class_images:
            data.append(img_array)
            labels.append(label)

    print("\n--- Завантажені зображення по класах ---")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} зображень")

    return np.array(data), np.array(labels), class_map

def optimized_cnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # 🔥 STRONGER CONVOLUTION BLOCKS
    x = Conv2D(64, (3, 3), padding="same", activation=swish)(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="same", activation=swish)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 🔥 DEEPER NETWORK (MORE LAYERS)
    x = SeparableConv2D(256, (3, 3), padding="same", activation=swish)(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(256, (3, 3), padding="same", activation=swish)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = SeparableConv2D(512, (3, 3), padding="same", activation=swish)(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(512, (3, 3), padding="same", activation=swish)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 🔥 IMPROVED FEATURE EXTRACTION
    x = GlobalAveragePooling2D()(x)  # ✅ GLOBAL POOLING WORKS BETTER
    x = Dense(512, activation=swish, name="dense_feature")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    # 🔥 STRONGER OPTIMIZER
    optimizer = AdamW(learning_rate=0.001, weight_decay=0.0001)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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

class IEITClassifier:
    def __init__(self):
        self.train_features = None
        self.class_means = None
        self.distance_matrix = None
        self.closest_pairs = None
        self.optimal_radii = {}  # Оптимальні радіуси для кожного класу

    def fit(self, train_features, train_labels):
        """
        Form rules for the classifier:
        - Compute binary mean vectors for each class
        - Find the closest class for each class
        - Compute reliability and error parameters for each radius
        """
        self.train_features = train_features
        self.train_labels = train_labels  # 🔴 STORE train_labels HERE

        unique_classes = np.unique(train_labels)

        # Compute binary mean vectors for each class
        self.class_means = []
        for class_label in unique_classes:
            class_features = train_features[train_labels == class_label]
            class_mean = np.mean(class_features, axis=0)
            binary_class_mean = (class_mean >= 0.5).astype(int)
            self.class_means.append(binary_class_mean)
        self.class_means = np.array(self.class_means)

        # Compute distance matrix
        self.distance_matrix = self.calculate_distance_matrix(self.class_means)

        # Find closest class for each class
        self.closest_pairs = self.find_all_closest_pairs(self.distance_matrix)

        # Compute reliability and error
        self.compute_reliability_and_error()

    @staticmethod
    def hamming_distance(vector1, vector2):
        """Розраховує гаммінгову відстань між двома бінарними векторами"""
        return np.sum(vector1 != vector2)

    def calculate_distance_matrix(self, class_means):
        """Створює матрицю гаммінгових відстаней між класами"""
        num_classes = class_means.shape[0]
        distance_matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                distance = self.hamming_distance(class_means[i], class_means[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        return distance_matrix

    def find_all_closest_pairs(self, distance_matrix):
        """Для кожного класу знаходить найближчий до нього клас"""
        num_classes = distance_matrix.shape[0]
        closest_pairs = {}
        for i in range(num_classes):
            distances = distance_matrix[i].copy()
            distances[i] = np.inf  # Ігноруємо відстань до самого себе
            closest_class = np.argmin(distances)
            closest_pairs[i] = closest_class
        return closest_pairs

    def compute_Jm(self, K1, K3):
        """
        Обчислення Jm згідно з формулою.
        """
        n = 1  # Припущення рівноімовірності
        r = 3
        epsilon = 10**(-r)  # Дуже мале число для стабілізації

        numerator = (2 * n + epsilon - (1 - K1 + K3))
        denominator = (1 - K1 + K3 + epsilon)

        if denominator == 0:
            return float('-inf')  # Запобігання діленню на 0

        log_term = np.log2(numerator / denominator)

        Jm = (1 / n) * log_term * (n - (1 - K1 + K3))

        return Jm


    def compute_reliability_and_error(self, output_dir="radius_logs"):
        """
        Computes reliability and error for each class and writes results to a file.
        Each class gets a separate log file.
        """
        num_features = self.train_features.shape[1]  # Number of features
        unique_classes = range(len(self.class_means))

        os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

        print("\n--- Calculating reliability and error for each radius ---")

        for class_label in unique_classes:
            neighbor_class_label = self.closest_pairs[class_label]
            mean_vector = self.class_means[class_label]

            class_vectors = self.train_features[np.where(self.train_labels == class_label)]
            neighbor_vectors = self.train_features[np.where(self.train_labels == neighbor_class_label)]

            optimal_radius = None

            log_filename = os.path.join(output_dir, f"class_{class_label}_radius_log.txt")

            with open(log_filename, "w") as log_file:
                log_file.write(f"Class {class_label} (Closest class: {neighbor_class_label})\n")
                log_file.write("Radius | k1 | k2 | Reliability | Beta Error | Jm | Optimal\n")
                log_file.write("-" * 80 + "\n")

                for radius in range(1, num_features + 1):
                    distances_self = np.array([self.hamming_distance(mean_vector, vec) for vec in class_vectors])
                    distances_neighbor = np.array([self.hamming_distance(mean_vector, vec) for vec in neighbor_vectors])

                    k1 = np.sum(distances_self <= radius)
                    k2 = np.sum(distances_neighbor <= radius)

                    reliability = k1 / len(class_vectors)
                    beta_error = k2 / len(neighbor_vectors)

                    is_optimal = reliability > 0.5 and beta_error < 0.5

                    if is_optimal:
                        optimal_radius = radius

                    kfe = self.compute_Jm(reliability, beta_error)

                    log_file.write(f"{radius:5d} | {k1:4d} | {k2:4d} | {reliability:.3f} | {beta_error:.3f} | {kfe:.3f} | {is_optimal}\n")

            self.optimal_radii[class_label] = optimal_radius
            print(f"✅ Log saved for class {class_label}: {log_filename}")

        print("\n--- Optimal Radii for Each Class ---")
        for class_label, radius in self.optimal_radii.items():
            print(f"Class {class_label}: Optimal Radius = {radius}")


    def predict(self, test_vector):
        """
        Передбачення одного вектора.

        Parameters:
        test_vector: np.array
            Вхідний вектор ознак (бінаризований).

        Returns:
        class_prediction: int
            Найбільш ймовірний клас.
        """
        best_score = None
        best_class = None

        #Процес розпізнавання

        for class_label, mean_vector in enumerate(self.class_means):
            if class_label not in self.optimal_radii or self.optimal_radii[class_label] is None:
               #print(f"Клас {class_label} пропущений: немає оптимального радіуса")
                continue  # Пропускаємо, якщо немає оптимального радіуса

            radius = self.optimal_radii[class_label]
            distance = self.hamming_distance(mean_vector, test_vector)

            if radius == 0:  # Уникаємо ділення на нуль
                #print(f"Клас {class_label} пропущений: радіус = 0")
                continue

            score = 1 - (distance / radius)

            #print(f"Клас {class_label}: Відстань={distance}, Радіус={radius}, f={score:.3f}")

            if best_score is None or score > best_score:
                best_score = score
                best_class = class_label

        return best_class

if __name__ == "__main__":
    # Завантаження даних
    dataset_path = "/content/dataset/EuroSAT_RGB"  # Шлях до даних
    image_size = (64, 64)
    data, labels, class_map = load_images_from_directory(dataset_path, target_size=image_size)

    # Розподіл на навчальні та тестові набори
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    class_map = {idx: name for idx, name in enumerate(label_encoder.classes_)}
    print(f"Оновлений class_map: {class_map}")  # Переконайтеся, що класи правильні

    # 🔹 Оновлення кількості класів
    num_classes = len(np.unique(train_labels))
    print(f"Кількість класів після перекодування: {num_classes}")
    # Додаємо канал (видаляємо зайвий вимір, якщо він існує)
    train_data = np.squeeze(train_data)
    test_data = np.squeeze(test_data)
    train_data = train_data[..., np.newaxis] if train_data.ndim == 3 else train_data
    test_data = test_data[..., np.newaxis] if test_data.ndim == 3 else test_data

    # Перевіряємо форми
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    lr_scheduler = LearningRateScheduler(clr)
    # 🔹 WandbCallback для логування
    wandb_callback = WandbMetricsLogger()

    # Тренуємо оновлену CNN
    cnn = optimized_cnn(image_size + (1,), len(class_map))
    cnn.fit(
        train_data, train_labels,
        epochs=50, batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[lr_scheduler, wandb_callback]  # 🔥 W&B ДОДАНО СЮДИ
    )


    # Видаляємо останній шар (классифікацію) → отримаємо екстрактор ознак
    feature_extractor = Model(inputs=cnn.input, outputs=cnn.get_layer("dense_feature").output)
    feature_extractor.compile(optimizer='adam')

    # Отримуємо ознаки
    train_features = feature_extractor.predict(train_data, verbose=1)
    test_features = feature_extractor.predict(test_data, verbose=1)

    # Виведемо частину даних до бінаризації
    print("\n--- Дані до бінаризації (перші 2 рядки) ---")
    print(f"Train features (до бінаризації):\n{train_features[:2]}")
    print(f"Test features (до бінаризації):\n{test_features[:2]}")

    # Розрахунок порогів на основі навчальних даних
    thresholds = np.mean(train_features, axis=0) + 0.1 * np.std(train_features, axis=0)
    print(f"\nАдаптивні пороги для ознак (на основі навчальних даних):\n{thresholds}")

    # Бінаризація ознак
    train_features = binarize_features(train_features, thresholds)
    test_features = binarize_features(test_features, thresholds)

    # Виведемо частину даних після бінаризації
    print("\n--- Дані після бінаризації (перші 2 рядки) ---")
    print(f"Train features (після бінаризації):\n{train_features[:2]}")
    print(f"Test features (після бінаризації):\n{test_features[:2]}")

    # 1. Формування правил класифікатора (навчання)
    ieit = IEITClassifier()
    ieit.fit(train_features, train_labels)

    # 2. Розпізнавання тестових даних
    predicted_labels = np.array([ieit.predict(test_vector) for test_vector in test_features])

    # Оцінка точності
    accuracy = np.mean(predicted_labels == test_labels)
    print(f"\nAccuracy: {accuracy:.2f}")

    # Показуємо деякі результати
    print("\nSample predictions:")
    for i in range(10):
        true_label = class_map[test_labels[i]]
        predicted_label = class_map[predicted_labels[i]]  # Оновлено доступ до класу
        print(f"Test Image {i}: True Label = {true_label}, Predicted Label = {predicted_label}")


    plot_model(cnn, to_file="model_structure.png", show_shapes=True, show_layer_names=True)

# ml_models/image_analyzer.py
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model


class MedicalImageAnalyzer:
    """
    CNN-based medical image classifier using EfficientNetB0 transfer learning.
    Classifies chest X-rays into: Normal, Pneumonia, Tuberculosis, COVID-19.
    """

    CLASSES = ['Normal', 'Pneumonia', 'Tuberculosis', 'COVID-19']
    IMG_SIZE = (224, 224)

    def __init__(self, weights_path: str = None):
        self.model = self._build_model()
        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"✅ Loaded weights from {weights_path}")

    def _build_model(self) -> Model:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.IMG_SIZE, 3)
        )
        base_model.trainable = False  # Freeze for transfer learning

        inputs = layers.Input(shape=(*self.IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(len(self.CLASSES), activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, train_ds, val_ds, epochs: int = 20, fine_tune_epochs: int = 10):
        os.makedirs("models", exist_ok=True)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                'models/image_model.h5', save_best_only=True, verbose=1
            )
        ]

        print("Phase 1: Training top layers...")
        history = self.model.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs, callbacks=callbacks
        )

        # Fine-tuning: unfreeze last 20 layers
        print("\nPhase 2: Fine-tuning top layers of base model...")
        base_model = self.model.layers[1]
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        history_ft = self.model.fit(
            train_ds, validation_data=val_ds,
            epochs=fine_tune_epochs, callbacks=callbacks
        )
        return history, history_ft

    def preprocess_image(self, image_path: str) -> tf.Tensor:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.IMG_SIZE)
        arr = tf.keras.preprocessing.image.img_to_array(img)
        return tf.expand_dims(arr / 255.0, axis=0)

    def analyze_image(self, image_path: str) -> dict:
        """
        Analyze a medical image and return diagnosis probabilities.

        Args:
            image_path: Path to chest X-ray or scan image

        Returns:
            dict with image_diagnosis, confidence, and all_probabilities
        """
        img_tensor = self.preprocess_image(image_path)
        predictions = self.model.predict(img_tensor, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))

        return {
            "image_diagnosis": self.CLASSES[predicted_idx],
            "confidence": round(float(np.max(predictions)) * 100, 2),
            "all_probabilities": {
                cls: round(float(prob) * 100, 2)
                for cls, prob in zip(self.CLASSES, predictions)
            }
        }

    def load_dataset(self, data_dir: str, batch_size: int = 32):
        """Load image dataset from directory with train/val split."""
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=self.IMG_SIZE,
            batch_size=batch_size
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=self.IMG_SIZE,
            batch_size=batch_size
        )
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)
        return train_ds, val_ds


if __name__ == "__main__":
    analyzer = MedicalImageAnalyzer()
    analyzer.model.summary()
    print("\nModel built successfully. Provide chest X-ray dataset to train.")

"""
Artwork Classification by Creation Date
Based on the research paper by Ivan Bilan (2017)
Updated for 2025 with modern frameworks and best practices

This script classifies paintings into three historical periods:
- 1400-1759 (Renaissance, Baroque, Rococo)
- 1760-1870 (Romanticism, Realism)
- 1870-present (Impressionism, Post-Impressionism, Expressionism, etc.)

Usage:
- Prepare a dataset with the following folder structure:
  train_set_paintings/
    1400-1759/
      image1.jpg
      ...
    1760-1870/
      image1.jpg
      ...
    1870-present/
      image1.jpg
      ...
  validation_set_paintings/
    1400-1759/
      image1.jpg
      ...
    1760-1870/
      image1.jpg
      ...
    1870-present/
      image1.jpg
      ...
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import EfficientNetV2S, VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 224  # Base size for input images
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Create output directories
BASE_DIR = Path("artwork_classification_results")
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
PLOT_DIR = BASE_DIR / "plots"

for dir_path in [MODEL_DIR, LOG_DIR, PLOT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data paths - adjust these to your directory structure
TRAIN_DIR = Path('train_set_paintings')
VALIDATION_DIR = Path('validation_set_paintings')

def plot_training_history(history, model_name):
    """Plot and save training history"""
    # Create figure
    plt.figure(figsize=(16, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{model_name} - Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} - Loss')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{model_name}_history.png", dpi=300)
    plt.show()

def create_data_generators():
    """Create data generators with augmentation for training"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

def create_custom_cnn_model(input_shape, num_classes):
    """Create a custom CNN model following the paper's approach but modernized"""
    model = tf.keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_vgg19_transfer_model(input_shape, num_classes):
    """Create a transfer learning model using VGG19 as in the paper"""
    # Load pre-trained VGG19 model without top layers
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze first several layers (as mentioned in paper)
    for layer in base_model.layers[:15]:
        layer.trainable = False
    
    # Create new model with custom top layers
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model using SGD with momentum as in the paper
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_efficientnet_model(input_shape, num_classes):
    """Create a transfer learning model using EfficientNetV2 (2025 modern approach)"""
    # Load pre-trained EfficientNetV2S model without top layers
    base_model = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Create new model with custom top layers
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model with modern optimizer
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, validation_generator, model_name):
    """Train model with callbacks for early stopping and checkpointing"""
    # Calculate steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        MODEL_DIR / f"{model_name}_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=LOG_DIR / model_name,
        histogram_freq=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard]
    
    # Train model
    print(f"\nTraining {model_name}...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(MODEL_DIR / f"{model_name}_final.h5")
    
    # Plot training history
    plot_training_history(history, model_name)
    
    # Evaluate on validation set
    print(f"\nEvaluating {model_name} on validation set...")
    evaluation = model.evaluate(validation_generator)
    
    print(f"{model_name} - Validation Loss: {evaluation[0]:.4f}")
    print(f"{model_name} - Validation Accuracy: {evaluation[1]:.4f}")
    
    return history, evaluation

def main():
    """Main execution function"""
    # Print TensorFlow version and device info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Create data generators
    train_generator, validation_generator = create_data_generators()
    
    # Print class indices
    print("\nClass indices:")
    for class_name, class_idx in train_generator.class_indices.items():
        print(f"- {class_name}: {class_idx}")
    
    # Get input shape and number of classes
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    num_classes = len(train_generator.class_indices)
    
    # Create timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train custom CNN model (as in the paper)
    custom_cnn = create_custom_cnn_model(input_shape, num_classes)
    custom_cnn.summary()
    custom_cnn_history, custom_cnn_eval = train_model(
        custom_cnn,
        train_generator,
        validation_generator,
        f"custom_cnn_{timestamp}"
    )
    
    # Train VGG19 transfer learning model (as in the paper)
    vgg19_model = create_vgg19_transfer_model(input_shape, num_classes)
    vgg19_model.summary()
    vgg19_history, vgg19_eval = train_model(
        vgg19_model,
        train_generator,
        validation_generator,
        f"vgg19_transfer_{timestamp}"
    )
    
    # Train EfficientNetV2 model (2025 modern approach)
    efficientnet_model = create_efficientnet_model(input_shape, num_classes)
    efficientnet_model.summary()
    efficientnet_history, efficientnet_eval = train_model(
        efficientnet_model,
        train_generator,
        validation_generator,
        f"efficientnet_v2_{timestamp}"
    )
    
    # Compare results
    print("\n--- Final Results ---")
    print(f"Custom CNN Accuracy: {custom_cnn_eval[1]:.4f}")
    print(f"VGG19 Transfer Learning Accuracy: {vgg19_eval[1]:.4f}")
    print(f"EfficientNetV2 Accuracy: {efficientnet_eval[1]:.4f}")

if __name__ == "__main__":
    main()

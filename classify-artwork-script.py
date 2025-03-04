#!/usr/bin/env python3
"""
Artwork Classification Inference Script

This script loads a trained model and classifies a single artwork image
into one of three historical periods:
- 1400-1759 (Renaissance, Baroque, Rococo)
- 1760-1870 (Romanticism, Realism)
- 1870-present (Impressionism, Post-Impressionism, Expressionism, etc.)

Usage:
  python classify_artwork.py --image path/to/image.jpg --model path/to/model.h5

Options:
  --image     Path to the artwork image to classify
  --model     Path to the trained model file (.h5)
  --size      Image size for model input (default: 224)
  --info      Display additional information about the image

Example:
  python classify_artwork.py --image paintings/mona_lisa.jpg --model models/efficientnet_v2_20250304_best.h5
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import tensorflow as tf


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Classify artwork by historical period')
    parser.add_argument('--image', type=str, required=True, help='Path to the artwork image')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file (.h5)')
    parser.add_argument('--size', type=int, default=224, help='Image size for model input (default: 224)')
    parser.add_argument('--info', action='store_true', help='Display additional information about the image')
    return parser.parse_args()


def load_model(model_path):
    """Load the trained model"""
    try:
        # Handle custom objects if any were used during training
        custom_objects = {}
        
        # For models using custom optimizer like AdamW
        # Add proper imports if you use other custom objects
        optimizer_config = {'class_name': 'AdamW'}
        custom_objects['AdamW'] = tf.keras.optimizers.AdamW
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def load_and_preprocess_image(image_path, target_size):
    """Load and preprocess an image for model input"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Resize image
        img = img.resize((target_size, target_size))
        
        # Convert to array and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)


def display_image_info(image_path, img):
    """Display information about the image"""
    # Get basic file info
    file_size = os.path.getsize(image_path) / (1024 * 1024)  # in MB
    file_modified = os.path.getmtime(image_path)
    
    # Get image dimensions and format
    width, height = img.size
    img_format = img.format
    
    # Try to get EXIF data if available
    exif_data = {}
    try:
        if hasattr(img, '_getexif') and img._getexif() is not None:
            for tag, value in img._getexif().items():
                if tag in ExifTags.TAGS:
                    exif_data[ExifTags.TAGS[tag]] = value
    except Exception:
        pass
    
    # Print information
    print("\n--- Image Information ---")
    print(f"File: {os.path.basename(image_path)}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Dimensions: {width} x {height} pixels")
    print(f"Format: {img_format}")
    
    # Print relevant EXIF data if available
    if exif_data:
        print("\nEXIF Data:")
        if 'Make' in exif_data:
            print(f"Camera: {exif_data.get('Make')} {exif_data.get('Model', '')}")
        if 'DateTime' in exif_data:
            print(f"Date taken: {exif_data.get('DateTime')}")
        if 'Software' in exif_data:
            print(f"Software: {exif_data.get('Software')}")


def classify_artwork(model, img_array, class_labels=None):
    """Classify the artwork and return the predicted class and confidence"""
    # If class labels are not provided, use default
    if class_labels is None:
        class_labels = {
            0: "1400-1759 (Renaissance, Baroque, Rococo)",
            1: "1760-1870 (Romanticism, Realism)",
            2: "1870-present (Impressionism, Modern)"
        }
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Get class label
    predicted_class = class_labels.get(predicted_class_idx, f"Class {predicted_class_idx}")
    
    return predicted_class, confidence, predictions[0]


def display_results(image_path, img, predicted_class, confidence, all_probabilities, class_labels=None):
    """Display the classification results with the image"""
    if class_labels is None:
        class_labels = {
            0: "1400-1759",
            1: "1760-1870",
            2: "1870-present"
        }
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    
    # Plot the probabilities
    plt.subplot(1, 2, 2)
    bars = plt.barh(list(class_labels.values()), all_probabilities * 100)
    
    # Color the bars based on the highest probability
    max_idx = np.argmax(all_probabilities)
    for i, bar in enumerate(bars):
        if i == max_idx:
            bar.set_color('darkred')
        else:
            bar.set_color('steelblue')
    
    plt.xlabel('Probability (%)')
    plt.title('Classification Probabilities')
    plt.xlim(0, 100)
    plt.tight_layout()
    
    # Save the figure
    output_dir = 'classification_results'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_result.png')
    plt.savefig(output_filename, dpi=300)
    
    print(f"\nClassification result image saved to: {output_filename}")
    plt.show()


def main():
    """Main function to classify an artwork image"""
    # Parse arguments
    args = parse_args()
    
    # Print TensorFlow version and device info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {bool(tf.config.list_physical_devices('GPU'))}")
    
    # Load model
    model = load_model(args.model)
    
    # Load and preprocess image
    img_array, img = load_and_preprocess_image(args.image, args.size)
    
    # Display image info if requested
    if args.info:
        display_image_info(args.image, img)
    
    # Define class labels (customize if your model uses different labels)
    class_labels = {
        0: "1400-1759 (Renaissance, Baroque, Rococo)",
        1: "1760-1870 (Romanticism, Realism)",
        2: "1870-present (Impressionism, Modern)"
    }
    
    # Classify the artwork
    predicted_class, confidence, all_probabilities = classify_artwork(model, img_array, class_labels)
    
    # Print results
    print("\n--- Classification Results ---")
    print(f"Predicted Period: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display probabilities for all classes
    print("\nProbabilities for all periods:")
    for i, (idx, label) in enumerate(class_labels.items()):
        print(f"  {label}: {all_probabilities[i] * 100:.2f}%")
    
    # Display results with the image
    display_labels = {idx: label.split(' ')[0] for idx, label in class_labels.items()}
    display_results(args.image, img, predicted_class, confidence, all_probabilities, display_labels)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Assuming models.py is in the same directory or accessible via PYTHONPATH
from models import ConfigurableCNNBuilder, NnClassifier, device

def classify_images(architecture_strings, image_paths, root_directory, weights_dir='.'):
    """
    Loads pre-trained models for different architectures and classifies given images.

    Args:
        architecture_strings (list): List of strings defining the CNN architectures.
        image_paths (list): List of paths to the images to classify.
        root_directory (str): Path to the root dataset directory (to get class labels).
        weights_dir (str): Directory where the trained model weights (.pth files) are stored.
    """
    # --- Configuration ---
    num_classes = len(os.listdir(os.path.join(root_directory, 'train')))
    emotion_labels = {i: emotion for i, emotion in enumerate(os.listdir(os.path.join(root_directory, 'train')))}

    # Define the same transformations used during training/evaluation
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Classification Loop ---
    results = {} # {image_path: {architecture: prediction}}

    for arch_str in architecture_strings:
        print(f"\n--- Architecture: {arch_str} ---")
        model_builder = ConfigurableCNNBuilder(architecture_string=arch_str, num_classes=num_classes)
        classifier = NnClassifier(nnet_builder=model_builder, epochs=1) # Epochs not relevant for prediction

        weights_path = os.path.join(weights_dir, f"model_{arch_str}.pth")

        if not os.path.exists(weights_path):
            print(f"Warning: Weights file not found at {weights_path}. Skipping this architecture.")
            continue

        try:
            print(f"Loading weights from {weights_path}...")
            classifier.load_state(weights_path)
            classifier.model.eval() # Ensure model is in evaluation mode
        except Exception as e:
            print(f"Error loading weights for architecture {arch_str}: {e}. Skipping.")
            continue

        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found at {img_path}. Skipping this image for this architecture.")
                continue

            try:
                # Load and transform the image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0) # Add batch dimension

                # Create a dummy dataset for prediction method compatibility
                # The label doesn't matter for prediction
                class DummyImageDataset(torch.utils.data.Dataset):
                    def __init__(self, tensor):
                        self.tensor = tensor
                    def __len__(self):
                        return 1
                    def __getitem__(self, idx):
                        return self.tensor.squeeze(0), 0 # Return image tensor and dummy label

                dummy_dataset = DummyImageDataset(img_tensor)

                # Predict
                prediction_idx = classifier.predict(dummy_dataset)[0]
                predicted_label = emotion_labels.get(prediction_idx, "Unknown")

                print(f"Image: {os.path.basename(img_path)}, Predicted Emotion: {predicted_label}")

                # Store result
                if img_path not in results:
                    results[img_path] = {}
                results[img_path][arch_str] = predicted_label

            except Exception as e:
                print(f"Error classifying image {img_path} with architecture {arch_str}: {e}")

    return results

if __name__ == "__main__":
    # --- User Configuration ---
    # !!! Replace with your actual architecture strings !!!
    ARCHITECTURES = [
        'CPCPCPFFF', # Example 1
        'IIFFF',     # Example 2
        'CNPCPNFFF', # Example 3
        'CPCPFFFF',  # Example 4 (replace with actual ones)
        'CCCCCCPFFF',# Example 5
        'ININFFF',   # Example 6
    ]
    IMAGE_FILES = ['happy.jpg', 'serious.jpg', 'triste.jpg']
    # !!! Adjust if your data directory is different !!!
    DATA_ROOT = './data'
    # !!! Adjust if your weights are stored elsewhere !!!
    WEIGHTS_DIR = '.'

    # --- Run Classification ---
    if not os.path.exists(DATA_ROOT) or not os.path.isdir(os.path.join(DATA_ROOT, 'train')):
         print(f"Error: Data root directory '{DATA_ROOT}' not found or does not contain a 'train' subdirectory.")
         print("Please adjust the DATA_ROOT variable.")
    else:
        all_predictions = classify_images(
            architecture_strings=ARCHITECTURES,
            image_paths=IMAGE_FILES,
            root_directory=DATA_ROOT,
            weights_dir=WEIGHTS_DIR
        )

        # Optional: Print summary at the end
        print("\n--- Classification Summary ---")
        for img, preds in all_predictions.items():
            print(f"\nImage: {os.path.basename(img)}")
            for arch, pred_label in preds.items():
                print(f"  Architecture {arch}: {pred_label}")

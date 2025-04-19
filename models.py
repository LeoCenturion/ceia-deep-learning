import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, nnet_builder, epochs=300, lr=0.01, batch_size=32, rebalance=True, patience=10, min_delta=0.001, validation_size=0.2, random_state=None, patience_val_worse_than_train_score=2):
        super(NnClassifier, self).__init__()
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.nnet_builder = nnet_builder
        self.batch_size = batch_size
        self.rebalance = rebalance
        self.patience = patience
        self.min_delta = min_delta
        self.validation_size = validation_size
        self.random_state = random_state
        self.patience_val_worse_than_train_score = patience_val_worse_than_train_score # Patience for val_score < train_score
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0 # Counter for loss-based stopping
        self.val_worse_than_train_score_counter = 0 # Counter for val_score < train_score stopping
        self.best_model_state = None
        self.history_ = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}

    def feed_model(self, X):
        return self.model(X)

    def early_stopping_check(self, val_loss):
        """Checks if early stopping should be triggered."""

        self.model.train()  # Set back to train mode
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                print(f"Early stopping triggered (validation loss did not improve for {self.patience} epochs).")
                return True
            return False

    def early_stopping_val_worse_than_train_score_check(self, val_score, train_score):
        """Checks if early stopping based on validation score being worse than train score should be triggered."""
        self.model.train() # Set back to train mode
        if val_score < train_score:
            self.val_worse_than_train_score_counter += 1
            if self.val_worse_than_train_score_counter >= self.patience_val_worse_than_train_score:
                print(f"Early stopping triggered (Validation F1 < Training F1 for {self.patience_val_worse_than_train_score} consecutive epochs).")
                return True
        else:
            self.val_worse_than_train_score_counter = 0 # Reset counter if condition is not met
        return False

    def fit(self, X, y=None):
        labels = X.labels  # so we can rebalance the dataset
        train_dataset, val_dataset = self._split_data(X, labels)
        train_dataloader = self.create_dataloader(train_dataset, None)
        val_dataloader = self.create_dataloader(val_dataset, None)

        self.nnet = self.nnet_builder.build()
        self.model = self.nnet.to(self.device)

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr, weight_decay=0.0001,
            momentum = 0.9
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.patience // 2,
            verbose=True
        )

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            all_train_preds = []
            all_train_labels = []
            for images, labels in train_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.feed_model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(train_dataloader)
            epoch_train_acc = correct_train / total_train
            epoch_train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
            self.history_['train_loss'].append(epoch_loss)
            self.history_['train_acc'].append(epoch_train_acc)
            self.history_['train_f1'].append(epoch_train_f1)
            print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Training Acc: {epoch_train_acc:.4f}, Training F1: {epoch_train_f1:.4f}")

            val_loss = 0.0
            correct_val = 0
            total_val = 0
            all_val_preds = []
            all_val_labels = []
            self.model.eval()
            with torch.no_grad():
                for val_images, val_labels in val_dataloader:
                    val_images, val_labels = val_images.to(self.device), val_labels.to(self.device)
                    val_output = self.feed_model(val_images)
                    v_loss = criterion(val_output, val_labels)
                    val_loss += v_loss.item()

                    _, predicted = torch.max(val_output.data, 1)
                    total_val += val_labels.size(0)
                    correct_val += (predicted == val_labels).sum().item()
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(val_labels.cpu().numpy())

            epoch_val_loss = val_loss / len(val_dataloader)
            epoch_val_acc = correct_val / total_val
            epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
            self.history_['val_loss'].append(epoch_val_loss)
            self.history_['val_acc'].append(epoch_val_acc)
            self.history_['val_f1'].append(epoch_val_f1)
            print(f"Epoch {epoch + 1}, Validation Loss: {epoch_val_loss:.4f}, Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}")

            scheduler.step(epoch_val_loss)
            self.model.train()

            # Check for early stopping based on validation loss
            if self.early_stopping_check(epoch_val_loss):
                break

            # Check for early stopping based on validation F1 score being worse than training F1 score
            if self.early_stopping_val_worse_than_train_score_check(epoch_val_f1, epoch_train_f1):
                break

        self.is_fitted_ = True
        return self

    def _split_data(self, dataset, labels):
        if self.validation_size > 0:
            train_idx, val_idx = train_test_split(
                range(len(dataset)),
                test_size=self.validation_size,
                stratify=labels,
                random_state=self.random_state
            )
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            print(f"Using {self.validation_size*100:.0f}% of training data for validation.")
            return train_dataset, val_dataset
        else:
            return dataset, dataset # Use all data for both if no validation

    def load_state(self, save_path):
        self.nnet = self.nnet_builder.build()
        self.nnet.load_state_dict(torch.load(save_path))
        self.model = self.nnet.to(self.device)

    def save_state(self, save_path):
        torch.save(self.model.cpu().state_dict(), save_path)
        self.model.to(self.device)

    def predict(self, X):
        dataloader = DataLoader(X, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = self.feed_model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
        return np.array(all_preds)

    def predict_proba(self, X):
        dataloader = DataLoader(X, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = self.feed_model(images)
                probabilities = torch.softmax(outputs, dim=1)
                all_probs.extend(probabilities.cpu().numpy())
        return np.array(all_probs)

    def create_dataloader(self, dataset, labels=None):
        if labels is not None:
            class_counts = np.bincount(labels)
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            sample_weights = [class_weights[label] for label in labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

class RandomDataset(Dataset):
    def __init__(self, num_samples, input_shape, num_classes):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.labels = [torch.randint(0, num_classes, (1,)).item() for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(self.input_shape)
        label = self.labels[idx]
        return image, label


def smoke_test(nnet_builder, input_shape=(3, 224, 224), num_classes=7, num_samples=10, batch_size=2):
    """Performs a smoke test using random data."""

    random_dataset = RandomDataset(num_samples, input_shape, num_classes)
    model_builder = nnet_builder(num_classes)
    model = NnClassifier(model_builder, batch_size=batch_size, rebalance=False)

    try:
        print("Training.")
        model.fit(random_dataset)
        print("Fit test passed.")
    except Exception as e:
        print(f"Fit test failed: {e}")
        return False

    try:
        predictions = model.predict(random_dataset)
        print("Prediction test passed.")
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False

    try:
        probabilities = model.predict_proba(random_dataset)
        print("Probability test passed.")
        print(f"Probabilities shape: {probabilities.shape}")
    except Exception as e:
        print(f"Probability test failed: {e}")
        return False

    return True




class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.emotion_to_label = {emotion: i for i, emotion in enumerate(os.listdir(os.path.join(root_dir, 'train')))}

        split_dir = os.path.join(root_dir, split)
        for emotion in os.listdir(split_dir):
            emotion_dir = os.path.join(split_dir, emotion)
            for image_file in os.listdir(emotion_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(emotion_dir, image_file))
                    self.labels.append(self.emotion_to_label[emotion])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class EmotionCNNBuilder(nn.Module, BaseEstimator):
    def __init__(self, num_classes):
        super(EmotionCNNBuilder, self).__init__()
        self.num_classes = num_classes

    def build(self):
        new_instance = EmotionCNNBuilder(self.num_classes)
        new_instance.build_layers()
        return new_instance

    def build_layers(self):
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def eval_model(model, root_directory, save_path=None):
    """
    Evaluates a PyTorch model on the given datasets and prints classification metrics.
    Args:
        model: The PyTorch model to evaluate
    """
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageDataset(root_directory, transform, split='train')
    test_dataset = ImageDataset(root_directory, transform, split='train') #for test, change to test.


    classifier = NnClassifier(model)

    if save_path is not None and os.path.exists(save_path):
        print(f"Loading weights from {save_path}")
        classifier.load_state(save_path)
    else:
        print("No weights file found. Training model...")
        classifier.fit(train_dataset)
        if save_path is not None: classifier.save_state(save_path)
        print(f"Model weights saved to {save_path}")

    # --- Plotting and Saving History ---
    history_df = None
    history_csv_path = None
    if save_path is not None:
        history_csv_path = os.path.splitext(save_path)[0] + '_history.csv'
        # Check if history CSV exists and if the classifier history is empty (model was loaded)
        if os.path.exists(history_csv_path) and not classifier.history_['train_loss']:
             try:
                 history_df = pd.read_csv(history_csv_path)
                 print(f"Loaded training history from {history_csv_path}")
             except Exception as e:
                 print(f"Warning: Could not load history from {history_csv_path}. Error: {e}")
                 history_df = None # Ensure df is None if loading fails

    # If history wasn't loaded from CSV, create it from the classifier and save it
    if history_df is None:
        history_df = pd.DataFrame(classifier.history_)
        if save_path is not None and not history_df.empty: # Only save if there's history data
            try:
                history_df.to_csv(history_csv_path, index=False)
                print(f"Training history saved to {history_csv_path}")
            except Exception as e:
                 print(f"Warning: Could not save history to {history_csv_path}. Error: {e}")

    # Proceed with plotting only if history_df is valid and not empty
    epochs_range = range(1, len(history_df) + 1)
    plt.figure(figsize=(15, 10))

    # Plot Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, history_df['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history_df['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, history_df['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history_df['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Plot F1 Score
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, history_df['train_f1'], label='Training F1 Score')
    plt.plot(epochs_range, history_df['val_f1'], label='Validation F1 Score')
    plt.legend(loc='lower right')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (Macro)')

    plt.tight_layout()
    plt.show()

    # --- End Plotting ---


    predictions = classifier.predict(test_dataset)

    emotion_labels = os.listdir(os.path.join(root_directory, 'train'))
    true_labels = test_dataset.labels
    print(classification_report(true_labels, predictions, target_names=emotion_labels))

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8)) # New figure for confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels1x1, out_channels3x3reduce, out_channels3x3, out_channels5x5reduce, out_channels5x5, out_channels_pool):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3x3reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels3x3reduce, out_channels3x3, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels5x5reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels5x5reduce, out_channels5x5, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels_pool, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class ConfigurableCNNBuilder(nn.Module, BaseEstimator):
    def __init__(self, architecture_string, num_classes):
        super(ConfigurableCNNBuilder, self).__init__()
        self.architecture_string = architecture_string
        self.num_classes = num_classes

    def build(self):
        new_instance = ConfigurableCNNBuilder(self.architecture_string, self.num_classes)
        new_instance.build_layers()
        return new_instance

    def build_layers(self):
        layers = []
        in_channels = 3
        pool_count = 0
        conv_output_shape = (3, 100, 100)
        flatten_layer_added = False
        for i, layer_type in enumerate(self.architecture_string):
            if layer_type == 'C':
                out_channels = 64 * (2 ** pool_count)
                layers.append((f'conv_{i}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)))
                layers.append((f'relu_{i}', nn.ReLU(inplace=True)))
                in_channels = out_channels
                conv_output_shape = (out_channels, conv_output_shape[1], conv_output_shape[2]) #update output shape
            elif layer_type == 'P':
                layers.append((f'pool_{i}', nn.MaxPool2d(kernel_size=2, stride=2)))
                pool_count += 1
                conv_output_shape = (conv_output_shape[0], conv_output_shape[1] // 2, conv_output_shape[2] // 2) #update output shape
            elif layer_type == 'I':
                layers.append((f'inception_{i}', InceptionModule(in_channels, 64, 96, 128, 16, 32, 32)))
                in_channels = 256
                conv_output_shape = (in_channels, conv_output_shape[1], conv_output_shape[2]) #update output shape
            elif layer_type == 'N':
                layers.append((f'norm_{i}', nn.LayerNorm(conv_output_shape)))
            elif layer_type == 'F':
                if not flatten_layer_added: #flatten before first fully connected.
                    layers.append(('flatten', nn.Flatten()))
                    flatten_layer_added = True
                if i == self.architecture_string.rfind('F'): #Last F layer.
                    layers.append((f'fc_{i}', nn.Linear(torch.prod(torch.tensor(conv_output_shape)), self.num_classes)))
                else:
                    layers.append((f'fc_{i}', nn.Linear(torch.prod(torch.tensor(conv_output_shape)), 512)))
                    layers.append((f'relu_fc_{i}', nn.ReLU(inplace=True)))
                    layers.append((f'dropout_fc_{i}', nn.Dropout()))
                    conv_output_shape = (512,) #update output shape
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)

def classify_images(model_configs, image_paths, root_directory):
    """
    Loads pre-trained models based on specified configurations and classifies given images.

    Args:
        model_configs (list): A list of tuples, where each tuple contains:
                              (architecture_string, weights_path).
                              - architecture_string (str): The string defining the CNN architecture.
                              - weights_path (str): Path to the trained model weights (.pth file).
        image_paths (list): List of paths to the images to classify.
        root_directory (str): Path to the root dataset directory (to get class labels).
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

    for arch_str, weights_path in model_configs:
        print(f"\n--- Processing Architecture: {arch_str} (Weights: {weights_path}) ---")

        if not os.path.exists(weights_path):
            print(f"Warning: Weights file not found at {weights_path}. Skipping this configuration.")
            continue

        try:
            model_builder = ConfigurableCNNBuilder(architecture_string=arch_str, num_classes=num_classes)
            classifier = NnClassifier(nnet_builder=model_builder, epochs=1) # Epochs not relevant for prediction
        except Exception as e:
            print(f"Error building model for architecture {arch_str} using weights {weights_path}: {e}. Skipping.")
            continue

        if not os.path.exists(weights_path):
            print(f"Warning: Weights file not found at {weights_path}. Skipping this model.")
            continue

        try:
            print(f"Loading weights from {weights_path}...")
            classifier.load_state(weights_path)
            classifier.model.eval() # Ensure model is in evaluation mode
        except Exception as e:
            print(f"Error loading weights from {weights_path} for architecture {arch_str}: {e}. Skipping.")
            continue

        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found at {img_path}. Skipping this image for model {weights_path}.")
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
    # !!! Define your model configurations (architecture_string, weights_path) !!!
    MODEL_CONFIGS = [
        ('CPCPCPFFF', 'CPCPCPFFF.pth'),     # Arch matches filename
        ('IIFFF', 'IIFFF_best.pth'),        # Arch IIFFF, weights from IIFFF_best.pth
        ('CNPCPNFFF', 'CNPCPNFFF.pth'),
        ('CPCPFFFF', 'CPCPFFFF_v1.pth'),   # Arch CPCPFFFF, weights from CPCPFFFF_v1.pth
        ('CCCCCCPFFF', 'CCCCCCPFFF.pth'),
        ('ININFFF', 'ININFFF.pth'),
        # ('CPCPFF', 'CPCPFF-2.pth'), # Example from user request
        # Add more configurations as needed
    ]
    IMAGE_FILES = ['happy.jpg', 'serious.jpg', 'triste.jpg']
    # !!! Adjust if your data directory is different !!!
    DATA_ROOT = './data'

    # --- Run Classification ---
    if not os.path.exists(DATA_ROOT) or not os.path.isdir(os.path.join(DATA_ROOT, 'train')):
         print(f"Error: Data root directory '{DATA_ROOT}' not found or does not contain a 'train' subdirectory.")
         print("Please adjust the DATA_ROOT variable.")
    else:
        # Filter MODEL_CONFIGS to include only those where the weight file exists
        valid_model_configs = []
        skipped_configs = []
        for arch, w_path in MODEL_CONFIGS:
            if os.path.exists(w_path):
                valid_model_configs.append((arch, w_path))
            else:
                skipped_configs.append((arch, w_path))

        if not valid_model_configs:
            print("Error: No valid model configurations found (weight files do not exist).")
            print(f"Checked configurations: {MODEL_CONFIGS}")
        else:
            if skipped_configs:
                print("Warning: Some specified weight files were not found and their configurations will be skipped:")
                for arch, w_path in skipped_configs:
                    print(f"  - Arch: {arch}, Weights: {w_path}")

            all_predictions = classify_images(
                model_configs=valid_model_configs,
                image_paths=IMAGE_FILES,
                root_directory=DATA_ROOT
            )

            # Optional: Print summary at the end
            print("\n--- Classification Summary ---")
            for img, preds in all_predictions.items():
                print(f"\nImage: {os.path.basename(img)}")
                for arch, pred_label in preds.items():
                    print(f"  Architecture {arch}: {pred_label}")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import cycle
from math import ceil
from typing import List, Optional, Tuple, Union, Any, Dict



from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize

import tensorflow as tf



class NPZDatasetManager:
    def __init__(self, path: str):
        """
        Initializes the NPZ dataset manager.
        Args:
            path (str): Path to the directory containing .npz files.
        """
        self.path = path
        self.npz_files = self.list_npz_files()

    def list_npz_files(self) -> List[str]:
        """
        Lists all .npz files in the specified directory.
        Returns:
            List[str]: List of full paths of the .npz files.
        """
        npz_files = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.npz')]
        print(f'Number of .npz files in the path: {len(npz_files)}')
        for file in npz_files:
            print(file)
        return npz_files

    def load_npz_by_index(self, index: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Loads a .npz file based on its index in the file list.
        Args:
            index (int): Index of the .npz file to load.
        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary with data loaded from the .npz file, or None if the index is out of range.
        """
        if index < 0 or index >= len(self.npz_files):
            print("Index out of range.")
            return None
        file_path = self.npz_files[index]
        dataset = np.load(file_path, allow_pickle=True)
        print("Keys in the NPZ file:", list(dataset.keys()))
        return dict(dataset)

    def concatenate_and_save_all(self, save_path: str) -> None:
        """
        Concatenates 'jetImages' and 'jetClass' from all .npz files and
        saves the result in a single .npz file.
        Args:
            save_path (str): Path where the resulting .npz file will be saved.
        """
        jet_images = []
        jet_classes = []
        for file_path in self.npz_files:
            dataset = np.load(file_path, allow_pickle=True)
            if "jetImages" in dataset and "jetClass" in dataset:
                jet_images.append(dataset["jetImages"])
                jet_classes.append(dataset["jetClass"])
            else:
                print(f"The keys 'jetImages' and/or 'jetClass' not found in {file_path}")

        if jet_images and jet_classes:  # Check if lists are not empty
            jet_images = np.concatenate(jet_images)
            jet_classes = np.concatenate(jet_classes)
            np.savez(save_path, jetImages=jet_images, jetClass=jet_classes)
            print(f"Concatenated data saved in {save_path}")
        else:
            print("No valid data found to concatenate.")

class ClassDistributionPlotter:
    def __init__(self, labels):
        self.labels = labels
        self.class_names = None
        self.class_counts = None

    def map_labels(self, label_dict):
        self.class_names = list(label_dict.keys())
        self.class_counts = np.sum(self.labels, axis=0)

    def plot_distribution(self):
        if self.class_names is None or self.class_counts is None:
            raise ValueError("Class names and counts not set. Call map_labels() first.")

        data = pd.DataFrame({
            'Class': self.class_names,
            'Count': self.class_counts
        })

        # Set style and create a custom color palette
        sns.set_theme(style="whitegrid")
        num_classes = len(self.class_names)
        palette = sns.color_palette("pastel", n_colors=num_classes)

        # Create a bar plot with Seaborn
        plt.figure(figsize=(10, 6))
        bar_plot = sns.barplot(x='Class', y='Count', data=data, palette=palette)

        # Add annotations to the bars
        for p in bar_plot.patches:
            bar_plot.annotate(format(p.get_height(), '.0f'),
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='center',
                              xytext=(0, 9),
                              textcoords='offset points')

        plt.title('Class Distribution in the Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Examples')
        plt.show()

class BalancedDatasetSampler:
    """
    A class for balancing a dataset by randomly sampling a fixed number of examples from each class.
    This version specifically handles NumPy arrays.

    Attributes:
    - images (np.ndarray): NumPy array of image data.
    - labels (np.ndarray): NumPy array of labels.
    - balanced_images (np.ndarray): NumPy array to store balanced image data.
    - balanced_labels (np.ndarray): NumPy array to store balanced labels.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        """
        Initializes the BalancedDatasetSampler with NumPy arrays of image and label data.

        Args:
        - images (np.ndarray): NumPy array of image data.
        - labels (np.ndarray): NumPy array of labels.
        """
        if not isinstance(images, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("Expected NumPy arrays as arguments")

        self.images = images
        self.labels = labels
        self.balanced_images = None
        self.balanced_labels = None

    def balance_dataset(self, shuffle=True) -> None:
        """
        Balances the dataset by randomly sampling a fixed number of examples from each class.
        The balanced data is stored in the attributes balanced_images and balanced_labels.
        """
        # Find the minimum number of examples in all classes
        min_samples = np.min(np.sum(self.labels, axis=0))

        # Convert min_samples to an integer
        min_samples = int(min_samples)

        # Initialize lists to store balanced data
        balanced_images = []
        balanced_labels = []

        for i in range(self.labels.shape[1]):  # Iterate over each class
            # Get indices of all examples from the current class
            class_indices = np.where(self.labels[:, i] == 1)[0]

            # Ensure there are enough examples in the current class
            if len(class_indices) >= min_samples:
                # Randomly choose 'min_samples' from these indices
                selected_indices = np.random.choice(class_indices, min_samples, replace=False)

                # Add the selected images and labels to the lists
                balanced_images.extend(self.images[selected_indices])
                balanced_labels.extend(self.labels[selected_indices])

        # Convert the lists to NumPy arrays
        self.balanced_images = np.array(balanced_images)
        self.balanced_labels = np.array(balanced_labels)

        # Shuffle the balanced dataset: Optional
        if shuffle:
            shuffle_indices = np.random.permutation(len(self.balanced_images))
            self.balanced_images = self.balanced_images[shuffle_indices]
            self.balanced_labels = self.balanced_labels[shuffle_indices]

def generate_dataset_info(images, labels):
    """
    Generates information about a dataset, including length, shape, and the shape of the first element.
    Args:
    - images (array): Array of image data.
    - labels (array): Array of labels.
    Returns:
    - str: Formatted information about the length, shape, and shape of the first element of images and labels.
    """
    # Check that the arguments are arrays
    if not isinstance(images, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("Arrays are expected as arguments")
    info = (
        f"Length of images: {len(images)}\n"
        f"Shape of images: {images.shape}\n"
        f"Shape of the first element of images: {images[0].shape}\n"
        f"Length of labels: {len(labels)}\n"
        f"Shape of labels: {labels.shape}\n"
        f"Shape of the first element of labels: {labels[0].shape}"
    )

    return info
class DatasetSplitter:
    def __init__(self, train_size: float, val_size: float, test_size: float):
        """
        Initializes the DatasetSplitter object with sizes for the training, validation, and test sets.
        Args:
            train_size (float): Percentage of the dataset for the training set.
            val_size (float): Percentage of the dataset for the validation set.
            test_size (float): Percentage of the dataset for the test set.
        """
        if train_size + val_size + test_size != 1.0:
            raise ValueError("The sum of train_size, val_size, and test_size must be 1.")

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def split_data(self, images, labels) -> dict:
        """
        Splits the data into training, validation, and test sets.

        Args:
            images (np.ndarray): Array of images.
            labels (np.ndarray): Array of labels.
        Returns:
            dict: A dictionary containing the split datasets.
                'X_train' (np.ndarray): Training set images.
                'X_val' (np.ndarray): Validation set images.
                'X_test' (np.ndarray): Test set images.
                'y_train' (np.ndarray): Training set labels.
                'y_val' (np.ndarray): Validation set labels.
                'y_test' (np.ndarray): Test set labels.
        """
        from sklearn.model_selection import train_test_split
        import tensorflow as tf

        # First split into training+validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=self.test_size, shuffle=True, random_state=42)

        # Now split the training+validation into training and validation
        val_size_adjusted = self.val_size / (1 - self.test_size)  # Validation size adjustment
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, shuffle=True, random_state=42)

        print("Dimensions of the training set: ", X_train.shape)
        print("Dimensions of the validation set: ", X_val.shape)
        print("Dimensions of the test set: ", X_test.shape)

        return {
            "X_train": tf.convert_to_tensor(X_train, dtype=tf.float32),
            "X_val": tf.convert_to_tensor(X_val, dtype=tf.float32),
            "X_test": tf.convert_to_tensor(X_test, dtype=tf.float32),
            "y_train": tf.convert_to_tensor(y_train, dtype=tf.float32),
            "y_val": tf.convert_to_tensor(y_val, dtype=tf.float32),
            "y_test": tf.convert_to_tensor(y_test, dtype=tf.float32)
        }
class ImageTransformer:
    """
    A class for image transformations using TensorFlow.
    """
    def normalize_by_global_max(self, imgs):
        """
        Normalize each channel of the input image tensor by the maximum value across all images.
        Args:
            imgs (tf.Tensor): Input image tensor of shape (num_images, height, width, channels).
        Returns:
            tf.Tensor: Normalized image tensor.
        """
        if not isinstance(imgs, tf.Tensor):
            raise TypeError("Input must be a TensorFlow tensor")

        min_val = tf.reduce_min(imgs)
        max_val = tf.reduce_max(imgs)
        normalized_imgs = (imgs - min_val) / (max_val - min_val)

        return normalized_imgs

    def fft_transform_nch(self, img):
        """
        Apply FFT transformation to the input single-channel image tensor and add the FFT result as a second channel.

        Args:
            img (tf.Tensor): Input image tensor of shape (height, width, 1).

        Returns:
            tf.Tensor: Transformed image tensor of shape (height, width, 2).
        """
        if not isinstance(img, tf.Tensor):
            raise TypeError("Input must be a TensorFlow tensor")

        if img.shape[-1] != 1:
            raise ValueError("Input image must have a single channel")

        img = tf.squeeze(img, axis=-1)  # Remove the channel dimension
        fourier_transform = tf.signal.fft2d(tf.cast(img, tf.complex64))
        magnitude_spectrum = tf.math.abs(fourier_transform)

        img_with_fft = tf.stack([img, magnitude_spectrum], axis=-1)

        return img_with_fft

    def apply_transformset(self, X, y, transformation_list=None, normalize_function=False, apply_fft=False):
        """
        Apply a set of transformations to input images.

        Args:
            X (list): List of input images.
            y (list): List of labels corresponding to input images.
            transformation_list (list): List of transformation functions to apply.
            normalize_function (bool): Whether to normalize the images.
            apply_fft (bool): Whether to apply FFT transformation.

        Returns:
            tf.Tensor: Transformed input images tensor.
            tf.Tensor: Transformed labels tensor.
        """
        transformed_X = []
        transformed_y = []

        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

        if normalize_function:
            X_tensor = self.normalize_by_global_max(X_tensor)

        for x, label in zip(X_tensor, y):
            transformed_versions = [x]

            if transformation_list is not None:
                for trans in transformation_list:
                    x_trans = trans(x)
                    if x_trans.shape[0:2] != (100, 100):
                        x_trans = tf.image.resize(x_trans, [100, 100])
                    transformed_versions.append(x_trans)

            for x_trans in transformed_versions:
                if apply_fft:
                    x_trans = self.fft_transform_nch(x_trans)

                transformed_X.append(x_trans)
                transformed_y.append(label)

        return tf.stack(transformed_X), tf.convert_to_tensor(transformed_y, dtype=tf.float32)

def visualize_image_channels(images, labels, num_images=3):
    """
    Visualizes individual channels of the first 'num_images' images in a batch and checks pixel value normalization.
    Args:
        images (tf.Tensor): A batch of images with shape [batch_size, H, W, C].
        labels (tf.Tensor): Corresponding labels of the images.
        num_images (int): Number of images to visualize.

    The function iterates over each channel of the first 'num_images' images in the provided batch.
    It displays each channel as a grayscale image and prints the minimum and maximum pixel values for that channel.
    It also checks if the pixel values are normalized between 0 and 1.
    """
    for i in range(num_images):
        # Get the i-th image and its label from the batch
        current_image = images[i]
        current_label = labels[i].numpy()

        # Create a horizontal subplot for each channel in the image
        fig, axs = plt.subplots(1, current_image.shape[-1], figsize=(10, 5))

        # Check if axs is not iterable (single subplot)
        if not isinstance(axs, np.ndarray):
            axs = [axs]

        # Iterate over each channel in the image (assuming the format [H, W, C])
        for channel_idx in range(current_image.shape[-1]):
            image_channel = current_image[..., channel_idx]

            # Display the channel in the subplot without grid and axis ticks
            axs[channel_idx].imshow(image_channel.numpy(), cmap='gray')
            axs[channel_idx].set_title(f"Channel {channel_idx + 1}")
            axs[channel_idx].axis('off')  # Turn off axis ticks and labels

        # Set the overall title for the subplot
        fig.suptitle(f"Image {i + 1} - Label: {current_label}")

        plt.show()

        # Print minimum and maximum pixel values after showing the images
        for channel_idx in range(current_image.shape[-1]):
            image_channel = current_image[..., channel_idx]
            min_pixel_value = tf.reduce_min(image_channel).numpy()
            max_pixel_value = tf.reduce_max(image_channel).numpy()
            print(f"Min pixel value in channel {channel_idx + 1}: {min_pixel_value}")
            print(f"Max pixel value in channel {channel_idx + 1}: {max_pixel_value}")
            if 0 <= min_pixel_value <= 1 and 0 <= max_pixel_value <= 1:
                print(f"Channel {channel_idx + 1} is correctly normalized")
            else:
                print(f'Channel {channel_idx + 1} is not normalized correctly')

class TrainingPlotter:
    """Class to plot training metrics such as loss and accuracy."""
    def __init__(self, training_result: dict) -> None:
        """Initializes the plotter with training results.
        Args:
            training_result (dict): A history object containing training metrics.
        """
        self.result = training_result.history

    def plot_metrics(self) -> None:
        """Plots the training and validation loss and accuracy."""
        metrics = {
            'Loss': ('loss', 'val_loss'),
            'Accuracy': ('accuracy', 'val_accuracy')
        }

        plt.figure(figsize=(12, 4))

        for i, (title, (train_metric, val_metric)) in enumerate(metrics.items(), 1):
            plt.subplot(1, 2, i)
            plt.plot(self.result[train_metric], label=f'Training {title}')
            plt.plot(self.result[val_metric], label=f'Validation {title}')
            plt.title(f'Training and Validation {title}')
            plt.xlabel('Epochs')
            plt.ylabel(title)
            plt.legend()

        plt.tight_layout()
        plt.show()
def plot_multiclass_roc(models: Union[Any, List[Any]], class_names: List[str], val_dataset=None, data=None, each=False):
    if not isinstance(models, list):
        models = [models]  # Ensures that models is a list

    n_models = len(models)
    n_classes = len(class_names)
    colors = cycle(['limegreen','royalblue','darkorange','firebrick','darkviolet','darkgoldenrod','darkcyan','darkmagenta','darkolivegreen','darkslategray'])
    n=0
    if each:
        # Calculate the number of rows needed to fit all models in a 3-column grid
        n_rows = ceil(n_models / 3)
        fig, axs = plt.subplots(n_rows, 3, figsize=(15, n_rows * 5), constrained_layout=True)
        fig.suptitle('Multi-class ROC curves for each model', fontsize=16)

        for model_idx, model in enumerate(models):
            row = model_idx // 3
            col = model_idx % 3
            ax = axs[row, col] if n_rows > 1 else axs[col]
            plot_roc_for_model(model, model_idx, class_names, val_dataset, data, ax, next(colors),n='')

            # Set titles and labels for the current subplot
            ax.set_title(f'Model {model_idx + 1}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")

        # Hide empty subplots if any
        for idx in range(model_idx + 1, n_rows * 3):
            plt.delaxes(axs.flatten()[idx])

        plt.show()

    else:
        plt.figure(figsize=(10, 8))
        for model_idx, model in enumerate(models):
            n+=1
            plot_roc_for_model(model, model_idx, class_names, val_dataset, data, plt.gca(), next(colors),n)
        plt.legend(loc="lower right")
        plt.show()

def plot_multiclass_roc(models: Union[Any, List[Any]], class_names: List[str], val_dataset=None, data=None, each=False):
    if not isinstance(models, list):
        models = [models]  # Ensure that models is a list

    n_models = len(models)
    n_classes = len(class_names)
    colors = cycle(['limegreen', 'royalblue', 'darkorange', 'firebrick', 'darkviolet', 'darkgoldenrod', 'darkcyan', 'darkmagenta', 'darkolivegreen', 'darkslategray'])
    n = 0
    if each:
        # Calculate the number of rows needed to accommodate all models in a 3-column layout
        n_rows = ceil(n_models / 3)
        fig, axs = plt.subplots(n_rows, 3, figsize=(15, n_rows * 5), constrained_layout=True)
        fig.suptitle('Multi-class ROC curves for each model', fontsize=16)

        for model_idx, model in enumerate(models):
            row = model_idx // 3
            col = model_idx % 3
            ax = axs[row, col] if n_rows > 1 else axs[col]
            plot_roc_for_model(model, model_idx, class_names, val_dataset, data, ax, next(colors), n='')

            # Set titles and labels for the current subplot
            ax.set_title(f'Model {model_idx + 1}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")

        # Hide empty subplots if any
        for idx in range(model_idx + 1, n_rows * 3):
            plt.delaxes(axs.flatten()[idx])

        plt.show()

    else:
        plt.figure(figsize=(10, 8))
        for model_idx, model in enumerate(models):
            n += 1
            plot_roc_for_model(model, model_idx, class_names, val_dataset, data, plt.gca(), next(colors), n)
        plt.legend(loc="lower right")
        plt.show()

def plot_roc_for_model(model, model_idx, class_names, val_dataset, data, ax, color, n):
    n_classes = len(class_names)
    line_styles_with_alpha = [
        ('-', 0.9),  # Solid style with alpha 0.1
        ('--', 0.7),  # Dashed style with alpha 0.2
        ('--', 0.5),  # Dash-dot style with alpha 0.3
        ('--', 0.3),  # Dotted style with alpha 0.4
        ('--', 0.2)  # Custom style with alpha 0.5
    ]
    line_styles = cycle(line_styles_with_alpha)
    if data is not None:
        if isinstance(data, tuple) and len(data) == 2:
            y_score = model.predict(np.array(data[0]))
            y_true = data[1]
        else:
            raise ValueError("The 'data' parameter should be a tuple containing two numpy arrays (X, y).")
    elif val_dataset is not None:
        y_true, y_score = [], []
        for x_val, labels in val_dataset:
            y_score_batch = model.predict(x_val)
            y_true.extend(labels)
            y_score.extend(y_score_batch)
        y_true = np.array(y_true)
        y_score = np.array(y_score)
    else:
        raise ValueError("Either 'val_dataset' or 'data' must be provided.")

    # Binarize the labels in a one-vs-all format
    y_true = label_binarize(y_true, classes=range(n_classes))

    # Calculate the ROC curve and ROC area for each class
    for i in range(n_classes):
        line_style, alpha = next(line_styles)

        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        if isinstance(n, str):
            ti = ''
        else:
            ti = f'Model {n}'

        ax.plot(fpr, tpr, color=color, linestyle=line_style, alpha=alpha,
                label=f'{ti} Class {class_names[i]} (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')



def plot_confusion_matrices(models: Union[Any, List[Any]], class_names: List[str], val_dataset: Optional[Any] = None, data: Optional[Tuple[np.ndarray, np.ndarray]] = None, each: bool = False) -> None:
    if not isinstance(models, list):
        models = [models]

    n_classes = len(class_names)
    cols = 3

    for model_idx, model in enumerate(models):
        y_true = []
        y_pred = []

        if data is not None:
            if isinstance(data, tuple) and len(data) == 2:
                predictions = model.predict(np.array(data[0]))
                y_pred = np.argmax(predictions, axis=1)
                y_true = np.argmax(np.array(data[1]), axis=1)
            else:
                raise ValueError("The 'data' parameter should be a tuple containing two numpy arrays (X, y).")
        elif val_dataset is not None:
            for x_val, labels in val_dataset:
                predictions = model.predict(x_val)
                y_pred_batch = np.argmax(predictions, axis=1)
                y_true.extend(np.argmax(labels, axis=1))
                y_pred.extend(y_pred_batch)
        else:
            raise ValueError("Either 'val_dataset' or 'data' must be provided.")

        if each:
            rows_per_model = ceil(n_classes / cols)
            fig, axs = plt.subplots(rows_per_model, cols, figsize=(15, 5 * rows_per_model))
            fig.suptitle(f'Binary Confusion Matrices for Model {model_idx+1}', fontsize=16)

            for i, class_name in enumerate(class_names):
                row = i // cols
                col = i % cols
                ax = axs[row, col] if rows_per_model > 1 else axs[col]

                true_binary = np.array(y_true == i, dtype=int)
                pred_binary = np.array(y_pred == i, dtype=int)

                cm = confusion_matrix(true_binary, pred_binary)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Not {class_name}', class_name], yticklabels=[f'Not {class_name}', class_name], ax=ax)
                ax.set_title(f'Class: {class_name}')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')

            for i in range(n_classes, rows_per_model * cols):
                if rows_per_model > 1:
                    axs.flatten()[i].axis('off')
                else:
                    axs[i].axis('off')

        else:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Multiclass Confusion Matrix for Model {model_idx+1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

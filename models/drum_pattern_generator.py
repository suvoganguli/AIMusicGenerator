import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

class DrumPatternGenerator:
    def __init__(self, num_samples=1000, sequence_length=32, num_drums=3):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_drums = num_drums
        self.data = np.zeros((self.num_samples, self.sequence_length, self.num_drums))

    def generate_patterns(self):
        # Define the base pattern
        base_pattern = np.array([
            [1, 0, 1],  # Kick on first step, hi-hat always
            [0, 0, 1],  # Hi-hat only
            [0, 1, 1],  # Snare on third step, hi-hat
            [0, 0, 1]   # Hi-hat only
        ])

        # Repeat the base pattern to fill the sequence length
        for seq in self.data:
            repeated_pattern = np.tile(base_pattern, (self.sequence_length // 4, 1))
            seq[:self.sequence_length, :] = repeated_pattern

        print("Patterns generated.")  # Debug statement
        return self.data

    def split_data(self, test_size=0.2, val_size=0.25):
        # Split into input (X) and output (y)
        X = self.data[:, :-1, :]
        y = self.data[:, 1:, :]
        
        print("Generated pattern:")
        print(self.data)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Split train set further into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )

        print("Data split into training, validation, and test sets.")  # Debug statement
        print(f"Shapes after split: X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_dataset(self, X_train, y_train, batch_size=16):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_loader = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        print("Training dataset prepared.")  # Debug statement
        return train_loader
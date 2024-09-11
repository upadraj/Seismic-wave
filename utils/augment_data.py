import numpy as np
import pandas as pd

def add_noise(data, noise_level=0.05):
    noise = np.random.randn(*data.shape) * noise_level
    return data + noise


def time_shift(data, shift_max=100):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift, axis=1)


def scale_amplitude(data, scale_factor):
    return data * scale_factor


def augment_data(df):
    augmented_data = []
    augmented_labels = []

    label_df = pd.DataFrame(df['labels'])

    df = df.drop(columns=['labels'])

    unique_classes = np.unique(label_df['labels'])

    for target_class in unique_classes:

        # Isolate the target class
        target_indices = np.where(label_df['labels'] == target_class)[0]

        data_to_augment = df.iloc[target_indices]
        labels_to_augment = label_df.iloc[target_indices]

        # Augment each sample in the target class
        for i in range(data_to_augment.shape[0]):
            augmented_sample = data_to_augment.iloc[i].copy().to_numpy()

            # Apply augmentations
            augmented_sample = add_noise(augmented_sample, noise_level=0.05)

            if target_class == 'S':
                augmented_sample = scale_amplitude(augmented_sample, scale_factor=1.8)
            else:
                augmented_sample = scale_amplitude(augmented_sample, scale_factor=1.3)

            # Add the augmented sample and label to the lists
            augmented_data.append(augmented_sample)
            augmented_labels.append(target_class)

        # Step 2: Convert augmented data and labels to DataFrame
        augmented_data_df = pd.DataFrame(augmented_data, columns=df.columns)
        augmented_labels_df = pd.DataFrame(augmented_labels,
                                           columns=['labels'])  # Replace 'S' with the label column name

        # Step 3: Reintegrate augmented data into the original dataset
        train_features = pd.concat([df, augmented_data_df], axis=0)
        train_labels = pd.concat([label_df, augmented_labels_df], axis=0)

        # Step 4: Reset indices after concatenation to avoid index issues
        train_features = train_features.reset_index(drop=True)
        train_labels = train_labels.reset_index(drop=True)

        # Check final shape to verify augmentation
        print("Original dataset shape:", df.shape)
        print("Augmented dataset shape:", train_features.shape)

        return train_features, train_labels

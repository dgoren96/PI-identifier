"""
Augmentation script for placental insufficiency dataset.

Steps:
1. Load features and target CSVs.
2. Convert the 'frames' column to numpy arrays.
3. Resample all signals to a uniform sampling frequency (300 Hz).
4. Merge features with target labels for augmentation.
5. Compute the class balance and determine augmentation needs.
6. Apply random augmentations (noise, scaling, etc.) to both True and False classes.
7. Record the augmented data and save to a CSV for later use.
"""

import numpy as np
import AI_main as ai
import pandas as pd
import assisting_functions as af

if __name__ == '__main__':
    features_csv = fr'C:\Python_Projects\pythonProject\features.csv'
    target_csv = fr'C:\Python_Projects\pythonProject\target.csv'

    # Load and preprocess features + target
    df_fet_tar = ai.load_and_preprocess_data(features_csv, target_csv)
    target_df = pd.read_csv(target_csv)

    features_df = pd.read_csv(features_csv)
    # Convert frames column string to numpy array
    features_df['frames'] = features_df['frames'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
    df = features_df.copy()

    # Resample all signals to 300 Hz
    features_df['frames'] = features_df.apply(lambda row: af.resample_signal(row['frames'], row['fs'], 300), axis=1)
    features_df['fs'] = 300

    # Merge features with target labels
    df_to_augment = pd.merge(
        features_df[['id', 'id_frame_id', 'id_int', 'trimester', 'gw', 'gw_in_days',
                     'gw_delivery', 'gw_delivery_in_days', 'fs', 'frames']],
        target_df,
        on='id_int',
        how='inner'
    )

    # Display class counts before augmentation
    value_counts = df_to_augment['has_pi'].value_counts()
    num_true = value_counts.get(True, 0)
    num_false = value_counts.get(False, 0)
    print("\nDB pre-Augmentations:")
    print(f"num_true = {num_true}")
    print(f"num_false = {num_false}")

    balance_factor = int(num_false / num_true)

    # Initialize lists to store augmentation information
    augmentation_records = []

    # Augment the True class
    for i in range(1, 2):
        df_augmented_True = df_to_augment[df_to_augment['has_pi'] == True].copy()
        df_augmented_True['id_frame_id'] = df_augmented_True['id_frame_id'].apply(lambda x: f"{x}_aug_{i}")
        df_augmented_True['frames'] = df_to_augment[df_to_augment['has_pi'] == True]['frames'].apply(af.apply_random_augmentation)
        augmentation_records.append(df_augmented_True)

    # Augment the False class
    for i in range(1, 2):  # Adjust number of augmentations as needed
        df_augmented_False = df_to_augment[df_to_augment['has_pi'] == False].copy()
        df_augmented_False['id_frame_id'] = df_augmented_False['id_frame_id'].apply(lambda x: f"{x}_aug_{i}")
        df_augmented_False['frames'] = df_to_augment[df_to_augment['has_pi'] == False]['frames'].apply(af.apply_random_augmentation)
        augmentation_records.append(df_augmented_False)

    # Combine all augmented records
    df_augmentation_info = pd.concat(augmentation_records, axis=0).reset_index(drop=True)
    df_augmentation_info.drop('has_pi', axis=1, inplace=True)

    # Save augmented info to CSV
    file_path = fr'C:\Python_Projects\pythonProject\augmentation_info.csv'
    df_augmentation_info.to_csv(file_path, index=False)
    print("\nExported augmentation_info.csv")

    input("Press Enter to exit...")

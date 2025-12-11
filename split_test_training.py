import pandas as pd
import os
import shutil
import random

# Define paths
csv_file = "/media/edward/HDD/Workspace/Resnet/LEAF/human/resnet-leaf-top-and-angled-all-days.csv"
image_folder = "/media/edward/HDD/Workspace/Resnet/LEAF/human/training/images/"
test_folder = "/media/edward/HDD/Workspace/Resnet/LEAF/human/test/images/"
validation_folder = "/media/edward/HDD/Workspace/Resnet/LEAF/human/validation/images/"

# Function to determine CSV save path
def get_csv_save_path(image_folder, dataset_type, csv_file):
    parent_folder = os.path.dirname(os.path.dirname(image_folder))  # Get directory before the images folder
    csv_filename = os.path.basename(csv_file)  # Extract original CSV filename
    new_csv_filename = f"{dataset_type}_{csv_filename}"  # Prepend dataset type
    return os.path.join(parent_folder, new_csv_filename)  # Construct new save path

# Create directories if they don't exist
os.makedirs(test_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

# Set random seed for repeatability
random.seed(42)

# Load CSV file
df = pd.read_csv(csv_file)

# Get unique pot IDs
unique_pot_ids = df['Pot ID'].unique()

test_pot_ids = {'D-4-5', 'D-6-1', 'D-10-1', 'D-11-2', 'D-12-5', 'N-1-5', 'N-3-5', 'N-8-3', 'N-9-4'}
validation_pot_ids = {'D-4-4', 'D-5-1', 'D-10-3', 'D-12-2', 'N-1-2', 'N-2-4', 'N-7-5', 'N-8-5', 'N-9-5'}

print(f"Number of pot IDs: {len(unique_pot_ids)}")
print(f"Number of test pot IDs: {len(test_pot_ids)}")
print(f"Number of validation pot IDs: {len(validation_pot_ids)}")
print("Test pot IDs:", test_pot_ids)
print("Validation pot IDs:", validation_pot_ids)


# Split dataset
test_df = df[df['Pot ID'].isin(test_pot_ids)]
validation_df = df[df['Pot ID'].isin(validation_pot_ids)]
train_df = df[~df['Pot ID'].isin(test_pot_ids.union(validation_pot_ids))]

print(f"Number of images to be moved to test: {len(test_df)}")
print(f"Number of images to be moved to validation: {len(validation_df)}")

# Set a boolean flag to determine if images should be moved
move_images = True

test_count = 0
for image_name in os.listdir(image_folder):
    if any(str(pot_id) in image_name for pot_id in test_pot_ids):
        src_path = os.path.join(image_folder, image_name)
        dst_path = os.path.join(test_folder, image_name)
        if os.path.exists(src_path):
            test_count += 1
            if move_images:
                shutil.move(src_path, dst_path)
print(f"Moved {test_count} images to test folder.")

val_count = 0
for image_name in os.listdir(image_folder):
    if any(str(pot_id) in image_name for pot_id in validation_pot_ids):
        src_path = os.path.join(image_folder, image_name)
        dst_path = os.path.join(validation_folder, image_name)
        if os.path.exists(src_path):
            val_count += 1
            if move_images:
                shutil.move(src_path, dst_path)
print(f"Moved {val_count} images to validation folder.")

# Add the "drought" column
train_df['drought'] = train_df['Pot ID'].apply(lambda x: 1 if str(x).startswith('D') else 0)
test_df['drought'] = test_df['Pot ID'].apply(lambda x: 1 if str(x).startswith('D') else 0)
validation_df['drought'] = validation_df['Pot ID'].apply(lambda x: 1 if str(x).startswith('D') else 0)

# Save CSV files dynamically
train_csv_path = get_csv_save_path(image_folder, "train", csv_file)
test_csv_path = get_csv_save_path(test_folder, "test", csv_file)
validation_csv_path = get_csv_save_path(validation_folder, "validation", csv_file)

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)
validation_df.to_csv(validation_csv_path, index=False)

print(f"Training CSV saved to: {train_csv_path}")
print(f"Test CSV saved to: {test_csv_path}")
print(f"Validation CSV saved to: {validation_csv_path}")
print("Dataset split completed. Images moved (if enabled) and CSV files saved.")

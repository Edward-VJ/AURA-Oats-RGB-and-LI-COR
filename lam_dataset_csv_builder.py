import os
import pandas as pd

# Define the directory containing image files
base_dir = "/media/edward/HDD/Docker-Workspace/Data/GH4-Completed/"

dates = ['2024-08-29-Morning', '2024-08-29-Evening',
         '2024-09-02-Evening', '2024-09-03-Morning',
         '2024-09-03-Evening', '2024-09-04-Morning']
final_output_csv = os.path.join(base_dir, "resnet-leaf-top-and-angled-all-days.csv")  # Final combined CSV

all_dfs = []  # List to store dataframes for final concatenation

for date in dates:
    image_dir = os.path.join(base_dir, date, f"{date}_results_rgb_11/images")
    input_excel_file = os.path.join(base_dir, date, "processed_LiCOR_data_v2.xlsx")
    output_csv_file = os.path.join(base_dir, date, "resnet-leaf-top-and-angled.csv")

    # Check if the Excel file exists
    if not os.path.exists(input_excel_file):
        print(f"ERROR: Missing Excel file: {input_excel_file}")
        continue  # Skip this date and move to the next

    # Read the specified sheet from the Excel file
    try:
        df = pd.read_excel(input_excel_file)
    except Exception as e:
        print(f"ERROR: Failed to read Excel file {input_excel_file}. Exception: {e}")
        continue  # Skip this date and move to the next

    # Ensure the 'Pot ID' column exists
    if 'Pot ID' not in df.columns:
        print(f"ERROR: The Excel file {input_excel_file} must contain a 'Pot ID' column.")
        continue  # Skip processing for this date

    df['Pot ID'] = df['Pot ID'].astype(str)

    # Define the conditions for selecting images
    image_conditions = [
        #top
        ("RZ0.0", "RX180.0"),
        ("RZ180.0", "RX180.0"),
        #angled
        ("RZ180.0", "RX-120.0"),
        ("RZ0.0", "RX-120.0"),
        # Add more conditions here if needed
    ]

    # List to store the new rows
    filtered_rows = []

    # Check if the image directory exists
    if not os.path.exists(image_dir):
        print(f"WARNING: Image directory does not exist: {image_dir}")
        continue  # Skip this date and move to the next

    for _, row in df.iterrows():
        plant_pot = row['Pot ID']

        for rz, rx in image_conditions:
            matching_files = [
                f for f in os.listdir(image_dir)
                if plant_pot in f and rz in f and rx in f
            ]

            for img_file in matching_files:
                new_row = row.copy()
                new_row['image_file'] =  img_file 
                new_row['date'] = date  # Add the date column for clarity
                filtered_rows.append(new_row)

    # Create a new DataFrame from the filtered rows
    filtered_df = pd.DataFrame(filtered_rows)

    if filtered_df.empty:
        print(f"WARNING: No matching images found for {date}. No CSV generated.")
        continue  # Skip saving empty CSVs

    # Save the filtered DataFrame to a CSV file
    filtered_df.to_csv(output_csv_file, index=False)
    print(f"Filtered dataset with image_file column saved to {output_csv_file}.")

    # Store DataFrame for final concatenation
    all_dfs.append(filtered_df)

# Concatenate all DataFrames and save final CSV
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(final_output_csv, index=False)
    print(f"Final combined dataset saved to {final_output_csv}.")
else:
    print("No valid data found. Final CSV was not created.")

import os
from PIL import Image
import numpy as np
from lang_sam import LangSAM


def retrieve_combined_mask(image, masks, output_path):
    """
    Combines all masks into one and applies it to the image.
    Saves the result to the output path.

    Parameters:
        image (PIL.Image.Image): The original image.
        masks (np.ndarray): Array of masks, each as a 2D binary NumPy array.
        output_path (str): Path to save the resulting masked image.
    """
    # Combine all masks using logical OR
    combined_mask = np.any(masks, axis=0).astype(np.uint8)

    # Apply the mask to the image
    image_array = np.asarray(image)
    masked_image_array = np.zeros_like(image_array)
    for channel in range(3):  # Assuming RGB
        masked_image_array[:, :, channel] = image_array[:, :, channel] * combined_mask

    # Convert back to PIL image and save
    masked_image = Image.fromarray(masked_image_array)
    masked_image.save(output_path)




def process_images(input_folder, output_folder):
    """
    Detects "oat plant" on every image in the `input_folder`,
    retrieves the masked section closest to the centroid of the image,
    and saves the result in the `output_folder`.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save masked images.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize the model
    model = LangSAM(sam_type="sam2.1_hiera_large")

    # Set thresholds
    box_threshold = 0.3
    text_threshold = 0.25
    text_prompt = ["large beige plywood wooden plane"]

    # Process each image
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            if not matches_any_condition(filename, conditions):
                print(f"⏭️ Skipping {filename} (does not match any condition)")
                continue
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load image
            image = Image.open(input_path)

            # Perform prediction
            results = model.predict(
                [image],
                text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # Extract the masks from the results
            masks = results[0]["masks"]

            if len(masks) > 0:
                retrieve_combined_mask(image, masks, output_path)
                print(f"Masked and saved: {output_path}")
            else:
                print(f"No 'oat plant' detected in {filename}. Skipping.")


def matches_any_condition(filename, conditions):
    return any(all(part in filename for part in condition) for condition in conditions)


if __name__ == "__main__":
    input_folder = "/media/edward/HDD/Docker-Workspace/Data/GH4-Completed/2024-08-29-Evening/2024-08-29-Evening_original_images"
    output_folder = "/media/edward/HDD/Docker-Workspace/Data/GH4-Completed/2024-08-29-Evening/2024-08-29-Evening_plywood"
    conditions = [
    ("RX180.0", "RZ0.0"),
    ("RX180.0", "RZ180.0")
    ]

    process_images(input_folder, output_folder)

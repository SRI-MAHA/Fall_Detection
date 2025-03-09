import os
import cv2


def preprocess_image(image_path, output_dir, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian filter to an image and save it to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    image = cv2.imread(image_path)

    if image is not None:
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, blurred_image)
        print(f"✅ Processed and saved: {output_path}")
    else:a
        print(f"❌ Warning: Could not read {image_path}. Skipping.")


def process_directory(input_dir, output_subdir):
    """
    Process all images in the given directory.
    """
    if not os.path.isdir(input_dir):
        print(f"❌ Directory does not exist: {input_dir}")
        return

    output_dir = os.path.join(os.path.dirname(input_dir), "Preprocessed", output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📊 Processing images in: {input_dir}")
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        if os.path.isfile(image_path):
            preprocess_image(image_path, output_dir)


def main():
    # Hardcoded paths for testing
    fall_dir = r"E:\Personal\Fall_Detection\archive\fall_dataset\Images\Fall"
    not_fall_dir = r"E:\Personal\Fall_Detection\archive\fall_dataset\Images\Not_Fall"

    print("Starting preprocessing...\n")

    # Process 'Fall' images
    process_directory(fall_dir, "Fall")

    # Process 'Not_Fall' images
    process_directory(not_fall_dir, "Not_Fall")

    print("\n✅ Preprocessing completed successfully!")
    print(f"All preprocessed images are saved in: E:\\Personal\\Fall_Detection\\archive\\Preprocessed")


if __name__ == "__main__":
    main()

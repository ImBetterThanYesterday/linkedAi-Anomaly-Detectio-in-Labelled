import json
import os
import cv2

# Load the annotation file
with open('./tools-with-anomalies.json', 'r') as f:
    data = json.load(f)

# Directory where the images are stored
image_directory = './tools-images-with-anomalies'

# Directory to save cropped images
output_dir = './cropped_objects'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each image entry in the JSON
for entry in data:
    image_filename = entry['image']
    image_path = os.path.join(image_directory, image_filename)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Extract the base name without extension
    base_name = os.path.splitext(image_filename)[0]

    # Iterate over each tagged object in the image
    for tag in entry['tags']:
        class_name = tag['name']
        x, y, w, h = int(tag['pos']['x']), int(tag['pos']['y']), int(tag['pos']['w']), int(tag['pos']['h'])

        # Create directory for the class if it doesn't exist
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Crop the object from the image
        cropped_object = image[y:y+h, x:x+w]

        # Create a filename for the cropped object
        object_filename = f"{base_name}.png"
        object_path = os.path.join(class_dir, object_filename)

        # Save the cropped object
        cv2.imwrite(object_path, cropped_object)

print("Objects have been cropped and saved.")

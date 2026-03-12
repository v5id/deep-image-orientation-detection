import os
from PIL import Image, ExifTags

# Path to your dataset
dataset_dir = "./data/upright_images"

# Optional: save corrected images to a separate folder
# save_dir = "./data/upright_images_corrected"
# os.makedirs(save_dir, exist_ok=True)

# Find EXIF orientation tag
orientation_tag = None
for tag, value in ExifTags.TAGS.items():
    if value == "Orientation":
        orientation_tag = tag
        break

if orientation_tag is None:
    raise RuntimeError("Cannot find EXIF Orientation tag")

# Process every image
for filename in os.listdir(dataset_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(dataset_dir, filename)
    try:
        img = Image.open(path)

        # Read EXIF
        exif = img._getexif()
        rotated = False

        if exif is not None:
            orientation = exif.get(orientation_tag, 1)

            if orientation == 3:
                img = img.rotate(180, expand=True)
                rotated = True
            elif orientation == 6:
                img = img.rotate(-90, expand=True)
                rotated = True
            elif orientation == 8:
                img = img.rotate(90, expand=True)
                rotated = True

        # Save back (overwrite original)
        img.save(path)

        if rotated:
            print(f"Corrected orientation for: {filename}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
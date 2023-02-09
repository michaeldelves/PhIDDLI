import os
from PIL import Image

# Set the input directory
input_dir = "data/extracted/"

# Iterate over all subdirectories in the input directory
for subdir, _, _ in os.walk(input_dir):
    # Iterate over all files in the subdirectory
    for file_name in os.listdir(subdir):
        # Check if the file is a JPEG image
        if file_name.endswith(".jpg"):
            # Open the image file
            image_file = os.path.join(subdir, file_name)
            image = Image.open(image_file)

            # Mirror the image
            image_mirrored = image.transpose(Image.FLIP_LEFT_RIGHT)

            
            # Rotate the image 90 degrees
            image90 = image.rotate(90)

            # Save the rotated image with a new file name
            new_file_name = os.path.splitext(file_name)[0] + "90.jpg"
            new_file_path = os.path.join(subdir, new_file_name)
            image90.save(new_file_path)

            # Rotate the image 180 degrees
            image180 = image.rotate(180)

            # Save the rotated image with a new file name
            new_file_name = os.path.splitext(file_name)[0] + "180.jpg"
            new_file_path = os.path.join(subdir, new_file_name)
            image180.save(new_file_path)

            # Rotate the mirrored image 270 degrees
            image270 = image.rotate(270)

            # Save the rotated image with a new file name
            new_file_name = os.path.splitext(file_name)[0] + "270.jpg"
            new_file_path = os.path.join(subdir, new_file_name)
            image270.save(new_file_path)



            # Save the mirrored image with a new file name
            new_file_name = os.path.splitext(file_name)[0] + "m.jpg"
            new_file_path = os.path.join(subdir, new_file_name)
            image_mirrored.save(new_file_path)
            
            # Rotate the mirrored image 90 degrees
            imagem90 = image_mirrored.rotate(90)

            # Save the rotated image with a new file name
            new_file_name = os.path.splitext(file_name)[0] + "90m.jpg"
            new_file_path = os.path.join(subdir, new_file_name)
            imagem90.save(new_file_path)

            # Rotate the mirrored image 180 degrees
            imagem180 = image_mirrored.rotate(180)

            # Save the rotated image with a new file name
            new_file_name = os.path.splitext(file_name)[0] + "180m.jpg"
            new_file_path = os.path.join(subdir, new_file_name)
            imagem180.save(new_file_path)

            # Rotate the mirrored image 270 degrees
            imagem270 = image_mirrored.rotate(270)

            # Save the rotated image with a new file name
            new_file_name = os.path.splitext(file_name)[0] + "270m.jpg"
            new_file_path = os.path.join(subdir, new_file_name)
            imagem270.save(new_file_path)

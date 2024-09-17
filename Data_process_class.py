from six.moves import urllib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random
import pandas as pd
import zipfile
import cv2
import os
import csv

# Download the class for extracting data
# This class is designed to streamline the preparation of image data for further processing and analysis.
class Process_data:
    # Define the path
    def __init__(self, download_path=None, save_path=None, full_path=None):
        # Initialize default paths for downloading and saving images, customizable via parameters.
        self.download_path = download_path if download_path else "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
        self.save_path = save_path if save_path else "image_datasets/image"
        self.full_path = full_path if full_path else "image_datasets/image/GTSRB/Final_Training/Images"

    # File download and decompression
    def download_and_unzipped_file(self, file_name):
        # Download an image file and unzip it in the specified directory. Checks for existing data to avoid redundancy.
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        # Combine the download path and the archive save path
        image_url = os.path.join(self.download_path, file_name)
        zip_path = os.path.join(self.save_path, file_name)
        # If this folder is not detected, the folder will be downloaded and decompressed
        if not os.path.isdir(self.full_path):
            # Download the progress bar
            reporthook = self.creat_reporthook(file_name)
            urllib.request.urlretrieve(image_url, zip_path, reporthook=reporthook)
            print(" Download finished.")
            print("Start unzipping")
            # Unzip the zip file
            with zipfile.ZipFile(zip_path, "r") as image_zip:
                image_zip.extractall(path=self.save_path)
            print("Unzipping is complete")
            # Delete the zip file
            os.remove(zip_path)
        # If a folder exists, the download and decompression phase is skipped
        else:
            print("The dataset file is detected and the download is skipped")

    @staticmethod
    # Download progress
    def creat_reporthook(file_name):
        # Generates a callback to report the download progress of a file in percentage.
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r{file_name} Downloading: {percent}% complete", end="")

        return reporthook

    @staticmethod
    # Read the CSV file
    def read_csvfile(path):
        annotations = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    with open(csv_path, 'r') as csvfile:
                        reader = csv.DictReader(csvfile, delimiter=';')
                        for row in reader:
                            annotations.append(row)
        return annotations

    # Get a dataset with picture paths, points of interest, categories
    def get_processed_data(self, path):
        # Read annotations from CSV files located at the specified path
        annotations = self.read_csvfile(path)
        image_path = [] # Store the full paths to the images
        image_roi = [] # Store regions of interest from the images
        image_height = []# Store heights of the images
        image_width = []# Store widths of the images

        # Process each record from the CSV data
        for csv_data in annotations:
            folder_path = f"{int(csv_data['ClassId']):05d}" # Format the class ID to match folder naming
            # Create a tuple for the region of interest coordinates
            roi = (int(csv_data['Roi.X1']), int(csv_data['Roi.Y1']),
                   int(csv_data['Roi.X2']), int(csv_data['Roi.Y2']))
            height = int(csv_data['Height'])
            width = int(csv_data['Width'])

            # Compile full path to each image and collect all other data
            image_roi.append(roi)
            image_full_path = os.path.join(path, folder_path, csv_data['Filename'])
            image_path.append(image_full_path)
            image_height.append(height)
            image_width.append(width)

        # Create a DataFrame to organize and return the collected image data
        df_images = pd.DataFrame({
            'Image_Path': image_path,
            'Image_roi': image_roi,
            'Height': image_height,
            'Width': image_width,
            'ClassId': [a['ClassId'] for a in annotations]
        })
        return df_images


# Classes for image processing
class Image_preprocessing:
    def __init__(self):
        # Initialize the training set image generator
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest',
        )
        # Initialize the image generator for development and test sets
        self.dev_test_datagen = ImageDataGenerator(rescale=1. / 255)

    def train_data_process(self, dataset):
        # Load the training image
        generator = self.train_datagen.flow_from_dataframe(
            dataframe=dataset,
            x_col='Image_Path',
            y_col='ClassId',
            target_size=(64, 64),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=True
        )
        return generator

    def test_data_process(self, dataset):
        # Load tests as well as develop images
        generator = self.dev_test_datagen.flow_from_dataframe(
            dataframe=dataset,
            x_col='Image_Path',
            y_col='ClassId',
            target_size=(64, 64),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=False
        )
        return generator

    # Point of interest clipping
    def crop_image(self, data_set):
        print("Points of interest are being clipped")
        crop_path = [] # List to store paths of cropped images
        crop_class = []# List to store class IDs corresponding to each cropped image

        # Iterate through each row of the dataset containing image paths and ROI
        for row in data_set.itertuples(index=False):
            image_path = row.Image_Path # Path to the image
            image_class = row.ClassId # Class ID of the image
            roi = row.Image_roi # Region of interest (coordinates) for cropping
            height = row.Height  # Height of the image
            width = row.Width  # Width of the image

            # Crop the image based on ROI and save it back to the same path
            self.crop_and_save_image(image_path, height, width, roi, image_path)
            crop_path.append(image_path)# Store the path of the cropped image
            crop_class.append(image_class) # Store the class ID of the cropped image

        # Create a DataFrame to organize cropped image paths and their class IDs
        crop_df = pd.DataFrame({
            'Image_Path': crop_path,
            'ClassId': crop_class
        })
        print("Points of interest are cropped")
        return crop_df# Return the DataFrame containing paths and class IDs of the cropped images

    # Trim the points of interest and save the file
    @staticmethod
    def crop_and_save_image(image_path, height, width, roi, save_path):
        # Open the image file from the specified path
        image = Image.open(image_path)
        # Get the actual dimensions of the image
        image_width, image_height = image.size
        # Check if the actual dimensions match the expected dimensions
        if image_width == width and image_height == height:
            # Crop the image based on the region of interest (ROI) coordinates
            cropped_image = image.crop(roi)
            # Save the cropped image to the specified path
            cropped_image.save(save_path)

    @staticmethod
    def show_image(image, class_id):
        # Display an image with matplotlib, annotating it with the corresponding class ID
        plt.imshow(image) # Display the image
        plt.title(f'Class ID: {class_id}') # Set the title of the plot to the class ID
        plt.axis('off') # Turn off the axis
        plt.show()# Show the image plot


class Data_analysis:
    # Adjustable parameters
    def __init__(self):
        self.target_size = (30, 30)  # Size of the image
        self.blur_radius = 2  # Gaussian smoothing radius
        self.contrast_factor = 2  # Contrast intensity
        self.rotate = 15  # Random rotation angle
        self.translate_rate = 0.1  # Random translation rate

    @staticmethod
    def read_csvfile(path):
        # Initialize a list to hold all parsed CSV data entries
        annotations = []

        # os.walk is used to walk through directory trees, including subdirectories
        for root, _, files in os.walk(path):
            # Iterate through each file in directories
            for file in files:
                # Check if the file is a CSV by its extension
                if file.endswith('.csv'):
                    # Join the directory path and file name to form a full path
                    csv_path = os.path.join(root, file)
                    # Open the CSV file for reading
                    with open(csv_path, 'r') as csvfile:
                        # Create a CSV DictReader to automatically map information in each row to a dictionary with headers as keys
                        reader = csv.DictReader(csvfile, delimiter=';')
                        # Iterate over each row in the CSV file and append it to the annotations list
                        for row in reader:
                            annotations.append(row)

        # Return the list of annotations read from all CSV files found
        return annotations

    def get_processed_data(self, path):
        # Read CSV files in the given directory to extract image metadata
        annotations = self.read_csvfile(path)
        image_path = []# List to hold paths to individual images
        image_roi = []# List to hold region of interest tuples (x1, y1, x2, y2)
        image_height = [] # List to hold the height of each image
        image_width = [] # List to hold the width of each image

        # Process each entry in the CSV data
        for csv_data in annotations:
            folder_path = f"{int(csv_data['ClassId']):05d}" # Format class ID into a folder path
            # Create a tuple representing the ROI from CSV data points
            roi = (int(csv_data['Roi.X1']), int(csv_data['Roi.Y1']),
                   int(csv_data['Roi.X2']), int(csv_data['Roi.Y2']))
            height = csv_data['Height']# Image height from CSV
            width = csv_data['Width']# Image width from CSV

            # Append the ROI to the list
            image_roi.append(roi)
            # Build the full path for the image and append it to the list
            image_full_path = os.path.join(path, folder_path, csv_data['Filename'])
            image_path.append(image_full_path)
            # Append the image height and width to their respective lists
            image_height.append(int(height))
            image_width.append(int(width))

        # Create a DataFrame to organize the data extracted above
        df_images = pd.DataFrame({
            'Image_Path': image_path,# Path to each image
            'Image_roi': image_roi,# Regions of interest of each image
            'ClassId': [a['ClassId'] for a in annotations],# List of class IDs for each image
            'Width': image_width,# Width of each image
            'Height': image_height# Height of each image
        })
        return df_images

    # Gaussian blur
    def gaussian_blur(self, image):
        return image.filter(ImageFilter.GaussianBlur(self.blur_radius))

        # Contrast enhancement

    def enhance_contrast(self, image):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(self.contrast_factor)

    # Resizing images
    def resize_image(self, image):
        return image.resize(self.target_size)

    # Data augmentation
    def augment_data(self, image):
        img_array = np.array(image)
        rows, cols = self.target_size
        # Setting random and numpy's random seeds
        random.seed(42)
        np.random.seed(42)

        # Random rotate (default from -15 to 15 degrees)
        angle = random.randint(-self.rotate, self.rotate)
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rotated = cv2.warpAffine(img_array, rotation_matrix, self.target_size)

        # Random transform (default translation within 10% of width and height)
        tx = np.random.randint(-int(cols * self.translate_rate), int(cols * self.translate_rate))
        ty = np.random.randint(-int(rows * self.translate_rate), int(rows * self.translate_rate))
        m_translate = np.float32([[1, 0, tx], [0, 1, ty]])
        img_translated = cv2.warpAffine(img_rotated, m_translate, self.target_size)

        # Normalisation
        img_normalised = img_translated / 255.0

        return img_normalised

    # Entry function
    def process_image(self, image_path, annotation_roi):
        with Image.open(image_path) as img:
            # Image Processing Pipeline
            # 1 Conversion to greyscale
            # img = self.greyscale(img)
            # 2 ROI cutting (Assuming annotation_roi is a dict with keys 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2')
            img_cropped = img.crop(annotation_roi)
            # 3 Apply Gaussian filter
            img_smoothed = self.gaussian_blur(img_cropped)
            # 4 Image enhancement
            img_enhanced = self.enhance_contrast(img_smoothed)
            # 5 Resizing the cropped image
            img_resized = self.resize_image(img_enhanced)
            # 6 Data augmentation
            img_augmented = self.augment_data(img_resized)

            return img_augmented

    @staticmethod
    # Three-channel intensity
    def analyze_image_color(image_path):
        # Open the image from the provided path and convert it into a numpy array
        img = Image.open(image_path)
        img_array = np.array(img)

        # Define the list of color channels to analyze
        channels = ['red', 'green', 'blue']
        stats = {} # Dictionary to store statistics for each color channel

        # Loop through each color channel to compute statistics
        for i, color in enumerate(channels):
            # Extract the array for the current color channel
            channel = img_array[:, :, i]
            # Calculate and store statistics for the current color channel
            stats[color] = {
                'average_value': np.mean(channel),# Compute the average pixel intensity
                'standard_deviation': np.std(channel),# Compute the standard deviation of pixel intensity
                'minimum': np.min(channel), # Find the minimum pixel intensity
                'maximum': np.max(channel) # Find the maximum pixel intensity
            }
        return stats # Return the dictionary containing the stats for each channel

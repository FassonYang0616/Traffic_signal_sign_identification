from Data_process_class import Process_data, Image_preprocessing, Data_analysis
import matplotlib
from PIL import Image
import seaborn as sns

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Get a dataset for analysis
    process_data = Process_data(None, "image_datasets/image/analysis",
                                "image_datasets/image/analysis/GTSRB/Final_Training/Images")
    analysis_data = Data_analysis()
    # Download and unzip
    process_data.download_and_unzipped_file("GTSRB_Final_Training_Images.zip")
    # Processed into five columns of DF
    analysis_data_df = analysis_data.get_processed_data("image_datasets/image/analysis/GTSRB/Final_Training/Images")
    # Display data

    row_data = analysis_data_df.iloc[1]
    print(row_data)

    # Analyze and visualize the distribution of class IDs
    label_count = analysis_data_df["ClassId"].value_counts()
    # Count occurrences of each unique image height in the dataset
    height_count = analysis_data_df["Height"].value_counts()
    # Count occurrences of each unique image width in the dataset
    width_count = analysis_data_df["Width"].value_counts()
    # Convert the index of label counts to a list (for use in plotting or further analysis)
    label = label_count.index.tolist()
    label = label_count.index.tolist()
    # Convert the values of label counts to a list (for use in plotting or further analysis)
    value = label_count.values.tolist()

    # Print maximum, minimum, average, and median values of image heights
    print("Height max:", analysis_data_df['Height'].max())
    print("Height min:", analysis_data_df['Height'].min())
    print("Height average:", analysis_data_df['Height'].mean())
    print("Height median", analysis_data_df['Height'].median())

    # Print maximum, minimum, average, and median values of image widths
    print("Width max:", analysis_data_df['Width'].max())
    print("Width min:", analysis_data_df['Width'].min())
    print("Width average:", analysis_data_df['Width'].mean())
    print("Width median", analysis_data_df['Width'].median())

    # The distribution of class_id
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(value, labels=label, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.tight_layout()
    plt.title('Dataset label distribution')
    plt.tight_layout()
    plt.show()

    # Picture high distribution
    plt.figure(figsize=(13, 10))
    plt.bar(height_count.index, height_count.values, color='skyblue')
    plt.title('Image high distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.show()

    # Wide distribution of images
    plt.figure(figsize=(13, 10))
    plt.bar(width_count.index, width_count.values, color='skyblue')
    plt.title('Image width distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.show()

    # Calculate the value of the tricolor channel
    analysis_data_df['ImageStats'] = analysis_data_df['Image_Path'].apply(
        lambda x: analysis_data.analyze_image_color(x))
    stats_df = pd.json_normalize(analysis_data_df['ImageStats'])

    # Mean histogram
    plt.figure(figsize=(12, 6))
    for color in ['red', 'green', 'blue']:
        sns.histplot(stats_df[f'{color}.average_value'], bins=10, label=f'{color} average', alpha=0.5, kde=True)
    plt.title('Color Channel Average Values')
    plt.xlabel('Average Pixel Value')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    # Box plot showing mean, variance, maximum and minimum values
    plt.figure(figsize=(12, 6))
    color_stats = pd.DataFrame()
    for color in ['red', 'green', 'blue']:
        for stat in ['average_value', 'standard_deviation', 'minimum', 'maximum']:
            temp_df = pd.DataFrame({
                'Value': stats_df[f'{color}.{stat}'],
                'Statistic': [stat] * len(analysis_data_df),
                'Color': [color] * len(analysis_data_df)
            })
            color_stats = pd.concat([color_stats, temp_df], ignore_index=True)

    sns.boxplot(x='Color', y='Value', hue='Statistic', data=color_stats)
    plt.title('Statistics of Color Channels')
    plt.show()

    # Open three sample images from specified paths in the dataset
    img1 = Image.open(analysis_data_df['Image_Path'][1])
    img2 = Image.open(analysis_data_df['Image_Path'][1000])
    img3 = Image.open(analysis_data_df['Image_Path'][2000])

    # Process the same three images using custom image processing functions to enhance or alter features
    img1_process = analysis_data.process_image(analysis_data_df['Image_Path'][1],
                                               analysis_data_df['Image_roi'][1])
    img2_process = analysis_data.process_image(analysis_data_df['Image_Path'][1000],
                                               analysis_data_df['Image_roi'][1000])
    img3_process = analysis_data.process_image(analysis_data_df['Image_Path'][2000],
                                               analysis_data_df['Image_roi'][2000])

    # Setup a figure with a 2x3 subplot layout to display images in two rows
    fig, axes = plt.subplots(2, 3)

    # Display original images in the first row of the subplot
    axes[0, 0].imshow(img1)
    axes[0, 0].axis('off')# Turn off axis labels and ticks for a cleaner look

    axes[0, 1].imshow(img2)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(img3)
    axes[0, 2].axis('off')

    # Display processed images in the second row of the subplot
    axes[1, 0].imshow(img1_process)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img2_process)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(img3_process)
    axes[1, 2].axis('off')

    # Add descriptive text at the top and middle of the figure
    fig.text(0.5, 0.9, 'Original Images', ha='center', va='center', fontsize=12)
    fig.text(0.5, 0.47, 'Processed Images', ha='center', va='center', fontsize=12)

    plt.show()# Display the figure with the images


if __name__ == "__main__":
    main()

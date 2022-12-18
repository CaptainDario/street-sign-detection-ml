import sys
from os import path, listdir

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import utils


def annotate_sample(
    sample_path: str, 
    annotation_path: str, 
    output_dir: str
) -> path.abspath:
    """ Loads a sample from `sample_path` and an annotation from `annotation_path`.
        Draws the BBoxs from the annotations on the image and saves the image too.
        Returns the path to the annotated file.
    """
    with open(annotation_path, encoding="utf8", mode="r") as f:
        bboxs = [[float(__a) for __a in _a.split(" ")] for _a in f.read().split("\n")]

    im = Image.open(sample_path)
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(20, 20))
    # Display the image
    ax.imshow(im)
    # label, X, Y, Width, and Height
    for b in bboxs:
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (b[1] * im.width - b[3] * im.width / 2, b[2] * im.height - b[4] * im.height / 2), 
            b[3] * im.width,
            b[4] * im.height,
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    file_name = path.basename(sample_path)
    file_name = file_name.replace(".ppm", ".png")
    output_path = path.join(
        output_dir,
        f'{file_name}',
    )
    plt.savefig(output_path)
    return output_path

def annotate_all_samples(
    input_dir_samples: str,
    input_dir_coordinates: str,
    output_dir
):
    ''' annotates all samples from an `input directory` with coordinates from a `coordinates directory`.
        Expects sample files of type '.ppm' and coordinates of type '.txt'
    '''
    coordinate_elements = listdir(input_dir_coordinates)
    for index in range(len(coordinate_elements)):
        utils.print_progress_bar(
            iteration= index+1,
            total= len(coordinate_elements),
            prefix= '\tAnnotated Files:',
            suffix= 'Complete',
            length= 30
        )
        coordinate_path = path.join(
            input_dir_coordinates,
            coordinate_elements[index]
        )
        sample_path = path.join(
            input_dir_samples,
            coordinate_elements[index].replace('.txt', '.ppm')
        )
        annotate_sample(
            sample_path,
            coordinate_path,
            output_dir
        )


def parse_args():
    if len(sys.argv) == 4:
        return sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        print("Usage: python dataset_annotation.py <input_directory_samples> <input_directory_coordinates> <output_directory>")
        exit(1)


if __name__ == '__main__':
    input_dir_samples, input_dir_coordinates, output_dir = parse_args()
    annotate_all_samples(
        input_dir_samples,
        input_dir_coordinates,
        output_dir
    )
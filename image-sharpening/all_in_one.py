import argparse, os, cv2, traceback
import numpy as np

from time import time

from images.images import (
    loadImages,
    save_image,
)

from sharpen.sharpen import sharpen


def setup_args():
    parser = argparse.ArgumentParser(description="Process Image")
    parser.add_argument("input_dir", help="Input directory of images")
    parser.add_argument(
        "--internal_image_extension",
        default="png",
        help="Extension of images to process",
    )
    parser.add_argument(
        "--sharpen",
        help="Sharpen the postprocess image",
        choices=["filter_kernel", "unsharp_mask"],
        type=str,
    )
    parser.add_argument(
        "--sharpen_amount",
        type=float,
        default=1.0,
        help="Sharpen amount for unsharp_mask",
    )
    return parser.parse_args()


# Create main and do any processing if needed
def single_image(
    image,
    input_dir,
    image_extension="png",
    quality=95,
    sharpen_method=None,
    sharpen_amount=1.0,
):
    if sharpen_method:
        image = sharpen(image, sharpen_method, sharpen_amount)

    output_image = os.path.join(input_dir, f"main.{image_extension}")

    save_image(output_image, image, image_extension, quality)
    print(f"Saved {output_image}")

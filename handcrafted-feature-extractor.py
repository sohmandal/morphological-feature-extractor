import sys
import os
import pandas as pd
from PIL import Image
import numpy as np
from skimage.measure import regionprops_table
from scipy import ndimage
import centrosome.cpmorphology
import centrosome.zernike


def base(mask_array):
    desired_properties = [
        "label",
        "image",
        "area",
        "perimeter",
        "bbox",
        "bbox_area",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "centroid",
        "equivalent_diameter",
        "extent",
        "eccentricity",
        "convex_area",
        "solidity",
        "euler_number",
        "inertia_tensor",
        "inertia_tensor_eigvals",
        "moments",
        "moments_central",
        "moments_hu",
        "moments_normalized",
    ]

    props = regionprops_table(mask_array, properties=desired_properties)
    return props


def get_base(props):
    props_df = pd.DataFrame(props)
    props_df = props_df.drop(columns=["label", "image"])
    return props_df


def get_formfactor(props):
    formfactor = 4.0 * np.pi * props["area"] / props["perimeter"] ** 2
    formfactor_df = pd.DataFrame(formfactor, columns=["formfactor"])
    return formfactor_df


def get_compactness(props):
    denom = [max(x, 1) for x in 4.0 * np.pi * props["area"]]
    compactness = props["perimeter"] ** 2 / denom
    compactness_df = pd.DataFrame(compactness, columns=["compactness"])
    return compactness_df


def get_radius(props, nobjects):
    max_radius = np.zeros(nobjects)
    median_radius = np.zeros(nobjects)
    mean_radius = np.zeros(nobjects)

    for index, mini_image in enumerate(props["image"]):
        mini_image = np.pad(mini_image, 1)
        distances = ndimage.distance_transform_edt(mini_image)
        max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            ndimage.maximum(distances, mini_image)
        )
        mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            ndimage.mean(distances, mini_image)
        )
        median_radius[index] = centrosome.cpmorphology.median_of_labels(
            distances, mini_image.astype("int"), [1]
        )

    max_radius_df = pd.DataFrame(max_radius, columns=["max_radius"])
    mean_radius_df = pd.DataFrame(mean_radius, columns=["mean_radius"])
    median_radius_df = pd.DataFrame(median_radius, columns=["median_radius"])
    return max_radius_df, mean_radius_df, median_radius_df


def get_zernike_numbers(ZERNIKE_N=9):
    return centrosome.zernike.get_zernike_indexes(ZERNIKE_N + 1)


def get_zernike_name(zernike_index):
    return "Zernike_%d_%d" % (zernike_index[0], zernike_index[1])


def get_zernike(mask_array, indices):
    zernike_numbers = get_zernike_numbers()

    zf_val = centrosome.zernike.zernike(zernike_numbers, mask_array, indices)

    zf_idx = []
    for n, m in zernike_numbers:
        zf_idx.append(get_zernike_name((n, m)))

    zernike_df = pd.DataFrame(zf_val, columns=zf_idx)
    return zernike_df


def get_feret_diameter(mask_array, indices, nobjects):
    min_feret_diameter = np.zeros(nobjects)
    max_feret_diameter = np.zeros(nobjects)

    idx_feret = np.nonzero(mask_array)
    values = mask_array[idx_feret]
    ijv_feret = np.column_stack((idx_feret[0], idx_feret[1], values))

    chulls, chull_counts = centrosome.cpmorphology.convex_hull_ijv(ijv_feret, indices)

    min_feret_diameter, max_feret_diameter = centrosome.cpmorphology.feret_diameter(
        chulls, chull_counts, indices
    )

    min_feret_diameter_df = pd.DataFrame(
        min_feret_diameter, columns=["min_feret_diameter"]
    )
    max_feret_diameter_df = pd.DataFrame(
        max_feret_diameter, columns=["max_feret_diameter"]
    )
    return min_feret_diameter_df, max_feret_diameter_df


def mask_to_handcrafted_features(mask_path, output_dir):
    """
    Extracts a set of handcrafted morphological features from a given mask and saves them to a csv file. 
    Reuses features from the measureobjectsizeshape module from the CellProfiler(https://github.com/CellProfiler/CellProfiler).

    Args:
        mask_path (str): The path to the mask file.
        output_dir (str): The path to the output directory where the .csv containing the features will be saved.

    Returns:
        None
    """

    try:
        mask = Image.open(mask_path)
    except FileNotFoundError:
        print("Mask not found!")
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        sys.exit(1)

    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    max_label = np.max(mask_array)
    indices = np.arange(max_label).astype(np.int32) + 1
    nobjects = len(indices)

    props = base(mask_array)

    props_df = get_base(props)
    formfactor_df = get_formfactor(props)
    compactness_df = get_compactness(props)
    max_radius_df, mean_radius_df, median_radius_df = get_radius(props, nobjects)
    zernike_df = get_zernike(mask_array, indices)
    min_feret_diameter_df, max_feret_diameter_df = get_feret_diameter(
        mask_array, indices, nobjects
    )

    df_all = pd.concat(
        [
            props_df,
            formfactor_df,
            compactness_df,
            max_radius_df,
            mean_radius_df,
            median_radius_df,
            zernike_df,
            min_feret_diameter_df,
            max_feret_diameter_df,
        ],
        axis=1,
    )

    if df_all.isnull().values.any():
        df_all = df_all.dropna(axis=1, how="all")

    output_file = os.path.join(
        output_dir, os.path.splitext(os.path.basename(mask_path))[0] + ".csv"
    )
    df_all.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Argument missing!")
        sys.exit(1)
    mask_path = sys.argv[1]
    output_dir = sys.argv[2]

    mask_to_handcrafted_features(mask_path, output_dir)

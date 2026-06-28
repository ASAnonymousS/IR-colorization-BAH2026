import argparse
import os
import shutil
import subprocess

from utils.file_utils import find_file
from utils.logging_utils import setup_logging


def run_script(script_name, logger, *args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, "scripts")
    script_path = os.path.join(scripts_dir, script_name)
    command = ["python", script_path] + list(args)
    logger.info(f"Running: {' '.join(command)}")
    try:
        # Add the project root to PYTHONPATH so scripts can import from utils
        env = os.environ.copy()
        env["PYTHONPATH"] = base_dir + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, env=env
        )
        if result.stdout:
            logger.info(f"STDOUT from {script_name}:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR from {script_name}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="IR-Colorization Dataset Generation Baseline"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(base_dir, "dataset")
    input_root = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")

    output_downscale_dir = os.path.join(output_dir, "downscaled_data")
    output_rgb_dir = os.path.join(output_dir, "rgb_images")
    output_patches_dir = os.path.join(output_dir, "patches")

    for d in [output_downscale_dir, output_rgb_dir, output_patches_dir]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logging(output_dir)

    if not os.path.isdir(dataset_root):
        logger.error(f"Dataset root directory {dataset_root} not found")
        exit(1)

    if not os.path.isdir(input_root):
        logger.error(f"Input root directory {input_root} not found.")
        exit(1)

    datasets = [
        dataset
        for dataset in os.listdir(dataset_root)
        if os.path.isfile(os.path.join(dataset_root, dataset))
    ]

    for dataset in datasets:
        unique_id, extension = dataset.split(".")
        if extension.lower() != "tif":
            logger.error(f"Dataset {dataset} is not a TIF file")
            continue
        (
            project_and_satelite_details,
            processing_level,
            wrs_2_grid,
            acquisition_date,
            processing_date,
            collection_number,
            collection_category,
            product_type,
            band,
        ) = unique_id.split("_")
        if project_and_satelite_details not in ["LC08", "LC09"]:
            logger.error(f"Dataset {dataset} doesn't belong to Landsat 8/9")
            continue
        if processing_level != "L2SP":
            logger.error(
                f"Dataset {dataset} is not surface level product it cannot be used for training purposes"
            )
            continue

        match (collection_number):
            case "01":
                choice = input(
                    f"The dataset {dataset} is too old are you sure you want to use this to train your model ? Y/N"
                )
                if choice.lower() != "y":
                    logger.error(
                        f"Dataset {dataset} is too old and user aborted the dataset generation"
                    )
                    continue
            case "02":
                pass
            case _:
                logger.error(
                    f"Dataset collection_number {collection_number} is invalid"
                )
                continue
        match (collection_category):
            case "T1":
                pass
            case "T2":
                choice = input(
                    f"""The dataset collection_category {collection_category} is not of the best quality, it stands in acceptable zone.
                    Model might not perform at it's best efficiency
                    Do you want to use this dataset for training purposes? Y/N"""
                )
                if choice.lower() != "y":
                    logger.error(
                        f"Dataset {dataset} is Tier 2 (lower geometric quality) and user aborted processing."
                    )
                    continue
            case "RT":
                choice = input(
                    f"\n[WARNING] The dataset {dataset} is Real-Time (RT) data.\n"
                    f"It has not undergone final geometric calibration. Pixels may be misaligned.\n"
                    f"Do you want to use this dataset anyway? Y/N: "
                )
                if choice.lower() != "y":
                    logger.error(
                        f"Dataset {dataset} is uncalibrated Real-Time data and user aborted processing."
                    )
                    continue
        match (product_type):
            case "SR":
                if band not in ["B2", "B3", "B4"]:
                    logger.error(
                        f"\nThis project only works with bands B2, B3 and B4 as of now\n"
                        f"Dataset {dataset} doesn't belong to given bands"
                    )
                    continue
            case "ST":
                if band != "B10":
                    logger.error(
                        f"\nThis project only works with band B10 as of now\n"
                        f"Dataset {dataset} doesn't belong to given band"
                    )
                    continue
        dataset_dir = os.path.join(
            input_root,
            f"{project_and_satelite_details}_{wrs_2_grid}_{acquisition_date}_{collection_number}_{collection_category}",
        )
        os.makedirs(dataset_dir, exist_ok=True)
        try:
            # Choose move or copy2 based on your dataset workflow
            shutil.move(os.path.join(dataset_root, dataset), dataset_dir)
            logger.info(f"Successfully organized {dataset} -> {dataset_dir}")
        except Exception as e:
            logger.error(f"Failed to move file {dataset}: {e}")

    product_folders = [
        e for e in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, e))
    ]

    for product_id in product_folders:
        input_dir = os.path.join(input_root, product_id)
        logger.info(f"Processing product: {product_id}")

        band2_path = find_file(input_dir, "_B2")
        band3_path = find_file(input_dir, "_B3")
        band4_path = find_file(input_dir, "_B4")
        band10_path = find_file(input_dir, "_B10")

        if not all([band2_path, band3_path, band4_path, band10_path]):
            logger.warning(f"Skipping {product_id}: Missing required bands.")
            continue

        file_prefix = product_id

        try:
            # 1. Merge RGB (30m)
            rgb_output_path = os.path.join(output_rgb_dir, f"{file_prefix}_rgb_30m.tif")
            if not os.path.isfile(rgb_output_path):
                run_script(
                    "merge_rgb.py",
                    logger,
                    band4_path,
                    band3_path,
                    band2_path,
                    rgb_output_path,
                )
            else:
                logger.info(
                    f"Skipping {file_prefix}: It already has RGB image of 30m TIF"
                )

            # 2. Downscale RGB to 100m (3.33x)
            downscaled_rgb_100m = os.path.join(
                output_downscale_dir, f"{file_prefix}_rgb_100m.tif"
            )
            if not os.path.isfile(downscaled_rgb_100m):
                run_script(
                    "downscale.py", logger, rgb_output_path, downscaled_rgb_100m, "3.33"
                )
            else:
                logger.info(
                    f"Skipping {file_prefix}: It already has downscaled RGB image of 100m TIF"
                )

            # 3. Downscale TIR to 100m (3.33x)
            downscaled_tir_100m = os.path.join(
                output_downscale_dir, f"{file_prefix}_tir_100m.tif"
            )
            if not os.path.isfile(downscaled_tir_100m):
                run_script(
                    "downscale.py", logger, band10_path, downscaled_tir_100m, "3.33"
                )
            else:
                logger.info(
                    f"Skipping {file_prefix}: It already has downscaled TIR image of 100m TIF"
                )

            # 4. Downscale TIR to 200m (6.67x)
            downscaled_tir_200m = os.path.join(
                output_downscale_dir, f"{file_prefix}_tir_200m.tif"
            )
            if not os.path.isfile(downscaled_tir_200m):
                run_script(
                    "downscale.py", logger, band10_path, downscaled_tir_200m, "6.67"
                )
            else:
                logger.info(
                    f"Skipping {file_prefix}: It already has downscaled TIR image of 200m TIF"
                )

            # 5. Create Coregistered Patches
            run_script(
                "create_patches.py",
                logger,
                "--input_dir",
                output_downscale_dir,
                "--output_dir",
                output_patches_dir,
            )

            logger.info(f"Successfully generated dataset samples for {product_id}")

        except Exception as e:
            logger.error(f"Error processing {product_id}: {e}")

    logger.info("Dataset generation finished. Samples available in output/patches")


if __name__ == "__main__":
    main()

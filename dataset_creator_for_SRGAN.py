import argparse
import os

import numpy as np


def output_root(args):
    output_npz_root = os.path.join(os.path.abspath(args.output_root), "npz_files")
    os.makedirs(output_npz_root, exist_ok=True)
    output_patch_root = os.path.join(os.path.abspath(args.output_root), "patches")
    if not os.path.isdir(output_patch_root):
        print(f"No such {output_patch_root} directory exist")
        exit(1)

    output_patches = [
        os.path.join(output_patch_root, output_patch)
        for output_patch in os.listdir(output_patch_root)
    ]
    for output_patch in output_patches:
        scene(output_patch, output_npz_root)


def scene(single_output_patch, output_npz_save):
    subdirectories = [
        os.path.join(single_output_patch, subdirectory)
        for subdirectory in os.listdir(single_output_patch)
    ]
    for subdirectory in subdirectories:
        npydict = {}
        files = os.listdir(subdirectory)
        for f in files:
            filename, extension = f.split(".")
            if extension == "npy" and "rgb" not in filename.lower():
                value = np.load(os.path.join(subdirectory, f))
                if "tir_200m" in filename:
                    npydict["lr"] = value
                elif "tir_100m_512" in filename:
                    npydict["hr"] = value
        output_npydict = ".".join(
            [
                os.sep.join(
                    [output_npz_save, "_".join(subdirectory.split(os.sep)[-2:])]
                ),
                "npz",
            ]
        )
        np.savez_compressed(output_npydict, **npydict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merges all the different .npy files and creates a .npz archive for further processing by SRGAN"
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Path to the root directory of all output patches",
    )
    args = parser.parse_args()
    try:
        output_root(args)
    except Exception as e:
        print(e)
        exit(1)

from glob import glob
import numpy as np
import argparse


def get_ua():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input-dir", dest="input_dir", type=str,
    )
    ap.add_argument(
        "-o", "--output-file", dest="output_file", type=str,
    )
    uap = ap.parse_args()
    return uap


def main():
    ua = get_ua()

    res_dict = {
        "dataset_ids": [],
        "sample_idxs": [],
        "x_train_all": [],
        "y_train_all": [],
        "pos_train_all": [],
        "idx_train_all": [],
        "x_train_arr": [],
        "y_train_arr": []
    }

    for f in glob(f"{ua.input_dir}/train*.npz"):
        zdata = np.load(f)

        dataset_id = f.split("train")[-1].split("_")[0]
        sample_id = f.split("train")[-1].split("_")[1].split(".")[0]
        res_dict["dataset_ids"].append(dataset_id)
        res_dict["sample_idxs"].append(sample_id)

        res_dict["x_train_all"].append(zdata["x_train"])
        res_dict["y_train_all"].append(zdata["y_train"])
        res_dict["pos_train_all"].append(zdata["pos_train"])
        res_dict["idx_train_all"].append(zdata["idx_train"])
    
    res_dict["x_train_arr"] = np.stack(res_dict["x_train_all"])
    res_dict["y_train_arr"] = np.stack(res_dict["y_train_all"])

    del res_dict["x_train_all"]
    del res_dict["y_train_all"]

    np.savez_compressed(ua.output_file, **res_dict)


if __name__ == "__main__":
    main()

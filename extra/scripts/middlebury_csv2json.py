import csv
import glob
import json
import numpy as np

output_dict = {}
samples = []

input_files = sorted([x for x in glob.glob("midd_eval/*.csv")])

with open(input_files[0], "r") as fp:
    reader = csv.DictReader(fp, delimiter=" ", skipinitialspace=True)
    for row in reader:
        samples.append(row["sample"])
        output_dict[row["sample"]] = {
            "mae": float(row["avgErr"]),
            "mask": float(row["mask"]),
            "invalid": float(row["invalid"]),
            "bad_pixel_rate": {},
        }

for file in input_files:
    with open(file, "r") as fp:
        ths = file.split("/")[-1].split(".csv")[0]
        reader = csv.DictReader(fp, delimiter=" ", skipinitialspace=True)
        for row in reader:
            output_dict[row["sample"]]["bad_pixel_rate"][ths] = float(row["totbad"])

output_dict["average"] = {
    "mae": np.average([output_dict[sample]["mae"] for sample in samples]),
    "mask": np.average([output_dict[sample]["mask"] for sample in samples]),
    "invalid": np.average([output_dict[sample]["invalid"] for sample in samples]),
    "bad_pixel_rate": {
        ths: np.average(
            [output_dict[sample]["bad_pixel_rate"][ths] for sample in samples]
        )
        for ths in output_dict[samples[0]]["bad_pixel_rate"].keys()
    },
}

with open("results.json", "w") as fp:
    json.dump(output_dict, fp, indent=2)

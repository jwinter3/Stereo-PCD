import glob
import json

avg_files = sorted([x for x in glob.glob("errors_noc_avg/*_10.txt")])
bad_px_files = sorted([x for x in glob.glob("errors_noc_out/*_10.txt")])

avg_stat_file = "stats_noc_avg.txt"
bad_px_file = "stats_noc_out.txt"

output_dict = {}

for file in avg_files:
    sample = file.split("/")[-1].split("_10.txt")[0]
    with open(file, "r", encoding="ascii") as fp:
        line = fp.readline().split()
    output_dict[sample] = {
        "interpolated": {"mae": float(line[0])},
        "original": {"mae": float(line[1])},
    }

for file in bad_px_files:
    sample = file.split("/")[-1].split("_10.txt")[0]
    with open(file, "r", encoding="ascii") as fp:
        line = fp.readline().split()
    output_dict[sample]["density"] = float(line[-1])
    output_dict[sample]["interpolated"]["bad_pixel_rate"] = {
        x + 1: line[2 * x] for x in range(5)
    }
    output_dict[sample]["original"]["bad_pixel_rate"] = {
        x + 1: line[2 * x + 1] for x in range(5)
    }

with open(avg_stat_file, "r", encoding="ascii") as fp:
    lines = fp.readlines()
    for line, stat in zip(lines, ["average", "minimum", "maximum"]):
        splited_line = line.split()
        output_dict[stat] = {
            "interpolated": {"mae": float(splited_line[0])},
            "original": {"mae": float(splited_line[1])},
        }

with open(bad_px_file, "r", encoding="ascii") as fp:
    lines = fp.readlines()
    for line, stat in zip(lines, ["average", "minimum", "maximum"]):
        splited_line = line.split()
        output_dict[stat]["density"] = float(splited_line[-1])
        output_dict[stat]["interpolated"]["bad_pixel_rate"] = {
            x + 1: splited_line[2 * x] for x in range(5)
        }
        output_dict[stat]["original"]["bad_pixel_rate"] = {
            x + 1: splited_line[2 * x + 1] for x in range(5)
        }

with open("results.json", "w") as fp:
    json.dump(output_dict, fp, indent=2)

import csv
import re

input_file = "data/data.txt"
output_file = "data/cleaned_data.csv"

with open(input_file, "r", encoding="latin-1") as infile, open(
    output_file, "w", newline="", encoding="utf-8"
) as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["label", "text"])

    for line in infile:
        parts = line.strip().split("\t", 1)
        # print(parts[0:5])
        if len(parts) == 2:
            label, text = parts
            label = re.findall(r"\b(ham|spam)\b", label)
            print(text.strip())
            writer.writerow([label[0], text.strip()])
        else:
            continue
print("Cleaned", output_file)

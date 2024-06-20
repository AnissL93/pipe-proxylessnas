#!/usr/bin/env python

import matplotlib.pyplot as plt
import re


def get_acc(text):
    match_top1 = re.search(r"top-1 acc (\d+\.\d+)", text)
    if match_top1:
        top1_acc = float(match_top1.group(1))
        return top1_acc


def read_gradient_file(f):
    all_acc = []
    with open(f) as fp:
        lines = fp.readlines()
        for line in lines:
            if "Warmup" in line:
                continue
            acc = get_acc(line)
            all_acc.append(acc)

    return all_acc


lat_pipe = read_gradient_file("tiny_pipe1/logs/valid_console.txt")[0:1400]
lat_cpu = read_gradient_file("tiny_origin/logs/valid_console.txt")[0:1400]

plt.plot(range(len(lat_pipe)), lat_pipe, label="pipe")
plt.plot(range(len(lat_pipe)), lat_cpu, label="origin")

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Top-1 Accuracy")
plt.title("Accurary for TinyImagenet (BlockN = 16)")
plt.savefig("tiny_16-acc.png")

import pandas as pd

big = pd.read_csv("cpu_b.csv")
small = pd.read_csv("cpu_s.csv")
gpu = pd.read_csv("gpu.csv")

big["cpu_s"] = small["time"]
big["gpu"] = gpu["time"]
big.to_csv("latency1.csv")

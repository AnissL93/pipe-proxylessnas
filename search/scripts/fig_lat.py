import matplotlib.pyplot as plt
import re

def get_latency_tensor(text):
    pattern = r"tensor\(\[(\d\.\d+), (\d\.\d+), (\d\.\d+)\]"
    match = re.search(pattern, text)
    if match:
        values = [float(x) for x in match.groups()]
        return max(values)


def get_mobile_latency(text, suffix):
    if suffix == "mobile":
        match = re.search(r"Latency-mobile:\s+(\d+\.\d+)", text)
    else:
        match = re.search(r"Latency-soc:\s+(\d+\.\d+)", text)
        if not match:
            return get_latency_tensor(text)
            

    if match:
        latency = float(match.group(1))
        return latency
    else:
        print("Latency-mobile not found in ", text)
        return None


def get_pipe_latency(text):
    pattern = r'tensor\(\[(.*?)\]'
    matches = re.findall(pattern, text)
    numbers = [float(num) for num in matches[0].split(', ')]
    return max(numbers)

def read_gradient_file(f):
    max_lat = []
    with open(f) as fp:
        lines = fp.readlines()
        for line in lines:
            if "Warmup" in line:
                continue
            if "soc" in line:
                lat = get_mobile_latency(line, "soc")
                if lat is None:
                    lat = get_pipe_latency(line)
            elif "mobile" in line:
                lat = get_mobile_latency(line, "mobile")

            max_lat.append(lat)
                    
    return max_lat
            
lat_pipe = read_gradient_file("tiny_pipe1/logs/valid_console.txt")[0:1400]
# lat_cpu = read_gradient_file("cifar_search_origin_cpu_only-small/logs/valid_console.txt")[0:500]
# print(lat_cpu)

plt.plot(range(len(lat_pipe)), lat_pipe, label="pipe")
# plt.plot(range(len(lat_pipe)), lat_cpu, label="origin")

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Max stage latency")
plt.title("TinyImagenet (BlockN = 16)")
plt.savefig("tiny_16.png")

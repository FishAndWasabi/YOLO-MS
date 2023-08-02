import re
import pandas as pd


with open("benchmark/benchlist", "r") as f:
    paths = f.read().split("\n")

paths = [path.split(" ")[1] for path in paths]

results = {"path":[],
           "fps":[],
           "latency":[],
           'flops': [],
           'params': []
           }
for path in paths:
    with open("benchmark/"+path+".log", 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            if "Model Flops:" in line:
                flops = re.findall(r"\d+\.+\d+", line.split(":")[1])[0]
                

            if "Model Parameters:" in line:
                params = re.findall(r"\d+\.+\d+", line.split(":")[1])[0]

            if "Overall fps:" in line:
                results["path"].append(path)
                results["flops"].append(flops)
                results["params"].append(params)
                result = re.findall(r"\d+\.+\d+", line.split(":")[3])[0]
                results["fps"].append(float(result))
                result = re.findall(r"\d+\.+\d+", line.split(":")[-1])[0]
                results["latency"].append(float(result))

print(results)
df = pd.DataFrame(results).groupby("path").min()
print(df)
df.to_csv("benchmark.csv")
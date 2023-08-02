import os

with open("benchmark/benchlist", 'r') as f:
    lines = f.read().split("\n")


for line in lines:
    path,name = line.split(" ")
    os.system(f"bash benchmark/benchmark.sh {path} {name}")
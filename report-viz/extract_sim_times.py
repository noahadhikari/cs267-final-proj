import re

def extract_simulation_times(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    times = []
    for line in lines:
        match = re.search(r"Simulation Time = ([0-9\.]+) seconds", line)
        if match:
            times.append(match.group(1))
    
    print(",".join(times))

# Example usage
filename = "results/ss_1M.txt"  # Change this to the actual filename
extract_simulation_times(filename)

# Runs ./Bchmk.py iterating over the first parameter

import sys
import os
import subprocess

# Get the path to the Bchmk.py script
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, 'Bchmk.py')

# Get the number of iterations
n = int(sys.argv[1])
avg_cnt = int(sys.argv[2])

# Parse the output format
def parse_output(output):
    # Format:
    # RadResizeGen: time ms
    # Prop0: time ms
    # Prop1: time ms
    # Max Device Memory: bytes
    # Points: x, x

    # Split the output by lines
    lines = output.split('\n')

    # Get the RadResizeGen time
    rad_resize_gen = float(lines[0].split(' ')[-2])

    # Get the Prop0 time
    prop0 = float(lines[1].split(' ')[-2])

    # Get the Prop1 time
    prop1 = float(lines[2].split(' ')[-2])

    # Get the Max Device Memory
    max_device_memory = int(lines[3].split(' ')[-1])

    # Get the Points
    points = lines[4].split(' ')[-1].split(',')

    # Return the parsed output
    return {
        'rad_resize_gen': rad_resize_gen,
        'prop0': prop0,
        'prop1': prop1,
        'max_device_memory': max_device_memory,
        'points': points
    }


# Run Bchmk.py n times
for i in range(n):
    outputs = []
    for _ in range(avg_cnt):
        output = subprocess.run(['python', path, str(i * 2), sys.argv[3]], check=True, capture_output=True, text=True).stdout
        output = parse_output(output)
        outputs.append(output)
    
    # Get the average of the outputs
    rad_resize_gen = sum([output['rad_resize_gen'] for output in outputs]) / len(outputs)
    prop0 = sum([output['prop0'] for output in outputs]) / len(outputs)
    prop1 = sum([output['prop1'] for output in outputs]) / len(outputs)
    max_device_memory = sum([output['max_device_memory'] for output in outputs]) / len(outputs)
    points = outputs[0]['points']

    # Print the output
    print(f'{points[0]} {rad_resize_gen} {prop0} {prop1} {max_device_memory}', flush=True)


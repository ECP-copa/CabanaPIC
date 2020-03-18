#!/usr/bin/env python

# Usage:
# This script takes a piped input of lists of files which contain VPIC energy data
# It will (try to) read each file, and plot matching files
# Absolute paths are preferred

def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True

# TODO: Make output dir selectable

import sys  # For reading std in
import os.path  # For file detection

# For plotting
from plotly.offline import plot
from plotly.graph_objs import Scatter, Layout

# Read stdin
input_file_list = sys.stdin.readlines()

# Check which of the input files are valid
valid_files = []
for f in input_file_list:
    f = f.strip()
    if os.path.isfile(f):
        valid_files.append(f)
    else:
        print("'" + f + "' is not a valid file path")

if len(valid_files) == 0:
    sys.exit('No valid files detected!')

# Print valid files
print("Valid :")
print(valid_files)


# For each valid file, parse the interesting subsets of data and put them in a
# graph using .. plotly?


class EnergyData(object):
    e_field = 0.0
    b_field = 0.0
    total = 0.0
    file_name = ""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, kwargs[key])


# TODO: enable custom delim. For now we refer the default which is \w
#delim = ' '

skip_count = 0

# Grab steps from first file
steps = []
with open(valid_files[0], 'r') as fh:
    for _ in range(skip_count):  # skip first 3 lines
        next(fh)
    for line in fh:
        #split = line.split(delim)
        split = line.split()
        if len(split) > 0:
            if split[0].isdigit():
                steps.append(split[0])

# Scan all files for data
data = {}
for f in valid_files:
    data[f] = {}
    # print(f)
    with open(f, 'r') as fh:
        for _ in range(skip_count):
            next(fh)
        for line in fh:
            #split = line.split(delim)
            split = line.split()
            print(split)

            # print(split)
            if len(split) == 0:
                continue
            if not is_number(split[0]):
                continue


            time_step = int(split[0])
            e = float(split[2])
            b = float(split[3])

            total_sum = e + b
            this_data = EnergyData(
                step=time_step,
                e_field=e,
                b_field=b,
                total=total_sum,
                file_name=f,
            )

            data[f][time_step] = this_data

graphs_needed = ["e_field", "b_field", "total"]

x_data = steps

combine_graphs = 1
combined_plot_data = []

# Generate each graph one by one
for g in graphs_needed:
    # for d in data:
    # for this_step in steps:
    name = g
    plot_data = []
    layout = Layout(
        title=name,
        xaxis=dict(
            title='STEPS',
            autorange=True
        ),
        yaxis=dict(
            title="ENERGY",
            type='log',
            autorange=True
        )
    )

    for f in valid_files:
        y_data = []
        for s in steps:
            s = int(s)
            val = getattr(data[f][s], g)
            y_data.append(val)

        line_name = g + "-" + f

        trace = Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name=line_name
        )
        plot_data.append(trace)
        if combine_graphs:
            combined_plot_data.append(trace)

    # Do plot
    plot(
        {
            'data': plot_data,
            'layout': layout,
        },
        filename=name + ".html",
        auto_open=False,  # TODO: make this programmable?
    )

if combine_graphs:
    name = "Combined"
    layout = Layout(
        title=name,
        xaxis=dict(
            title='STEPS',
            autorange=True
        ),
        yaxis=dict(
            title="ENERGY",
            type='log',
            autorange=True
        )
    )
    plot(
        {
            'data': combined_plot_data,
            'layout': layout,
        },
        filename="combined.html",
        auto_open=False,  # TODO: make this programmable?
    )

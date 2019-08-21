import matplotlib.pyplot as plt

filename = "out"
filename2 = "out2"

# Import data as a list of numbers
def read_file(filename):
    data = []
    with open(filename) as textFile:
        for line in textFile:
            if ( line.strip().startswith('#') ):
                # Skip comments
                continue
            if ( line.strip() == ""):
                # skip empty lines
                continue

            print(line)
            l = line.strip().split()[2]
            data.append(l)
        print(data)
        return data

data = read_file(filename)
data2 = read_file(filename2)

# Plot as a time series plot
plt.plot(data)
plt.plot(data2)
plt.show()

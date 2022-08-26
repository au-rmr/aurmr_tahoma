import matplotlib.pyplot as plt
import csv
import statistics

# program to get box and whisker plot of manipulability %'s one one graph
def boxPlot_multiple():
    file = open("boxPlot_data.csv", 'r')
    data = csv.reader(file, delimiter = ',')
    emptyLine = next(data)

    data_dict = {}

    # key: distance of pod relative to bin
    # value: 10 manipulability %'s 
    for line in data:
        distance = float(line[0])
        percents = [float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]),float(line[9]), float(line[10])]
        data_dict[distance] = percents

        print(str(distance))
        print("Manipulability % MEAN: " + str(statistics.mean(percents)))
        print("Manipulability %  STDEV: " + str(statistics.stdev(percents)))
        print("---------------------------------------------")

    fig,ax = plt.subplots()

    ax.set_title('Box plot of manipulability percents for multiple distances')
    ax.set_xlabel('distance of pod relative to robot(m)')
    ax.set_ylabel('manipulability %')

    ax.boxplot(data_dict.values())
    ax.set_xticklabels(data_dict.keys())

    plt.show()

#program for determining manipulability threshold 
def manip_threshold():
    file = open("manipData.txt", "r")
    data = csv.reader(file, delimiter = ",")

    pts = [42,38,36,38,38,46,38,32,32,40]

    for pt in data:
        for val in pt:
            if (val != ' ') :
                pts.append(float(val))

    n,bins,patches = plt.hist(pts, 20, range = (30,50), density = True)

    plt.show() 

#program to get box and whisker plot of manipulability %'s
def boxPlot_single():
    header = []
    manipPercent = []
    green = []
    white = []

    file = open("looped_Data.csv", 'r')
    data = csv.reader(file, delimiter = ',')
    header = next(data)

    for line in data:
        manipPercent.append(float(line[0]))
        green.append(float(line[1]))
        white.append(float(line[2]))

    print("Manipulability % MEAN: " + str(statistics.mean(manipPercent)))
    print("Manipulability %  STDEV: " + str(statistics.stdev(manipPercent)))

    #box = plt.boxplot(manipPercent)
    #fig = plt.figure()
    #ax = fig.add_subplot()

    pts = plt.scatter(green, white)

# robot position relative to bin gets hardcoded in fc.launch file
if __name__ == '__main__':
    boxPlot_multiple()

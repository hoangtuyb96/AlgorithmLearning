import csv
import random
import math
import operator

def loadData(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        data_set = list(lines)
        for x in range(len(data_set)-1):
            for y in range(4):
                data_set[x][y] = float(data_set[x][y])
            if random.random()<split:
                trainingSet.append(data_set[x])
            else:
                testSet.append(data_set[x])

def getDistance(data_1, data_2, length):
    distance = 0
    for x in range(length):
        distance += pow(data_1[x] - data_2[x], 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, data_test, k):
    distances = []
    neighbors = []
    length = len(data_test) - 1
    for x in range(len(trainingSet)):
        distance = getDistance(data_test, trainingSet[x], length)
        distances.append((trainingSet[x], distance))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getClassResult(neighbors):
    classes = {}
    for x in range(len(neighbors)):
        class_result = neighbors[x][-1]
        if class_result in classes:
            classes[class_result] += 1
        else:
            classes[class_result] = 1
    sortedClasses = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClasses[0][0]

def getAccuracy(testSet, results):
    tmp = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == results[x]:
            tmp += 1
    return (float(tmp/len(testSet)))

def main():
    trainingSet = []
    testSet = []
    results = []
    loadData('iris.data', 0.67, trainingSet, testSet)
    print('Train Data: ' +  str(len(trainingSet)) + ' files')
    print('Test Data: ' + str(len(testSet)) + ' files')

    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], 3)
        result = getClassResult(neighbors)
        results.append(result)
        print('Origin Data: ' + str(testSet[x][-1]) + ', Predicted Data: ' + str(result))
    acc = getAccuracy(testSet, results)
    print('Accuracy: ' + str(acc*100) + '%')

main()


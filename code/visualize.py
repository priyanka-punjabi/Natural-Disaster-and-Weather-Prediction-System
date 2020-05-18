# import library
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os


# read dataset
def readDB():
    dataFrame2 = pd.read_csv('newTao.csv')
    dbTao = np.array(dataFrame2)
    return dbTao


# create scatter plots
def createTaoPlots(db):
    coord = np.column_stack((db[:, 5], db[:, 6])).astype(np.float64)
    keys = np.unique(coord, axis=0)
    c = ['black', 'red', 'green', 'blue', 'brown']
    for i in range(5):
        j = 1
        for coord in keys:
            val = db[db[:, 5] == coord[0]]
            buoy = val[val[:, 6] == coord[1]]
            name = 'buoy--' + str(j) + '--attr--' + str(i+1) + '.png'
            j += 1
            plt.title('Attribute vs Time')
            plt.scatter(buoy[:, 4], buoy[:, 7+i], c=c[i], s=2)
            plt.savefig(os.getcwd() + '\\vis\\tao\\scatter\\' + name)
            plt.clf()


# get highest covariance
def covariance(db):
    val = db[:, 7:]
    cov = np.cov(val.T)
    covVals = []
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if i != j:
                if cov[i][j] not in covVals:
                    covVals.append(cov[i][j])
    index = np.where(np.max(covVals) == cov)[0]
    print("Attribute ", index[0], " and ", index[1], " has the highest covariance")


# scatter and histogram plots
def attrRelationPlots(db):
    # air temp vs ss temp
    plt.scatter(db[:, 10], db[:, 11], s=1)
    plt.title('Air temp vs S S temp')
    plt.savefig(os.getcwd() + '\\vis\\tao\\attrRelation\\AirvsSS.png')
    plt.clf()

    # zonal winds vs mer winds
    plt.scatter(db[:, 7], db[:, 8], s=1)
    plt.title('Zonal Winds vs Mer Winds')
    plt.savefig(os.getcwd() + '\\vis\\tao\\attrRelation\\zWindsvsmWinds.png')
    plt.clf()

    # humidity vs air temp
    plt.scatter(db[:, 9], db[:, 10], s=1)
    plt.title('Humidity vs Air temp')
    plt.savefig(os.getcwd() + '\\vis\\tao\\attrRelation\\humvsAirtemp.png')
    plt.clf()

    # histogram
    c = ['blue', 'red']
    plt.hist((db[:, 7], db[:, 8]), color=c)
    plt.title('Zonal Winds vs Mer Winds')
    plt.legend(['Zonal Winds', 'Meridian Winds'])
    plt.savefig(os.getcwd() + '\\vis\\tao\\Hist\\zWindsvsmWinds.png')
    plt.clf()

    plt.hist((db[:, 10], db[:, 11]), color=c)
    plt.title('Air temp vs S S temp')
    plt.legend(['Air temp', 'SS Temp'])
    plt.savefig(os.getcwd() + '\\vis\\tao\\Hist\\AirvsSS.png')
    plt.clf()


# main
if __name__ == '__main__':
    dbTao = readDB()
    createTaoPlots(dbTao)
    covariance(dbTao)
    attrRelationPlots(dbTao)
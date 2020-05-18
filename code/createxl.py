# import library
import pandas as pd


# read the dat file and create csv file
def createDB():
    cols = []
    for line in open('elnino.col', 'r'):
        cols.append(line.rstrip())

    dataFrame = pd.read_csv('elnino', sep='\s+', names=cols, engine='python')
    dataFrame.to_csv('elnino.csv', index=False)

    cols = []
    for line in open('tao-all2.col', 'r'):
        cols.append(line.rstrip())

    dataFrame = pd.read_csv('tao-all2.dat', sep='\s+', names=cols, engine='python')
    dataFrame.to_csv('tao.csv', index=False)


# main
if __name__ == '__main__':
    createDB()
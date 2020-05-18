# import library
import numpy as np
import pandas as pd
import copy


# read the db
def getdb():
    dataFrame1 = pd.read_csv('elnino.csv')
    dbEl = np.array(dataFrame1)

    dataFrame2 = pd.read_csv('tao.csv')
    dbTao = np.array(dataFrame2)
    return dbTao, dbEl, dataFrame1, dataFrame2


# read mean long and lat
def getMeanCoord():
    dataFrame = pd.read_csv('Approx Lat Long brackets - Sheet1.csv')
    meanlongitudes = np.array(dataFrame)[2, 1:][:9]
    meanLattitudes = np.array(dataFrame)[8, 1:]
    return meanLattitudes, meanlongitudes


# replace long and lat by mean long and lat
def LatLongmeanReplace(db, lat, long):
    for row in db:
        oldlat, oldLong = row[5], row[6]
        row[5] = lat[np.argsort(np.abs(np.subtract(oldlat, lat)))[0]]
        row[6] = long[np.argsort(np.abs(np.subtract(oldLong, long)))[0]]
    return db


# replace by mean
def fixattr(vals, db):
    if len(vals[vals == '.']) == len(vals):
        mean = np.mean(db[db != '.'].astype(np.float64))
    else:
        mean = np.mean(vals[vals != '.'].astype(np.float64))
    vals[vals == '.'] = mean
    return vals


# fix missing values for tao
def fixMissing(db, coords):
    newdb = copy.deepcopy(db)
    for coord in coords:

        latvals = db[:, 7:][db[:, 5] == coord[0]]

        for i in range(latvals.shape[1]):
            newdb[:, 7 + i][db[:, 5] == coord[0]] = fixattr(latvals[:, i], newdb[:, 7+i])

        longvals = db[:, 7:][db[:, 6] == coord[1]]

        for i in range(longvals.shape[1]):
            newdb[:, 7 + i][db[:, 6] == coord[1]] = fixattr(longvals[:, i], newdb[:, 7+i])

    return newdb


# fix tao
def fixTao(dbTao, dF):
    mLat, mLong = getMeanCoord()

    newdbTao = LatLongmeanReplace(dbTao, mLat, mLong)

    coord = np.column_stack((newdbTao[:, 5], newdbTao[:, 6])).astype(np.float64)
    keys = np.unique(coord, axis=0)

    db = fixMissing(newdbTao, keys)

    new = pd.DataFrame(db, columns=dF.columns)
    new.to_csv('newTao.csv', index=False)


# fix el
def fixEl(dbEl, dF):
    newdb = copy.deepcopy(dbEl)
    bouys = list(set(dbEl[:, 0]))
    for bouy in bouys:
        data = dbEl[:, 4:][dbEl[:, 0] == bouy]
        for i in range(data.shape[1]):
            newdb[:, 4 + i][newdb[:, 0] == bouy] = fixattr(data[:, i], dbEl[:, 4 + i])

    new = pd.DataFrame(newdb, columns=dF.columns)
    new.to_csv('newEl.csv', index=False)


# main function
if __name__ == '__main__':
    dbTao, dbEl, dF1, dF2 = getdb()
    fixTao(dbTao, dF2)
    fixEl(dbEl, dF1)
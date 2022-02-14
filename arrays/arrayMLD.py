from pathlib import Path

import pandas as pd
import numba
import numexpr as ne
import numpy as np

import altair as alt

# %load_ext line_profiler

dataPath = Path("data/raw/")
interimDataPath = Path("data/interim/")


def loadData():
    df = pd.read_csv(dataPath / "atlanticInterpolated.csv")
    z = df.z.values
    dropColumns = []
    for col in df.columns:
        if df.loc[30, col] > df.loc[0, col] - 0.1:
            dropColumns.append(col)
    df = df.drop(columns=dropColumns)
    df = pd.concat([df for _ in range(100)], axis=1)
    temps = df.iloc[:, 1:].values
    tempsC = np.ascontiguousarray(temps)
    surfaceTemps = temps[:2, :].mean(axis=0)
    surfaceTempsC = tempsC[:2, :].mean(axis=0)
    return df, temps, z, tempsC, surfaceTemps, surfaceTempsC


print("Load data")
df, temps, z, tempsC, surfaceTemps, surfaceTempsC = loadData()
print("Data loaded")
thresholdTemperatureDifference = 0.1


def numpyNonZeroColumnWise(
    temps: np.ndarray,
    surfaceTemps: np.ndarray,
    thresholdTemperatureDifference: float,
    z: np.ndarray,
):
    assert temps.shape[1] == surfaceTemps.shape[0]
    condition = temps < surfaceTemps - thresholdTemperatureDifference
    mldIndex = np.array(
        [condition[:, colIdx].nonzero()[0][0] for colIdx in range(temps.shape[1])]
    )
    mldDepth = np.array([z[idx] for idx in mldIndex])[:, np.newaxis]
    mldDepth = z[mldIndex][:, np.newaxis]
    mldTemp = np.array([temps[idx, colIdx] for colIdx, idx in enumerate(mldIndex)])[
        :, np.newaxis
    ]
    return mldDepth, mldTemp


mldnumpyNonZeroColumnWise, numpymldTempNonZeroColumnWise = numpyNonZeroColumnWise(
    temps=temps, surfaceTemps=surfaceTemps, thresholdTemperatureDifference=0.1, z=z
)
mldnumpyNonZeroColumnWise


def numExprNonZeroColumnWise(
    temps: np.ndarray,
    surfaceTemps: np.ndarray,
    thresholdTemperatureDifference: float,
    z: np.ndarray,
):
    assert temps.shape[1] == surfaceTemps.shape[0]
    condition = ne.evaluate("temps<surfaceTemps-thresholdTemperatureDifference")
    mldIndex = np.array(
        [condition[:, colIdx].nonzero()[0][0] for colIdx in range(temps.shape[1])]
    )
    mldDepth = np.array([z[idx] for idx in mldIndex])[:, np.newaxis]
    mldDepth = z[mldIndex][:, np.newaxis]
    mldTemp = np.array([temps[idx, colIdx] for colIdx, idx in enumerate(mldIndex)])[
        :, np.newaxis
    ]
    return mldDepth, mldTemp


mldnumExprNonZeroColumnWise, numExprmldTempNonZeroColumnWise = numExprNonZeroColumnWise(
    temps=temps, surfaceTemps=surfaceTemps, thresholdTemperatureDifference=0.1, z=z
)
mldnumExprNonZeroColumnWise
np.testing.assert_array_almost_equal(
    mldnumpyNonZeroColumnWise, mldnumExprNonZeroColumnWise
)
np.testing.assert_array_almost_equal(
    numpymldTempNonZeroColumnWise, numExprmldTempNonZeroColumnWise
)


@numba.njit()
def numbaConditionLoop(
    temps: np.ndarray,
    surfaceTemps: np.ndarray,
    thresholdTemperatureDifference: float,
    z: np.ndarray,
):
    mlDepth = np.empty_like(surfaceTemps)
    mldTemp = np.empty_like(surfaceTemps)
    for col in range(temps.shape[1]):
        row = 0
        temperature = temps[row, col]
        surfaceTemp = surfaceTemps[col]
        threshold = surfaceTemp - thresholdTemperatureDifference
        while (temperature > threshold) and row < temps.shape[0]:
            row += 1
            temperature = temps[row, col]
        mlDepth[col] = z[int(row)]
        mldTemp[col] = temps[int(row), col]
    return mlDepth, mldTemp


@numba.njit(parallel=True)
def numbaConditionLoopParallel(
    temps: np.ndarray,
    surfaceTemps: np.ndarray,
    thresholdTemperatureDifference: float,
    z: np.ndarray,
):
    mlDepth = np.empty_like(surfaceTemps)
    mldTemp = np.empty_like(surfaceTemps)
    for col in numba.prange(temps.shape[1]):
        row = 0
        temperature = temps[row, col]
        surfaceTemp = surfaceTemps[col]
        threshold = surfaceTemp - thresholdTemperatureDifference
        while (temperature > threshold) and row < temps.shape[0]:
            row += 1
            temperature = temps[row, col]
        mlDepth[col] = z[int(row)]
        mldTemp[col] = temps[int(row), col]
    return mlDepth, mldTemp


print("Compile numba loop")
mldnumbaConditionLoop, mldTempnumbaConditionLoop = numbaConditionLoop(
    temps=temps, surfaceTemps=surfaceTemps, thresholdTemperatureDifference=0.1, z=z
)
(
    mldnumbaConditionLoopParallel,
    mldTempnumbaConditionLoopParallel,
) = numbaConditionLoopParallel(
    temps=temps, surfaceTemps=surfaceTemps, thresholdTemperatureDifference=0.1, z=z
)
np.testing.assert_array_almost_equal(
    mldnumpyNonZeroColumnWise, mldnumExprNonZeroColumnWise
)
np.testing.assert_array_almost_equal(
    numpymldTempNonZeroColumnWise, numExprmldTempNonZeroColumnWise
)

# %timeit -n 1 -r 5 (temps<surfaceTemps-thresholdTemperatureDifference)
# %timeit -n 1 -r 5 (tempsC<surfaceTemps-thresholdTemperatureDifference)
# %timeit -n 1 -r 5 (tempsC<surfaceTempsC-thresholdTemperatureDifference)

# %timeit -n 1 -r 5 ne.evaluate("(temps<surfaceTemps-thresholdTemperatureDifference)")
# %timeit -n 1 -r 5 ne.evaluate("tempsC<surfaceTemps-thresholdTemperatureDifference")
# %timeit -n 1 -r 5 ne.evaluate("tempsC<surfaceTempsC-thresholdTemperatureDifference")
temps32 = temps.astype(np.float32)
%timeit -n 1 -r 3 numpyNonZeroColumnWise(temps=temps,surfaceTemps=surfaceTemps,thresholdTemperatureDifference=0.1,z=z)
%timeit -n 1 -r 3 numExprNonZeroColumnWise(temps=temps,surfaceTemps=surfaceTemps,thresholdTemperatureDifference=0.1,z=z)
%timeit -n 1 -r 3 numExprNonZeroColumnWise(temps=temps32,surfaceTemps=surfaceTemps,thresholdTemperatureDifference=0.1,z=z)
%timeit -n 1 -r 3 numbaConditionLoop(temps=temps,surfaceTemps=surfaceTemps,thresholdTemperatureDifference=0.1,z=z)
%timeit -n 1 -r 3 numbaConditionLoopParallel(temps=temps,surfaceTemps=surfaceTemps,thresholdTemperatureDifference=0.1,z=z)

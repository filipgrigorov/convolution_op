import numpy as np

def Conv2d(arr, kernel):
    rows, cols = arr.shape
    krows, kcols = kernel.shape
    res = np.zeros(shape=(rows - krows + 1, cols - kcols + 1))
    print(f'{rows}-{cols} :: {krows}-{kcols}')
    for row in range(0, rows - krows + 1):
        for col in range(0, cols - kcols + 1):
            for krow in range(0, krows):
                for kcol in range(0, kcols):
                    res[row][col] += arr[row + krow][col + kcol] * kernel[krow][kcol]
    return res

if __name__ == '__main__':
    arr = np.array([
        [ 1, 2, 1 ],
        [ 2, 3, 2 ],
        [ 3, 2, 3 ]
    ])

    kernel = np.array([
        [ -1, 1 ],
        [ 1, 1 ]
    ])
    arr = Conv2d(arr, kernel)
    print(arr)


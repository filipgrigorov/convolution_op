import numpy as np

np.set_printoptions(precision=2, suppress=True)

def Conv2d(tensor, nkernels, ksize, stride=1, padding=0):
    kernels = []
    depth = tensor.shape[0]
    for _ in range(0, nkernels):
        kernels.append(np.random.random((depth, ksize, ksize)))

    feature_maps = []
    for kernel in kernels:
        feature_maps.append(convolve(tensor, kernel, stride, padding))
    return np.stack(feature_maps)

def convolve(tensor, kernel, stride=1, padding=0):
    depth, arows, acols = tensor.shape
    _, krows, kcols = kernel.shape

    print(f'tensor shape: {tensor.shape}\nkernel shape: {kernel.shape}\n')
        
    pad = 2 * padding
    src = np.zeros(shape=(depth, arows + pad, acols + pad))
    _, rows, cols = src.shape
    print(src.shape)
    src[:, padding : -padding, padding : -padding] = tensor
    print(src)

    if __debug__:
        print(f'src: \n{src}')
   
    dst = np.zeros(shape=(
        int(((arows - krows + padding) / stride)) + 1, 
        int(((acols - kcols + padding) / stride)) + 1
    ))

    if __debug__:
        print(f'dst size: {dst.shape}')

    print(f'{rows}-{cols} :: {krows}-{kcols}')
    for row in range(0, rows - krows, stride):
        for col in range(0, cols - kcols, stride):
            for d in range(0, depth):
                for krow in range(0, krows):
                    kkrow = krow + row
                    for kcol in range(0, kcols):
                        for kd in range(0, depth):
                            dst[row][col] += src[d][kkrow][col + kcol] * \
                                kernel[d][krow][kcol]
    return dst

if __name__ == '__main__':
    tensor = np.array([
        [ 1, 2, 1 ],
        [ 2, 3, 2 ],
        [ 3, 2, 3 ]
    ])

    tensor = np.expand_dims(tensor, axis=0)

    print(f'tensor: \n{tensor}\n')

    feature_map = Conv2d(tensor, nkernels=1, ksize=2, stride=2, padding=1)
    print(feature_map)


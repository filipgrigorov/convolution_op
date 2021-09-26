import numpy as np
import time

np.set_printoptions(precision=2, suppress=True)

def Conv2d(tensor, nkernels, ksize, stride=1, padding=0):
    kernels = []
    depth = tensor.shape[0]
    for _ in range(0, nkernels):
        kernels.append(np.random.random((depth, ksize, ksize)))

    feature_maps = []
    for kernel in kernels:
        feature_maps.append(convolve_2d(tensor, kernel, stride, padding))
    return np.stack(feature_maps)

def convolve_2d(tensor, kernel, stride=1, padding=0):
    depth, arows, acols = tensor.shape
    _, krows, kcols = kernel.shape

            
    pad = 2 * padding
    src = np.zeros(shape=(depth, arows + pad, acols + pad))
    _, rows, cols = src.shape
    src[:, padding : -padding, padding : -padding] = tensor

    out_w = ((arows - krows + pad) / stride)
    out_h = ((acols - kcols + pad) / stride)
    if __debug__:
        print(f'\nOUT: {out_w}x{out_h}\n')

    dst = np.zeros(shape=(
        int(np.ceil(out_w)) + 1, 
        int(np.ceil(out_h)) + 1
    ))
    
    if __debug__:
        print(f'tensor shape: {tensor.shape}\nkernel shape: {kernel.shape}\n')

    if __debug__:
        print(f'src:\n{src}')
        print(f'kernel:\n{kernel}')
        print(f'dst size: {dst.shape}')

    for row in np.arange(0, rows - krows, stride):
        for col in np.arange(0, cols - kcols, stride):
            for d in np.arange(0, depth):
                for krow in np.arange(0, krows):
                    kkrow = krow + row
                    for kcol in np.arange(0, kcols):
                        for kd in np.arange(0, depth):
                            if __debug__:
                                print(f'{row}x{col} = {kkrow}x{col+kcol} + {krow}x{kcol}')

                            dst[row][col] += src[d][kkrow][col + kcol] * kernel[d][krow][kcol]

    if out_w % 2 != 0 or out_h % 2 != 0:
        if __debug__:
            print('*** ODD ***')
        return np.delete(np.delete(dst, dst.shape[0] // 2, axis=0), dst.shape[1] // 2, axis=1)
    return dst

if __name__ == '__main__':
    tensor = np.array([
        [ 1, 2, 1 ],
        [ 2, 3, 2 ],
        [ 3, 2, 3 ]
    ])

    tensor = np.expand_dims(tensor, axis=0)

    print(f'tensor: \n{tensor}\n')

    time_pnt = time.time()

    feature_map = Conv2d(tensor, nkernels=1, ksize=2, stride=2, padding=1)

    print(f'elapsed: {round((time.time() - time_pnt) * 1e3, 4)}us')

    print(feature_map.shape)
    print(feature_map)


import numpy as np
import util as ut
from math import fabs

def div_difference(samples: np.ndarray, inclusive_start: int, inclusive_end: int):

    if (inclusive_start >= inclusive_end):
        return samples[inclusive_start, 1]

    xb = samples[inclusive_end, 0]
    xa = samples[inclusive_start, 0]

    yb = div_difference(samples, inclusive_start + 1,  inclusive_end)
    ya = div_difference(samples, inclusive_start, inclusive_end - 1)

    return (yb - ya) / (xb - xa)

def estimate_error(samples: np.ndarray, argument: np.float64):
    dif_product = 1.0
    for x in samples[:, 0]:
        dif_product *= (argument - x)
    dif_product = fabs(dif_product)
    return dif_product * fabs(div_difference(samples, 0, samples.shape[0] - 1))

def interpolate(samples: np.ndarray, argument: np.float64):

    SIZE = samples.shape[0]

    image = 0.0
    for i in range(SIZE):
        product = 1.0
        for j in range(i):
            product *= argument - samples[j, 0]
        image += div_difference(samples, 0, i) * product

    return image
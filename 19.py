def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.aum(t * np.log(y + delta))
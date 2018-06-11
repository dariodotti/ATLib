import pickle

def load_matrix_pickle(filename):

    with open(filename, 'rU') as handle:
        file = pickle.load(handle)
    return file


def get_coordinate_points(occurance):

    xs = map(lambda line: int(float(line.split(' ')[2])),occurance)
    ys = map(lambda line: int(float(line.split(' ')[3])),occurance)
    ids =map(lambda line: str(line.split(' ')[0]),occurance)


    return xs,ys,ids
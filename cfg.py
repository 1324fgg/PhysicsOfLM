# input: a corpus dataset
# output: cfg rules

def rules3b():
    rules3b = {
        1: {22: [[21, 20], [20, 19]]},
        2: {19: [[16, 17, 18], [17, 18, 16]], 
            20: [[17, 16, 18], [16, 17]], 
            21: [[18, 16], [16, 18, 17]]},
        3: {16: [[15, 13], [13, 15, 14]], 
            17: [[14, 13, 15], [15, 13, 14]], 
            18: [[15, 14, 13], [14, 13]]},
        4: {13: [[11, 12], [12, 11]], 
            14: [[11, 10, 12], [10, 11, 12]], 
            15: [[12, 11, 10], [11, 12, 10]]},
        5: {10: [[7, 9, 8], [9, 8, 7]], 
            11: [[8, 7, 9], [7, 8, 9]], 
            12: [[8, 9, 7], [9, 7, 8]]},
        6: {7: [[3, 1], [1, 2, 3]], 
            8: [[3, 2], [3, 1, 2]], 
            9: [[3, 2, 1], [2, 1]]}
    }
    return rules3b


def summriseRule(dataset='handcrafted'):
    '''
    Given a dataset, this function summary it's distribution
    and give cfg rule depending on arguments

    dataset: the whole dataset
    parser:  the parser of the dataset if dataset is not gold
    rule:    the final cfg rule from the dataset
    '''
    if dataset == 'handcrafted':
        return rules3b()
    rule = ''
    return rule
    pass

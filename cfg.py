import os

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

def updateRule(rule, parent, child):
    if parent not in rule:
        rule[parent] = {}
    if child not in rule[parent]:
        rule[parent][child] = 0
    print(child)
    rule[parent][child] += 1
    return rule

def readTree(tree):
    # may have bug
    print(tree)
    stack = []
    pStack = []
    currentRule = {}
    current = ''
    for c in tree:
        if c == '(':
            stack.append(c)
            current = current.strip() 
            if current != '':
                print(current)
                pStack.append(current)
            current = ''
        elif c == ')':
            stack.pop(-1)
            current = current.strip() 
            if current != '':
                current = current.split(' ')[0]
                currentRule = updateRule(currentRule, pStack[-1], current)
            else:
                try:
                    current = pStack.pop(-1)
                    currentRule = updateRule(currentRule, pStack[-1], current)
                except IndexError:
                    pass
            current = ''
        else:
            current += c
    
    print(currentRule)
    return currentRule

def mergeRule(x, y):
    # may have bug
    merged = x.copy()
    for k, v in y.items():
        if k not in merged:
            merged[k] = v
        else:
            for sub_k, sub_v in v.items():
                if sub_k not in merged[k]:
                    merged[k][sub_k] = sub_v
                else:
                    merged[k][sub_k] += sub_v
    return merged

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
    else:
        rules = {}

        for root, _, files in os.walk(dataset):
            for file in files:
                if file.endswith('.mrg'):
                    with open(os.path.join(root, file)) as f:
                        rules = mergeRule(rules, readTree(f.read()))
            #         innerTestCount += 1
            #     if innerTestCount == 2:
            #         break
            # testCount += 1
            # if testCount == 2:
            #     break

        print(rules)
        # save rules, hard code for now, upgrade to args later
        return rules
    pass

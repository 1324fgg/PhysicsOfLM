import os
import pickle
from tqdm import tqdm

# input: a corp  us dataset
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
    # print(child)
    rule[parent][child] += 1
    return rule

def readTree(tree):
    # may have bug

    # for child frequency
    # print(tree)
    stack = []
    pStack = []
    currentRule = {}
    current = ''

    # for children combination frequency
    currentChildernRule = {}
    dfsStack = {}

    for c in tree:
        if c == '(':
            stack.append(c)
            current = current.strip() 
            if current != '':
                # print(current)
                pStack.append(current)
            current = ''
        elif c == ')':
            stack.pop(-1)
            current = current.strip() 
            # print('current: |{}|'.format(current))
            # print('pStack: |{}|'.format(pStack))
            # use path to node since node can be child of itself
            currentNode = '>'.join(pStack)
            if current != '':
                current = current.split(' ')[0]
                currentRule = updateRule(currentRule, pStack[-1], current)

                if currentNode not in dfsStack:
                    dfsStack[currentNode] = []
                dfsStack[currentNode].append(current)
            else:
                try:
                    current = pStack.pop(-1)
                    currentRule = updateRule(currentRule, pStack[-1], current)
                    # print(dfsStack)
                    # print(pStack)
                    # print(current)
                    currentChildernRule = updateRule(
                        currentChildernRule, 
                        current, 
                        '|'.join(dfsStack[currentNode])
                    )
                    # delete the used node as left and right child can be the same
                    dfsStack.pop(currentNode, None)

                    # update dfsStack
                    currentNode = '>'.join(pStack)
                    if currentNode not in dfsStack:
                        dfsStack[currentNode] = []
                    dfsStack[currentNode].append(current)

                except IndexError:
                    # print('rooting: ', dfsStack, '\tcurrent Node: ', currentNode)
                    currentChildernRule = updateRule(
                        currentChildernRule, 
                        current, 
                        '|'.join(dfsStack[currentNode])
                    )
                    # delete the used node as left and right child can be the same
                    dfsStack.pop(currentNode, None)

                    # print('after root del', dfsStack)

                    # update dfsStack
                    if currentNode != '':
                        currentNode = '>'.join(pStack)
                        if currentNode not in dfsStack:
                            dfsStack[currentNode] = []
                        dfsStack[currentNode].append(current)

                    # print('after root add', dfsStack)
                    pass
            current = ''
        else:
            current += c
    
    # print(currentRule)
    # print(currentChildernRule)
    return currentRule, currentChildernRule

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

def summriseRule(dataset='handcrafted', savePath='./results/'):
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
        childRules = {}
        childComboRules = {}

        testCount = 0

        for root, _, files in os.walk(dataset):
            for file in files:
                if file.endswith('.mrg'):
                    with open(os.path.join(root, file)) as f:
                        childFreq, childComboFreq = readTree(f.read())
                        childRules = mergeRule(childRules, childFreq)
                        childComboRules = mergeRule(childComboRules, childComboFreq)
                #     innerTestCount += 1
                # if innerTestCount == 2:
                    # break
            # testCount += 1
            # if testCount == 2:
            #     break

        # print(rules)
        # save rules, hard code for now, upgrade to args later
        childRulesPath = savePath + 'childRules.pkl'
        childComboRulesPath = savePath + 'childComboRules.pkl'
        print('childRules', childRules)
        # print('='*50)
        # print('childComboRules', childComboRules)
        with open(childRulesPath, 'wb') as f:
            pickle.dump(childRules, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(childComboRulesPath, 'wb') as f:
            pickle.dump(childComboRules, f, protocol=pickle.HIGHEST_PROTOCOL)
        return childRules
    pass

First version createDataset.ipynb done by Fiona

Using Python 3.11

# data

PTB location: /metadata/data/corpora at machine songcpu0

extra parser data: /metadata/data/corpora/aser/datas

# RL reward

reward1: design a new reward based on goal

reward2: 1. cfg rule -> generate data -> judge using llm judge, distillate cfg tree, to see the preplexity

# RL model 1

1. collect all cfg rules appeared in corpus(only type of word, e.g. S, NP, VP, ...)
2. Use the collected cfg rules to generate sentence, reward based on the generation results

## How to evaluate the amount of reward for a generation?

### train each gold tree

- ask the RL model to choice a corresponding next level nodes, compare with the ground truth

# RL model 2

1. collect all cfg rules appeared in corpus(only actual word)
2. Use the collected cfg rules to generate sentence, reward based on the generation results

## How to evaluate the amount of reward for a generation?

### train each gold tree

- ask the RL model to choice a corresponding next level word in cfg form, and ask LLM judge the perplexity ???


# Todo

recreate [Physics of LM 1](https://arxiv.org/pdf/2305.13673)
|task|status|
|--|--|
|createDataset|Done|
|complete tasks.py|just created|
|complete cfg.py|to automate cfg summary for input corpus|
|complete eval.py|accept input model for test|

# Usage

py eval.py to perform evaluation on given corpus for it's physics on given model

# Files

|dir|Usage|
|--|--|
|data|source corpus for cfg rules generation|
|result|resulted cfg rules|
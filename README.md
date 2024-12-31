First version createDataset.ipynb done by Fiona

Using Python 3.11

PTH is not open sourced and no similar gold is found

PTB location: /metadata/data/corpora at machine songcpu0

reward1: design a new reward based on goal
reward2: 1. cfg rule -> generate data -> judge using llm judge, distillate cfg tree, to see the preplexity
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
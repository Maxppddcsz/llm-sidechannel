## Basic Usage:
> You can use `fine-tuned-model` to obtain a better fine-tuned predictor. Our fine-tuned predictor is trained on GPU NVIDIA 1080 Ti and it is not so advanced.

- First, accumulate data in `sampling`, different Machine may have different time latency even with the same setting of our environment mentioned in the paper.
- Second, use code in `classifier` to train your classifier.
- Lastly, use the code in `strategy` to predict the next token. 

The path of the file might be different, have a closer look on the source code.
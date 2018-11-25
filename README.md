# deep-robust-nlp

A simple sentiment analysis using [a large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/). 
The input sentences are pre-processed using [feed-forward Neural-Net Language Models](https://tfhub.dev/google/nnlm-en-dim128/1) and feeded to a classifier. 
Here, we mainly focus on the robustness of classifiers when the train output labels are randomly flipped. 


This is the official implementation of paper "*LogitSpec*: Accelerating Retrieval-based Speculative Decoding via Next Next Token Speculation". 

## Get Start

Run the following command to prepare the environment.

```shell
conda create -n logitspec python=3.9
conda activate logitspec
cd LogitSpec
pip install -r requirements.txt
```

## Model Weights

Our *LogitSpec* is a retrieval-based speculative decoding method, which does not need additional draft model. Currently, our code only supports model family of Llama 2, including Llama 2, Vicuna, CodeLlama and so on. (All of these model weights can be found at Huggingface.)

## Reproduction

To reproduce the reported results in our paper, run the command:

```shell
sh eval.sh
```

## Acknowledgements

Our code is bulit on the official repo of Spec-Bench. Thanks for their excellent codebase! 
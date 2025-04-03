TEMP=0.0
GPU_DEVICES=0

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

Vicuna_PATH=/mnt/hwfile/sport/huggingface/hub/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6/
MODEL_NAME=vicuna-7b

FILE_PATH=data/spec_bench/model_answer/$MODEL_NAME-logitspec-$torch_dtype-temp-$TEMP.jsonl
BASE_PATH=data/spec_bench/model_answer/$MODEL_NAME-vanilla-$torch_dtype-temp-$TEMP.jsonl

echo -e "\033[33m $MODEL_NAME \033[0m"
python evaluation/speed.py --file-path $FILE_PATH --base-path $BASE_PATH --tokenizer-path $Vicuna_PATH

Vicuna_PATH=/mnt/hwfile/sport/huggingface/hub/models--lmsys--vicuna-13b-v1.3/snapshots/6566e9cb1787585d1147dcf4f9bc48f29e1328d2/
MODEL_NAME=vicuna-13b

FILE_PATH=data/spec_bench/model_answer/$MODEL_NAME-logitspec-$torch_dtype-temp-$TEMP.jsonl
BASE_PATH=data/spec_bench/model_answer/$MODEL_NAME-vanilla-$torch_dtype-temp-$TEMP.jsonl

echo -e "\033[33m $MODEL_NAME \033[0m"
python evaluation/speed.py --file-path $FILE_PATH --base-path $BASE_PATH --tokenizer-path $Vicuna_PATH

Vicuna_PATH=/mnt/hwfile/sport/huggingface/hub/models--lmsys--vicuna-33b-v1.3/snapshots/ef8d6becf883fb3ce52e3706885f761819477ab4/
MODEL_NAME=vicuna-33b

FILE_PATH=data/spec_bench/model_answer/$MODEL_NAME-logitspec-$torch_dtype-temp-$TEMP.jsonl
BASE_PATH=data/spec_bench/model_answer/$MODEL_NAME-vanilla-$torch_dtype-temp-$TEMP.jsonl

echo -e "\033[33m $MODEL_NAME \033[0m"
python evaluation/speed.py --file-path $FILE_PATH --base-path $BASE_PATH --tokenizer-path $Vicuna_PATH
Vicuna_PATH=/your_own_path/vicuna-7b-v1.3

MODEL_NAME=vicuna-7b-v1.3
TEMP=0.0
GPU_DEVICES=0

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_logitspec --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-logitspec-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP

BASE_PATH=data/$bench_NAME/model_answer/$MODEL_NAME-vanilla-$torch_dtype-temp-$TEMP.jsonl
LogitSpec_PATH=data/$bench_NAME/model_answer/$MODEL_NAME-logitspec-$torch_dtype-temp-$TEMP.jsonl

python evaluation/speed.py --file-path $LogitSpec_PATH --base-path $BASE_PATH --tokenizer-path $Vicuna_PATH
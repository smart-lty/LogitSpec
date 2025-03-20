Vicuna_PATH=/mnt/hwfile/sport/huggingface/hub/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6/
MODEL_NAME=vicuna-7b-v1.3
TEMP=0.0
GPU_DEVICES=1

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_logitspec --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-logitspec-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_logitspec --model-path $Vicuna_PATH --model-id test2 --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP
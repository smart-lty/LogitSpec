TEMP=0.0
GPU_DEVICES=0

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]


Vicuna_PATH=/mnt/hwfile/sport/huggingface/hub/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6/
MODEL_NAME=vicuna-7b

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_logitspec --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-logitspec-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP

# Vicuna_PATH=/mnt/hwfile/sport/huggingface/hub/models--lmsys--vicuna-13b-v1.3/snapshots/6566e9cb1787585d1147dcf4f9bc48f29e1328d2/
# MODEL_NAME=vicuna-13b

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_logitspec --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-logitspec-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP

# Vicuna_PATH=/mnt/hwfile/sport/huggingface/hub/models--lmsys--vicuna-33b-v1.3/snapshots/ef8d6becf883fb3ce52e3706885f761819477ab4/
# MODEL_NAME=vicuna-33b

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_logitspec --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-logitspec-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --dtype $torch_dtype --temperature $TEMP
"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer
from model.logitspec.logitspec import logitspec_generate


def logitspec_forward(inputs, model, tokenizer, max_new_tokens, temperature, max_ngram_size, num_pred_tokens, draft_tree_capacity):
    input_ids = inputs.input_ids
    output_ids, idx, accept_length_list = model.logitspec_generate(
        input_ids=inputs.input_ids, 
        max_length=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        max_ngram_size=max_ngram_size,
        num_pred_tokens=num_pred_tokens,
        draft_tree_capacity=draft_tree_capacity,
        )
    input_len = len(input_ids[0])
    new_token = len(output_ids[0][input_len:])
    if tokenizer.eos_token_id in output_ids[0, input_len:].tolist():
        for i, id in enumerate(output_ids[0, input_len:]):
            if id == tokenizer.eos_token_id:
                eos_token_ids_index = i
        invalid_len = len(output_ids[0, input_len:]) - eos_token_ids_index - 1
        if invalid_len > 0:
            accept_length_list[-1] -= invalid_len
            new_token -= invalid_len
    return output_ids, new_token, idx+1, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max_ngram_size", type=int, default=3, 
    )
    parser.add_argument(
        "--num_pred_tokens", type=int, default=6, 
    )
    parser.add_argument(
        "--draft_tree_capacity", type=int, default=64, 
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    
    question_file = f"data/{args.bench_name}/question.jsonl"
    
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")


    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            
    if not hasattr(tokenizer, "pad_token_id"):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.logitspec_generate = logitspec_generate.__get__(model, type(model))

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=logitspec_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        max_ngram_size=args.max_ngram_size,
        num_pred_tokens=args.num_pred_tokens,
        draft_tree_capacity=args.draft_tree_capacity,
    )

    reorg_answer_file(answer_file)
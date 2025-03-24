from typing import List, Optional, Union

import torch
import ipdb
import time
from transformers.generation.stopping_criteria import StoppingCriteriaList
from .generate_drafts_util import Pool, generate_draft_tokens
from .prepare_attn_util import prepare_attention_inputs
from .logit_util import LogitProcessor

device = torch.device('cuda:0')


@torch.no_grad()
def logitspec_generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        **kwargs,
):
    # init values
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    
    logit_processor = LogitProcessor(temperature=kwargs["temperature"])
    
    step = 0
    accept_length_list = []
    
    # prefilling stage, get past_key_values and last_logit
    outputs = self.forward(input_ids)
    past_key_values = outputs.past_key_values
    last_logit = outputs.logits[:, -1, :]
    verify_input_ids = input_ids[:, :]
    cur_len = verify_input_ids.shape[1]
    
    pool = Pool(input_ids=verify_input_ids, max_ngram_size=kwargs["max_ngram_size"], num_pred_tokens=kwargs["num_pred_tokens"])
    
    # * decoding stage, using draft-then-verify speculative decoding
    for _ in range(max_length):
        # ? 1. get draft ids of multi-sequence, maybe optimized
        next_token, candidate_list, num_draft_tokens = generate_draft_tokens(pool=pool.pool,
                                                                             input_ids=verify_input_ids, 
                                                                             last_logit=last_logit, 
                                                                             logit_processor=logit_processor,
                                                                             max_ngram_size=kwargs["max_ngram_size"],
                                                                             draft_tree_capacity=kwargs["draft_tree_capacity"])
        
        # 2. prepare attention mask and position ids for these draft tokens
        draft_ids, tree_attention_mask, tree_position_ids, search_path = prepare_attention_inputs(past_len=cur_len, 
                                                                                     next_token=next_token, 
                                                                                     candidate_list=candidate_list, 
                                                                                     num_draft_tokens=num_draft_tokens,
                                                                                     device=device,
                                                                                     pad_length=kwargs["num_pred_tokens"]+1)

        # 3. model forward, get all results for post-evaluation
        outputs = self(input_ids=draft_ids, 
                       attention_mask=tree_attention_mask, 
                       position_ids=tree_position_ids, 
                       past_key_values=past_key_values)

        # ? 4. speculative decoding verification, maybe optimized
        logits = outputs.logits
        
        model_res = torch.argmax(logits, dim=-1)
        all_input_path = draft_ids[0][search_path]
        all_input_path[search_path==-1] = -100
        all_output_path = model_res[0][search_path]
        reward = torch.cumprod(all_input_path[:, 1:].eq(all_output_path[:, :-1]), dim=-1).sum(dim=-1)
        best_reward = reward.max()
        
        accept_len = 1 + best_reward
        accept_length_list.append(accept_len.item())
        best_path_index = torch.argmax(reward, dim=-1).to(torch.long)
        index_path = search_path[best_path_index][:accept_len]

        # ? 5. update, maybe optimized
        update_ids = torch.index_select(draft_ids, index=index_path, dim=1)
        verify_input_ids = torch.cat([verify_input_ids, update_ids], dim=-1)
        select_kv_idx = torch.cat([torch.arange(cur_len, device=device), index_path + cur_len], dim=-1)
        past_key_values = [(k[:, :, select_kv_idx, :], v[:, :, select_kv_idx, :]) for k,v in outputs.past_key_values]
        pool.update_pool(update_ids)
        last_logit = logits[:, index_path[-1], :]
        
        # 6. check
        cur_len = verify_input_ids.shape[1]
        if (update_ids == eos_token_id).any() or cur_len >= max_length or (last_logit.argmax(dim=-1) == eos_token_id).any():
            break
        
        step += 1
    
    return verify_input_ids, step, accept_length_list
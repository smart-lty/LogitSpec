import torch


def prepare_attention_inputs(past_len, next_token, candidate_list, num_draft_tokens, device, dtype=torch.float16):
    """
    LogitSpec organizes draft tokens in a tree manner. Each sub-sequence corresponds to a local causal mask.
    """
    # for some model, need to change the fill value
    fill_value_false, fill_value_true = 0, 1 # torch.finfo(dtype).min, 0
    seq_len = num_draft_tokens + 1
    causal_mask = torch.full((seq_len, past_len + seq_len), fill_value=fill_value_false, dtype=dtype, device=device)
    causal_mask[:, :past_len+1] = fill_value_true
    
    draft_ids = [next_token] + [token for sub in candidate_list for token in sub]
    draft_ids = torch.tensor(draft_ids, device=device, dtype=torch.long).unsqueeze(0)
    position_ids = torch.zeros((1, seq_len), device=device, dtype=torch.long)
    
    idx = 1
    for sub_sequence in candidate_list:
        l = len(sub_sequence)
        sub_mask = torch.tril(torch.ones((l, l), dtype=dtype, device=device))
        causal_mask[idx:idx+l, idx+past_len:idx+past_len+l] = sub_mask
        position_ids[0, idx:idx+l] = torch.arange(l) + 1
        idx += l
    
    position_ids += past_len
    causal_mask = causal_mask[None, None, :, :].expand(1, 1, -1, -1)
    
    return draft_ids, causal_mask, position_ids 
import torch


class Pool:
    """Cache the ngrams in the input ids into a pool"""
    def __init__(self, input_ids, max_ngram_size=3, num_pred_tokens=6):
        if isinstance(input_ids, torch.Tensor):
            assert len(input_ids.shape) == 2
            self.input_ids_list = input_ids[0].tolist()
        elif isinstance(input_ids, list):
            self.input_ids_list = input_ids
        else:
            raise TypeError("Pool: input ids need to be list or torch.Tensor!")
        
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens
        
        self.pool = [{} for _ in range(self.max_ngram_size)]
        
        self.construct_pool()
        
    def __repr__(self):
        length = [len(_) for _ in self.pool]
        return ", ".join([f"{i+1}: {l}" for i, l in enumerate(length)])
    
    def construct_pool(self):
        len_s = len(self.input_ids_list)
        # iteratively add each ngram into the pool
        for i in range(len_s):
            for l in range(1, self.max_ngram_size + 1):
                end = i + l
                if end >= len_s:
                    continue
                substr = tuple(self.input_ids_list[i: end])
                if substr not in self.pool[l-1] or len(self.input_ids_list[end: end+self.num_pred_tokens]) >= len(self.pool[l-1][substr]):
                    self.pool[l-1][substr] = self.input_ids_list[end: end+self.num_pred_tokens]
    
    def update_pool(self, new_input_ids):
        if isinstance(new_input_ids, torch.Tensor):
            new_input_ids_list = new_input_ids[0].tolist()
        elif isinstance(new_input_ids, list):
            new_input_ids_list = new_input_ids
        else:
            raise TypeError("Pool: NEW input ids need to be list or torch.Tensor!")
        
        self.input_ids_list += new_input_ids_list
        len_s = len(self.input_ids_list)
        
        # iteratively add each ngram into the pool
        for i in range(len_s - 2 * self.num_pred_tokens, len_s):
            for l in range(1, self.max_ngram_size + 1):
                end = i + l
                if end >= len_s:
                    continue
                substr = tuple(self.input_ids_list[i: end])
                if substr not in self.pool[l-1] or len(self.input_ids_list[end: end+self.num_pred_tokens]) >= len(self.pool[l-1][substr]):
                    self.pool[l-1][substr] = self.input_ids_list[end: end+self.num_pred_tokens]
                    

def get_draft_length_via_rank(rank: int):
    """ 
    rank 1-5: 5
    rank 5-30: 3
    rank 30-50: 2
    rank >50: 1
    """
    assert isinstance(rank, int)
    if rank <= 8:
        return 4
    elif rank <= 32:
        return 3
    else:
        return 1

def generate_draft_tokens(pool, input_ids, last_logit, max_ngram_size=3, draft_tree_capacity=64):
    """
    LogitSpec generates draft tokens in 3 steps:
    1. generate next token as the 0-layer root of the draft tree
    2. expand the 1-layer nodes of the draft tree via last logit, which predicts the next-next-token (in BFS)
    3. Find candidate ngrams from prompt, expand each node into a sequence (similar to DFS)
    """
    next_token = last_logit.argmax().item()
    draft_token_layer_1 = last_logit.topk(k=draft_tree_capacity, dim=-1).indices[0].tolist()
    
    temp_input_ids_list = input_ids[0].tolist()
    
    num_draft_tokens = 0
    candidate_list = []
    
    temp_input_ids_list.append(next_token)
    
    # first, generate candidate sequence for the next token
    for ngram_size in range(max_ngram_size, 0, -1):
        ngram = tuple(temp_input_ids_list[-ngram_size:])
        if ngram in pool[ngram_size-1]:
            candidate_list.append(pool[ngram_size-1][ngram])
            num_draft_tokens += len(pool[ngram_size-1][ngram])
            break
    
    # second, generate candidate for 1-layer draft tokens
    for idx, draft_token in enumerate(draft_token_layer_1):
        if draft_token == next_token:
            continue
            
        temp = temp_input_ids_list + [draft_token]
        
        draft_sequence = [draft_token]
        
        for ngram_size in range(max_ngram_size, 0, -1):
            ngram = tuple(temp[-ngram_size:])
            if ngram in pool[ngram_size-1]:
                draft_sequence += pool[ngram_size-1][ngram]
                break
        
        draft_len = get_draft_length_via_rank(idx)
        draft_sequence = draft_sequence[:draft_len]
        draft_len = len(draft_sequence)
        
        if draft_len >= draft_tree_capacity-num_draft_tokens-1:
            candidate_list.append(draft_sequence[:draft_tree_capacity-num_draft_tokens-1])
            num_draft_tokens += draft_tree_capacity-num_draft_tokens-1
        else:
            candidate_list.append(draft_sequence)
            num_draft_tokens += len(draft_sequence)
            
        if num_draft_tokens >= draft_tree_capacity-1:
            break
        
    return next_token, candidate_list, num_draft_tokens


sub_mask_zoo = {l: torch.tril(torch.ones((l, l)))  for l in range(1, 30)}


def prepare_attention_inputs(past_len, next_token, candidate_list, num_draft_tokens, device, pad_length, dtype=torch.float16):
    """
    LogitSpec organizes draft tokens in a tree manner. Each sub-sequence corresponds to a local causal mask.
    """
    seq_len = num_draft_tokens + 1
    
    draft_ids = [next_token] + [token for sub in candidate_list for token in sub]
    draft_ids = torch.tensor(draft_ids, device=device, dtype=torch.long).unsqueeze(0)
    
    position_ids = torch.zeros((1, seq_len), dtype=torch.long)
    search_path = []
    
    causal_mask = torch.full((seq_len, past_len + seq_len), fill_value=0)
    causal_mask[:, :past_len+1] = 1
    
    idx = 1
    for sub_sequence in candidate_list:
        l = len(sub_sequence)
        if l == 1:
            causal_mask[idx, idx+past_len] = 1
            position_ids[0, idx] = 1
            search_path.append([0, idx] + [-1] * (pad_length-2))
            idx += 1
        else:
            sub_mask = sub_mask_zoo[l]
            causal_mask[idx:idx+l, idx+past_len:idx+past_len+l] = sub_mask
            position_ids[0, idx:idx+l] = torch.arange(l) + 1
            search_path.append([0] + [i for i in range(idx, idx+l)] + [-1] * (pad_length-l-1))
            idx += l
    
    position_ids += past_len
    position_ids = position_ids.to(device)
    causal_mask = causal_mask[None, None, :, :].expand(1, 1, -1, -1).to(dtype).to(device)
    search_path = torch.tensor(search_path, dtype=torch.long, device=device)
    
    return draft_ids, causal_mask, position_ids, search_path
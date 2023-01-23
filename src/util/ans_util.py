

# find the index, of the answer for raw tokens;
def _subfinder(words_list, answer_list):  
    # print('input words:',words_list)  e.g., [A few years ago, steve jobs founded the apple company]
    # print('input ans:',answer_list)   e.g., ['steve', 'jobs']
    # 1. edge case
    if not words_list or not answer_list:
        return None, 0, 0
    # 2. main 
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if (i+len(answer_list))>len(words_list): break # cannot exceed the length of the words

        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)     
    # 3. return res
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0

# important raw matching function!
def _raw_ans_word_idx_range(words, answers):
    # Match trial 1: try to find one of the answers in the context, return first match
    words_example = [word.lower() for word in words]
    for answer in answers:
        match, ans_word_idx_start, ans_word_idx_end = _subfinder(words_example, answer.lower().split())
        if match:
            break
    # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
    if not match:
        for answer in answers:
            for i in range(len(answer)):
                # drop the ith character from the answer
                answer_i = answer[:i] + answer[i+1:]
                # check if we can find this one in the context
                match, ans_word_idx_start, ans_word_idx_end = _subfinder(words_example, answer_i.lower().split())
                if match:
                    break
    # END OF EXPERIMENT
    if not match:
        for answer in answers:
            strs = answer.split()
            if len(strs)<3: continue

            drop_start = ' '.join(strs[1:])
            drop_tail = ' '.join(strs[:-1])
            cands = [drop_start, drop_tail]
            if len(strs)>3: 
                for i in range(1,len(strs)-1):
                    cand = strs[:i]+strs[i+1:]
                    cands.append(' '.join(cand))
            for answer_i in cands:
                # check if we can find this one in the context
                match, ans_word_idx_start, ans_word_idx_end = _subfinder(words_example, answer_i.lower().split())
                if match:
                    break
    # END OF EXPERIMENT
    return match, ans_word_idx_start, ans_word_idx_end
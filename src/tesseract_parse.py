from PIL import Image
import pytesseract
import pickle
import os
from transformers import LayoutLMv3FeatureExtractor
import json
from datasets import Dataset, load_from_disk
import numpy as np

feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained('/home/ubuntu/resources/layoutlmv3.base',apply_ocr=False)


def _get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox
def _normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]
def _load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def _adjust_shared_bbox(doc_dict):
    seg_ids = doc_dict['seg_ids']
    bboxes = doc_dict['bboxes']
    doc_dict['share_bboxes'] = []
    if not seg_ids or not bboxes or len(seg_ids)!=len(bboxes):
        return 

    block_num = seg_ids[0]  # 11
    window_bboxes = [bboxes[0]]
    l = 0
    for i in range(1,len(seg_ids)):
        curr_id = seg_ids[i]
        if curr_id!=block_num:
            new_bboxes = _get_line_bbox(window_bboxes)
            doc_dict['share_bboxes'] += new_bboxes
            # reset the params
            l = i
            block_num = curr_id
            window_bboxes = [bboxes[i]]
        else:
            window_bboxes.append(bboxes[i])
    # process the last one
    new_bboxes = _get_line_bbox(window_bboxes)
    doc_dict['share_bboxes'] += new_bboxes
    return doc_dict
def _pixel_feature(image_path):
    image, size = _load_image(image_path)
    encoding = feature_extractor(image)
    pixel_values = encoding.pixel_values[0]
    return pixel_values


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


def json_to_doc(base, docid_page):
    json_path = os.path.join(base, 'ocr_results/'+docid_page +'.json')
    img_path = os.path.join(base, 'documents/'+docid_page +'.png')

    with open(json_path, "r", encoding="utf8") as f:
        data = json.load(f)
    recognitionResults = data['recognitionResults']
    # usually just one page
    for page in recognitionResults:
        # each line = block = segment 
         # save to:
        one_doc = {'tokens':[],'bboxes':[],'widths':[],'heights':[], 'seg_ids':[],'image':None}

        width = page['width']
        height = page['height']
        size = [width,height]

        tokens = []
        bboxes = []
        seg_ids = []
        widths = []
        heights = []

        seg_id = 0
        for line in page['lines']:
            cur_line_bboxes = []
            text = line['text']
            words = line['words']
            # boundingBox = _normalize_bbox(line['boundingBox'],size)

            words = [w for w in words if w["text"].strip() != ""]
            if len(words) == 0:
                continue
            for word in words:
                tokens.append(word['text'])
                seg_ids.append(seg_id)
                cur_line_bboxes.append(_normalize_bbox(word["boundingBox"], size))
            cur_line_bboxes = _get_line_bbox(cur_line_bboxes)
            bboxes.extend(cur_line_bboxes)
            widths.extend([(x1-x0) for x0,y0,x1,y1 in cur_line_bboxes])
            heights.extend([(y1-y0) for x0,y0,x1,y1 in cur_line_bboxes])
        
            seg_id +=1
        pixel_values = _pixel_feature(img_path)
        # produce one sample
        one_doc['tokens'] = tokens
        one_doc['bboxes'] = bboxes
        one_doc['seg_ids']=seg_ids
        one_doc['widths'] = widths
        one_doc['heights'] = heights
        one_doc['image'] = pixel_values

        return one_doc




def image_to_doc(image_path, adjust_share_bbox=True):
    image, size = _load_image(image_path)
    data = pytesseract.image_to_data(Image.open(image_path),output_type='dict')

    texts = data['text']
    page_nums = data['page_num']
    block_nums = data['block_num']
    line_nums = data['line_num']
    x0s = data['left']
    y0s = data['top']
    hs = data['height']
    ws = data['width']

    # save to:
    one_doc = {'tokens':[],'bboxes':[],'widths':[],'heights':[], 'seg_ids':[],'image':None}

    encoding = feature_extractor(image)
    one_doc['image'] = encoding.pixel_values[0]    # image object, get the first one, cause there is only one!
    # one_doc['image'] = image
    for i,word in enumerate(texts):
        # token
        token = word.strip()
        if token=='': continue
        # height and width
        height, width = hs[i],ws[i]
        # coordinate
        x0 = x0s[i]
        y0 = y0s[i]
        x1 = x0 + width
        y1 = y0 + height
        # page, line, block, block_id
        page_num, line_num, block_num = page_nums[i],line_nums[i], block_nums[i]

        # produce one sample
        one_doc['tokens'].append(token)
        one_doc['bboxes'].append(_normalize_bbox([x0,y0,x1,y1], size))
        one_doc['seg_ids'].append(block_num)
        one_doc['widths'].append(width)
        one_doc['heights'].append(height)

    # adjust the shared box
    if adjust_share_bbox:
        one_doc = _adjust_shared_bbox(one_doc)
    return one_doc


def get_img2doc_data(img_dir):
    res = {}    # a dict of dict, i.e., {docID_pageNO : {one_doc_info}}
    for doc_idx, file in enumerate(sorted(os.listdir(img_dir))):
        print('process:',doc_idx,file)
        image_path = os.path.join(img_dir, file)
        one_doc = image_to_doc(image_path)
        docID_pageNO = file.replace(".png", "")
        res[docID_pageNO] = one_doc
        # print(one_doc)
        # if doc_idx>50:
        #     break
    return res

def get_json2doc_data(base_dir):
    res = {}    # a dict of dict, i.e., {docID_pageNO : {one_doc_info}}
    for doc_idx, file in enumerate(sorted(os.listdir(base_dir+'ocr_results'))):
        # print('process json:',doc_idx,file)
        json_path = os.path.join(base_dir, 'ocr_results', file)
        one_doc = json_to_doc(base_dir,file.replace('.json',''))
        
        docID_pageNO = file.replace(".json", "")
        res[docID_pageNO] = one_doc
        # print(one_doc)
        # if doc_idx>50:
        #     break
    return res

def output_to_pickle(my_dict,output_path):
    with open(output_path,'wb') as fw:
        pickle.dump(my_dict,fw)

def load_pickle(picke_path):
    with open(picke_path,'rb') as fr:
        res = pickle.load(fr)
    return res


def get_question_pairs(base,split='val'):
    # from json of questions and answers
    file_path = os.path.join(base, split+'_v1.0.json')
    
    with open(file_path) as fr:
        data = json.load(fr)
    id2trip = {}
    for sample in data['data']:
        qID = sample['questionId']  # numeric e.g., 8366
        question = sample['question']
        # for test set, there is no answersr
        answers = []
        if 'answers' in sample.keys():
            answers = sample['answers']

        ucsf_doc_id = sample['ucsf_document_id']   # e.g.,: txpp0227
        ucsf_doc_page = sample['ucsf_document_page_no'] # e.g.,: 10
        docID_page = ucsf_doc_id + '_' + ucsf_doc_page
        trip_object = (docID_page, question, answers)
        id2trip[qID] = trip_object
    return id2trip


def generator_based_on_questions(base, split):
    
    id2trip = get_question_pairs(base,split)
    id2doc = get_json2doc_data(base)
    print('q num:',len(id2trip.keys()))
    print('doc num:', len(id2doc.keys()))

    for qID,(docID_page, question, answers) in id2trip.items():
        doc = id2doc[docID_page]    # {keys: tokens, bboxes, seg_ids, widths, heights, image}

        if answers:
            match, ans_word_idx_start, ans_word_idx_end = _raw_ans_word_idx_range(doc['tokens'], answers)
        else:
            ans_word_idx_start, ans_word_idx_end = 0,0

        yield {
            "qID": qID,'question':question, 'answers':answers, 
            'ans_range':(ans_word_idx_start, ans_word_idx_end),
            "words": doc['tokens'], "boxes": doc['bboxes'],
            "seg_ids": doc["seg_ids"], "widths": doc["widths"], "heights": doc['heights'],
            "image_pixel_values": doc['image'],
        }

def wrap_and_save(base, split):
    mydataset = Dataset.from_generator(generator_based_on_questions,gen_kwargs={'split':split, 'base':base})
    # save to disk
    mydataset.save_to_disk(split+'.hf')
    del mydataset


if __name__=='__main__':

    # image_path = "/home/ubuntu/python_projects/GraphVRDU/data/FUNSD/testing_data/images/82491256.png"
    # one_doc = image_to_doc(image_path)

    # 1. produce pickle
    # val_dir = '/home/ubuntu/resources/shared_efs/vrdu/datasets/docvqa/test/'
    # val_res = get_img2doc_data(val_dir + 'documents')
    # output_to_pickle(val_res, 'test_pickle.pickle')
    # pickle is a dict of dict, i.e., {'docID_pageNO' : {one_doc_info}}

    # 2. load 
    # pickle_path = '/home/ubuntu/resources/shared_efs/vrdu/datasets/docvqa/pickles/val_pickle.pickle'
    # id2doc = load_pickle(pickle_path)
    # for k,doc in id2doc.items():
    #     print(doc['image'][0])
    #     print((doc['image'][0]).shape)
    #     print(doc['tokens'])
    # print('tokens:===',one_doc['tokens'])
    # print('bboxes:===',one_doc['bboxes'])
    # print('shared_bboxes:===',one_doc['share_bboxes'])


    # produce hf dataset
    for split in ['train','test','val']:
        base = '/home/ubuntu/air/vrdu/datasets/docvqa/'+split+'/'
        wrap_and_save(base, split)
    # wrap generater
    # 
    # one_doc = json_to_doc(base)
    # for k,v in one_doc.items():
    #     print(k,v,'\n')
    # for k,v in one_doc.items():
    #     print(k,np.array(v).shape)

    # load from disk
    # df_path = '/home/ubuntu/air/vrdu/datasets/docvqa/hfs/'
    # split = 'val'
    # dataset = load_from_disk(df_path + split + '.hf')
    # print(dataset)


from PIL import Image
import pytesseract
import pickle
import os
from transformers import LayoutLMv3FeatureExtractor
import json
from datasets import Dataset, load_from_disk
import numpy as np


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


def produce_based_on_questions(base, split):
    
    id2trip = get_question_pairs(base,split)
    id2doc = get_json2doc_data(base)


    for qID,(docID_page, question, answers) in id2trip.items():
        doc = id2doc[docID_page]    # {keys: tokens, bboxes, seg_ids, widths, heights, image}

        if answers:
            match, ans_word_idx_start, ans_word_idx_end = _raw_ans_word_idx_range(doc['tokens'], answers)
        else:
            ans_word_idx_start, ans_word_idx_end = 0,0
        id2trip[qID] = (docID_page, question, answers, ans_word_idx_start, ans_word_idx_end)
    return id2trip, id2doc


def wrap_and_save(base, split):
    # mydataset = Dataset.from_generator(generator_based_on_questions,gen_kwargs={'split':split, 'base':base})
    id2queryinfo, id2doc = produce_based_on_questions(base, split)
    print('q num:',len(id2queryinfo.keys()))
    print('doc num:', len(id2doc.keys()))
    output_to_pickle([id2queryinfo,id2doc],split+'.pickle')
    # save to disk


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
    # for split in ['train','test','val']:
    #     base = '/home/ubuntu/air/vrdu/datasets/docvqa/'+split+'/'
    #     wrap_and_save(base, split)

    split = 'val'
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


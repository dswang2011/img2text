




feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained('/home/ubuntu/resources/layoutlmv3.base',apply_ocr=False)

# block 1: shared functions: normalize bbox, normalize segment boxes, load images; 

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


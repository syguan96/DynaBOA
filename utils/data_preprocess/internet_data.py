
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from utils.kp_utils import *

scaleFactor = 1.


def load_json(filepath):
    """
    return data: list
                 data[0]:{'image_id': 'seq01_000001.png', 
                            'category_id': 1, 
                            'keypoints': [1320.1944580078125, 477.17205810546875, 0.9276900887489319, ...], 
                            'score': 2.907074451446533, 
                            'box': [1279.7008056640625, 440.28424072265625, 125.80419921875, 353.02001953125], 
                            'idx': [0.0]}

    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def get_bbox(j2d):
    bbox = [min(j2d[:,0]), min(j2d[:,1]),
                        max(j2d[:,0]), max(j2d[:,1])]
    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
    return center, scale

def get_person_height(j2d):
    vis = j2d[:, 2] > 0.3
    min_j = np.min(j2d[vis,:2], 0)
    max_j = np.max(j2d[vis,:2], 0)
    person_height = np.linalg.norm(max_j - min_j)
    return person_height

def internet_data_extract(in_path):    
    # spin joints is body25+smpl joints(24). 
    perm_idx = get_perm_idxs('spin', 'coco') 

    seqs = [os.path.basename(name)[:-5] for name in glob.glob(os.path.join(in_path, '*.json'))]#['seq01_c01', 'seq02_c01', 'seq03_c01', 'seq04_c01', 'seq07_c01', 'seq10_c01']
    seqs.sort()
    for seq in seqs:
        imagenames = []
        scales, centers = [], []
        j2ds = []

        jsonfile = f'{in_path}/{seq}.json'
        annots = load_json(jsonfile)
        for annot in tqdm(annots, total=len(annots)):
            imagename = os.path.join(seq, annot['image_id'])
            kps2d = np.array(annot['keypoints']).reshape(-1,3)
            score = annot['score']
            height = get_person_height(kps2d)
            # filter out low-quality detection results and the person with only a small region.
            if score < 2.5 or height < 250:
                continue
            assert kps2d.shape == (17, 3), print(kps2d.shape)

            # get bbox
            center, scale = get_bbox(kps2d)

            # process 2d keypoints
            kps2d[:,2] = kps2d[:,2] > 0.3
            part = np.zeros([49, 3])
            part[perm_idx] = kps2d

            imagenames.append(imagename)
            centers.append(center)
            scales.append(scale)
            j2ds.append(part)
        out_file = os.path.join(in_path, f'{seq}.npz')
        print(f'{seq} Total Images:', len(glob.glob(os.path.join(in_path, 'images', seq, '*.png'))), ', in fact:', len(imagenames))
        np.savez(out_file, imgname=imagenames, center=centers, scale=scales, part=j2ds)

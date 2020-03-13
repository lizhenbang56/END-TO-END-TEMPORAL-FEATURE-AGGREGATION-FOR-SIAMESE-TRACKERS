from os.path import join, realpath, dirname, exists
import logging
import glob
import numpy as np
import json
import cv2
import collections


def load_dataset(dataset, base_path, json_path=None):
    info = collections.OrderedDict()
    if 'VOT' in dataset:
        # base_path = join(realpath(dirname(__file__)), '../dataset', dataset)
        if not exists(base_path):
            logging.error("Please download test dataset!!!")
            exit()
        list_path = join(base_path, 'list.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]
        videos = sorted(videos)
        print(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            if len(image_files) == 0:  # VOT2018
                image_path = join(video_path, 'color', '*.jpg')
                image_files = sorted(glob.glob(image_path))
            gt_path = join(video_path, 'groundtruth.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            if gt.shape[1] == 4:
                gt = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3],
                                      gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3], gt[:, 0] + gt[:, 2], gt[:, 1]))
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    elif 'OTB' in dataset:
        # base_path = join(realpath(dirname(__file__)), '../dataset', dataset)
        # json_path = join(realpath(dirname(__file__)), '../dataset', dataset + '.json')
        info = json.load(open(json_path, 'r'))
        for v in info.keys():
            path_name = info[v]['name']
            info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
            info[v]['gt'] = np.array(info[v]['gt_rect'])-[1,1,0,0]
            info[v]['name'] = v
    elif 'UAV' in dataset:
        import os
        if not exists(base_path):
            logging.error("Please download test dataset!!!")
            exit()
        videos = os.listdir(base_path)
        videos = sorted(videos)
        print(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, '*.jpg')
            image_files = sorted(glob.glob(image_path))
            gt_path = join(base_path, '../../anno', dataset, video + '.txt')
            gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
            gt = gt - [1, 1, 0, 0]
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    elif 'LASOT' in dataset:
        import os
        if not exists(base_path):
            logging.error("Please download test dataset!!!")
            exit()
        classes = os.listdir(base_path)
        classes = sorted(classes)
        for class_ in classes:
            print(class_)
            class_path = join(base_path, class_)
            videos = os.listdir(class_path)
            videos = sorted(videos)
            for video in videos:
                video_path = join(class_path, video)
                image_path = join(video_path, 'img', '*.jpg')
                image_files = sorted(glob.glob(image_path))
                gt_path = join(video_path, 'groundtruth.txt')
                gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)
                gt = gt - [1, 1, 0, 0]
                info[video] = {'image_files': image_files, 'gt': gt, 'name': video}
    else:
        logging.error('Not support')
        exit()
    return info


def load_video_info_im_gt(base_path, video):

    video_path = join(base_path, video)
    image_path = join(video_path, 'color', '*.jpg')
    image_files = sorted(glob.glob(image_path))

    if len(image_files) == 0:
        image_path = join(video_path, '*.jpg')
        image_files = sorted(glob.glob(image_path))

    gt_path = join(video_path, 'groundtruth.txt')
    gt = np.loadtxt(gt_path, delimiter=',').astype(np.float64)  # float64 otherwise nan with np.linalg.norm
    if len(gt) == 4:
        gt[0] -= 1  # 0-index OTB
        gt[1] -= 1  # 0-index OTB
        gt = np.expand_dims(gt, axis=0)
    if gt.shape[1] == 4:
        gt = np.column_stack((gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3],
                              gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3], gt[:, 0] + gt[:, 2], gt[:, 1]))

    ims = []
    for f in image_files:
        im = cv2.imread(f)
        if im.shape[2] == 1:
            cv2.cvtColor(im, im, cv2.COLOR_GRAY2RGB)
        ims.append(im)

    return ims, gt
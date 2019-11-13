"""根据proposals得到可视化的结果视频"""

import os, cv2, glob, json
import numpy as np

def run_per_image(anno, img_path, proposals_):
    x, y, w, h = [int(i) for i in anno]
    img = cv2.imread(img_path)
    proposals = [proposal[:-1] for proposal in proposals_]
    scores = [proposal[-1] for proposal in proposals_]
    for proposal, score in zip(proposals, scores):
        if score > 0.7:
            x1, y1, w1, h1 = [int(i) for i in proposal]
            img = cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
            img = cv2.putText(img, str('%.02f'%score), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
            img = cv2.putText(img, str('%.02f'%score), (x1, y1+h1), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    height, width, layers = img.shape
    size = (width, height)
    return img, size


def run_per_video(video_name):
    anno_path = os.path.join(anno_root, video_name, '*001.txt')
    anno_path = glob.glob(anno_path)[0]
    annos = np.loadtxt(anno_path, delimiter=',')
    imgs = sorted(glob.glob(os.path.join(img_root, video_name, '*.jpg')))
    proposals_path = os.path.join(proposals_root, video_name+'.json')
    with open(proposals_path, 'r') as f:
        proposal_dict = json.load(f)
    img_array = []
    size = None
    for anno, img in zip(annos, imgs):
        img_name = img.split('/')[-1]
        proposals = proposal_dict[img_name]
        img, size = run_per_image(anno, img, proposals)
        img_array.append(img)
    save_path = os.path.join('/tmp/7', video_name + '.avi')
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def main():
    video_root = sorted(os.listdir(anno_root))
    for video_name in video_root:
        if 'GOT' not in video_name:
            continue
        if single_video is not None:
            if video_name != single_video:
                continue
        print(video_name)
        run_per_video(video_name)


if __name__ == '__main__':
    img_root = '/home/zhbli/Dataset/data2/got10k/test'
    root = '/home/etvuz/project3/siamrcnn2/experiments/got10k_v8'
    thresh = 0.9
    single_video = 'GOT-10k_Test_000101'
    anno_root = os.path.join(root, 'result', str(thresh))
    proposals_root = os.path.join(root, 'proposals')
    main()
import os, cv2, glob, json
import numpy as np

def run_per_image(anno, img_path):
    x, y, w, h = [int(i) for i in anno]
    img = cv2.imread(img_path)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    height, width, layers = img.shape
    size = (width, height)
    return img, size


def run_per_video(video_name):
    anno_path = os.path.join(anno_root, video_name, '*001.txt')
    anno_path = glob.glob(anno_path)[0]
    annos = np.loadtxt(anno_path, delimiter=',')
    imgs = sorted(glob.glob(os.path.join(img_root, video_name, '*.jpg')))
    img_array = []
    size = None
    for anno, img in zip(annos, imgs):
        img, size = run_per_image(anno, img)
        img_array.append(img)
    save_path = os.path.join(save_root, video_name + '.avi')
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
    # img_root = '/home/zhbli/Dataset/data2/got10k/test'
    img_root = '/home/yyshi/zhbli/dataset/got10k/test'
    single_video = 'GOT-10k_Test_000004'
    anno_root = '/tmp/siamrcnn2'
    save_root = os.path.join(anno_root, 'visualization_video')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    main()
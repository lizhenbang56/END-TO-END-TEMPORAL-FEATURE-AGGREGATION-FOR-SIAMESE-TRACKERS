from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor import COCODemo
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList

from PIL import Image
import numpy as np
import cv2, os, glob, argparse, torch, time


def process_template(img, box):
    """
    img: PIL Image
    box: np vector, (4,) xywh
    """
    template_size = 400
    x, y, w, h = box
    '''裁剪图像'''
    img = img.crop((x, y, x+w, y+h))
    img = img.resize((template_size, template_size))
    box = np.array([0,0,template_size,template_size])
    return img, box


def re_track(model, video_dir, last_img_name, box, current_image):
    """
    box: 上一帧跟踪结果 xywh
    """
    frame_path = os.path.join(video_dir, last_img_name)
    pil_image = Image.open(frame_path).convert("RGB")
    pil_image, last_dummy_targets = process_template(pil_image, box)
    last_img = np.array(pil_image)[:, :, [2, 1, 0]]
    predictions, proposals = model.run_on_opencv_image(last_img, last_dummy_targets, current_image)
    return predictions, proposals


def get_tracking_result(last_box, proposals):
    """
    last_box: xywh
    proposals: BoxList, mode=xyxy
    """
    last_box = torch.from_numpy(last_box).reshape(1, 4)
    last_box = BoxList(last_box, proposals.size, mode="xywh").convert("xyxy")
    '''计算IoU'''
    overlaps = boxlist_iou(proposals, last_box)
    proposals = proposals.convert("xywh")
    res = proposals.bbox[torch.argmax(overlaps)].cpu().numpy()
    return res


def visualization(is_first, video_name, img_name, predictions):
    save_dir = os.path.join(cfg.OUTPUT_DIR, 'visualization', video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, img_name)
    if not is_first:
        cv2.imwrite(save_path, predictions)
    else:  # 对于第一帧，predictions是图像+矩形框
        image, box = predictions
        x, y, w, h = [int(i) for i in box]
        image = np.array(image)
        image = cv2.rectangle(
            image, (x,y), (x+w,y+h), (0,0,255), 1
        )
        cv2.imwrite(save_path, image)


def run_per_video(model, video_name):
    """
    跟踪一段视频
    frame_paths: list, 该段视频中所有帧的路径
    """
    global_start = time.time()
    global first_img, dummy_targets
    video_dir = os.path.join(video_root, video_name)
    imgs = sorted(os.listdir(video_dir))
    gt = np.loadtxt(os.path.join(video_dir, 'groundtruth.txt'), delimiter=',') # [xmin，ymin，width，height]
    frame_num = len(imgs) - 1  # 排除 groundtruth.txt
    boxes = np.zeros((frame_num, 4))  # xywh
    times = np.zeros(frame_num)
    for i in range(len(imgs)):
        if i == 0:
            is_first = True
            boxes[0] = gt
        else:
            is_first = False
        img_name = imgs[i]
        if img_name == 'groundtruth.txt':
            continue
        frame_path = os.path.join(video_dir, img_name)
        pil_image = Image.open(frame_path).convert("RGB")
        start_time = time.time()
        if is_first:
            # visualization(is_first, video_name, img_name, (pil_image, gt))
            pil_image, dummy_targets = process_template(pil_image, gt)
            first_img = np.array(pil_image)[:, :, [2, 1, 0]]
        else:
            image = np.array(pil_image)[:, :, [2, 1, 0]]
            predictions, proposals = model.run_on_opencv_image(first_img, dummy_targets, image)
            '''若无检测结果则利用上一帧重新检测（特殊情况是物体出画面，重检也捡不到）'''
            if predictions is None:
                predictions, proposals = re_track(model, video_dir, imgs[i-1], boxes[i-1], image)
            '''获得跟踪结果'''
            if predictions is None:  # 这是两次都没检测到的情况
                boxes[i] = boxes[i-1]
            else:
                boxes[i] = get_tracking_result(boxes[i-1], proposals)
            '''保存可视化结果'''
            visualization(is_first, video_name, img_name, predictions)
        times[i] = time.time() - start_time
    '''保存该帧跟踪结果'''
    record_file = os.path.join(
        cfg.OUTPUT_DIR, 'result', video_name,
        '%s_%03d.txt' % (video_name, 1))
    record_dir = os.path.dirname(record_file)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)
    np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    '''保存时间文件'''
    time_file = record_file[:record_file.rfind('_')] + '_time.txt'
    times = times[:, np.newaxis]
    if os.path.exists(time_file):
        exist_times = np.loadtxt(time_file, delimiter=',')
        if exist_times.ndim == 1:
            exist_times = exist_times[:, np.newaxis]
        times = np.concatenate((exist_times, times), axis=1)
    np.savetxt(time_file, times, fmt='%.8f', delimiter=',')
    '''打印速度'''
    global_time = (time.time()-global_start)/frame_num
    print(('%.3f'%global_time)+'秒每帧')

def main():
    """"""
    '''创建网络'''
    model = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
    )
    '''获得一段视频的路径'''
    with open(os.path.join(video_root, 'list.txt'), 'r') as f:
        videos = f.read().splitlines()
    print('start tracking')
    for video in videos:
        video_id = int(video.split('_')[-1])
        if video_id < args.start or video_id > args.end:
            continue
        else:
            print(video, end=' ', flush=True)
        run_per_video(model, video)


if __name__ == '__main__':
    '''定义全局变量'''
    video_root = '/home/zhbli/Dataset/data2/got10k/test'
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=180)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--weight", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg.merge_from_file(args.config_file)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.WEIGHT = args.weight
    main()
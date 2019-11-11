from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor import COCODemo
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList

from PIL import Image
import numpy as np
import cv2, os, glob, argparse, torch, time, json


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


def save_proposals(proposal_dict, proposals, img_name):
    if proposals is not None:
        boxes = proposals.convert("xywh").bbox.numpy()  # xywh
        scores = proposals.get_field('scores').numpy().reshape(-1,1)
        res = np.concatenate((boxes, scores), axis=1).tolist()
    else:
        res = [[-1., -1., -1., -1., -1]]
    proposal_dict[img_name] = res


def run_per_video(model, video_name):
    """
    跟踪一段视频
    frame_paths: list, 该段视频中所有帧的路径
    """
    proposal_dict = {}
    global first_img, dummy_targets
    video_dir = os.path.join(video_root, video_name)
    imgs = sorted(os.listdir(video_dir))
    gt = np.loadtxt(os.path.join(video_dir, 'groundtruth.txt'), delimiter=',') # [xmin，ymin，width，height]
    frame_num = len(imgs) - 1  # 排除 groundtruth.txt
    boxes = np.zeros((frame_num, 4))  # xywh
    for i in range(len(imgs)):
        img_name = imgs[i]
        if img_name == 'groundtruth.txt':
            continue
        if i == 0:
            is_first = True
            boxes[0] = gt
            proposal_dict[img_name] = [list(gt) + [1.0]]  # xywh, score
        else:
            is_first = False
        frame_path = os.path.join(video_dir, img_name)
        pil_image = Image.open(frame_path).convert("RGB")
        if is_first:
            pil_image, dummy_targets = process_template(pil_image, gt)
            first_img = np.array(pil_image)[:, :, [2, 1, 0]]
        else:
            image = np.array(pil_image)[:, :, [2, 1, 0]]
            predictions, proposals = model.run_on_opencv_image(first_img, dummy_targets, image)
            '''若无检测结果则利用上一帧重新检测（特殊情况是物体出画面，重检也捡不到）'''
            if predictions is None:
                predictions, proposals = re_track(model, video_dir, imgs[i-1], boxes[i-1], image)
            '''获得跟踪结果'''
            save_proposals(proposal_dict, proposals, img_name)
    save_dir = os.path.join(cfg.OUTPUT_DIR, 'proposals')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, video_name + '.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(proposal_dict, indent=4))

def main():
    """"""
    '''创建网络'''
    model = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.1,
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
            print(video, flush=True)
        run_per_video(model, video)


def track_per_video(video_name):
    print(video_name)
    json_path = os.path.join(root, video_name+'.json')
    with open(json_path, 'r') as f:
        proposal_dict = json.load(f)
    gt = None
    frame_num = len(proposal_dict)
    boxes = np.zeros((frame_num, 4))  # xywh
    times = np.zeros(frame_num)
    i = 0
    for img_name, proposals_ in proposal_dict.items():
        start_time = time.time()
        if img_name == '00000001.jpg':
            gt = proposals_[0][:-1]
            boxes[0] = gt
            times[0] = time.time() - start_time
            gt = torch.Tensor(gt).reshape(1, 4)
            gt = BoxList(gt, (-1,-1), mode="xywh").convert("xyxy")
            i += 1
            continue
        proposals = [proposal[:-1] for proposal in proposals_]
        scores = [proposal[-1] for proposal in proposals_]
        proposals = torch.Tensor(proposals)
        proposals = BoxList(proposals, (-1,-1), mode="xywh").convert("xyxy")
        last_box = torch.Tensor(boxes[i-1]).reshape(1, 4)
        last_box = BoxList(last_box, (-1,-1), mode="xywh").convert("xyxy")
        overlaps = boxlist_iou(proposals, last_box).squeeze()
        scores_ = []
        if max(scores) < threshold: # 若没有大于threshold的，则找得分最高的。
            selected_id = torch.argmax(torch.Tensor(scores))
        elif sum(np.array(scores)>threshold) == 1:  # 若只有一个大于阈值的，则直接选那个
            selected_id = np.where(np.array(scores)>threshold)[0][0]
        else:
            for score in scores:
                if score > threshold:  # 多个候选框，卡IoU
                    scores_.append(1)  # 若小于阈值则按overlap比。因为是相乘。
                elif 0 <= score <= threshold:
                    scores_.append(0)  # 多个候选框，卡IoU
                elif score == -1:
                    scores_.append(-1)
                else:
                    assert False
            selected_id = torch.argmax(overlaps*torch.Tensor(scores_))
        if scores[selected_id] == -1:
            boxes[i] = boxes[i-1]
        else:
            proposals = proposals.convert("xywh")
            res_box = proposals.bbox[selected_id].cpu().numpy()
            boxes[i] = res_box
        times[i] = time.time() - start_time
        i += 1
    '''保存该帧跟踪结果'''
    record_file = os.path.join(cfg.OUTPUT_DIR,
        'result/{}'.format(str(threshold)), video_name,
        '%s_%03d.txt' % (video_name, 1))
    record_dir = os.path.dirname(record_file)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)
    np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
    '''保存时间文件'''
    time_file = record_file[:record_file.rfind('_')] + '_time.txt'
    times = times[:, np.newaxis]
    np.savetxt(time_file, times, fmt='%.8f', delimiter=',')


def track_proposals():
    global root
    root = os.path.join(os.path.join(cfg.OUTPUT_DIR, 'proposals'))
    videos = sorted(os.listdir(root))
    for video in videos:
        video_name = video.split('.')[0]
        video_id = int(video_name.split('_')[-1])
        if video_id < args.start or video_id > args.end:
            print('跳过！！！！！！！！！！！！！！！！！')
            continue
        track_per_video(video_name)


if __name__ == '__main__':
    '''定义全局变量'''
    threshold = 0.85
    video_root = '/home/zhbli/Dataset/data2/got10k/test'
    config_file = 'experiments/got10k_v3/e2e_faster_rcnn_R_50_FPN_1x.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=180)
    args = parser.parse_args()
    cfg.merge_from_file(config_file)
    # main()
    track_proposals()
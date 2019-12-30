from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor import COCODemo
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_nms
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers import nms as box_nms

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
            last_image = np.array(pil_image)[:, :, [2, 1, 0]]
            pil_image, dummy_targets = process_template(pil_image, gt)
            first_img = np.array(pil_image)[:, :, [2, 1, 0]]
        else:
            image = np.array(pil_image)[:, :, [2, 1, 0]]
            predictions, proposals = model.run_on_opencv_image(first_img, [dummy_targets, dummy_targets], [last_image, image])
            last_image = image
            '''若无检测结果则利用上一帧重新检测（特殊情况是物体出画面，重检也捡不到）'''
            # if predictions is None:
            #     predictions, proposals = re_track(model, video_dir, imgs[i-1], boxes[i-1], image)
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

def visualization(video_name, img_name, proposals, res, last_box):
    img_path = os.path.join(video_root, video_name, img_name)
    save_dir = os.path.join(cfg.OUTPUT_DIR, 'visualization_nms', video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, img_name)
    image = cv2.imread(img_path)
    for proposal in proposals:
        x, y, w, h = [int(i) for i in proposal]
        image = cv2.rectangle(
            image, (x,y), (x+w,y+h), (0,0,255), 1
        )
    x, y, w, h = [int(i) for i in res]
    image = cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1)
    x, y, w, h = [int(i) for i in last_box]
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite(save_path, image)


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
        scores = torch.Tensor(scores)
        proposals = torch.Tensor(proposals)
        proposals = BoxList(proposals, (-1,-1), mode="xywh").convert("xyxy")
        proposals.add_field('objectness', scores)
        '''对proposals执行nms，保留top_n个样本。'''
        proposals_nms = boxlist_nms(
            proposals,
            0.1,
            max_proposals=10,
            score_field="objectness",
        )
        last_box = torch.Tensor(boxes[i - 1]).reshape(1, 4)
        last_box = BoxList(last_box, (-1, -1), mode="xywh").convert("xyxy")
        overlaps = boxlist_iou(proposals_nms, last_box).squeeze(0)
        selected_id = torch.argmax(overlaps)
        if overlaps[selected_id] == 0:
            print('消失')
            selected_id = torch.argmax(proposals_nms.extra_fields['objectness'])
        proposals_nms = proposals_nms.convert("xywh")
        res_box = proposals_nms.bbox[selected_id].cpu().numpy()
        boxes[i] = res_box
        # visualization(video_name, img_name, proposals_nms.bbox, res_box, boxes[i - 1])
        times[i] = time.time() - start_time
        i += 1
    '''保存该帧跟踪结果'''
    record_file = os.path.join(cfg.OUTPUT_DIR,
        'result', video_name,
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
    video_root = '/home/zhbli/Dataset/data2/got10k/test'
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=180)
    # parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--phase", type=str)
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg.merge_from_file(args.config_file)
    # threshold = args.threshold
    if args.output_dir is not None:
        cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.WEIGHT = args.weight
    if args.phase == 'run_model':
        main()
    elif args.phase == 'gen_result':
        track_proposals()
    else:
        assert False

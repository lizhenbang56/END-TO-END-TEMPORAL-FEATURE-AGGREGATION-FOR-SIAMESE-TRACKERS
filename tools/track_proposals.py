"""根据得到的proposals，获得新跟踪结果。如果没检测到东西，则放大了重新检测。"""
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor import COCODemo
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_nms
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers import nms as box_nms

import PIL
from PIL import Image
import numpy as np
import cv2, os, glob, argparse, torch, time, json

def visualization(video_name, img_name, proposals, res, last_box, proposals_1):
    img_path = os.path.join(video_root, video_name, img_name)
    save_dir = os.path.join(cfg.OUTPUT_DIR, 'visualization_nms', video_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, img_name)
    image = cv2.imread(img_path)
    for (proposal, score) in zip(proposals.bbox, proposals.extra_fields['objectness']):
        x, y, w, h = [int(i) for i in proposal]
        image = cv2.rectangle(
            image, (x,y), (x+w,y+h), (0,0,255), 1  # 红色
        )
        image = cv2.putText(image, '%.03f'%score, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
    if proposals_1 is not None:
        for (proposal, score) in zip(proposals_1.bbox, proposals_1.extra_fields['scores']):
            x, y, w, h = [int(i) for i in proposal]
            image = cv2.rectangle(
                image, (x,y), (x+w,y+h), (0,255,255), 1
            )
            image = cv2.putText(image, '%.03f' % score, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 2)
    x, y, w, h = [int(i) for i in res]
    image = cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1)  # 蓝色
    x, y, w, h = [int(i) for i in last_box]
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 绿色
    cv2.imwrite(save_path, image)


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


def re_track(model, video_name, img_name, last_box, first_img, dummy_targets):
    """
    last_box: xywh
    """
    '''准备当前帧图像。先将当前帧候选区域裁剪出来，然后将patch放大。这时设计坐标的变换，头脑要清晰。
    可以使用仿射函数，一步到位。'''
    video_dir = os.path.join(video_root, video_name)
    img_path = os.path.join(video_dir, img_name)
    pil_image = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_image.size
    x1, y1, w, h = last_box
    x2 = x1 + w
    y2 = y1 + h
    patch_x1 = x1 - w
    patch_x2 = x2 + w
    patch_y1 = y1 - h
    patch_y2 = y2 + h
    patch_w = patch_x2 - patch_x1
    patch_h = patch_y2 - patch_y1
    patch = pil_image.crop((patch_x1, patch_y1, patch_x2, patch_y2))
    scale = 400 / max(w, h)
    patch_w = int(patch_w * scale)
    patch_h = int(patch_h * scale)
    patch = patch.resize((patch_w, patch_h), PIL.Image.LANCZOS)
    current_image = np.array(patch)[:, :, [2, 1, 0]]
    predictions, proposals = model.run_on_opencv_image(first_img, dummy_targets, current_image)
    if predictions is not None:
        # cv2.imwrite('/tmp/a.jpg', predictions)
        # return predictions, proposals
        boxes = proposals.bbox  # xyxy
        boxes /= scale
        boxes[:, 0] += patch_x1
        boxes[:, 2] += patch_x1
        boxes[:, 1] += patch_y1
        boxes[:, 3] += patch_y1
        proposals.bbox = boxes
        proposals.size = (-1,-1)
        proposals = proposals.convert('xywh')
    return proposals


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

            frame_path = os.path.join(video_root, video_name, img_name)
            pil_image = Image.open(frame_path).convert("RGB")
            pil_image, dummy_targets = process_template(pil_image, boxes[0])
            first_img = np.array(pil_image)[:, :, [2, 1, 0]]
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
        proposals_nms = proposals_nms.convert("xywh")
        proposals_1 = None
        if overlaps[selected_id] == 0:  # 情况1：没找到重叠的物体
            score = torch.max(proposals_nms.extra_fields['objectness'])
            print('{}消失'.format(img_name), end=' ')
            proposals_1 = re_track(model, video_name, img_name, boxes[i-1], first_img, dummy_targets)
            if proposals_1 is None:  # 情况1.1 再次搜索也没找到重叠的物体：全局选最高分
                print('还是没找着。')
                selected_id = torch.argmax(proposals_nms.extra_fields['objectness'])
                res_box = proposals_nms.bbox[selected_id].cpu().numpy()
            else:
                overlaps_1 = boxlist_iou(proposals_1, last_box).squeeze(0)
                selected_id_1 = torch.argmax(overlaps_1)
                score_1 = proposals_1.extra_fields['scores'][selected_id_1]
                if overlaps_1[selected_id_1] == 0:  # 情况1.2 再次搜索也没找到重叠的物体：全局选最高分
                    print('还是没找着。')
                    selected_id = torch.argmax(proposals_nms.extra_fields['objectness'])
                    res_box = proposals_nms.bbox[selected_id].cpu().numpy()
                else:  # 情况2.2 再次搜索找到重叠物体
                    print('我回来了',end=' ')
                    if score_1 > score:  # 情况2.2.1 在此搜索找到重叠物体，且得分最高
                        print('得分很高！')  # 被录用
                        res_box = proposals_1.bbox[selected_id_1].cpu().numpy()
                    else:
                        print('得分低。')  # 情况2.2.2 再次搜索找到重叠物体，但得分不是最高：相当于没有找到，选全局最高分
                        selected_id = torch.argmax(proposals_nms.extra_fields['objectness'])
                        res_box = proposals_nms.bbox[selected_id].cpu().numpy()
        else:
            res_box = proposals_nms.bbox[selected_id].cpu().numpy()
        boxes[i] = res_box
        visualization(video_name, img_name, proposals_nms, res_box, boxes[i - 1], proposals_1)
        times[i] = time.time() - start_time
        i += 1
    '''保存该帧跟踪结果'''
    record_file = os.path.join(cfg.OUTPUT_DIR,
        'result/nms', video_name,
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
    config_file = 'experiments/got10k_v14/e2e_faster_rcnn_R_50_FPN_1x.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=180)
    args = parser.parse_args()
    cfg.merge_from_file(config_file)
    model = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.1,
    )
    track_proposals()
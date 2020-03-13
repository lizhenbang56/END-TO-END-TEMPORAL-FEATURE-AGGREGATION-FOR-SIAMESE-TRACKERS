import torch, os, random
import torch.utils.data
import numpy as np
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList


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


def data_augment(box, img):
    img_w, img_h = img.size
    x, y, w, h = box
    can_pad = np.array([False, False, False, False]) # pad_left, pad_right, pad_top, pad_down = [False, False, False, False]
    '''计算要扩增的边界'''
    left = x - w
    right = x + 2*w
    top = y - h
    down = y + 2*h
    if left > 0: can_pad[0] = True  # pad_left = True
    if right < img_w: can_pad[1] = True  # pad_right = True
    if top > 0: can_pad[2] = True  # pad_top = True
    if down < img_h: can_pad[3] = True  # pad_down = True

    if random.random() > 0.5 and any(can_pad):
        patch = img.crop((int(x), int(y), int(x+w), int(y+h)))
        new_boxes =[[left, y, w, h],
                  [x+w, y, w, h],
                  [x, top, w, h],
                  [x, y+h, w, h]]
        pad_id = random.choice(np.where(can_pad)[0])
        new_box = np.array(new_boxes[pad_id])
        n_x1, n_y1, n_w, n_h = [int(i) for i in new_box]
        img.paste(patch, (n_x1, n_y1))
        box = torch.from_numpy(np.stack((box, new_box)))
        target = BoxList(box, img.size, mode="xywh").convert("xyxy")
        target.add_field("labels", torch.Tensor([1, 1]).long())
    else:
        box = torch.from_numpy(box).reshape(1, 4)
        target = BoxList(box, img.size, mode="xywh").convert("xyxy")
        target.add_field("labels", torch.Tensor([1]).long())
    return img, target





class Uav20lDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, template_transforms):
        self.root = root
        self._transforms = transforms
        self._template_transforms = template_transforms
        self.videos = os.listdir(self.root)

        '''排除错误视频'''
        if 'train' in root:
            with open(os.path.join(self.root, 'error_list.txt')) as f:
                error_videos = f.read().splitlines()
            self.videos = list(set(self.videos) - set(error_videos))

    def __getitem__(self, item):
        """"""
        '''随机选择一段视频'''
        video_name = random.choice(self.videos)
        video_path = os.path.join(self.root, video_name)

        '''读入该视频的gt/cover/absence/cut_by_image信息'''
        gts = [np.loadtxt(os.path.join(os.path.dirname(video_path).replace('data_seq', 'anno'), video_name + '.txt'),
                             delimiter=',')[0]] # [?, 4] [xmin，ymin，width，height]
        template_gt_id = 0
        search_gt_id = 0
        template_img, template_target = self.getitem1(video_path, gts, template_gt_id, is_template=True)
        search_img, search_target = self.getitem1(video_path, gts, search_gt_id)
        '''生成相邻帧'''
        template_gt_id1 = 0
        search_gt_id1 = 0
        template_img1, template_target1 = self.getitem1(video_path, gts, template_gt_id1, is_template=True)
        search_img1, search_target1 = self.getitem1(video_path, gts, search_gt_id1)

        '''返回'''
        return [(template_img, template_target, template_gt_id, search_img, search_target, search_gt_id),
                (template_img1, template_target1, template_gt_id1, search_img1, search_target1, search_gt_id1)]

    def __len__(self):
        return len(self.videos)

    def getitem1(self, video_path, gts, gt_id, is_template=False):
        img_id = gt_id + 1  # 易错：gt与图像的对应关系：gt从第0行开始，而图像编号从1开始。
        gt = gts[gt_id]  # np vector, (4,) xywh
        img_path = os.path.join(video_path, '%06d.jpg' % img_id)
        img = Image.open(img_path).convert('RGB')
        '''对于target，裁剪并缩放'''
        if is_template:
            img, gt = process_template(img, gt)

        box = torch.from_numpy(gt).reshape(1, 4)
        target = BoxList(box, img.size, mode="xywh").convert("xyxy")
        target.add_field("labels", torch.Tensor([1]).long())

        '''进行clip_to_image/transform'''
        target = target.clip_to_image(remove_empty=True)
        if target.bbox.shape[0] == 0:
            print('无目标', img_path)
        if self._transforms is not None:
            if is_template:
                img, target = self._template_transforms(img, target)
            else:
                img, target = self._transforms(img, target)
        return img, target


def vis_got10k():
    root = '/data/zhbli/Dataset/got10k/train'
    video_id = '%06d' % 8636
    save_root = '/tmp/got10k/{}'.format(video_id)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    video_path = os.path.join(root, 'GOT-10k_Train_' + video_id)
    imgs = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
    imgs = imgs[:100]
    gts = np.loadtxt(os.path.join(video_path, 'groundtruth.txt'), delimiter=',')  # [?, 4] [xmin，ymin，width，height]
    for img_path, gt in zip(imgs, gts):
        print(img_path)
        img_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        x1, y1, w, h = gt
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0))
        save_path = os.path.join(save_root, img_name)
        cv2.imwrite(save_path, img)


def filter_wrong_videos():
    root = '/data/zhbli/Dataset/got10k/train'
    gt_paths = sorted(glob.glob(os.path.join(root, '*/groundtruth.txt')))
    for gt_path in gt_paths:
        gts = np.loadtxt(gt_path, delimiter=',')  # [?, 4] [xmin，ymin，width，height]
        video_name = gt_path.split('/')[-2]
        img_path = os.path.join(root, video_name, '00000001.jpg')
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        x1s = gts[:, 0]
        y1s = gts[:, 1]
        ws = gts[:, 2]
        hs = gts[:, 3]
        x2s = x1s + ws
        y2s = y1s + hs

        if any((ws <= 3) & (ws >= 0)) or any((hs <= 3) & (hs >= 0)):
            print('目标尺寸为0', gt_path)
        elif max(x2s) > img_w or max(y2s) > img_h:
            print('出界', gt_path)


def check_video_diff():
    root = '/data/zhbli/Dataset/got10k/train'
    with open(os.path.join(root, 'list.txt')) as f:
        videos = f.read().splitlines()
    for video_name in videos:
        video_path = os.path.join(root, video_name)
        check_img_diff(video_path)


def check_img_diff(video_path='/data/zhbli/Dataset/got10k/train/GOT-10k_Train_008636'):
    img1_name = os.path.join(video_path, '00000001.jpg')
    img2_name = os.path.join(video_path, '00000002.jpg')
    img1 = cv2.imread(img1_name).astype(np.float)
    img2 = cv2.imread(img2_name).astype(np.float)
    diff = np.abs(img1 - img2) < 30
    diff_len = len(np.where(diff==False)[0])
    if diff.all():
        print('帧重复', video_path)


if __name__ == '__main__':
    import cv2, glob
    # vis_got10k()
    # filter_wrong_videos()
    # check_img_diff()
    check_video_diff()
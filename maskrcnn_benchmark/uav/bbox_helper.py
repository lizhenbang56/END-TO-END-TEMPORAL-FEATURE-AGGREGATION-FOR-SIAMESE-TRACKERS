#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy as np
from collections import namedtuple
from shapely.geometry import Polygon, box

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """
    :param corner: Corner or np.array 4*N
    :return: Center or 4 np.array N
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """
    :param center: Center or np.array 4*N
    :return: Corner or np.array 4*N
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def aug_apply(bbox, param, shape, inv=False, rd=False):
    """
    apply augmentation
    :param bbox: original bbox in image
    :param param: augmentation param, shift/scale
    :param shape: image shape, h, w, (c)
    :param inv: inverse
    :param rd: round bbox
    :return: bbox(, param)
        bbox: augmented bbox
        param: real augmentation param
    """
    if not inv:
        center = corner2center(bbox)
        original_center = center

        real_param = {}
        if 'scale' in param:
            scale_x, scale_y = param['scale']
            imh, imw = shape[:2]
            h, w = center.h, center.w

            scale_x = min(scale_x, float(imw) / w)
            scale_y = min(scale_y, float(imh) / h)

            # center.w *= scale_x
            # center.h *= scale_y
            center = Center(center.x, center.y, center.w * scale_x, center.h * scale_y)

        bbox = center2corner(center)

        if 'shift' in param:
            tx, ty = param['shift']
            x1, y1, x2, y2 = bbox
            imh, imw = shape[:2]

            tx = max(-x1, min(imw - 1 - x2, tx))
            ty = max(-y1, min(imh - 1 - y2, ty))

            bbox = Corner(x1 + tx, y1 + ty, x2 + tx, y2 + ty)

        if rd:
            bbox = Corner(*map(round, bbox))

        current_center = corner2center(bbox)

        real_param['scale'] = current_center.w / original_center.w, current_center.h / original_center.h
        real_param['shift'] = current_center.x - original_center.x, current_center.y - original_center.y

        return bbox, real_param
    else:
        if 'scale' in param:
            scale_x, scale_y = param['scale']
        else:
            scale_x, scale_y = 1., 1.

        if 'shift' in param:
            tx, ty = param['shift']
        else:
            tx, ty = 0, 0

        center = corner2center(bbox)

        center = Center(center.x - tx, center.y - ty, center.w / scale_x, center.h / scale_y)

        return center2corner(center)


def _IoU(rect1, rect2):
    def inter(rect1, rect2):
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = min(rect1[3], rect2[3])
        return max(x2 - x1 + 1, 0) * max(y2 - y1 + 1, 0) * 1.

    def area(rect):
        x1, y1, x2, y2 = rect
        return (x2 - x1 + 1) * (y2 - y1 + 1)

    ii = inter(rect1, rect2)
    iou = ii / (area(rect1) + area(rect2) - ii)
    return iou


def IoU(rect1, rect2):
    # overlap

    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)

    target_a = (tx2-tx1) * (ty2 - ty1)

    inter = ww * hh
    overlap = inter / (area + target_a - inter)

    return overlap


def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), np.array([rect[2], rect[3]])  # 0-index


def cxy_wh_2_rect1(pos, sz):
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])  # 1-index


def rect1_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), np.array([rect[2], rect[3]])  # 0-index


def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2

    return cx, cy, w, h


def get_min_max_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        w = x2 - x1
        h = y2 - y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


# for VOT reset
def judge_overlap_poly(poly1, poly2):
    xy1 = poly1.reshape(-1, 2)
    polygon_shape1 = Polygon(xy1)
    xy2 = poly2.reshape(-1, 2)
    polygon_shape2 = Polygon(xy2)
    # The intersection
    overlap = polygon_shape1.intersection(polygon_shape2).area
    return True if overlap > 1 else False


def judge_overlap_rect(poly, rect):
    xy = poly.reshape(-1, 2)
    polygon_shape = Polygon(xy)
    gridcell_shape = box(rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3])
    # The intersection
    overlap = polygon_shape.intersection(gridcell_shape).area
    return True if overlap > 1 else False


def judge_overlap(poly, xy):
    if len(xy) == 4:
        return judge_overlap_rect(poly, xy)
    else:
        return judge_overlap_poly(poly, xy)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.DTA.consistency_losses import get_consistency_loss
from maskrcnn_benchmark.modeling.DTA.dropout import create_adversarial_dropout_mask, calculate_jacobians
from maskrcnn_benchmark.modeling.DTA.ramps import linear_rampup


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.dropout = nn.Dropout2d(0.1)
        self.target_consistency_criterion = get_consistency_loss('kld')

    def forward(self, obj_feature, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        '''以下是与DTA相关的代码'''
        if self.training:
            '''进行两次随机的dropout'''
            x1 = self.feature_extractor.forward1(obj_feature, features, proposals)
            x1_1 = self.dropout(x1)
            x1_2 = self.dropout(x1)
            target_features1 = self.feature_extractor.forward2(x1_1)
            # target_features2 = self.feature_extractor.forward2(x1_2)
            target_logits1 = self.predictor.forward_cls(target_features1)
            jacobian_for_cnn_adv_drop, clean_target_logits = calculate_jacobians(
                x1_2.detach(), target_logits1.detach(), self.predictor.cls_score, self.predictor.drop_size,
                self.target_consistency_criterion, self.reset_grad, self.feature_extractor.forward2)
            target_cnn_dropout_mask, _ = create_adversarial_dropout_mask(
                torch.ones_like(jacobian_for_cnn_adv_drop),
                jacobian_for_cnn_adv_drop, 0.01)
            target_logits_cnn_drop = self.predictor.forward_cls(self.feature_extractor.forward2(target_cnn_dropout_mask * x1_2))
            target_consistency_loss = self.target_consistency_criterion(target_logits_cnn_drop, target_logits1)
        else:
            x1 = self.feature_extractor.forward1(obj_feature, features, proposals)
            target_features1 = self.feature_extractor.forward2(x1)
            target_logits1 = self.predictor.forward_cls(target_features1)
            target_consistency_loss = None
        '''以上是与DTA相关的代码'''
        box_regression = self.predictor.forward_reg(target_features1)

        if not self.training:
            result = self.post_processor((target_logits1, box_regression), proposals)
            return target_features1, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [target_logits1], [box_regression]
        )

        if self.training:
            loss_classifier += target_consistency_loss

        return (
            target_features1,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )

    def reset_grad(self, keys=None):
        self.optimizers.zero_grad()


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)

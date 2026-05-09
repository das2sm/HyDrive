from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk
from .motion_blocks import RouteEncoder, FlattenRouteEncoder


@HEADS.register_module()
class MotionPlanningHeadV6(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_mode=18,
        motion_anchor=None,
        plan_anchor_config=None,
        plan_config=None,
        embed_dims=256,
        decouple_attn=False,
        anchor_reference_group=None,
        match_reference_group=None,
        instance_queue=None,
        cond_encoder=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        mode_graph_model=None,
        cond_graph_model=None,
        norm_layer=None,
        mode_norm_layer=None,
        ffn=None,
        motion_pred_layer=None,
        plan_pred_layer=None,
        deformable_model=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        planning_sampler_refine=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=20,
        attn_anchor_embed=False,
    ):
        super(MotionPlanningHeadV6, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_mode = ego_fut_mode
        self.plan_config = plan_config

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.anchor_reference_group = anchor_reference_group
        self.match_reference_group = match_reference_group

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.cond_encoder = build(cond_encoder, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.planning_sampler_refine = build(planning_sampler_refine, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "mode_gnn": [mode_graph_model, ATTENTION],
            "cond_cross_attn": [cond_graph_model, ATTENTION],
            "deformable": [deformable_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "mode_norm": [mode_norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "motion_pred": [motion_pred_layer, PLUGIN_LAYERS],
            "plan_pred": [plan_pred_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        self.plan_anchor = dict()
        for key, value in plan_anchor_config.items():
            if value is not None:
                anchor = np.load(value)
                self.plan_anchor[key] = nn.Parameter(
                    torch.tensor(anchor, dtype=torch.float32),
                    requires_grad=False,
                )
            else:
                self.plan_anchor[key] = None
        ref_plan_anchor = self.plan_anchor[self.anchor_reference_group]
        if ref_plan_anchor is not None:
            self.plan_anchor_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1, embed_dims * ref_plan_anchor.shape[1]),
                Linear(embed_dims, embed_dims),
            )
        else:
            self.plan_mode_query = nn.Embedding(self.ego_fut_mode, embed_dims)

        self.num_det = num_det
        self.num_map = num_map
        self.attn_anchor_embed = attn_anchor_embed

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):   
        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )

        map_instance_feature = map_output["instance_feature"]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        device = instance_feature.device
        (
            ego_feature,
            ego_anchor,
            temp_instance_feature,
            temp_anchor,
            temp_mask,
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            bs,
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== motion init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))

        # =========== plan init ===========
        plan_anchor = dict()
        for key, value in self.plan_anchor.items():
            if value is not None:
                plan_anchor[key] = value[None].repeat(bs, 1, 1, 1).to(device)
            else:
                plan_anchor[key] = None
        ref_plan_anchor = plan_anchor[self.anchor_reference_group]
        if ref_plan_anchor is not None:
            plan_pos = gen_sineembed_for_position(ref_plan_anchor)
            plan_mode_query = self.plan_anchor_encoder(plan_pos.flatten(2, 3))
        else:
            plan_mode_query = torch.tile(self.plan_mode_query.weight[None], (bs, 1 ,1))

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # =========== encoder condition ===========
        cond_embedding = self.cond_encoder(metas)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_results = []
        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_anchor_embed,
                    temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    anchor_embed_selected,
                    anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_anchor_embed_selected,
                )            
            elif op == "mode_gnn" or op == "mode_norm":
                plan_query = self.layers[i](plan_query)
            elif op == "cond_cross_attn":
                plan_query = self.layers[i](
                    plan_query,
                    key=cond_embedding,
                )
            elif op == "deformable":
                plan_anchor_flat = plan_anchor[self.anchor_reference_group].flatten(-2)
                plan_query = self.layers[i](
                    plan_query,
                    plan_anchor_flat,
                    None,
                    feature_maps,
                    metas,
                    None,
                )        
            elif op == "motion_pred":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                motion_cls, motion_reg = self.layers[i](motion_query)
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
            elif op == "get_plan_query":
                ego_feature = instance_feature[:, -1]
                ego_anchor_embed = anchor_embed[:, -1]
                plan_query = plan_mode_query + (ego_feature + ego_anchor_embed).unsqueeze(1)
            elif op == "plan_pred":
                plan_result, plan_anchor = self.layers[i](
                    plan_query,
                    plan_anchor,
                    ego_feature,
                    ego_anchor_embed,
                )
                planning_results.append(plan_result)

        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        status = plan_result.get("ego_status_reg")
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "planning_results": planning_results,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output
    
    def loss(self,
        motion_model_outs, 
        planning_model_outs,
        data, 
        motion_loss_cache
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        planning_results = model_outs["planning_results"]
        output = {}
        ## align matching
        if self.match_reference_group is not None:
            key = self.match_reference_group
            reg = planning_results[-1][key + "_" + "reg"]
            gt_reg = data[f"gt_{key}"]
            gt_mask = data[f"gt_{key}_mask"]
            mode_idx, dist = self.planning_sampler.get_mode_idx(reg, gt_reg, gt_mask)
        else:
            mode_idx = dist = None

        for key, value in self.plan_config.items():
            for decoder_idx, planning_result in enumerate(planning_results):
                if "temporal" in key or "spatial" in key:
                    cls = planning_result[key + "_" + "cls"]
                    reg = planning_result[key + "_" + "reg"]
                    gt_reg = data[f"gt_{key}"]
                    gt_mask = data[f"gt_{key}_mask"]
                    if "reg" in value["pred_types"]:
                        cls_target_type = "wta"
                    else:
                        cls_target_type = "hydra"
                    (
                        cls, 
                        cls_target, 
                        cls_weight,
                        reg,
                        reg_target,
                        reg_weight
                    ) = self.planning_sampler.sample(
                        cls,
                        reg,
                        gt_reg,
                        gt_mask,
                        cls_target_type,
                        mode_idx,
                        dist,
                    )
                    cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight) * value["cls_weight"]
                    output[f"planning_{key}_cls_loss_{decoder_idx}"] = cls_loss
                    if "reg" in value["pred_types"]:
                        reg_loss = self.plan_loss_reg(reg, reg_target, weight=reg_weight) * value["reg_weight"]
                        output[f"planning_{key}_reg_loss_{decoder_idx}"] = reg_loss
                elif "target_speed" in key:
                    cls = planning_result[key + "_" + "cls"]
                    gt_cls = data[f"gt_{key}"]
                    bs = cls.shape[0]
                    bs_indices = torch.arange(bs, device=cls.device)
                    best_cls = cls[bs_indices, mode_idx].softmax(dim=-1)
                    cls_loss = F.binary_cross_entropy(
                        best_cls, gt_cls
                    )
                    output[f"planning_{key}_cls_loss_{decoder_idx}"] = cls_loss * value["cls_weight"]
        
        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )
        return motion_result, planning_result
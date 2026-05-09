from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

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
class MotionPlanningHeadV2(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        mode_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        croase2fine_layer=None,
        deformable_model=None,
        refine_plan_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        planning_sampler_refine=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
        use_tp=True,
        use_route=False,
        route_dropout=False,
        route_flatten=False,
        route_cross_attn=False,
        route_input_dim=20,
        mask_route_p=0,
        mask_tp_p=0,
        route_scale_noise_p=0,
    ):
        super(MotionPlanningHeadV2, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
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
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
            "croase2fine": [croase2fine_layer, PLUGIN_LAYERS],
            "deformable": [deformable_model, ATTENTION],
            "refine_plan": [refine_plan_layer, PLUGIN_LAYERS],
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
        self.plan_loss_status = build_loss(plan_loss_status)

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
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, embed_dims * fut_ts),
            Linear(embed_dims, embed_dims),
        )
        self.use_tp = use_tp
        self.use_route = use_route
        if self.use_tp:
            self.target_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 2)
            )
        if self.use_route:
            if route_flatten:
                self.route_encoder = FlattenRouteEncoder(hidden=embed_dims, in_dim=route_input_dim)
            else:
                self.route_encoder = RouteEncoder(hidden=embed_dims, route_dropout=route_dropout)
        self.route_cross_attn = route_cross_attn
        if self.route_cross_attn:
            self.route_anchor_embed = nn.Embedding(1, embed_dims)

        self.mask_route_p = mask_route_p
        self.mask_tp_p = mask_tp_p
        self.route_scale_noise_p = route_scale_noise_p

        self.num_det = num_det
        self.num_map = num_map

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
        # anchor_idx = torch.randint(0, self.plan_anchor.shape[0], (bs, self.ego_fut_mode)).to(self.plan_anchor.device)
        # if self.training:
        #     gt_traj = metas['gt_ego_fut_trajs'].cumsum(-2)
        #     gt_mask = metas['gt_ego_fut_masks']
        #     dist = torch.linalg.norm(self.plan_anchor[None] - gt_traj[:, None], dim=-1)
        #     dist = (dist * gt_mask[:, None]).mean(dim=-1)
        #     best_match_index = dist.argmin(dim=1)
        #     anchor_idx[:, -1] = best_match_index
        #     for i in range(bs):
        #         if best_match_index[i] not in anchor_idx[i]:
        #             anchor_idx[i][-1] = best_match_index[i]
        # plan_anchor = self.plan_anchor[anchor_idx]
        plan_anchor = self.plan_anchor[None].repeat(bs, 1, 1, 1)
        plan_pos = gen_sineembed_for_position(plan_anchor)
        plan_mode_query = self.plan_anchor_encoder(plan_pos.flatten(2, 3)).unsqueeze(1)

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        paths = []
        plan_query = None
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),
                    key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask,
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                if self.use_route and self.route_cross_attn:
                    route_embedding = self.get_route_emb(metas)[:, None]
                    map_key = torch.cat([map_instance_feature_selected, route_embedding], dim=1)
                    map_key_pos = torch.cat([map_anchor_embed_selected, self.route_anchor_embed.weight[None].repeat(bs, 1, 1)], dim=1)
                else:
                    map_key = map_instance_feature_selected
                    map_key_pos = map_anchor_embed_selected
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_key,
                    query_pos=anchor_embed,
                    key_pos=map_key_pos,
                )
            elif op == "mode_gnn":
                if plan_query is None:
                    plan_query = self.get_plan_query(plan_mode_query, instance_feature, anchor_embed, num_anchor, metas)
                plan_query = self.layers[i](
                    plan_query[:, 0],
                )[:, None]
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                if plan_query is None:
                    plan_query = self.get_plan_query(plan_mode_query, instance_feature, anchor_embed, num_anchor, metas)
                (
                    motion_cls,
                    motion_reg,
                    plan_cls,
                    plan_status,
                    path,
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                plan_reg = torch.cat([plan_anchor.new_zeros([bs, self.ego_fut_mode, 1, 2]), plan_anchor], dim=-2)
                plan_reg = plan_reg[..., 1:, :] - plan_reg[..., :-1, :]
                plan_reg = plan_reg.unsqueeze(1)
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
                paths.append(path)
            elif op == "croase2fine":
                plan_anchor, plan_cls = self.layers[i](plan_anchor, plan_cls, plan_query, metas, feature_maps)
                plan_reg = torch.cat([plan_anchor.new_zeros([bs, plan_anchor.shape[1], 1, 2]), plan_anchor], dim=-2)
                plan_reg = plan_reg[..., 1:, :] - plan_reg[..., :-1, :]
                plan_reg = plan_reg.unsqueeze(1)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
            elif op == "deformable":
                if plan_query is None:
                    plan_query = self.get_plan_query(plan_mode_query, instance_feature, anchor_embed, num_anchor, metas)
                plan_query = self.layers[i](
                    plan_query[:,0],
                    plan_anchor.flatten(-2, -1),
                    None,
                    feature_maps,
                    metas,
                    None,
                ).unsqueeze(1)
            elif op == "refine_plan":
                plan_cls, plan_reg = self.layers[i](refined_plan_query.unsqueeze(1))
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(None)
                paths.append(None)
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "paths": paths,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output

    def get_route_emb(self, metas):
        route = metas["route"].float()
        if self.training and self.route_scale_noise_p > 0:
            mask = torch.rand(route.shape[0], 1) < self.route_scale_noise_p
            noise = torch.rand(route.shape[0], 2) * 2 - 1
            noise *= mask
            noise = noise.unsqueeze(1).tile(1, route.shape[1], 1).to(device=route.device, dtype=route.dtype)
            route = route + noise
        route_embedding = self.route_encoder(metas["route"].float())
        if self.training and self.mask_route_p > 0:
            mask = torch.rand(route_embedding.shape[0], 1)
            mask = mask.to(device=route_embedding.device, dtype=route_embedding.dtype) > self.mask_route_p
            route_embedding = route_embedding * mask
        return route_embedding

    def get_plan_query(self, plan_mode_query, instance_feature, anchor_embed, num_anchor, metas):
        if self.use_tp:
            target_embedding = self.target_encoder(metas["tp_near"].float())
            if self.training and self.mask_tp_p > 0:
                mask = torch.rand(target_embedding.shape[0], 1)
                mask = mask.to(device=target_embedding.device, dtype=target_embedding.dtype) > self.mask_tp_p
                target_embedding = (target_embedding * mask)
            target_embedding = target_embedding[:, None, None]
        else:
            target_embedding = 0
        if self.use_route and not self.route_cross_attn:
            route_embedding = self.get_route_emb(metas)[:, None, None]
        else:
            route_embedding = 0
        plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2) + target_embedding + route_embedding
        return plan_query

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
        if "croase2fine" in self.operation_order:
            cls_scores = model_outs["classification"][:-1]
            reg_preds = model_outs["prediction"][:-1]
            refine_cls_scores = model_outs["classification"][-1]
            refine_reg_preds = model_outs["prediction"][-1]
        else:
            cls_scores = model_outs["classification"]
            reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        path_preds = model_outs["paths"]
        output = {}
        for decoder_idx, (cls, reg, status, path) in enumerate(
            zip(cls_scores, reg_preds, status_preds, path_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            bs = cls.shape[0]
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=bs * self.ego_fut_mode)

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                }
            )
            if status is not None:
                status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])
                output.update({f"planning_loss_status_{decoder_idx}": status_loss,})
            if path is not None:
                bs = path.shape[0]
                mask = data["path_mask"].bool()
                path = path.reshape(bs, -1, 2).cumsum(dim=1)
                path = path[mask]
                path_label = data["path"][mask].float()
                path_loss = F.smooth_l1_loss(path, path_label, reduction="none").sum() / (mask.sum() + 1e-5)
                output.update({f"planning_loss_path_{decoder_idx}": path_loss})
        
        if "croase2fine" in self.operation_order:
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler_refine.sample(
                refine_cls_scores,
                refine_reg_preds,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            # bs, num_mode = cls.shape[:2]
            # cls = cls.flatten(end_dim=1)
            # cls_target = cls_target.flatten(end_dim=1)
            # cls_weight = cls_weight.flatten(end_dim=1)
            # cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=bs * num_mode)

            # reg_weight = reg_weight.flatten(end_dim=1)
            # reg_pred = reg_pred.flatten(end_dim=1)
            # reg_target = reg_target.flatten(end_dim=1)
            # reg_loss = self.plan_loss_reg(
            #     reg_pred, reg_target, weight=reg_weight
            # )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )

            output.update(
                {
                    f"planning_loss_cls_refine": cls_loss,
                    f"planning_loss_reg_refine": reg_loss,
                }
            )

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
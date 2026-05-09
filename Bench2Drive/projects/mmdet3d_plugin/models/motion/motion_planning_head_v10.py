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
from projects.mmdet3d_plugin.datasets.pipelines.transform import interp_paths_speeds
from projects.mmdet3d_plugin.core.box3d import *

from ..attention import gen_sineembed_for_position, gen_sineembed_for_position_multi
from ..blocks import linear_relu_ln
from ..instance_bank import topk, topk_with_indice
from .motion_blocks import RouteEncoder, FlattenRouteEncoder
from .rescore_utils import interp_anchor_to_traj, interp_feature, interp_feature_lowmem

@HEADS.register_module()
class MotionPlanningHeadV10(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        lat_fut_ts=15,
        lat_fut_mode=1024,
        lon_fut_ts=1,
        lon_fut_mode=45,
        motion_anchor=None,
        plan_config=None,
        plan_refine_config=None,
        embed_dims=256,
        decouple_attn=False,
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
        plan_refine_layer=None,
        deformable_model=None,
        lon_deformable_model=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        planning_sampler_refine=None,
        lat_plan_loss_cls=None,
        lon_plan_loss_cls=None,
        traj_plan_loss_cls=None,
        col_plan_loss_cls=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=20,
        det_attn_conf=False,
        map_attn_conf=False,
        det_attn_dist=False,
        map_attn_dist=False,
        his_dropout_p=0.0,
        lon_deform=False,
        lon_cumsum=False,
        motion_emb=False,
        sel_num_lat=None,
        sel_num_lon=None,
        lat_topk=None,
        lon_topk=None,
        decoder_weight=[1,]*6,
    ):
        super(MotionPlanningHeadV10, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.lat_fut_ts = lat_fut_ts
        self.lat_fut_mode = lat_fut_mode
        self.lon_fut_ts = lon_fut_ts
        self.lon_fut_mode = lon_fut_mode
        self.plan_config = plan_config
        self.plan_refine_config = plan_refine_config

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order

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
            "lat_agent_gnn": [cross_graph_model, ATTENTION],
            "lat_map_gnn": [cross_graph_model, ATTENTION],
            "lon_agent_gnn": [cross_graph_model, ATTENTION],
            "lon_map_gnn": [cross_graph_model, ATTENTION],
            "mode_gnn": [mode_graph_model, ATTENTION],
            "lat_mode_gnn": [mode_graph_model, ATTENTION],
            "lon_mode_gnn": [mode_graph_model, ATTENTION],
            "lat_lon_mode_gnn": [mode_graph_model, ATTENTION],
            "motion_gnn": [graph_model, ATTENTION],
            "cond_cross_attn": [cond_graph_model, ATTENTION],
            "lat_cond_cross_attn": [cond_graph_model, ATTENTION],
            "lon_cond_cross_attn": [cond_graph_model, ATTENTION],
            "deformable": [deformable_model, ATTENTION],
            "lat_deformable": [deformable_model, ATTENTION],
            "lat_deformable_interp": [deformable_model, ATTENTION],
            "lon_deformable": [lon_deformable_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "mode_norm": [mode_norm_layer, NORM_LAYERS],
            "lat_mode_norm": [mode_norm_layer, NORM_LAYERS],
            "lon_mode_norm": [mode_norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "motion_pred": [motion_pred_layer, PLUGIN_LAYERS],
            "plan_pred": [plan_pred_layer, PLUGIN_LAYERS],
            "plan_refine": [plan_refine_layer, PLUGIN_LAYERS],
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
        self.lat_plan_loss_cls = build_loss(lat_plan_loss_cls)
        self.lon_plan_loss_cls = build_loss(lon_plan_loss_cls)
        if traj_plan_loss_cls is not None:
            self.traj_plan_loss_cls = build_loss(traj_plan_loss_cls)
        self.col_plan_loss_cls = build_loss(col_plan_loss_cls)

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
        lat_anchor = np.load(plan_config["lat"]["anchor"])
        self.lat_anchor = nn.Parameter(
            torch.tensor(lat_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        lon_anchor = np.load(plan_config["lon"]["anchor"])
        self.lon_anchor = nn.Parameter(
            torch.tensor(lon_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        time_points = torch.tensor(self.plan_config["lon"]["time_points"])
        time_interval = time_points[:, 1] - time_points[:, 0]
        lon_dist = (self.lon_anchor * time_interval).cumsum(dim=-1)
        traj_anchor, traj_anchor_mask = interp_anchor_to_traj(self.lat_anchor, lon_dist)
        self.traj_anchor = nn.Parameter(
            torch.tensor(traj_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.traj_anchor_mask = nn.Parameter(
            torch.tensor(traj_anchor_mask, dtype=torch.float32),
            requires_grad=False,
        )

        # plan mode query
        self.lat_plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, lat_fut_ts * 2),
            Linear(embed_dims, embed_dims),
        )
        self.lon_plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1, lon_fut_ts),
            Linear(embed_dims, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map
        self.det_attn_conf = det_attn_conf
        self.map_attn_conf = map_attn_conf
        self.det_attn_dist = det_attn_dist
        self.map_attn_dist = map_attn_dist
        self.his_dropout_p = his_dropout_p
        self.lon_deform = lon_deform
        self.lon_cumsum = lon_cumsum
        self.motion_emb = motion_emb
        if motion_emb:
            self.motion_embedder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1, 4*2),
                Linear(embed_dims, embed_dims),
            )
            self.traj_embedder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1, 4*2),
                Linear(embed_dims, embed_dims),
            )
        
        self.sel_num_lat = sel_num_lat
        self.sel_num_lon = sel_num_lon
        self.lat_topk = lat_topk
        self.lon_topk = lon_topk
        self.decoder_weight = decoder_weight

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
        _, (instance_feature_selected, anchor_embed_selected, det_confidence_selected, det_anchors_selected), det_indices = topk_with_indice(
            det_confidence, self.num_det, instance_feature, anchor_embed, det_confidence, det_anchors
        )
        if not self.det_attn_conf:
            det_confidence_selected = None
        else:
            det_confidence_selected = det_confidence_selected[..., 0]
        if not self.det_attn_dist:
            det_anchors_selected = None

        map_instance_feature = map_output["instance_feature"]
        num_map_anchor = map_instance_feature.shape[1]
        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected, map_confidence_selected, map_anchors_selected), map_indices = topk_with_indice(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed, map_confidence, map_anchors
        )
        if not self.map_attn_conf:
            map_confidence_selected = None
        else:
            map_confidence_selected = map_confidence_selected[..., 0]
        if not self.map_attn_dist:
            map_anchors_selected = None

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
        if self.his_dropout_p > 0 and self.training:
            mask = torch.rand(bs, num_anchor + 1, 1, 1, device=ego_anchor.device) >= self.his_dropout_p
            temp_instance_feature *= mask
            temp_anchor_embed *= mask

        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)

        # =========== motion init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))

        # =========== plan init ===========
        lat_anchor = torch.tile(self.lat_anchor, (bs, 1 ,1, 1)) # bs, num_lat_mode, num_pts, 2
        lon_anchor = torch.tile(self.lon_anchor, (bs, 1 ,1)) # bs, num_lon_mode, num_vel
        traj_anchor = torch.tile(self.traj_anchor, (bs, 1 ,1, 1, 1)) # bs, num_lat_mode, num_lon_mode, num_vel, 2
        traj_anchor_mask = torch.tile(self.traj_anchor_mask, (bs, 1 ,1, 1))

        if self.sel_num_lat is not None and self.training:
            device = lat_anchor.device
            num_lat_mode = lat_anchor.size(1)
            num_lon_mode = lon_anchor.size(1)

            # 随机不重复采样索引（全 batch 共享）
            lat_idx = torch.randperm(num_lat_mode, device=device)[:self.sel_num_lat]  # (sel_num_lat,)
            lon_idx = torch.randperm(num_lon_mode, device=device)[:self.sel_num_lon]  # (sel_num_lon,)

            # 选子集
            lat_anchor = lat_anchor.index_select(1, lat_idx)                 # (bs, sel_num_lat, 15, 2)
            lon_anchor = lon_anchor.index_select(1, lon_idx)                 # (bs, sel_num_lon, 4)

            traj_anchor = traj_anchor.index_select(1, lat_idx)               # (bs, sel_num_lat, num_lon_mode, 4, 2)
            traj_anchor = traj_anchor.index_select(2, lon_idx)           # (bs, sel_num_lat, sel_num_lon, 4, 2)

            traj_anchor_mask = traj_anchor_mask.index_select(1, lat_idx)       # (bs, sel_num_lat, num_lon_mode, 4)
            traj_anchor_mask = traj_anchor_mask.index_select(2, lon_idx)   # (bs, sel_num_lat, sel_num_lon, 4)

        lat_mode_query = self.lat_plan_anchor_encoder(lat_anchor.flatten(-2))
        lon_mode_query = self.lon_plan_anchor_encoder(lon_anchor)
        lat_anchor_embed = ego_anchor_embed.repeat(1, lat_anchor.shape[1], 1)
        lon_anchor_embed = ego_anchor_embed.repeat(1, lon_anchor.shape[1], 1)

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)
        if det_confidence_selected is not None:
            det_confidence_selected = torch.cat([det_confidence_selected, det_confidence_selected.new_ones([bs, 1])], dim=1)
        if det_anchors_selected is not None:
            det_anchors_selected = torch.cat([det_anchors_selected, ego_anchor], dim=1)

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)
        cat_anchors = torch.cat([det_anchors, ego_anchor], dim=1)
        if not (self.det_attn_dist or self.map_attn_dist):
            cat_anchors = None

        # =========== encode condition ===========
        cond_embedding = self.cond_encoder(metas)

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_results = []
        planning_refine_results = []
        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn":
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
                    confidence=det_confidence_selected,
                    anchors_tgt=cat_anchors,
                    anchors_src=det_anchors_selected,
                )
                if hasattr(self.layers[i], "attn_weight"):
                    det_attn_weight = self.layers[i].attn_weight[:, -1, :self.num_det]
                else:
                    det_attn_weight = None
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                    confidence=map_confidence_selected,
                    anchors_tgt=cat_anchors,
                    anchors_src=map_anchors_selected,
                )
                if hasattr(self.layers[i], "attn_weight"):
                    map_attn_weight = self.layers[i].attn_weight[:, -1]
                else:
                    map_attn_weight = None
            elif op == "motion_pred":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                motion_cls, motion_reg = self.layers[i](motion_query)
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
            elif op == "get_plan_query":
                ego_feature = instance_feature[:, -1]
                ego_anchor_embed = anchor_embed[:, -1]
                if "ego_feature" in self.cond_encoder.cond_config:
                    cond_embedding = torch.cat([cond_embedding, (ego_feature + ego_anchor_embed).unsqueeze(1)], dim=1)
                if "ego_feature" not in self.cond_encoder.cond_config:
                    lat_mode_query = lat_mode_query + (ego_feature + ego_anchor_embed).unsqueeze(1)
                if "ego_feature" not in self.cond_encoder.cond_config:
                    lon_mode_query = lon_mode_query + (ego_feature + ego_anchor_embed).unsqueeze(1)
            elif op == "lat_mode_gnn" or op == "lat_mode_norm":
                lat_mode_query = self.layers[i](lat_mode_query)
            elif op == "lon_mode_gnn" or op == "lon_mode_norm":
                lon_mode_query = self.layers[i](lon_mode_query)
            elif op == "lat_lon_mode_gnn":
                num_lat_mode = lat_mode_query.shape[1]
                mode_query = torch.cat([lat_mode_query, lon_mode_query], dim=1)
                mode_query = self.layers[i](mode_query)
                lat_mode_query = mode_query[:, :num_lat_mode]
                lon_mode_query = mode_query[:, num_lat_mode:]
            elif op == "motion_gnn":
                best_motion_cls = motion_cls.argmax(dim=-1)
                best_motion_reg = motion_reg[
                    torch.arange(bs)[:, None],
                    torch.arange(num_anchor)[None, :],
                    best_motion_cls,
                ]
                motion_emb = self.motion_embedder(best_motion_reg.flatten(-2))
                _, (motion_emb_selected, ) = topk(
                    det_confidence, self.num_det, motion_emb
                )
                plan_emb = self.plan_embedder(ref_plan_anchor.flatten(-2))
                plan_query = self.graph_model(
                    i,
                    plan_query,
                    instance_feature_selected[:, :-1],
                    instance_feature_selected[:, :-1],
                    query_pos=plan_emb,
                    key_pos=motion_emb_selected,
                )
            elif op == "cond_cross_attn":
                plan_query = self.layers[i](
                    plan_query,
                    key=cond_embedding,
                )
            elif op == "lat_cond_cross_attn":
                lat_mode_query = self.layers[i](
                    lat_mode_query,
                    key=cond_embedding,
                )
            elif op == "lat_agent_gnn":
                lat_mode_query = self.layers[i](
                    lat_mode_query,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=lat_anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "lat_map_gnn":
                lat_mode_query = self.layers[i](
                    lat_mode_query,
                    map_instance_feature_selected,
                    map_instance_feature_selected,
                    query_pos=lat_anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "lon_cond_cross_attn":
                lon_mode_query = self.layers[i](
                    lon_mode_query,
                    key=cond_embedding,
                )
            elif op == "lon_agent_gnn":
                lon_mode_query = self.layers[i](
                    lon_mode_query,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=lon_anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "lon_map_gnn":
                lon_mode_query = self.layers[i](
                    lon_mode_query,
                    map_instance_feature_selected,
                    map_instance_feature_selected,
                    query_pos=lon_anchor_embed,
                    key_pos=map_anchor_embed_selected,
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
            elif op == "lat_deformable":
                lat_anchor_flat = lat_anchor.flatten(-2)
                lat_mode_query = self.layers[i](
                    lat_mode_query,
                    lat_anchor_flat,
                    None,
                    feature_maps,
                    metas,
                    None,
                )
            elif op == "lat_deformable_interp":
                lat_anchor_flat = lat_anchor.flatten(-2)
                lat_mode_query, kps_features = self.layers[i](
                    lat_mode_query,
                    lat_anchor_flat,
                    None,
                    feature_maps,
                    metas,
                    None,
                    return_kps_features=True,
                )
                kps_features = kps_features.unflatten(2, (self.lat_fut_ts, -1)).sum(3)
                time_points = torch.tensor(self.plan_config["lon"]["time_points"], device=kps_features.device)
                time_interval = time_points[:, 1] - time_points[:, 0]
                lon_dist = (self.lon_anchor * time_interval).cumsum(dim=-1)
                kps_features, kps_features_mask = interp_feature_lowmem(kps_features, lat_anchor[0], lon_dist)
                kps_features *= kps_features_mask.unsqueeze(-1)
                kps_features = kps_features.sum(dim=3)
                
            # elif op == "lon_deformable":
            #     lon_anchor = self.lon_anchor[None].repeat(bs, 1, 1, 1, 1)
            #     lon_plan_anchor_flat = lon_anchor.flatten(-2).flatten(1, 2)
            #     lon_plan_query = torch.repeat_interleave(plan_query, 45, dim=1)
            #     lon_plan_query = self.layers[i](
            #         lon_plan_query,
            #         lon_plan_anchor_flat,
            #         None,
            #         feature_maps,
            #         metas,
            #         None,
            #     )
            #     lon_plan_query = lon_plan_query.unflatten(1, (-1 ,45))
            #     if self.lon_cumsum:
            #         lon_plan_query = lon_plan_query.cumsum(-2)
            #         scale = torch.arange(1, 46).unsqueeze(-1).to(device)
            #         lon_plan_query /= scale
            elif op == "merge_lat":
                lon_mode_query = lat_mode_query.mean(dim=1, keepdim=True) + lon_mode_query
            elif op == "merge_lon":
                lat_mode_query = lon_mode_query.mean(dim=1, keepdim=True) + lat_mode_query
            elif op == "get_traj_query":
                traj_query = lat_mode_query.unsqueeze(2) + lon_mode_query.unsqueeze(1)
            elif op == "add_kps_feat":
                traj_query = traj_query + kps_features
            elif op == "plan_pred":
                plan_result = self.layers[i](
                    lat_mode_query,
                    lon_mode_query,
                    traj_query,
                    lat_anchor,
                    lon_anchor,
                    traj_anchor,
                    traj_anchor_mask,
                )
                planning_results.append(plan_result)
            elif op == "plan_refine":
                plan_result, plan_anchor = self.layers[i](
                    plan_query,
                    plan_anchor,
                    ego_feature,
                    ego_anchor_embed,
                )
                planning_refine_results.append(plan_result)
            elif op == "filter":
                bs_idx = torch.arange(bs, device=plan_query.device).view(-1, 1)  # (bs, 1)
                plan_cls = plan_result[f"{self.anchor_reference_group}_cls"]
                _, anchor_idx = torch.topk(plan_cls, k=self.ego_fut_mode_refine, dim=1)
                plan_query = plan_query[bs_idx, anchor_idx]
                for key, value in plan_anchor.items():
                    if value is not None:
                        plan_anchor[key] = value[bs_idx, anchor_idx]
            elif op == "topk":
                bs_idx = torch.arange(bs, device=traj_query.device).view(-1, 1)
                ## lat:
                lat_cls = plan_result["lat_cls"].squeeze(-1)
                _, lat_idx = torch.topk(lat_cls, k=self.lat_topk, dim=1)
                lat_mode_query = lat_mode_query[bs_idx, lat_idx]
                lat_anchor_embed = lat_anchor_embed[bs_idx, lat_idx]
                lat_anchor = lat_anchor[bs_idx, lat_idx]
                ## lon:
                lon_cls = plan_result["lon_cls"].squeeze(-1)
                _, lon_idx = torch.topk(lon_cls, k=self.lon_topk, dim=1)
                lon_mode_query = lon_mode_query[bs_idx, lon_idx]
                lon_anchor_embed = lon_anchor_embed[bs_idx, lon_idx]
                lon_anchor = lon_anchor[bs_idx, lon_idx]
                ## traj:
                traj_query = traj_query[bs_idx, lat_idx]  # (bs, lat_topk, num_lon)
                traj_query = torch.gather(traj_query, 2, lon_idx.unsqueeze(1).expand(-1, self.lat_topk, -1).unsqueeze(-1))
                traj_anchor = traj_anchor[bs_idx, lat_idx]  # (bs, lat_topk, num_lon, fut_ts, 2)
                traj_anchor = torch.gather(traj_anchor, 2, lon_idx.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(-1, self.lat_topk, -1, traj_anchor.size(-2), traj_anchor.size(-1)))
                traj_anchor_mask = traj_anchor_mask[bs_idx, lat_idx]  # (bs, lat_topk, num_lon, fut_ts)
                traj_anchor_mask = torch.gather(traj_anchor_mask, 2, lon_idx.unsqueeze(1).unsqueeze(-1).expand(-1, self.lat_topk, -1, traj_anchor_mask.size(-1)))

        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        status = plan_result.get("ego_status_reg")
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], status)

        def map_topk_attn_to_full(attn_weights, det_indices, fill_len, fill_value=0.0):
            """
            Args:
                attn_weights: [bs, 50] 或 [bs, 50, dim]
                det_indices: [bs, 50]，值为 0~899 的索引
                fill_value: 填充值，默认 0，也可设为 -inf
            Returns:
                full_attn: [bs, 900] 或 [bs, 900, dim]
            """
            if attn_weights is None:
                return None
            
            bs, k = det_indices.shape
            device = attn_weights.device
            dtype = attn_weights.dtype

            # 构造全量张量
            if attn_weights.dim() == 3:
                dim = attn_weights.shape[-1]
                full_attn = torch.full((bs, fill_len, dim), fill_value, dtype=dtype, device=device)
                full_attn.scatter_(1, det_indices.unsqueeze(-1).expand(-1, -1, dim), attn_weights)
            else:
                full_attn = torch.full((bs, fill_len), fill_value, dtype=dtype, device=device)
                full_attn.scatter_(1, det_indices, attn_weights)

            return full_attn

        det_attn_weights = map_topk_attn_to_full(det_attn_weight, det_indices, num_anchor, fill_value=0.0)
        map_attn_weights = map_topk_attn_to_full(map_attn_weight, map_indices, num_map_anchor, fill_value=0.0)

        det_output["attn_weight"] = det_attn_weights
        map_output["attn_weight"] = map_attn_weights

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }
        planning_output = {
            "planning_results": planning_results,
            "planning_refine_results": planning_refine_results,
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
        planning_refine_loss = self.loss_planning_refine(planning_model_outs, data)
        loss.update(planning_refine_loss)
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

        for decoder_idx, planning_result in enumerate(planning_results):
            (
                lat_cls, 
                lat_cls_target, 
                lat_cls_weight,
                lon_cls, 
                lon_cls_target, 
                lon_cls_weight,
                traj_cls,
                traj_cls_target,
                traj_cls_weight,
                col_cls,
                col_label,
            ) = self.planning_sampler.sample(
                planning_result,
                data,
                self.traj_anchor,
                self.traj_anchor_mask,
            )
            lat_cls_loss = self.lat_plan_loss_cls(lat_cls, lat_cls_target, lat_cls_weight) * self.decoder_weight[decoder_idx]
            output[f"lat_cls_loss_{decoder_idx}"] = lat_cls_loss 
            lon_cls_loss = self.lon_plan_loss_cls(lon_cls, lon_cls_target, lon_cls_weight) * self.decoder_weight[decoder_idx]
            output[f"lon_cls_loss_{decoder_idx}"] = lon_cls_loss
            traj_cls_loss = self.traj_plan_loss_cls(traj_cls, traj_cls_target, traj_cls_weight) * self.decoder_weight[decoder_idx]
            output[f"traj_cls_loss_{decoder_idx}"] = traj_cls_loss
            if col_cls is not None:
                col_cls_loss = self.col_plan_loss_cls(col_cls, col_label) * self.decoder_weight[decoder_idx]
                output[f"col_cls_loss_{decoder_idx}"] = col_cls_loss * self.plan_config["collision"]["weight"]

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning_refine(self, model_outs, data):
        planning_results = model_outs["planning_refine_results"]
        output = {}
        if len(planning_results) == 0:
            return output
        ## align matching
        if self.match_reference_group is not None:
            key = self.match_reference_group
            reg = planning_results[-1][key + "_" + "reg"]
            gt_reg = data[f"gt_{key}"]
            gt_mask = data[f"gt_{key}_mask"]
            mode_idx, dist = self.planning_sampler.get_mode_idx(reg, gt_reg, gt_mask)
        else:
            mode_idx = dist = None

        for key, value in self.plan_refine_config.items():
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
                        data,
                    )
                    cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight) * value["cls_weight"]
                    output[f"planning_refine_{key}_cls_loss_{decoder_idx}"] = cls_loss
                    if "reg" in value["pred_types"]:
                        reg_loss = self.plan_loss_reg(reg, reg_target, weight=reg_weight) * value["reg_weight"]
                        output[f"planning_refine_{key}_reg_loss_{decoder_idx}"] = reg_loss
                elif "target_speed" in key:
                    cls = planning_result[key + "_" + "cls"]
                    gt_cls = data[f"gt_{key}"]
                    bs = cls.shape[0]
                    bs_indices = torch.arange(bs, device=cls.device)
                    best_cls = cls[bs_indices, mode_idx].softmax(dim=-1)
                    cls_loss = F.binary_cross_entropy(
                        best_cls, gt_cls
                    )
                    output[f"planning_refine_{key}_cls_loss_{decoder_idx}"] = cls_loss * value["cls_weight"]
        
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
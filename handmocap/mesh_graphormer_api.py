# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import torch
import numpy as np
import cv2
import gc
from torchvision.transforms import transforms

from handmocap.hand_modules.mesh_graphormer_options import MeshGraphormerOptions
from handmocap.hand_modules.h3dw_model import H3DWModel
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm

from .hand_modules.MeshGraphormer.src.modeling.bert import BertConfig, Graphormer
from .hand_modules.MeshGraphormer.src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from .hand_modules.MeshGraphormer.src.modeling._mano import MANO, Mesh
from .hand_modules.MeshGraphormer.src.modeling.hrnet.config import config as hrnet_config
from .hand_modules.MeshGraphormer.src.modeling.hrnet.config import update_config as hrnet_update_config
from .hand_modules.MeshGraphormer.src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import mocap_utils.general_utils as gnu


transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_visualize = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])


class HandMocap:
    def __init__(self, regressor_checkpoint, smpl_dir, device=torch.device("cuda"), use_smplx=False):
        # For image transform
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.normalize_transform = transforms.Compose(transform_list)

        # Load Hand network
        self.opt = MeshGraphormerOptions().parse([])

        # Default options
        self.opt.single_branch = True
        self.opt.main_encoder = "resnet50"
        # self.opt.data_root = "/home/hjoo/dropbox/hand_yu/data/"
        self.opt.model_root = "./extra_data"
        self.opt.smplx_model_file = os.path.join(smpl_dir, "SMPLX_NEUTRAL.pkl")

        self.opt.batchSize = 1
        self.opt.phase = "test"
        self.opt.nThreads = 0
        self.opt.which_epoch = -1
        self.opt.checkpoint_path = regressor_checkpoint

        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.process_rank = -1

        # self.opt.which_epoch = str(epoch)
        # Load Mesh Graphormer model
        # Load pretrained model
        trans_encoder = []

        input_feat_dim = [int(item) for item in self.opt.input_feat_dim.split(",")]
        hidden_feat_dim = [int(item) for item in self.opt.hidden_feat_dim.split(",")]
        output_feat_dim = input_feat_dim[1:] + [3]

        # which encoder block to have graph convs
        which_blk_graph = [int(item) for item in self.opt.which_gcn.split(",")]

        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            config = config_class.from_pretrained(
                self.opt.config_name if self.opt.config_name else self.opt.model_name_or_path
            )

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]
            self.opt.hidden_size = hidden_feat_dim[i]
            self.opt.intermediate_size = int(self.opt.hidden_size * 2)

            if which_blk_graph[i] == 1:
                config.graph_conv = True
                # logger.info("Add Graph Conv")
            else:
                config.graph_conv = False

            config.mesh_type = self.opt.mesh_type

            # update model structure if specified in arguments
            update_params = ["num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size"]
            for idx, param in enumerate(update_params):
                arg_param = getattr(self.opt, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    # logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            # logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # create backbone model
        if self.opt.arch == "hrnet":
            "handmocap/hand_modules/MeshGraphormer/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
            hrnet_yaml = (
                "./handmocap/hand_modules/MeshGraphormer/models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
            )
            hrnet_checkpoint = (
                "./handmocap/hand_modules/MeshGraphormer/models/hrnet/hrnetv2_w40_imagenet_pretrained.pth"
            )
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            # logger.info('=> loading hrnet-v2-w40 model')
        elif self.opt.arch == "hrnet-w64":
            hrnet_yaml = (
                "./handmocap/hand_modules/MeshGraphormer/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
            )
            hrnet_checkpoint = (
                "./handmocap/hand_modules/MeshGraphormer/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
            )
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
            # logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(self.opt.arch))
            backbone = model.__dict__[self.opt.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        # logger.info('Graphormer encoders total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        # logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end Graphormer network (CNN backbone + multi-layer Graphormer encoder)
        self.model_regressor = Graphormer_Network(self.opt, config, backbone, trans_encoder)

        graphormer_model_path = (
            "./handmocap/hand_modules/MeshGraphormer/models/graphormer_release/graphormer_hand_state_dict.bin"
        )
        # for fine-tuning or resume training or inference, load weights from checkpoint
        # logger.info("Loading state dict from checkpoint {}".format(graphormer_model_path))
        # workaround approach to load sparse tensor in graph conv.
        state_dict = torch.load(graphormer_model_path)
        self.model_regressor.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        # update configs to enable attention outputs
        setattr(self.model_regressor.trans_encoder[-1].config, "output_attentions", True)
        setattr(self.model_regressor.trans_encoder[-1].config, "output_hidden_states", True)
        self.model_regressor.trans_encoder[-1].bert.encoder.output_attentions = True
        self.model_regressor.trans_encoder[-1].bert.encoder.output_hidden_states = True
        for iter_layer in range(4):
            self.model_regressor.trans_encoder[-1].bert.encoder.layer[
                iter_layer
            ].attention.self.output_attentions = True
        for inter_block in range(3):
            setattr(self.model_regressor.trans_encoder[-1].config, "device", device)

        self.model_regressor.to(device)

        self.mano_model = MANO().to(device)
        self.mano_model.layer = self.mano_model.layer.cuda()
        self.mesh_sampler = Mesh()

    def __pad_and_resize(self, img, hand_bbox, add_margin, final_size=224):
        """
        Box margin for mesh graphormer is 1.0
        """
        ori_height, ori_width = img.shape[:2]
        min_x, min_y = hand_bbox[:2].astype(np.int32)
        width, height = hand_bbox[2:].astype(np.int32)
        max_x = min_x + width
        max_y = min_y + height

        if width > height:
            margin = (width - height) // 2
            min_y = max(min_y - margin, 0)
            max_y = min(max_y + margin, ori_height)
        else:
            margin = (height - width) // 2
            min_x = max(min_x - margin, 0)
            max_x = min(max_x + margin, ori_width)

        # add additional margin
        if add_margin:
            margin = int(1.0 * (max_y - min_y))  # if use loose crop, change 0.3 to 1.0
            min_y = max(min_y - margin, 0)
            max_y = min(max_y + margin, ori_height)
            min_x = max(min_x - margin, 0)
            max_x = min(max_x + margin, ori_width)

        img_cropped = img[int(min_y) : int(max_y), int(min_x) : int(max_x), :]
        new_size = max(max_x - min_x, max_y - min_y)
        new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
        # new_img = np.zeros((new_size, new_size, 3))
        new_img[: (max_y - min_y), : (max_x - min_x), :] = img_cropped
        bbox_processed = (min_x, min_y, max_x, max_y)

        # resize to 224 * 224
        new_img = cv2.resize(new_img, (final_size, final_size))

        ratio = final_size / new_size
        return new_img, ratio, (min_x, min_y, max_x - min_x, max_y - min_y)

    def __process_hand_bbox(self, raw_image, hand_bbox, hand_type, add_margin=True):
        """
        args:
            original image,
            bbox: (x0, y0, w, h)
            hand_type ("left_hand" or "right_hand")
            add_margin: If the input hand bbox is a tight bbox, then set this value to True, else False
        output:
            img_cropped: 224x224 cropped image (original colorvalues 0-255)
            norm_img: 224x224 cropped image (normalized color values)
            bbox_scale_ratio: scale factor to convert from original to cropped
            bbox_top_left_origin: top_left corner point in original image cooridate
        """
        # print("hand_type", hand_type)

        assert hand_type in ["left_hand", "right_hand"]
        img_cropped, bbox_scale_ratio, bbox_processed = self.__pad_and_resize(raw_image, hand_bbox, add_margin)

        # horizontal Flip to make it as right hand
        if hand_type == "left_hand":
            img_cropped = np.ascontiguousarray(img_cropped[:, ::-1, :], img_cropped.dtype)
        else:
            assert hand_type == "right_hand"

        # img normalize
        norm_img = self.normalize_transform(img_cropped).float()
        # return
        return img_cropped, norm_img, bbox_scale_ratio, bbox_processed

    def regress(self, img_original, hand_bbox_list, add_margin=False):
        """
        args:
            img_original: original raw image (BGR order by using cv2.imread)
            hand_bbox_list: [
                dict(
                    left_hand = [x0, y0, w, h] or None
                    right_hand = [x0, y0, w, h] or None
                )
                ...
            ]
            add_margin: whether to do add_margin given the hand bbox
        outputs:
            To be filled
        Note:
            Output element can be None. This is to keep the same output size with input bbox
        """
        pred_output_list = list()
        hand_bbox_list_processed = list()
        # load smplx-hand faces
        hand_info_file = osp.join(self.opt.model_root, self.opt.smplx_hand_info_file)

        self.hand_info = gnu.load_pkl(hand_info_file)
        self.right_hand_faces_holistic = self.hand_info["right_hand_faces_holistic"]
        self.right_hand_faces_local = self.hand_info["right_hand_faces_local"]

        for hand_bboxes in hand_bbox_list:

            if hand_bboxes is None:  # Should keep the same size with bbox size
                pred_output_list.append(None)
                hand_bbox_list_processed.append(None)
                continue

            pred_output = dict(left_hand=None, right_hand=None)
            hand_bboxes_processed = dict(left_hand=None, right_hand=None)

            for hand_type in hand_bboxes:
                bbox = hand_bboxes[hand_type]

                if bbox is None:
                    continue
                else:
                    img_cropped, norm_img, bbox_scale_ratio, bbox_processed = self.__process_hand_bbox(
                        img_original, hand_bboxes[hand_type], hand_type, add_margin
                    )
                    hand_bboxes_processed[hand_type] = bbox_processed

                    # img_tensor = transform(to_pil_image(norm_img))
                    img_tensor = transform(to_pil_image(img_cropped))

                    batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
                    ### 여기에 모델 인퍼런스 입력하면 될듯
                    ## 입력 224x224 이미지
                    with torch.no_grad():
                        # pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))
                        regress_result = self.model_regressor(batch_imgs, self.mano_model, self.mesh_sampler)
                        if len(regress_result) == 4:
                            cam, pred_joints, pred_verticies_sub, pred_verts_origin = regress_result
                        else:
                            cam, pred_joints, pred_verticies_sub, pred_verts_origin, _, __ = regress_result

                        pred_verticies_sub = torch.squeeze(pred_verticies_sub, dim=0)
                        pred_verts_origin = torch.squeeze(pred_verts_origin, dim=0)
                        pred_joints = torch.squeeze(pred_joints, dim=0)
                        faces = self.right_hand_faces_local

                        if hand_type == "left_hand":
                            cam[1] *= -1
                            faces = faces[:, ::-1]
                            pred_verts_origin[:, 0] *= -1
                            pred_joints[:, 0] *= -1

                        pred_output[hand_type] = dict()
                        pred_output[hand_type][
                            "pred_vertices_smpl"
                        ] = pred_verts_origin  # SMPL-X hand vertex in bbox space
                        pred_output[hand_type]["pred_joints_smpl"] = pred_joints
                        pred_output[hand_type]["faces"] = faces

                        pred_output[hand_type]["bbox_scale_ratio"] = bbox_scale_ratio
                        pred_output[hand_type]["bbox_top_left"] = np.array(bbox_processed[:2])
                        pred_output[hand_type]["pred_camera"] = cam
                        pred_output[hand_type]["img_cropped"] = img_cropped

                        # Convert vertices into bbox & image space
                        cam_scale = cam[0]
                        cam_trans = cam[1:]
                        vert_smplcoord = pred_verts_origin.clone()
                        joints_smplcoord = pred_joints.clone()

                        vert_bboxcoord = convert_smpl_to_bbox(
                            vert_smplcoord, cam_scale, cam_trans, bAppTransFirst=True
                        )  # SMPL space -> bbox space
                        joints_bboxcoord = convert_smpl_to_bbox(
                            joints_smplcoord, cam_scale, cam_trans, bAppTransFirst=True
                        )  # SMPL space -> bbox space

                        hand_boxScale_o2n = pred_output[hand_type]["bbox_scale_ratio"]
                        hand_bboxTopLeft = pred_output[hand_type]["bbox_top_left"]

                        vert_imgcoord = convert_bbox_to_oriIm(
                            vert_bboxcoord,
                            hand_boxScale_o2n,
                            hand_bboxTopLeft,
                            img_original.shape[1],
                            img_original.shape[0],
                        )
                        pred_output[hand_type]["pred_vertices_img"] = vert_imgcoord

                        joints_imgcoord = convert_bbox_to_oriIm(
                            joints_bboxcoord,
                            hand_boxScale_o2n,
                            hand_bboxTopLeft,
                            img_original.shape[1],
                            img_original.shape[0],
                        )
                        pred_output[hand_type]["pred_joints_img"] = joints_imgcoord

            pred_output_list.append(pred_output)
            hand_bbox_list_processed.append(hand_bboxes_processed)

        assert len(hand_bbox_list_processed) == len(hand_bbox_list)
        return pred_output_list

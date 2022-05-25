# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import torch
import numpy as np
import cv2
import gc
from torchvision.transforms import transforms
from bodymocap.body_bbox_detector import BodyPoseEstimator
from bodymocap.utils.imutils import process_image_bbox

from handmocap.hand_modules.test_options import TestOptions
from handmocap.hand_modules.h3dw_model import H3DWModel
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import mocap_utils.general_utils as gnu

from .hand_modules.Hand4Whole_RELEASE.main.model import get_model
from torch.nn.parallel.data_parallel import DataParallel
import smplx
from .hand_modules.Hand4Whole_RELEASE.main import config as cfg
from .hand_modules.Hand4Whole_RELEASE.common.utils.preprocessing import generate_patch_image

def extract_hand_output(output, hand_type, hand_info, top_finger_joints_type='ave', use_cuda=True):
    assert hand_type in ['left', 'right']

    if hand_type == 'left':
        wrist_idx, hand_start_idx, middle_finger_idx = 20, 25, 28
    else:
        wrist_idx, hand_start_idx, middle_finger_idx = 21, 40, 43

    vertices = output.vertices
    joints = output.joints
    vertices_shift = vertices - joints[:, hand_start_idx:hand_start_idx+1, :]

    hand_verts_idx = torch.Tensor(hand_info[f'{hand_type}_hand_verts_idx']).long()
    if use_cuda:
        hand_verts_idx = hand_verts_idx.cuda()

    hand_verts = vertices[:, hand_verts_idx, :]
    hand_verts_shift = hand_verts - joints[:, hand_start_idx:hand_start_idx+1, :]

   # Hand joints
    if hand_type == 'left':
        hand_idxs =  [20] + list(range(25,40)) + list(range(66, 71)) # 20 for left wrist. 20 finger joints
    else:
        hand_idxs = [21] + list(range(40,55)) + list(range(71, 76)) # 21 for right wrist. 20 finger joints
    smplx_hand_to_panoptic = [0, 13,14,15,16, 1,2,3,17, 4,5,6,18, 10,11,12,19, 7,8,9,20] 
    hand_joints = joints[:, hand_idxs, :][:, smplx_hand_to_panoptic, :]
    hand_joints_shift = hand_joints - joints[:, hand_start_idx:hand_start_idx+1, :]

    output = dict(
        wrist_idx = wrist_idx,
        hand_start_idx = hand_start_idx,
        middle_finger_idx = middle_finger_idx,
        vertices_shift = vertices_shift,
        hand_vertices = hand_verts,
        hand_vertices_shift = hand_verts_shift,
        hand_joints = hand_joints,
        hand_joints_shift = hand_joints_shift
    )
    return output

class HandMocap:
    def __init__(self, regressor_checkpoint, smpl_dir, device=torch.device("cuda"), use_smplx=False):
        # For image transform
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.normalize_transform = transforms.Compose(transform_list)

        # Load Hand network
        self.opt = TestOptions().parse([])

        # Default options
        self.opt.single_branch = True
        self.opt.main_encoder = "resnet50"
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

        # Get cfg for hands4whole model
        self.cfg = cfg
        self.cfg.set_args(self.opt.gpu_ids)
        # Get body bbox detector
        self.bbox_detector = BodyPoseEstimator()

        # Get hand4whole model
        # snapshot load
        model_path = './snapshot_6.pth.tar'
        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        print('Load checkpoint from {}'.format(model_path))
        self.model_regressor = get_model('test')
        self.model_regressor = DataParallel(self.model_regressor).cuda()
        ckpt = torch.load(model_path)
        self.model_regressor.load_state_dict(ckpt['network'], strict=False)
        self.model_regressor.eval()

         # load smplx-hand faces
        hand_info_file = osp.join(self.opt.model_root, self.opt.smplx_hand_info_file)

        self.hand_info = gnu.load_pkl(hand_info_file)
        self.right_hand_faces_holistic = self.hand_info["right_hand_faces_holistic"]
        self.right_hand_faces_local = self.hand_info["right_hand_faces_local"]
        
        smplx_model_path = self.opt.smplx_model_file
        self.smplx = smplx.create(smplx_model_path, 
            model_type = "smplx", 
            batch_size = self.batch_size,
            gender = 'neutral',
            num_betas = 10,
            use_pca = False,
            ext='pkl').cuda()

    def get_smplx_output(self, pose_params, shape_params=None):
        hand_rotation = pose_params[:, :3]
        hand_pose = pose_params[:, 3:]
        body_pose = torch.zeros((self.batch_size, 63)).float().cuda() 
        body_pose[:, 60:] = hand_rotation # set right hand rotation

        output = self.smplx(
            global_orient = self.global_orient,
            body_pose = body_pose,
            right_hand_pose = hand_pose,
            betas = shape_params,
            return_verts = True)
        
        hand_output = extract_hand_output(
            output, 
            hand_type = 'right', 
            hand_info = self.hand_info,
            top_finger_joints_type = self.top_finger_joints_type, 
            use_cuda=True)

        pred_verts = hand_output['vertices_shift']
        pred_joints_3d = hand_output['hand_joints_shift']
        return pred_verts, pred_joints_3d

    def __pad_and_resize(self, img, hand_bbox, add_margin, final_size=224):
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
        
        with torch.no_grad():
            out = self.model_regressor(inputs, targets, meta_info, "test")

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

                    ### 여기에 모델 인퍼런스 입력하면 될듯
                    ## 입력 Whole Body image
                    ## 출력: smplx whole body -> 각 손별로 convert_smpl to bbox해야할듯
                    with torch.no_grad():
                        # pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))
                        cam = out["cam_param"]
                        faces = self.right_hand_faces_local
                        pred_verts_origin = 


                        if hand_type == "left_hand":
                            cam[1] *= -1
                            faces = faces[:, ::-1]
                            pred_verts_origin[:, 0] *= -1

                        pred_output[hand_type] = dict()
                        pred_output[hand_type][
                            "pred_vertices_smpl"
                        ] = pred_verts_origin  # SMPL-X hand vertex in bbox space
                        if hand_type =="left_hand":
                            pred_output[hand_type]["pred_joints_smpl"] = out["smplx_lhand_pose"]
                        else:
                            pred_output[hand_type]["pred_joints_smpl"] = out["smplx_rhand_pose"] 
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

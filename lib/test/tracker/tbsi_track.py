import math

from lib.models.tbsi_track import build_tbsi_track
from lib.test.tracker.tta import pura, tent, eata, adabn
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.grad_cam import reshape_transform


class TBSITrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TBSITrack, self).__init__(params)
        network = build_tbsi_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()

        # NOTE: PURA
        pura.replace_batchnorm(self.network.box_head)
        pura.configure_model(self.network)

        # NOTE: AdaBN
        # adabn.replace_batchnorm(self.network.box_head)
        # adabn.configure_model(self.network)

        # NOTE: Tent
        # model = tent.configure_model(self.network)
        # tta_params, tta_param_names = tent.collect_params(model)
        # optimizer = torch.optim.AdamW(tta_params, lr=1e-3)
        # self.network = tent.Tent(model, optimizer)

        # NOTE: ETA
        # model = eata.configure_model(self.network)
        # tta_params, tta_param_names = eata.collect_params(model.box_head)
        # optimizer = torch.optim.SGD(tta_params, lr=0.00025, momentum=0.9)
        # self.network = eata.EATA(model, optimizer, e_margin=math.log(1000)*0.40, d_margin=0.05)

        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = None
        self.grad_cam_root = '/data/zzf/proj/TBSI/output/grad_cam/'

        # update zzf 模板更新超参数
        self.online_template = None
        self.max_pred_score = -1
        self.max_pred_score_tir = -1

        self.online_max_template = None
        self.online_max_template_rgb = None
        self.online_max_template_tir = None
        self.update_id = 0

        self.max_score_decay = 0.999
        self.update_interval = 30
        self.threshold = 0.9

        self.rgb_id = 0
        self.tir_id = 0
        self.rgb_max_id = 0
        self.tir_max_id = 0

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        # from PIL import Image
        # image = Image.fromarray(z_patch_arr[:, :, :3])
        # image.save("/data/zzf/proj/TBSI/{:2d}.png".format(12))

        # image = Image.fromarray(z_patch_arr[:, :, 3:])
        # image.save("/data/zzf/proj/TBSI/{:2d}.png".format(13))

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template.tensors
            self.online_template = template.tensors.clone()

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        # self.online_template #1,6,128,128

        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():  # enable_grad  no_grad
            x_dict = search
            # merge the template and the search
            # run the transformer
            a1 = self.z_dict1[:, :3, :, :].unsqueeze(0)
            a2 = self.online_template[:, :3, :, :].unsqueeze(0)
            b1 = self.z_dict1[:, 3:, :, :].unsqueeze(0)
            b2 = self.online_template[:, 3:, :, :].unsqueeze(0)
            cur_template = [torch.cat([a1, a2], dim=0), torch.cat([b1, b2], dim=0)]

            # NOTE: Uncomment this block when using Tent or ETA
            # model_inputs = {
            #     "template": cur_template,
            #     "search": [x_dict.tensors[:, :3, :, :], x_dict.tensors[:, 3:, :, :]],
            #     "ce_template_mask": self.box_mask_z
            # }
            # out_dict = self.network(model_inputs)

            # NOTE: Uncomment this block when using PURA or AdaBN
            out_dict = self.network.forward(
                template=cur_template,
                search=[x_dict.tensors[:, :3, :, :], x_dict.tensors[:, 3:, :, :]], ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if 1:
            # 更新模板
            cls_score = out_dict['cls_score'].view(1).sigmoid().item()
            cls_score_tir = out_dict['cls_score_tir'].view(1).sigmoid().item()

            self.max_pred_score = self.max_pred_score * self.max_score_decay
            self.max_pred_score_tir = self.max_pred_score_tir * self.max_score_decay
            if cls_score > self.threshold and cls_score > self.max_pred_score:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                            self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                self.online_max_template_rgb = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors
                self.max_pred_score = cls_score
                self.rgb_max_id = self.frame_id
                # if self.frame_id % self.update_interval == 0:
                #     self.online_template[:, :3, ...] = self.online_max_template_rgb[:, :3, ...]
                #     self.rgb_max_id = self.frame_id
                #     self.max_pred_score = -1
            if cls_score_tir > self.threshold and cls_score_tir > self.max_pred_score_tir:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                            self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                self.online_max_template_tir = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors
                self.max_pred_score_tir = cls_score_tir
                self.tir_max_id = self.frame_id
                # if self.frame_id % self.update_interval == 0:
                #     self.tir_max_id = self.frame_id
                #     self.online_template[:, 3:, ...] = self.online_max_template_tir[:, 3:, ...]
                #     self.max_pred_score_tir = -1
            if self.frame_id % self.update_interval == 0:
                if self.max_pred_score > 0:
                    self.online_template[:, :3, ...] = self.online_max_template_rgb[:, :3, ...]
                    self.max_pred_score = -1
                    self.rgb_id = self.rgb_max_id
                if self.max_pred_score_tir > 0:
                    self.online_template[:, 3:, ...] = self.online_max_template_tir[:, 3:, ...]
                    self.max_pred_score_tir = -1
                    self.tir_id = self.tir_max_id

        # from PIL import Image
        # image = Image.fromarray(z_patch_arr[:, :, 3:])
        # image.save("/data/zzf/proj/TBSI/{:d}.png".format(self.frame_id + 1))

        # image = Image.fromarray(z_patch_arr[:, :, :3])
        # image.save("/data/zzf/proj/TBSI/{:d}_00.png".format(self.frame_id + 1))

        # print('+++++++++++++++++++++++', cls_score, self.update_id)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image[:, :, :3], info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register((image[:, :, 3:], info['gt_bbox'].tolist(), self.state), 'Tracking', 1,
                                     'TrackingI')

                self.visdom.register(torch.from_numpy(x_patch_arr[:, :, :3]).permute(2, 0, 1), 'image', 1,
                                     'search_region')
                self.visdom.register(torch.from_numpy(x_patch_arr[:, :, 3:]).permute(2, 0, 1), 'image', 1,
                                     'search_regionI')
                self.visdom.register(torch.from_numpy(self.z_patch_arr[:, :, :3]).permute(2, 0, 1), 'image', 1,
                                     'template')
                self.visdom.register(torch.from_numpy(self.z_patch_arr[:, :, 3:]).permute(2, 0, 1), 'image', 1,
                                     'templateI')

                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return TBSITrack

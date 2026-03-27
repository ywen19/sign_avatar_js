import os
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
#from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--multi_person', action='store_true')
    args = parser.parse_args()
    return args

def to_numpy_dict(d):
    out_dict = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out_dict[k] = v.detach().cpu().numpy()
        else:
            out_dict[k] = v
    return out_dict

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    # output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    # os.makedirs(output_folder, exist_ok=True)
    output_folder = osp.join(root_dir, 'demo', 'output_smplx', args.file_name)
    os.makedirs(output_folder, exist_ok=True)
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()

    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path",
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    # accumulate one person across all frames
    frame_ids = []
    global_orient_list = []
    body_pose_list = []
    left_hand_pose_list = []
    right_hand_pose_list = []
    jaw_pose_list = []
    betas_list = []
    expression_list = []

    for frame in tqdm(range(start, end)):

        # prepare input image
        img_path = osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]

        # detection, xyxy
        yolo_bbox = detector.predict(
                                original_img,
                                device='cuda',
                                classes=0,
                                conf=cfg.inference.detection.conf,
                                save=cfg.inference.detection.save,
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox) < 1:
            continue
        elif not args.multi_person:
            num_bbox = 1
        else:
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
            yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
            yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
            yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])

            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh,
                                img_width=original_img_width,
                                img_height=original_img_height,
                                input_img_shape=cfg.model.input_img_shape,
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))
            img, _, _ = generate_patch_image(cvimg=original_img,
                                                bbox=bbox,
                                                scale=1.0,
                                                rot=0.0,
                                                do_flip=False,
                                                out_shape=cfg.model.input_img_shape)

            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            if frame == start and bbox_id == 0:
                print("OUT KEYS:", sorted(out.keys()))
                for k, v in out.items():
                    if torch.is_tensor(v):
                        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
                    else:
                        print(f"{k}: type={type(v)}")

            # keep only one person track
            frame_ids.append(np.int32(frame))
            global_orient_list.append(out['smplx_root_pose'].detach().cpu().numpy()[0].astype(np.float32))
            body_pose_list.append(out['smplx_body_pose'].detach().cpu().numpy()[0].astype(np.float32))
            left_hand_pose_list.append(out['smplx_lhand_pose'].detach().cpu().numpy()[0].astype(np.float32))
            right_hand_pose_list.append(out['smplx_rhand_pose'].detach().cpu().numpy()[0].astype(np.float32))
            jaw_pose_list.append(out['smplx_jaw_pose'].detach().cpu().numpy()[0].astype(np.float32))
            betas_list.append(out['smplx_shape'].detach().cpu().numpy()[0].astype(np.float32))
            expression_list.append(out['smplx_expr'].detach().cpu().numpy()[0].astype(np.float32))

            break  # only save one person per frame

        # save rendered image
        # frame_name = os.path.basename(img_path)
        # cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])

    save_path = os.path.join(
        output_folder,
        f"{args.file_name}_person00_smplx_pose.npz"
    )

    np.savez_compressed(
        save_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        person_id=np.int32(0),
        global_orient=np.stack(global_orient_list, axis=0),
        body_pose=np.stack(body_pose_list, axis=0),
        left_hand_pose=np.stack(left_hand_pose_list, axis=0),
        right_hand_pose=np.stack(right_hand_pose_list, axis=0),
        jaw_pose=np.stack(jaw_pose_list, axis=0),
        betas=np.stack(betas_list, axis=0),
        expression=np.stack(expression_list, axis=0),
    )


if __name__ == "__main__":
    main()
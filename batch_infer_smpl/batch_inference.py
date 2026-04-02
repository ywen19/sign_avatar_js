import argparse
from pathlib import Path
import json
import cv2
import subprocess
import shutil
import time
import sys
from tqdm import tqdm
import datetime
import os
import os.path as osp

import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from main.base import Tester
from main.config import Config
from human_models.human_models import SMPLX
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.inference_utils import non_max_suppression


def get_all_subfolder_names(vocab_root):
    vocab_root = Path(vocab_root)
    return sorted([p.name for p in vocab_root.iterdir() if p.is_dir()])


def get_all_files_in_subfolder(vocab_root, subfolder_name):
    subfolder_path = Path(vocab_root) / subfolder_name
    return sorted([p.name for p in subfolder_path.iterdir() if p.is_file()])


def get_video_fps(video_path, default_fps=30.0):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return default_fps

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 1e-6:
        return default_fps

    return fps


def convert_video_to_frames(
    vocab_root, subfolder_name, file_name, frame_output_dir, fps=30.0
):
    if fps is None or fps <= 0:
        raise RuntimeError(f"Invalid FPS for {file_name}: {fps}")

    full_file_path = Path(vocab_root) / subfolder_name / file_name
    frame_output_dir = Path(frame_output_dir)

    reset_frame_output_dir(frame_output_dir)
    output_pattern = frame_output_dir / "%06d.jpg"

    cmd = [
        "ffmpeg",
        "-i", str(full_file_path),
        "-f", "image2",
        "-vf", f"fps={fps}/1",
        "-qscale", "0",
        str(output_pattern),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed for {full_file_path}: {e}")

    extracted_frames = sorted(frame_output_dir.glob("*.jpg"))
    if len(extracted_frames) == 0:
        raise RuntimeError(f"No frames extracted for {full_file_path}")


def reset_frame_output_dir(frame_output_dir):
    frame_output_dir = Path(frame_output_dir)
    if frame_output_dir.exists():
        shutil.rmtree(frame_output_dir)
    frame_output_dir.mkdir(parents=True, exist_ok=True)


def extract_smpl_data(cfg, detector, demoer, file_name, frame_output_dir, output_npz_path, start=1, end=1):
    # print("into extract smpl data function ... \n")
    # accumulate one person across all frames
    frame_ids = []
    global_orient_list = []
    body_pose_list = []
    left_hand_pose_list = []
    right_hand_pose_list = []
    jaw_pose_list = []
    betas_list = []
    expression_list = []

    demoer.logger.info(f'Inference [{file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # step through each frame
    for frame in tqdm(range(start, end+1)):
        # prepare input frame image
        img_path = osp.join(frame_output_dir, f'{int(frame):06d}.jpg')
        # image preprocess
        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]

        # bbox detection from yolo
        yolo_bbox = detector.predict(
            original_img,
            device='cuda',
            classes=0,
            conf=cfg.inference.detection.conf,
            save=cfg.inference.detection.save,
            verbose=cfg.inference.detection.verbose
        )[0].boxes.xyxy.detach().cpu().numpy()
        # print(f"detected yolo box is: {yolo_bbox} \n")

        # check if bbox detected or if bbox exist
        if len(yolo_bbox) < 1:
            print("no bbox exists \n")
            continue
        # just track the first person
        # don't suppose for our videos there exist multi-people cases
        bbox_id = 0
        # xywh
        yolo_bbox_xywh = np.zeros((4))
        yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
        yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
        yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
        yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
        # print(f"bbox info: {yolo_bbox_xywh} \n")

        # get the image patch for the person of the associated bbox
        bbox = process_bbox(
            bbox=yolo_bbox_xywh,
            img_width=original_img_width,
            img_height=original_img_height,
            input_img_shape=cfg.model.input_img_shape,
            ratio=getattr(cfg.data, "bbox_ratio", 1.25
        ))
        # print("finished process the bbox \n")
        # print(bbox)
        img, _, _ = generate_patch_image(
            cvimg=original_img,
            bbox=bbox,
            scale=1.0,
            rot=0.0,
            do_flip=False,
            out_shape=cfg.model.input_img_shape
        )
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        # print("image preprared for inference \n")
        # print(inputs)
        targets = {}
        meta_info = {}

        # inference
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')
        # print(out)
        
        frame_ids.append(np.int32(frame))
        global_orient_list.append(out['smplx_root_pose'].detach().cpu().numpy()[0].astype(np.float32))
        body_pose_list.append(out['smplx_body_pose'].detach().cpu().numpy()[0].astype(np.float32))
        left_hand_pose_list.append(out['smplx_lhand_pose'].detach().cpu().numpy()[0].astype(np.float32))
        right_hand_pose_list.append(out['smplx_rhand_pose'].detach().cpu().numpy()[0].astype(np.float32))
        jaw_pose_list.append(out['smplx_jaw_pose'].detach().cpu().numpy()[0].astype(np.float32))
        betas_list.append(out['smplx_shape'].detach().cpu().numpy()[0].astype(np.float32))
        expression_list.append(out['smplx_expr'].detach().cpu().numpy()[0].astype(np.float32))

    # save data to npz
    np.savez_compressed(
        output_npz_path,
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


def process_file(
    cfg, detector, demoer, vocab_root, subfolder_name, file_name, frame_output_dir, output_npz_path
):
    full_file_path = Path(vocab_root) / subfolder_name / file_name
    print(f"  Processing file: {full_file_path}")

    if not full_file_path.exists():
        raise RuntimeError(f"Missing file: {full_file_path}")

    fps = get_video_fps(full_file_path, default_fps=30.0)
    print(f"    FPS: {fps}")

    try:
        convert_video_to_frames(
            vocab_root=vocab_root,
            subfolder_name=subfolder_name,
            file_name=file_name,
            frame_output_dir=frame_output_dir,
            fps=fps,
        )
        print(f"Frames written to: {frame_output_dir} \n")
        end = len(sorted(Path(frame_output_dir).glob("*.jpg")))
        print(f"End frame is {end} \n")
    except Exception:
        reset_frame_output_dir(frame_output_dir)
        raise

    time.sleep(0.2)
    extract_smpl_data(cfg, detector, demoer, file_name, frame_output_dir, output_npz_path, end=end)
    reset_frame_output_dir(frame_output_dir)


def append_progress(progress_file, subfolder_name, processed_files):
    with open(progress_file, "a", encoding="utf-8") as f:
        json.dump({subfolder_name: processed_files}, f)
        f.write("\n")
        f.flush()


def append_missing(missing_file, subfolder_name, missing_files):
    with open(missing_file, "a", encoding="utf-8") as f:
        json.dump({subfolder_name: missing_files}, f)
        f.write("\n")
        f.flush()


def get_resume_info(progress_file):
    progress_path = Path(progress_file)
    if not progress_path.exists():
        return set(), None, None

    records = []
    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return set(), None, None

    completed_subfolders = set()
    for record in records[:-1]:
        key = next(iter(record))
        completed_subfolders.add(key)

    last_record = records[-1]
    last_subfolder = next(iter(last_record))
    last_processed_files = last_record[last_subfolder]

    return completed_subfolders, last_subfolder, last_processed_files


def process_subfolder(
    cfg, detector, demoer, vocab_root, progress_file, missing_file, subfolder_name, frame_output_dir,
    subfolder_smpl_root
):
    print(f"Processing: {subfolder_name}")
    all_files = get_all_files_in_subfolder(vocab_root, subfolder_name)
    processed_files = []
    missing_files = []

    for file_name in all_files:
        # first get the npz file path to store extracted smpl data
        output_npz_path = Path(subfolder_smpl_root) / f"{Path(file_name).stem}.npz"
        print(f"Out npz path is {output_npz_path} \n")
        # actual process the smpl data inference
        try:
            process_file(
                cfg, detector, demoer, vocab_root, subfolder_name, file_name, frame_output_dir,
                output_npz_path
            )
            processed_files.append(file_name)
        except Exception:
            missing_files.append(file_name)

    append_progress(progress_file, subfolder_name, processed_files)

    if missing_files:
        append_missing(missing_file, subfolder_name, missing_files)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab_root", type=str, default="./demo/vocabs"
    )
    parser.add_argument(
        "--progress_file", type=str, default="./progress.jsonl"
    )
    parser.add_argument(
        "--missing_file", type=str, default="./missing.jsonl"
    )
    parser.add_argument(
        "--frame_output_dir", type=str, default="/media/ywen/DATA800/tmp_frames"
    )
    parser.add_argument(
        "--smpl_output_dir", type=str, default="/media/ywen/DATA800/smpl_data"
    )
    parser.add_argument(
        "--ckpt_name", type=str, default="smplest_x_h"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # output root folder for frame extraction from input videos
    output_root = Path(args.frame_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # output root folder for storing all extracted smpl data
    smpl_outpot_root = Path(args.smpl_output_dir)
    smpl_outpot_root.mkdir(parents=True, exist_ok=True)

    all_vocabs = get_all_subfolder_names(args.vocab_root)
    print(f"We have in total {len(all_vocabs)} vocab folders to be prcossed \n")
    print("printing first 20 as examples: \n")
    print(all_vocabs[:20])

    # init smplx configuration for inference
    ckpt_name = args.ckpt_name
    cudnn.benchmark = True
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', ckpt_name, 'config_base.py')
    exp_name = f'inference_{ckpt_name}_{time_str}'

    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join(
        './pretrained_models', ckpt_name, f'{ckpt_name}.pth.tar'
    )
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

    # init yolo detector
    bbox_model = getattr(cfg.inference.detection, "model_path",
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)
    print(detector)
    print("detector loaded \n")

    # get resume information
    completed_subfolders, last_subfolder, last_processed_files = get_resume_info(args.progress_file)

    for subfolder_name in all_vocabs[:3]:
        # skip if the subfolder has been processed
        if subfolder_name in completed_subfolders:
            print(f"Skipping completed subfolder: {subfolder_name}")
            continue
        # skip the last tracked subfolder only if there exists files that haven't been 
        # recorded as completed; here is no such file exist, skip to the next subfolder
        if subfolder_name == last_subfolder:
            all_files = get_all_files_in_subfolder(args.vocab_root, subfolder_name)
            if all_files == last_processed_files:
                print(f"Skipping completed last subfolder: {subfolder_name}")
                continue
        
        # create directory for storing output for the current subfolder
        subfolder_smpl_root = Path(smpl_outpot_root) / subfolder_name
        subfolder_smpl_root.mkdir(parents=True, exist_ok=True)

        process_subfolder(
            cfg,
            detector, 
            demoer,
            args.vocab_root,
            args.progress_file,
            args.missing_file,
            subfolder_name,
            output_root, # for frame extraction
            subfolder_smpl_root, # for smpl extraction
        )


if __name__ == "__main__":
    main()
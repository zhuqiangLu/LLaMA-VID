import argparse
import torch

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from collections import defaultdict
import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from PIL import Image
import imageio
from torchvision.transforms.functional import InterpolationMode
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
import cv2

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        # for frame_index in frame_indices:
        #     img = Image.fromarray(vr[frame_index].asnumpy())
        #     images_group.append(img)
        spare_frames = vr.get_batch(frame_indices).asnumpy()
         
        # torch_imgs = self.transform(spare_frames)
        return spare_frames
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {

            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }







def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    # parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    # parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--model-max-length", type=int, default=None)

    return parser.parse_args()


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames




def run_mvbench_inference(args, mvbench):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)

    results = defaultdict(list)
    for sample in tqdm(mvbench):
        video = sample['video']
        video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
        video = [video]
        question = sample['question']
        answer = sample['answer']
        task_type = sample['task_type']

        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt) 
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        cur_prompt = question
        with torch.inference_mode():
            model.update_prompt([[cur_prompt]])
            output_ids = model.generate(
                input_ids,
                images=video,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        results[task_type].append(dict(
            question = question,
            gt = answer,
            pred = outputs,
        ))

    return results

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

if __name__ == "__main__":

    args = parse_args()
    print(args)
    # run_inference(args)
    data_root = "/nvme/luzhuiang/MVBench/video"
    data_list = {
        "Action Sequence": ("action_sequence.json", f"{data_root}/star/Charades_v1_480/", "video", True), # has start & end
        "Action Prediction": ("action_prediction.json", f"{data_root}/star/Charades_v1_480/", "video", True), # has start & end
        # "Action Antonym": ("action_antonym.json", f"{data_root}/ssv2_video/", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", f"{data_root}/Moments_in_Time_Raw/videos/", "video", False),
        # "Unexpected Action": ("unexpected_action.json", f"{data_root}/FunQA_test/test/", "video", False),
        # "Object Existence": ("object_existence.json", f"{data_root}/clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", f"{data_root}/star/Charades_v1_480/", "video", True), # has start & end
        # "Object Shuffle": ("object_shuffle.json", f"{data_root}/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", f"{data_root}/clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", f"{data_root}/sta/sta_video/", "video", True),  # has start & end
        "Scene Transition": ("scene_transition.json", f"{data_root}/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", f"{data_root}/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", f"{data_root}/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", f"{data_root}/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", f"{data_root}/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", f"{data_root}/nturgbd/", "video", False),
        # "Character Order": ("character_order.json", f"{data_root}/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", f"{data_root}/vlnqa/", "video", False),
        # "Episodic Reasoning": ("episodic_reasoning.json", f"{data_root}/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
        # "Counterfactual Inference": ("counterfactual_inference.json", f"{data_root}/clevrer/video_validation/", "video", False),
    }

    data_dir = "/nvme/luzhuiang/MVBench/json"
    num_frame = 16
    resolution = 224
    mvbench = MVBench_dataset(data_dir, data_list, num_segments=num_frame, resolution=resolution)
    all_result = run_mvbench_inference(args, mvbench)
    acc_dict = {}
    for task_type, result in all_result.items():
        acc_count = 0
        for r in result:
            gt = r['gt']
            pred = r['pred']
            if check_ans(pred, gt):
                acc_count += 1
        acc_dict[task_type] = acc_count / len(result)
    
    print(acc_dict)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, args.output_name), "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": all_result
        }, f, indent=4)




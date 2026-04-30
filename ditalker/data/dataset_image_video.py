import csv
import io
import json
import math
import os
import random
from threading import Thread

import albumentations
import cv2
import gc
import numpy as np
import torch
import torchvision.transforms as transforms

from func_timeout import func_timeout, FunctionTimedOut
from decord import VideoReader
from PIL import Image
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset
from contextlib import contextmanager

VIDEO_READER_TIMEOUT = 20

def get_random_mask(shape):
    f, c, h, w = shape
    
    if f != 1:
        mask_index = np.random.choice([0, 1, 2, 3, 4], p = [0.05, 0.3, 0.3, 0.3, 0.05]) # np.random.randint(0, 5)
    else:
        mask_index = np.random.choice([0, 1], p = [0.2, 0.8]) # np.random.randint(0, 2)
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if mask_index == 0:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
        block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)
        mask[:, :, start_y:end_y, start_x:end_x] = 1
    elif mask_index == 1:
        mask[:, :, :, :] = 1
    elif mask_index == 2:
        mask_frame_index = np.random.randint(1, 5)
        mask[mask_frame_index:, :, :, :] = 1
    elif mask_index == 3:
        mask_frame_index = np.random.randint(1, 5)
        mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
    elif mask_index == 4:
        center_x = torch.randint(0, w, (1,)).item()
        center_y = torch.randint(0, h, (1,)).item()
        block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
        block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

        start_x = max(center_x - block_size_x // 2, 0)
        end_x = min(center_x + block_size_x // 2, w)
        start_y = max(center_y - block_size_y // 2, 0)
        end_y = min(center_y + block_size_y // 2, h)

        mask_frame_before = np.random.randint(0, f // 2)
        mask_frame_after = np.random.randint(f // 2, f)
        mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
    else:
        raise ValueError(f"The mask_index {mask_index} is not define")
    return mask

class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]
from typing import List
@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame




def get_audio_window(audio, win_size:int=5):
    """

    Args:
        audio (numpy.ndarray): (N,)

    Returns:
        audio_wins (numpy.ndarray): (N, W)
    """
    num_frames = len(audio)
    ph_frames = []
    for rid in range(0, num_frames):
        ph = []
        for i in range(rid - win_size, rid + win_size + 1):
            if i < 0:
                ph.append(74)
            elif i >= num_frames:
                ph.append(74)
            else:
                ph.append(audio[i])

        ph_frames.append(ph)

    audio_wins = np.array(ph_frames)


    return audio_wins




from scipy.io import loadmat
def get_video_style_clip(video_path, style_max_len, start_idx="random", dtype=torch.float32):
    if video_path[-3:] == "mat":
        face3d_all = loadmat(video_path)["coeff"]
        face3d_exp = face3d_all[:, 80:144]  # expression 3DMM range
    elif video_path[-3:] == "txt":
        face3d_exp = np.loadtxt(video_path)
    else:
        raise ValueError("Invalid 3DMM file extension")

    face3d_exp = torch.tensor(face3d_exp, dtype=dtype)

    length = face3d_exp.shape[0]
    if length >= style_max_len:
        clip_num_frames = style_max_len
        if start_idx == "random":
            clip_start_idx = np.random.randint(low=0, high=length - clip_num_frames + 1)
        elif start_idx == "middle":
            clip_start_idx = (length - clip_num_frames + 1) // 2
        elif isinstance(start_idx, int):
            clip_start_idx = start_idx
        else:
            raise ValueError(f"Invalid start_idx {start_idx}")

        face3d_clip = face3d_exp[clip_start_idx : clip_start_idx + clip_num_frames]
        pad_mask = torch.tensor([False] * style_max_len)
    else:
        padding = torch.zeros(style_max_len - length, face3d_exp.shape[1])
        face3d_clip = torch.cat((face3d_exp, padding), dim=0)
        pad_mask = torch.tensor([False] * length + [True] * (style_max_len - length))

    return face3d_clip, pad_mask



class ImageVideoDataset(Dataset):
    def __init__(
            self,
            ann_path, data_root=None,
            video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
            image_sample_size=512,
            video_repeat=0,
            text_drop_ratio=-1,
            enable_bucket=False,
            video_length_drop_start=0.1, 
            video_length_drop_end=0.9,
            enable_inpaint=False,
            enable_audio_emb = False,
            audio_emb_dir = None,
            enable_style_attn=False,
            enable_pose_adapter=False,
            finetune_emo = False,
            enable_sync_lip_loss=False,
            classifier_free_p =0.9,
        ):
        self.classifier_free_p = classifier_free_p
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        self.dataset = []
        for data in dataset:
            if data.get('type', 'image') != 'video':
                self.dataset.append(data)
        if video_repeat > 0:
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint  = enable_inpaint
        self.enable_style_attn = enable_style_attn
        self.enable_pose_adapter = enable_pose_adapter
        self.enable_sync_lip_loss = enable_sync_lip_loss
        self.finetune_emo = finetune_emo
        if self.text_drop_ratio < 0:
            if self.finetune_emo:
                self.text_drop_ratio = 0.5
            elif self.enable_style_attn:
                self.text_drop_ratio = 0.99

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
        self.attn_transform_64 = transforms.Compose(
            [
                transforms.Resize(
                    (64, 64)),
                transforms.ToTensor(),
            ]
        )
        self.lip_mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (512, 512)),
                transforms.ToTensor(),
            ]
        )
        self.pose_transforms =transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
        self.enable_audio_emb = enable_audio_emb
        self.audio_emb_dir = audio_emb_dir
        # print(self.audio_emb_dir,self.enable_audio_emb)
        # input("test")

    
    def augmentation(self, images, transform, state=None):
        """
        Apply the given transformation to the input images.
        
        Args:
            images (List[PIL.Image] or PIL.Image): The input images to be transformed.
            transform (torchvision.transforms.Compose): The transformation to be applied to the images.
            state (torch.ByteTensor, optional): The state of the random number generator. 
            If provided, it will set the RNG state to this value before applying the transformation. Defaults to None.

        Returns:
            torch.Tensor: The transformed images as a tensor. 
            If the input was a list of images, the tensor will have shape (f, c, h, w), 
            where f is the number of images, c is the number of channels, h is the height, and w is the width. 
            If the input was a single image, the tensor will have shape (c, h, w), 
            where c is the number of channels, h is the height, and w is the width.
        """
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor
    
    def get_batch(self, idx):
        syncnet_T = 5
        syncnet_mel_step_size = 16
        data_info = self.dataset[idx % len(self.dataset)]
        # print(data_info)
        # input("test")
        if data_info.get('type', 'image')=='video':
            video_id, text = data_info['file_path'], data_info['text']

            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)
            # print(video_dir)
            if self.enable_audio_emb:
                # get whisper emb path
                # audio_emb_dir = video_dir.replace("videos", "whisper_audio_emb").replace(".mp4", ".pt")
                audio_emb_dir =  data_info["audio_emb_path"]
                audio_emb = torch.load(audio_emb_dir)
                # print(audio_emb.shape)
                # get mask path
                lip_mask_union_path = video_dir.replace("videos_25", "lip_mask").replace(".mp4", ".png") #ok
                # print(lip_mask_union_path)
                eye_mask_union_path = video_dir.replace("videos_25", "eye_masks").replace(".mp4", ".png") #ok 
                full_mask_union_path = video_dir.replace("videos_25", "face_mask").replace(".mp4", ".png") #ok

                
                # lip_mask_union_path = video_dir.replace("videos", "lip_mask").replace(".mp4", ".png")
                # # print(lip_mask_union_path)
                # eye_mask_union_path = video_dir.replace("videos", "eye_masks").replace(".mp4", ".png")
                # full_mask_union_path = video_dir.replace("videos", "face_mask").replace(".mp4", ".png")

                pixel_values_eye_mask = torch.zeros(3,64,64)
                lip_masks_list = [Image.open(lip_mask_union_path)] * 1
                state = torch.get_rng_state()
                pixel_values_lip_mask = self.augmentation(lip_masks_list, self.lip_mask_transform, state) 
                assert lip_masks_list[0] is not None, "Fail to load lip mask, no such file or directory"
                # pixel_values_full_mask= torch.zeros(3,64,64)

                

                try :
                    eye_masks_list = [Image.open(eye_mask_union_path)] * 1
                    # lip_masks_list = [Image.open(lip_mask_union_path)] * 1
                    full_masks_list = [Image.open(full_mask_union_path)] * 1

                    assert eye_masks_list[0] is not None, "Fail to load eye mask."
                    # assert lip_masks_list[0] is not None, "Fail to load lip mask."
                    assert full_masks_list[0] is not None, "Fail to load full mask."
                # print(face_masks_list)
                    state = torch.get_rng_state()
                    pixel_values_eye_mask = self.augmentation(eye_masks_list, self.attn_transform_64, state)               
                #     pixel_values_lip_mask = self.augmentation(lip_masks_list, self.attn_transform_64, state)               
                    pixel_values_full_mask = self.augmentation(full_masks_list, self.lip_mask_transform, state) 
                    # print(pixel_values_full_mask.shape)
                except Exception as e:
                    print(f"Error: {e}")
                    pixel_values_eye_mask = torch.zeros(3,64,64)
                    # pixel_values_lip_mask = torch.zeros(3,64,64)
                    pixel_values_full_mask= torch.zeros(3,512,512)
                    print("what happened audio or mask")

                # print(pixel_values_face_mask)
                # print(pixel_values_face_mask.shape)
                # input("test")
                # self.text_drop_ratio = 0.5
            else:
                audio_emb = torch.zeros(1280,50,1)
                pixel_values_eye_mask = torch.zeros(3,64,64)
                pixel_values_lip_mask = torch.zeros(3,64,64)
                pixel_values_full_mask= torch.zeros(3,64,64)
                print("what happened audio or mask")
                
            if self.enable_style_attn:
                phoneme_dir =  data_info["phoneme_dir"]
                # print(phoneme_dir)
                with open(phoneme_dir, "r") as f:
                     phoneme = json.load(f)           
                phoneme_win = get_audio_window(phoneme, 5)
                phoneme_win = torch.tensor(phoneme_win)

                # phoneme_win=torch.zeros(780,11)
                # print(phoneme_win.shape)
                style_clip_path =  data_info["3dmm_dir"]
                # print(style_clip_path)
                # style_clip_path = "/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/fenghe/datasets/mead/3dmm/videos/W016_fear_016.mat"
                style_clip, pad_mask = get_video_style_clip(style_clip_path, style_max_len=256, start_idx=0)
                # print(style_clip.shape)
                # print(pad_mask.shape)

                # style_clip, pad_mask = get_video_style_clip(style_clip_path, style_max_len=256, start_idx=0)
                # style_clip = torch.zeros(256)
                # pad_mask = None
                # print(style_clip.shape)
                # input("test")
            else:
                phoneme_win = torch.zeros(64,11)
                style_clip = torch.zeros(256)
                pad_mask = None
                print("what happened phoneme or 3dmm")
                # print(torch.load(audio_emb_dir).shape)

            


            with VideoReader_contextmanager(video_dir, num_threads=3) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start))
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                if self.enable_sync_lip_loss:
                    # res_dp_path = video_dir.replace('videos_25', 'deepspeech_25').replace('.mp4', '.txt')
                    res_dp_path = video_dir.replace("videos", "deepspeech_25").replace(".mp4", ".txt")
                    # mel_emb_path = video_dir.replace('videos_25', 'mel_dit').replace('.mp4', '.pt')
                    mel_emb_path = video_dir.replace('videos', 'mel_dit').replace('.mp4', '.pt')
                    loaded_data_deepspeech = np.loadtxt(res_dp_path)
                    deepspeech_emb = torch.from_numpy(loaded_data_deepspeech).float().cuda()
                    deepspeech_emb = deepspeech_emb.permute(1,0)
                    mel_emb=torch.load(mel_emb_path).cuda()
                    # start_idx = int(80. * (start_frame_num / float(hparams.fps)))
                    # end_idx = start_idx + syncnet_mel_step_size
                    # return spec[start_idx : end_idx, :]
                    # print(deepspeech_emb.shape)
                else:
                    deepspeech_emb = torch.zeros(video_length,29).permute(1,0)
                    mel_emb_path = video_dir.replace('videos', 'mel_dit').replace('.mp4', '.pt')
                    mel_emb=torch.load(mel_emb_path).cuda()
                # print(start_idx)

                
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)
                # state = torch.get_rng_state()
                pose_pil_image_list = []
                if self.enable_pose_adapter:
                    pose_dir = video_dir.replace("videos", "videos_dwpose")
                    pose_dir = video_dir.replace("videos_25", "videos_25_dwpose")
                    pose_reader = VideoReader(pose_dir)
                    for index in batch_index:
                        img = pose_reader[index]
                        pose_pil_image_list.append(Image.fromarray(img.asnumpy()))
                    
                    
                    pixel_values_pose = self.augmentation(
                        pose_pil_image_list, self.pose_transforms, state
                    )
                else:
                    pixel_values_pose = torch.zeros(3,self.video_sample_n_frames,256,256)
                # print(batch_index)
                #audio_emb =audio_emb[batch_index]
                # print(audio_emb.shape)
                # print(start_idx,start_idx+clip_length - 1)
                # print(batch_index)

                audio_emb = audio_emb[start_idx: start_idx + clip_length ,:,:]
                phoneme_win = phoneme_win[start_idx: start_idx + clip_length,:]
                deepspeech_emb= deepspeech_emb[:,start_idx: start_idx + clip_length]
                start_num = int(80. * (start_idx / float(25)))
                end_num = start_num + syncnet_mel_step_size
                # mel_emb = mel_emb[:,start_num:end_num].unsqueeze(0)
                mel_emb = mel_emb.unsqueeze(0)
                # print(mel_emb.shape)
                

                # return spec[start_num : end_idx, :]
                if phoneme_win.shape != (64, 11):
                    padding_rows = 64 - phoneme_win.size(0)
                    padding_cols = 11 - phoneme_win.size(1)
                    phoneme_win = torch.nn.functional.pad(phoneme_win, (0, max(0, padding_cols), 0, max(0, padding_rows)), 'constant', value=31)

                
                # print(phoneme_win.shape)
                # print(audio_emb.shape)
                # print(audio_emb.shape)
                # input("test")
                # print(audio_emb.shape)
                # input("test")

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    resized_frames = []
                    for i in range(len(pixel_values)):
                        frame = pixel_values[i]
                        resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                        resized_frames.append(resized_frame)
                    pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    # print(pixel_values.shape)
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    # print(pixel_values.shape)
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''
            # print(pixel_values.shape)
            return pixel_values, text, 'video', audio_emb[:self.video_sample_n_frames], pixel_values_eye_mask,pixel_values_lip_mask,pixel_values_full_mask,phoneme_win,style_clip, pad_mask, pixel_values_pose,deepspeech_emb,mel_emb
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)
            if random.random() < self.text_drop_ratio:
                text = ''
            return image, text, 'image', audio_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, name, data_type , audio_emb,pixel_values_eye_mask,pixel_values_lip_mask,pixel_values_full_mask,phoneme_win,style_clip, pad_mask, pixel_values_pose,deepspeech_emb,mel_emb= self.get_batch(idx)
                sample["pixel_values"] = pixel_values

                    
                    
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                sample["audio_encoder_hidden_states"] = audio_emb
                sample["eye_mask_values"] =pixel_values_eye_mask
                sample["lip_mask_values"] =pixel_values_lip_mask
                sample["full_mask_values"] = pixel_values_full_mask
                sample["phoneme_win"] = phoneme_win
                sample["style_clip"] = style_clip
                sample["pad_mask"]= pad_mask
                sample['pixel_values_pose'] = pixel_values_pose
                sample['deepspeech_emb'] = deepspeech_emb
                sample['mel_emb'] = mel_emb


                if self.enable_audio_emb and audio_emb.shape[0]<self.video_sample_n_frames:
                    # print(audio_emb.shape)
                    idx = random.randint(0, self.length-1)
                    print("sample another data ")
                    sample = self.__getitem__(idx)

                # print(sample["audio_emb"].shape)

                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.ones_like(pixel_values) * -1 * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values

        if pixel_values.shape[0]<self.video_sample_n_frames or audio_emb.shape[0]<self.video_sample_n_frames:
            idx = random.randint(0, self.length-1)
            sample = self.__getitem__(idx)
        if self.enable_audio_emb:
            classifier_free_mask = torch.bernoulli(torch.ones_like(sample["audio_encoder_hidden_states"]), self.classifier_free_p)
            sample["audio_encoder_hidden_states"] = sample["audio_encoder_hidden_states"] * classifier_free_mask
        
        if self.enable_style_attn:
            classifier_free_mask = torch.bernoulli(torch.ones_like(sample["style_clip"]), self.classifier_free_p)
            sample["style_clip"] = sample["style_clip"] * classifier_free_mask
            classifier_free_mask = torch.bernoulli(torch.ones_like(sample["phoneme_win"]), self.classifier_free_p)
            sample["phoneme_win"] = sample["phoneme_win"] * classifier_free_mask

        if self.enable_pose_adapter:
            classifier_free_mask = torch.bernoulli(torch.ones_like(sample["pixel_values_pose"]), self.classifier_free_p)
            sample['pixel_values_pose'] = sample['pixel_values_pose'] * classifier_free_mask
            classifier_free_mask = torch.bernoulli(torch.ones_like(sample["mel_emb"]), self.classifier_free_p)
            sample["mel_emb"] = sample["mel_emb"] * classifier_free_mask
            
            # print( sample['pixel_values_pose'].shape)

        # print(sample['pixel_values'].shape)
        return sample




    

if __name__ == "__main__":
    dataset = ImageVideoDataset(
        ann_path="data/annotations.json"  # Update to your annotation path
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16)
    for idx, batch in enumerate(dataloader):
        if batch["audio_encoder_hidden_states"].shape[2] <= 64:
            print("bug data")
        print(batch["pixel_values"].shape, len(batch["text"]))

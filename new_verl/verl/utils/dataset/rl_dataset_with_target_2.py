# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask



from omegaconf import ListConfig
import os
from typing import List, Union, Optional

import pandas as pd
import copy 

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, ProcessorMixin
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import pad_sequence_to_length


logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    将一个包含多个样本字典的列表，整理成适合神经网络训练的批次化数据格式。

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.
            包含多个样本字典的列表

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
         一个字典，其中张量被堆叠，非张量被转换为numpy数组
    """
    # 使用默认字典来分别存储张量数据和非张量数据。
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    # 遍历每个样本，将张量数据和非张量数据分别存储到不同的字典中。
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    # 使用 torch.stack() 将所有张量沿着新的第0维（批次维度）堆叠起来。
    '''
        输入: 5个形状为 [3, 224, 224] 的图像张量
        输出: 形状为 [5, 3, 224, 224] 的批次张量
    '''
    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    # 使用字典解包将两个字典合并后返回。
    return {**tensors, **non_tensors}


class RLHFDatasetWithTarget_2(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt") # 
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)


        # 在config 中已经加入了 target_key
        self.target_key = config.get("target_key", "target") # 
        self.max_target_length = config.get("max_target_length", 8192)

        # 尚未在config中加入
        self.filter_targets=config.get("filter_targets", False),
        self.sample_target_ratio=config.get("sample_target_ratio", 1.0)
        self.target_list_key = config.get("target_list_key", "target_lst") # 
        self.max_num_targets=config.get("max_num_targets", 5)
        self.target_probs_key=config.get("target_probs_key", 'target_ds_qwen_7b_probs')




        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """

        # 某一条data
        row_dict: dict = self.dataframe[item]
        # 将当前 data 用 chat template 包裹
        messages = self._build_messages(row_dict)
        model_inputs = {}

        # 获取 input_ids, attention_mask
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # 获取 position_ids
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # 处理 tgt
        tgt = row_dict.pop(self.target_key) # 从数据行字典中提取并移除目标文本。pop 方法同时获取值并从字典中删除该键。
        sample = np.random.rand() < self.sample_target_ratio  # 基于 sample_target_ratio 概率决定是否使用这个目标文本进行训练（课程学习或概率性使用目标）。
        if tgt is not None and sample is True: # 只有当目标文本存在且采样结果为True时才处理目标文本。
            tgt = tgt[0]
            # 从目标数据结构中提取第一个元素（可能是列表或元组中的第一个字典）。
            
            # 特殊标记清理, 避免在 prompt 和 target 中出现重复的 <think>
            if isinstance(tgt, dict):
                tgt = tgt['content']

            if not tgt.startswith('<think>\n'):
                tgt = '<think>\n' + tgt

            if raw_prompt.endswith('<think>\n') and tgt.startswith('<think>\n'):
                tgt = tgt[len('<think>\n'):]

            tgt_input_ids = self.tokenizer(tgt, add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1) # [1, l]
            tgt_input_ids = tgt_input_ids.reshape(1, -1)
        #  若 根据概率，不使用tgt
        else:
            # tgt_input_ids 被设置为 空 tensor
            tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0) # empty target, will be pad to max_target_length

        # padding or truncate
        sequence_length = tgt_input_ids.shape[-1] # 将标记ID重新整形为 [1, sequence_length] 格式，适合批量处理。
        
        # off-policy 数据太短，则padding
        if sequence_length < self.max_target_length:
            # right pad for tgt_input_ids
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                            max_seq_len=self.max_target_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            left_pad=False)
        else: # 数据过长，则截断
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        tgt_input_ids = tgt_input_ids.squeeze(0)
        # off-policy 的分词结果保存在 row_dict['tgt_input_ids'] 当中
        row_dict['tgt_input_ids'] = tgt_input_ids


        # ---  process target_list
        if getattr(self, 'target_list_key', "target_list_key") in row_dict:
            target_list = row_dict.pop(self.target_list_key)
            if target_list is None:
                tgt_input_ids_lst = [torch.zeros_like(tgt_input_ids).fill_(self.tokenizer.pad_token_id)] * self.max_num_targets
            else:
                tgt_input_ids_lst = [self._process_target(tgt, prompt_with_chat_template, add_eos=True) for tgt in target_list]
                if len(tgt_input_ids_lst) <= self.max_num_targets:
                    tgt_input_ids_lst.extend([torch.zeros_like(tgt_input_ids_lst[0]).fill_(self.tokenizer.pad_token_id)] * (self.max_num_targets - len(tgt_input_ids_lst)))
                else:
                    tgt_input_ids_lst = tgt_input_ids_lst[:self.max_num_targets]
            row_dict['tgt_input_ids_lst'] = torch.stack(tgt_input_ids_lst, dim=0) # [max_num_targets, max_target_length]
        
        if getattr(self, 'target_probs_key', "target_probs_key") in row_dict:
            target_probs = row_dict.pop(self.target_probs_key)
            if target_probs is not None:
                target_probs_pt = torch.tensor(target_probs, dtype=torch.float32, device=tgt_input_ids.device)
                target_probs_pt = target_probs_pt.reshape(1, -1)
                # truncation
                tgt_len = (tgt_input_ids != self.tokenizer.pad_token_id).sum()
                try:
                    assert target_probs_pt.shape[-1] == tgt_len+1
                except Exception as e:
                    breakpoint()
                
                # same padding as tgt_input_ids
                if target_probs_pt.shape[-1] < self.max_target_length:
                    target_probs_pt = pad_sequence_to_length(target_probs_pt,
                                                max_seq_len=self.max_target_length,
                                                pad_token_id=-1,
                                                left_pad=False)
                else:
                    assert self.truncation in ('right', 'error')
                    target_probs_pt = target_probs_pt[:, :self.max_target_length]
                row_dict['target_probs'] = target_probs_pt.squeeze(0) # [max_target_length]
            else:
                row_dict['target_probs'] = torch.zeros_like(tgt_input_ids, dtype=torch.float32, device=tgt_input_ids.device).fill_(-1)
        # ----- 



        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()


    def _process_target(self, tgt: str, prompt: str, add_eos=False) -> torch.Tensor:
        if prompt.endswith('<think>\n') and tgt.startswith('<think>\n'):
            tgt = tgt[len('<think>\n'):]
        tgt_input_ids = self.tokenizer(tgt, add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1) # [1, l]
        if add_eos:
            tgt_input_ids = torch.cat([tgt_input_ids, torch.tensor([self.tokenizer.eos_token_id], device=tgt_input_ids.device, dtype=tgt_input_ids.dtype).reshape(-1)])

        tgt_input_ids = tgt_input_ids.reshape(1, -1)
        # padding or truncate
        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.max_target_length:
            # right pad for tgt_input_ids
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                            max_seq_len=self.max_target_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            left_pad=False)
        else:
            assert self.truncation in ('right', 'error')
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        tgt_input_ids = tgt_input_ids.squeeze(0)

        return tgt_input_ids
    

    def remove_data(self, remove_item_list):
        """
        Remove data corresponding to all items in remove_item_list (all values are of type int)
        
        Args:
            remove_item_list (List[int]): List of data indices to be removed
        """
        if not remove_item_list:
            return
        
        # 确保索引在有效范围内
        valid_indices = [idx for idx in remove_item_list if 0 <= idx < len(self.dataframe)]
        
        if not valid_indices:
            return
            
        # 从dataframe中删除指定索引的行
        self.dataframe = self.dataframe.drop(self.dataframe.index[valid_indices]).reset_index(drop=True)
        
        logger.info(f"Removed {len(valid_indices)} items from dataset. New dataset size: {len(self.dataframe)}")
    

    def random_get(self, num):
        """
        Randomly select and process num data samples, with processing identical to __getitem__ function
        
        Args:
            num (int): Number of data samples to randomly select
            
        Returns:
            dict: Batch data containing num samples, where each key's value is a list or stacked tensor of all samples' values for that key
        """
        if num <= 0:
            raise ValueError("num must be positive")
        
        # 确保不超过数据集大小
        num = min(num, len(self.dataframe))
        
        # 随机选择num个不重复的索引
        indices = np.random.choice(len(self.dataframe), size=num, replace=False)
        
        # 收集所有样本的数据
        batch_data = []
        for idx in indices:
            sample_data = self.__getitem__(idx)
            batch_data.append(sample_data)
        
        # 使用collate_fn整理数据
        return collate_fn(batch_data)

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import logging
import os
import pickle
import socket
from contextlib import contextmanager
from types import MethodType
from typing import Any
from typing import List

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import DictConfig, ListConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length, pad_sequence_to_length
from verl.workers.config import RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


def get_eos_mask(response_id: torch.Tensor, eos_token: int = 2, dtype=torch.int64):
    '''
    e.g. end of sentence token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    '''
    eos_mask = response_id.eq(eos_token).long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

# 
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    '''
        去掉padding 操作
    '''
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids



def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    '''
       # 某些模型或处理流程使用右填充
        batch = [
            ["Hello", "world", pad, pad, pad],    # 较短序列
            ["How", "are", "you", "?", pad],      # 较长序列  
            ["I'm", "fine", pad, pad, pad]        # 中等序列
        ]

        # 处理后：移除右侧填充，保留有效内容
        [
            ["Hello", "world"],
            ["How", "are", "you", "?"],
            ["I'm", "fine"]
        ]
    '''
    # 移除左侧填充，只保留右侧填充的有效token
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if len(non_pad_index) == 0:
        return []
    else:
        # 找到最后一个非填充token的位置
        last_non_pad_index = non_pad_index[-1][0]
        # 保留从开始到这个位置的所有token
        token_ids = prompt_token_ids[:last_non_pad_index+1].tolist()
    return token_ids


class MIXvLLMRollout(vLLMRollout):
    '''
        继承自 vLLMRollout
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 新的超参整理
        self.tokenizer = kwargs.get('tokenizer')
        self.prefix_strategy = self.config.get('prefix_strategy', 'random')
        self.prefix_steps = self.config.get('prefix_steps', 300)
        self.prefix_linear_max_ratio = self.config.get('prefix_linear_max_ratio', 0.8)
        if self.prefix_strategy == 'linear':
            pass
        elif self.prefix_strategy == 'linear_max':
            self.prefix_ratio_windows = [(0, i*self.prefix_linear_max_ratio/10) for i in range(10, 0, -1)]
            self.prefix_step_windows = [(i*self.prefix_steps/10, (i+1)*self.prefix_steps/10) for i in range(10)]
        elif self.prefix_strategy == 'linear_variance':
            self.prefix_lienar_max_var = self.config.get('prefix_lienar_max_var', 0.1)
        elif self.prefix_strategy == 'reverse_linear':
            self.prefix_ratio_windows = [(0, (i+1)*self.prefix_linear_max_ratio/10) for i in range(10)]
            self.prefix_step_windows = [(i*self.prefix_steps/10, (i+1)*self.prefix_steps/10) for i in range(10)]
        elif self.prefix_strategy == 'fixed':
            assert self.config.prefix_share_across_samples == False, "Fixed strategy could not work with prefix_share_across_samples=True ! "
            # self.prefix_fixed_num = self.config.get('prefix_fixed_num', 2)
            n_prefix = self.config.n_prefix if self.config.n_prefix != -1 else self.config.n
            ratio_step = (self.config.max_prefix_ratio - self.config.min_prefix_ratio) / (n_prefix-1)
            self.prefix_fix_ratios = [self.config.min_prefix_ratio + i*ratio_step for i in range(n_prefix)]

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        print('Love from MIXvLLMRollout.generate_sequences')
        idx = prompts.batch['input_ids'] # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']


        # we use repeat to get n generations for each prompt
        # Pre-process input token ids
        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch

        # print(f'non_tensor_batch.keys() firest {non_tensor_batch.keys()}')
        

        # # 获取每个prompt 的raw_prompt_ids
        # if "raw_prompt_ids" not in non_tensor_batch:
        #     print('"raw_prompt_ids" not in non_tensor_batch')
        #     # non_tensor_batch["raw_prompt_ids"] 对应 UPT 中的 idx_list 
        #     non_tensor_batch["raw_prompt_ids"] = np.array(
        #         [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
        #     )
        #     print('non_tensor_batch["raw_prompt_ids"]', non_tensor_batch["raw_prompt_ids"])
        # if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
        #     raise RuntimeError("vllm sharding manager is not work properly.")

        # if "multi_modal_data" in non_tensor_batch:
        #     vllm_inputs = []
        #     for raw_prompt_ids, multi_modal_data in zip(
        #         non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
        #     ):
        #         vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        # else:
        #     vllm_inputs = [
        #         {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
        #     ]
        
        # for input_data in vllm_inputs:
        #     # Ensure token IDs are lists or numpy arrays
        #     if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
        #         raise TypeError(
        #             f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
        #         )

        #     input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        
        
        # 这段代码是强化学习文本生成中处理前缀比例（prefix ratio）的核心逻辑，主要用于控制生成文本时使用多少目标文本作为前缀。
        # print('non_tensor_batch.keys: ', non_tensor_batch.keys())
        idx_list = non_tensor_batch["raw_prompt_ids"]  # 原始 prompt ids

        print('idx_list before repeat shape: ', idx_list.shape)
        if not is_validate:
            rollout_num = self.config.n # 每个提示生成多少个样本
        else:
            rollout_num = self.config.val_kwargs.n


        if do_sample: # 非贪心采样
            # 将每个prompt_ids 重复了 rollout_num 次
            idx_list = sum([[idx_list[i]] * rollout_num for i in range(len(idx_list))], [])
        
        print('idx_list after repeat len: ', len(idx_list))
        
        prefix_ratios = None
        # logger.info('after idx_list length', len(idx_list))
        tgt_input_ids = None
        # 若 tgt_input_ids 在 prompts.batch 中
        if 'tgt_input_ids' in prompts.batch: # in train mode
            # 获得target 的input_ids, 每个batch data 都有一个对应的 tgt_input_ids
            tgt_input_ids = prompts.batch['tgt_input_ids']  # [bsz, tgt_len]
            print('tgt_input_ids shape: ', tgt_input_ids.shape)
            # tgt_input_ids shape:  torch.Size([32, 1024])

            # 步骤1: 清理每个目标序列的右侧 padding
            tgt_list = [
                _pre_process_inputs_right_pad(self.pad_token_id, tgt_input_ids[i]) for i in range(batch_size)
            ]

            # 为非空目标序列添加EOS token ('<|endoftext|>' ) tgt_list 和 prompt 的size 一一对应
            tgt_list = [
                        tgt_list[i] + [self.tokenizer.eos_token_id,] if len(tgt_list[i]) > 0 else tgt_list[i]
                        for i in range(batch_size)
                    ]
            
            # 为每个目标序列生成n个副本（用于多样性采样）
            # [序列A] * 3 = [序列A, 序列A, 序列A], 此时每个rollout data 都有一个 参考目标
            tgt_list = sum([[tgt_list[i]] * rollout_num for i in range(len(tgt_list))], [])


            # 将训练步数从1-based转换为0-based
            global_steps = prompts.meta_info['global_steps'] - 1 # we start from 1
            import random

            '''
                假设：
                有一个提示："写一个关于人工智能的故事"
                目标文本："人工智能正在改变世界，它让生活更便捷"
                rollout_num = 3（生成3个样本）

                共享前缀（所有样本相同） 前缀比例: 0.6 (所有样本都一样)
                样本1: "写一个关于人工智能的故事" + "人工智能正在改变"
                样本2: "写一个关于人工智能的故事" + "人工智能正在改变"  
                样本3: "写一个关于人工智能的故事" + "人工智能正在改变"

                不共享前缀（每个样本独立）
                样本1前缀比例: 0.8 → "写一个关于人工智能的故事" + "人工智能正在改变世界，它让"
                样本2前缀比例: 0.4 → "写一个关于人工智能的故事" + "人工智能正在"
                样本3前缀比例: 0.0 → "写一个关于人工智能的故事" (完全自主生成)
                
                有些样本使用较多前缀（保守策略）
                有些样本使用较少前缀（冒险策略）
                有些样本完全自主生成（最大探索）
            '''

            #  # 不共享前缀 (prefix_share_across_samples=False)
            if not self.config.prefix_share_across_samples:
                assert self.config.prefix_strategy != 'linear', "Linear strategy is not implemented with prefix_share_across_samples=True ! "
                if self.config.n_prefix == -1: # n_prefix == -1（所有样本都使用前缀）
                    # 为每个tgt 参考随机生成前缀比例
                    if self.config.prefix_strategy == 'random':
                        # 此时 tgt 已经被复制过了， prefix_ratios 有 bz * rollout_num 元素
                        prefix_ratios = [random.uniform(self.config.min_prefix_ratio, self.config.max_prefix_ratio) for _ in range(len(tgt_list))]
                    
                    # 反向线性策略 (reverse_linear/linear_max)， 分配前缀比例
                    elif self.config.prefix_strategy == 'reverse_linear' or self.config.prefix_strategy == 'linear_max':
                        # 基于训练步数的线性策略
                        w_idx = -1
                        for i in range(len(self.prefix_step_windows)):
                            if global_steps >= self.prefix_step_windows[i][0] and global_steps <= self.prefix_step_windows[i][1]:
                                w_idx = i
                                break
                        prefix_ratios = [random.uniform(self.prefix_ratio_windows[w_idx][0], self.prefix_ratio_windows[w_idx][1]) for _ in range(len(tgt_list))]
                    # 使用预设的固定比例列表
                    elif self.config.prefix_strategy == 'fixed':
                        # 例如：prefix_fix_ratios = [0.8, 0.5, 0.3]
                        prefix_ratios = sum([self.prefix_fix_ratios for i in range(batch_size)], [])
                else: # 部分样本使用前缀
                    # 只有n_prefix个样本使用前缀，其余为0， 对于LUFFY，n_prefix=1， prefix_ratio=1
                    assert self.config.n_prefix <= rollout_num, f"n_prefix {self.config.n_prefix} must be less than or equal to n {rollout_num}"
                    assert len(tgt_list) == rollout_num * batch_size
                    prefix_ratios = []  
                    for i in range(batch_size):
                        if self.config.prefix_strategy == 'random':
                            prefix_ratios.extend([random.uniform(self.config.min_prefix_ratio, self.config.max_prefix_ratio) for _ in range(self.config.n_prefix)])
                        elif self.config.prefix_strategy == 'reverse_linear' or self.config.prefix_strategy == 'linear_max':
                            w_idx = -1
                            for i in range(len(self.prefix_step_windows)):
                                if global_steps >= self.prefix_step_windows[i][0] and global_steps <= self.prefix_step_windows[i][1]:
                                    w_idx = i
                                    break
                            prefix_ratios.extend([random.uniform(self.prefix_ratio_windows[w_idx][0], self.prefix_ratio_windows[w_idx][1]) for _ in range(self.config.n_prefix)])
                        elif self.config.prefix_strategy == 'fixed':
                            prefix_ratios.extend(self.prefix_fix_ratios[:])
                        else: raise NotImplementedError(f"Prefix strategy {self.config.prefix_strategy} is not implemented! ")

                        prefix_ratios.extend([0.0] * (rollout_num - self.config.n_prefix))
                    assert len(prefix_ratios) == len(tgt_list)
            else:
                # 共享前缀（同一提示的所有样本相同）
                if self.config.prefix_strategy == 'linear':
                    # 前缀比例基础值计算
                    ratio = min((global_steps / self.prefix_steps), 1.0) # 当前训练进度（0.0到1.0）
                    # 例如：最大比例0.8，训练完成50%时 → 0.8 × (1-0.5) = 0.4
                    prefix_ratio_base = self.prefix_linear_max_ratio * (1-ratio)
                else: # default, use random prefix ratio
                    prefix_ratio_base = None # 标记使用随机策略
                    
                assert self.config.n_prefix <= self.config.n, f"n_prefix {self.config.n_prefix} must be less than or equal to n {self.config.n}"
                assert len(tgt_list) == self.config.n * batch_size
                prefix_ratios = []
                for i in range(batch_size):
                    prefix_ratio = prefix_ratio_base if prefix_ratio_base is not None else random.uniform(self.config.min_prefix_ratio, self.config.max_prefix_ratio)
                    if self.config.n_prefix > 0: # 部分样本使用前缀
                        '''
                            batch_size=2, n=3, n_prefix=2, prefix_ratio=0.6
                            结果：[0.6, 0.6, 0.0, 0.6, 0.6, 0.0]
                            每个提示的3个样本：2个使用60%前缀，1个完全自主生成
                        '''
                        prefix_ratios.extend([prefix_ratio] * self.config.n_prefix)
                        prefix_ratios.extend([0.0] * (self.config.n - self.config.n_prefix))
                    else:
                        '''
                            n_prefix = 0（所有样本使用相同比例）
                            所有样本使用相同的前缀比例
                            示例：[0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
                        '''
                        logger.info(f"Prefix share across samples enabled! n_prefix is 0, n is set to {self.config.n}")
                        prefix_ratios.extend([prefix_ratio] * self.config.n)
                assert len(prefix_ratios) == len(tgt_list) # 确保生成的比例数量与目标列表长度匹配 len(tgt_list) = batch_size × n
            
            '''
                tgt_list = 
                    [
                        [10, 20, 30, 40, 50],  # target序列1
                        [60, 70, 80, 90]       # target序列2
                    ]
                
                prefix_ratios：
                    计算好的前缀比例
                    [0.6, 0.5]  # 序列1用60%，序列2用50%

                idx_list：原始提示序列
                    [
                        [1, 2, 3],  # 提示1
                        [4, 5, 6]   # 提示2
                    ]

                步骤1：计算前缀部分
                prefix_list = [
                    [10, 20, 30, 40, 50][:int(5*0.6)] = [10, 20, 30],  # 前3个token
                    [60, 70, 80, 90][:int(4*0.5)] = [60, 70]           # 前2个token
                ]

                idx_list = [
                    [1, 2, 3] + [10, 20, 30] = [1, 2, 3, 10, 20, 30],
                    [4, 5, 6] + [60, 70] = [4, 5, 6, 60, 70]
                ]
            '''
            
            # 从 tgt_list 中获得 prefix_list
            prefix_list = [tgt_list[i][:int(len(tgt_list[i]) * prefix_ratios[i])] for i in range(len(tgt_list))]
            # 把 prefix_list 拼接到 idx_list 后面
            idx_list = [idx_list[i] + prefix_list[i] for i in range(len(idx_list))]
        else: 
            # 没有 tgt_input_ids 的情况，比如eval阶段
            # in eval mode, we don't have tgt_input_ids
            tgt_list = None
        
        # print(f'prefix_list with len {len(prefix_list)}: {prefix_list}')

        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        
        # we use n=1 because we have repeated the idx_list to get n generations for each prompt
        kwargs['n'] = 1

        # print("=== Debug VLLM input for validation===")
        # print('prompts: ', prompts)
        if is_validate:
            # 如果 idx_list 中的元素已经是 list，直接使用
            idx_list = [list(item) if not isinstance(item, list) else item for item in idx_list]
        else:
            # 如果需要其他转换
            pass
            

        # print('idx_list to be inferred: ', idx_list)

        # Generate sequences
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)
        
        # Process outputs
        # print('check output[0]: ', output[0])
        # print('check output[1]: ', output[1])
        # print('output len:', len(output))

        # response = output[0].to(idx.device)
        if 'tgt_input_ids' in prompts.batch:
            # put the prefix back to the response
            try:
                resp_list = []
                for output_request in output:
                    response_ids = torch.tensor(output_request.outputs[0].token_ids, device = idx.device)
                    resp = _pre_process_inputs_right_pad(self.pad_token_id, response_ids)
                    resp_list.append(resp)

                # resp_list = [
                #     _pre_process_inputs_right_pad(self.pad_token_id, resp)
                #     for resp in response
                # ]
            except:
                breakpoint()

            '''
                假设：
                resp_list = [[30, 40, 50], [60, 70]]（生成的内容）
                prefix_list = [[10, 20], [80, 90, 100]]（前缀）
                response_length = 6（最大响应长度）
                pad_token_id = 0

                步骤1：拼接响应
                concat_resp_list = [
                    [10, 20, 30, 40, 50],    # 长度5
                    [80, 90, 100, 60, 70]    # 长度5
                ]

                创建前缀掩码
                prefix_mask = [
                    [True, True, False, False, False, False],  # 前2个是前缀
                    [True, True, True, False, False, False]    # 前3个是前缀
                ]

                # 最大长度5，所以不需要额外填充
                response = [
                    [10, 20, 30, 40, 50, 0],    # 实际长度5，填充到6
                    [80, 90, 100, 60, 70, 0]     # 实际长度5，填充到6
                ]

            '''
            
            # get prefix_mask and concat_resp_list
            concat_resp_list = [] # 用于存储拼接后的响应（前缀+生成内容）
            # 创建全False的掩码矩阵，形状为 [批次大小, 响应最大长度]
            prefix_mask = torch.zeros([len(resp_list), self.config.response_length], dtype=torch.bool).to(idx.device)
            # 将每个样本的前缀和生成响应拼接起来
            # prefix_list[i] = [10, 20] + resp_list[i] = [30, 40, 50] = [10, 20, 30, 40, 50]
            for i in range(len(resp_list)):
                concat_resp_list.append(torch.tensor(prefix_list[i] + resp_list[i]))
                prefix_len = min(len(prefix_list[i]), self.config.response_length)
                prefix_mask[i, :prefix_len] = True # 创建前缀掩码, 在掩码中将前缀部分标记为True
                '''
                    例如：前缀长度3 → [True, True, True, False, False, ...]
                '''
            # 找到所有拼接响应中的最大长度
            resp_max_len = max([len(resp) for resp in concat_resp_list])
            # prepare batch response, right pad to the max length # 创建填充矩阵，用pad_token_id填充
            # 将每个响应复制到填充矩阵中, 保持原始数据，其余部分保持填充值
            tt = torch.ones(len(concat_resp_list), resp_max_len).fill_(self.pad_token_id)
            for i in range(len(concat_resp_list)):
                tt[i][:len(concat_resp_list[i])] = concat_resp_list[i].clone().detach()
            response = tt.to(idx.device)[:, :self.config.response_length].to(idx.dtype) #  截断并转换数据类型
        else:
            resp_list = []
            for output_request in output:
                response_ids = torch.tensor(output_request.outputs[0].token_ids, device = idx.device)
                resp = _pre_process_inputs_right_pad(self.pad_token_id, response_ids)
                resp_list.append(resp)
            
            response = pad_2d_list_to_length(resp_list, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )

            prefix_mask = torch.tensor([]) # empty dummy tensor
        
        # Pad sequences if needed
        # if not is_validate:
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(
                        response, self.config.response_length, self.pad_token_id)
        # else:
        #     pass


        # Handle multiple samples per prompt
        if rollout_num > 1 and do_sample:
            idx = idx.repeat_interleave(rollout_num, dim=0)
            if tgt_input_ids is not None:
                tgt_input_ids = tgt_input_ids.repeat_interleave(
                   rollout_num, dim=0)
            else:
                tgt_input_ids = None
            attention_mask = attention_mask.repeat_interleave(
                rollout_num, dim=0)
            position_ids = position_ids.repeat_interleave(
                rollout_num, dim=0)
            batch_size = batch_size * rollout_num
        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)
        # Create position IDs and attention mask for full sequence
        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(
            batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids],
                                dim=-1)
        
        print('response shape: ', response.shape)
        # print('eos_token_id: ', eos_token_id)

        # response_attention_mask = get_eos_mask(
        #     response_id=response,
        #     eos_token=eos_token_id,
        #     dtype=attention_mask.dtype)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )

        attention_mask = torch.cat(
            (attention_mask, response_attention_mask), dim=-1)

        # Construct output batch
        batch = TensorDict(
                {
                    'prompts': idx,
                    'responses': response,
                    'input_ids': seq,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                },
                batch_size=batch_size)
        if tgt_input_ids is not None:
            batch['tgt_input_ids'] = tgt_input_ids
        
        if prefix_mask.shape[0] > 0:
            batch['prefix_mask'] = prefix_mask
        
        # if self.config.calculate_log_probs:
        #     # we will recompute old log prob with actor
        #     batch["rollout_log_probs"] = rollout_log_probs
        
        # # Free cache if configured
        # if self.config.free_cache_engine:
        #     self.inference_engine.free_cache_engine()

        # 在 return DataProto 之前添加调试信息
        print("=== Debug batch info ===")
        print(f"Expected batch size: unknown (will be inferred)")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"Key: {key}, Shape: {value.shape}, Type: {type(value)}")
            elif isinstance(value, (list, tuple)):
                print(f"Key: {key}, Length: {len(value)}, Type: {type(value)}")
            else:
                print(f"Key: {key}, Value: {value}, Type: {type(value)}")

        # if non_tensor_batch:
        #     print("=== Non-tensor batch info ===")
        #     for key, value in non_tensor_batch.items():
        #         if isinstance(value, (list, tuple)):
        #             print(f"Non-tensor Key: {key}, Length: {len(value)}, Type: {type(value)}")
        #         else:
        #             print(f"Non-tensor Key: {key}, Value: {value}, Type: {type(value)}")

        problematic_keys = []
        for key, value in non_tensor_batch.items():
            if hasattr(value, '__len__') and len(value) != batch['input_ids'].shape[0]:
                problematic_keys.append(key)
                
        if prefix_ratios is not None:
            meta_info = {
                'prefix_ratios': prefix_ratios,
            }
            if problematic_keys:
                for key in problematic_keys:
                    meta_info[key] = non_tensor_batch.pop(key)
            
            return DataProto(batch=batch, meta_info=meta_info, non_tensor_batch=non_tensor_batch)
        else:
            return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @torch.no_grad()
    def generate_off_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences using off-policy data without vLLM sampling.
        
        Args:
            prompts (DataProto): Input prompts containing batch data with input_ids, attention_mask,
                position_ids, tgt_input_ids and meta_info.
            max_retries (int, optional): Not used in this function but kept for consistency.
            **kwargs: Additional parameters (not used in off-policy generation).
            
        Returns:
            DataProto: Generated sequences containing:
                - prompts: Original input token ids
                - responses: Target response token ids (from tgt_input_ids)
                - input_ids: Concatenated prompt and response tokens
                - attention_mask: Attention mask for full sequence
                - position_ids: Position ids for full sequence
                - tgt_input_ids: Original target input ids
                - prefix_mask: Mask indicating all tokens are from off-policy data
        """
        # Extract input tensors from prompt batch
        print('Love from MIXvLLMRollout.generate_off_sequences')
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        tgt_input_ids = prompts.batch['tgt_input_ids']  # [bsz, tgt_len]
        non_tensor_batch = prompts.non_tensor_batch
        
        batch_size = idx.size(0)
        
        # Process target input ids - add eos token if needed
        tgt_list = [
            _pre_process_inputs_right_pad(self.pad_token_id, tgt_input_ids[i]) for i in range(batch_size)
        ]
        tgt_list = [
            tgt_list[i] + [self.tokenizer.eos_token_id,] if len(tgt_list[i]) > 0 else tgt_list[i]
            for i in range(batch_size)
        ]
        
        # For off-policy data, prefix_ratio is always 1.0 (use all target data)
        # No repetition needed for off-policy data
        prefix_ratios = [1.0] * len(tgt_list)
        
        # Use entire target as response (prefix_ratio = 1.0)
        response_list = tgt_list
        
        # Prepare response tensor
        resp_max_len = max([len(resp) for resp in response_list]) if response_list else 0
        response = torch.ones(len(response_list), max(resp_max_len, self.config.response_length)).fill_(self.pad_token_id)
        
        # Fill response tensor and create prefix mask
        prefix_mask = torch.zeros([len(response_list), self.config.response_length], dtype=torch.bool).to(idx.device)
        
        for i in range(len(response_list)):
            resp_len = min(len(response_list[i]), self.config.response_length)
            if resp_len > 0:
                response[i][:resp_len] = torch.tensor(response_list[i][:resp_len])
                # All tokens are from off-policy data (prefix)
                prefix_mask[i, :resp_len] = True
        
        response = response.to(idx.device)[:, :self.config.response_length].to(idx.dtype)
        
        # No repetition for off-policy data - keep original batch_size
        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)
        
        # Create position IDs and attention mask for full sequence
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # response_attention_mask = get_eos_mask(
        #     response_id=response,
        #     eos_token=eos_token_id,
        #     dtype=attention_mask.dtype)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )

        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # Construct output batch
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'tgt_input_ids': tgt_input_ids,
                'prefix_mask': prefix_mask,
            },
            batch_size=batch_size)
        
        meta_info = {
            'prefix_ratios': prefix_ratios,
        }
        
        return DataProto(batch=batch, meta_info=meta_info,  non_tensor_batch=non_tensor_batch)


    @torch.no_grad()
    def generate_on_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences using vLLM engine with retry logic for failures.

        Args:
            prompts (DataProto): Input prompts containing batch data with input_ids, attention_mask,
                position_ids and meta_info.
            max_retries (int, optional): Maximum number of retries on failure. Defaults to 1e9.
            **kwargs: Additional sampling parameters to override defaults.

        Returns:
            DataProto: Generated sequences containing:
                - prompts: Original input token ids
                - responses: Generated response token ids
                - input_ids: Concatenated prompt and response tokens
                - attention_mask: Attention mask for full sequence
                - position_ids: Position ids for full sequence

        Raises:
            RuntimeError: If generation fails after max_retries attempts.
        """
        # if on_num is None:
        #     on_num = self.config.n

        # max_retries = 1e9
        # for attempt in range(max_retries):
        #     try:
        # Rebuild vLLM cache engine if configured
        # if self.config.free_cache_engine:
        #     self.inference_engine.init_cache_engine()
            
        # Extract input tensors from prompt batch
        print('Love from MIXvLLMRollout.generate_on_sequences')
        on_num = prompts.meta_info['on_num']
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        is_validate = prompts.meta_info.get("validate", False)

        # we use repeat to get n generations for each prompt
        # Pre-process input token ids
        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch
        idx_list = [
            _pre_process_inputs(self.pad_token_id, idx[i])
            for i in range(batch_size)
        ]
        # repeat idx_list to get n generations for each prompt
        do_sample = prompts.meta_info.get('do_sample', True)
        if do_sample:
            idx_list = sum([[idx_list[i]] * on_num for i in range(len(idx_list))], [])
        
        prefix_ratios = None

        # Configure sampling parameters
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1
            }
        if prompts.meta_info.get('val_temperature', None):
            kwargs['temperature'] = prompts.meta_info['val_temperature']

        # we use n=1 because we have repeated the idx_list to get n generations for each prompt
        kwargs['n'] = 1

        # Generate sequences
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # Process outputs
        # response = output[0].to(idx.device)
        resp_list = []
        for output_request in output:
            response_ids = torch.tensor(output_request.outputs[0].token_ids, device = idx.device)
            resp = _pre_process_inputs_right_pad(self.pad_token_id, response_ids)
            resp_list.append(resp)
        
        response = pad_2d_list_to_length(resp_list, self.pad_token_id, max_length=self.config.response_length).to(
            idx.device
        )

        
        prefix_mask = torch.zeros([batch_size, self.config.response_length], dtype=torch.bool).to(idx.device)
        
        # Pad sequences if needed
        # if not is_validate: 
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(
                response, self.config.response_length, self.pad_token_id)

        
        print(f"=== Debug non_tensor_batch before expansion ===")
        print(f"original batch_size: {batch_size}")
        print(f"on_num: {on_num}")
        print(f"do_sample: {do_sample}")

        for key, value in non_tensor_batch.items():
            if isinstance(value, (list, tuple)):
                print(f"  {key}: type=list/tuple, len={len(value)}")
                if len(value) > 0:
                    print(f"    first item type: {type(value[0])}")
            elif hasattr(value, 'shape'):
                print(f"  {key}: type=tensor, shape={value.shape}")
            else:
                print(f"  {key}: type={type(value)}, value={value}")
        
        
        # Handle multiple samples per prompt
        if on_num > 1 and do_sample:
            idx = idx.repeat_interleave(on_num, dim=0)
            prefix_mask = prefix_mask.repeat_interleave(on_num, dim=0)
            
            tgt_input_ids = None
            attention_mask = attention_mask.repeat_interleave(
                on_num, dim=0)
            position_ids = position_ids.repeat_interleave(
                on_num, dim=0)

            # 扩展 non_tensor_batch 中的数据
            expanded_non_tensor_batch = {}
            original_batch_size = batch_size
            
            for key, value in non_tensor_batch.items():
                if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == original_batch_size:
                    if key == 'raw_prompt_ids':
                        # 特殊处理 raw_prompt_ids：它是包含列表的数组
                        # expanded_list = []
                        # for i in range(original_batch_size):
                        #     # 对每个原始item重复 on_num 次
                        #     for _ in range(on_num):
                        #         expanded_list.append(value[i])
                        # expanded_non_tensor_batch[key] = np.array(expanded_list, dtype=object)
                        # print(f"Expanded object array {key}: {value.shape} -> {expanded_non_tensor_batch[key].shape}")
                        print(f"Excluding {key} from output")
                        continue
                    elif isinstance(value, np.ndarray):
                        # 对普通 numpy 数组进行扩展
                        expanded_non_tensor_batch[key] = np.repeat(value, on_num)
                        print(f"Expanded numpy {key}: {value.shape} -> {expanded_non_tensor_batch[key].shape}")
                    elif hasattr(value, 'repeat_interleave'):
                        # 对 torch 张量进行扩展
                        expanded_non_tensor_batch[key] = value.repeat_interleave(on_num, dim=0)
                        print(f"Expanded tensor {key}: {value.shape} -> {expanded_non_tensor_batch[key].shape}")
                    else:
                        expanded_non_tensor_batch[key] = value
                        print(f"Kept {key} unchanged: type={type(value)}, shape={value.shape}")
                else:
                    # 其他情况保持原样
                    # expanded_non_tensor_batch[key] = value
                    expanded_non_tensor_batch = {k: v for k, v in non_tensor_batch.items() if k != 'raw_prompt_ids'}
                    print(f"Kept {key} unchanged: type={type(value)}")
                    
            batch_size = batch_size * on_num
            
            print(f"=== After expansion ===")
            print(f"new batch_size: {batch_size}")
            for key, value in expanded_non_tensor_batch.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}")
                    if key == 'raw_prompt_ids' and len(value) > 0:
                        print(f"    first item type: {type(value[0])}, first item: {value[0][:10] if hasattr(value[0], '__len__') else value[0]}")
        else:
            expanded_non_tensor_batch = non_tensor_batch

        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)

        # Create position IDs and attention mask for full sequence
        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(
            batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids],
                                dim=-1)
        # response_attention_mask = get_eos_mask(
        #     response_id=response,
        #     eos_token=eos_token_id,
        #     dtype=attention_mask.dtype)

        response_attention_mask = get_response_mask(
                response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
            )

        attention_mask = torch.cat(
            (attention_mask, response_attention_mask), dim=-1)

        # Construct output batch
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            batch_size=batch_size)
        
        if prefix_mask.shape[0] > 0:
            batch['prefix_mask'] = prefix_mask

        # Free cache if configured
        # if self.config.free_cache_engine:
        #     self.inference_engine.free_cache_engine()
        problematic_keys = []
        for key, value in expanded_non_tensor_batch.items():
            if hasattr(value, '__len__') and len(value) != batch['input_ids'].shape[0]:
                problematic_keys.append(key)

        if prefix_ratios is not None:
            meta_info = {
                'prefix_ratios': prefix_ratios,
            }
            if problematic_keys:
                for key in problematic_keys:
                    meta_info[key] = expanded_non_tensor_batch.pop(key)

            # 在 return DataProto 之前添加
            print(f"=== Final consistency check ===")
            print(f"batch['input_ids'].shape[0]: {batch['input_ids'].shape[0]}")
            print(f"batch_size: {batch_size}")

            for key, value in (expanded_non_tensor_batch if 'expanded_non_tensor_batch' in locals() else non_tensor_batch).items():
                if hasattr(value, '__len__'):
                    print(f"non_tensor_batch['{key}'] length: {len(value)}")
            
            return DataProto(batch=batch, meta_info=meta_info, non_tensor_batch = expanded_non_tensor_batch)
        else:
            return DataProto(batch=batch, non_tensor_batch = expanded_non_tensor_batch)

            # except Exception as e:
            #     traceback.print_exc()
            #     print("Restarting vLLM due to error: ", e)
            #     print("Retrying...")

            #     # Clean up and restart engine
            #     torch.cuda.empty_cache()
            #     if hasattr(self.inference_engine, 'free_cache_engine'):
            #         self.inference_engine.free_cache_engine()
            #     del self.inference_engine

            #     # Reinitialize engine with same parameters
            #     self.inference_engine = LLM(
            #         self.actor_module,
            #         tokenizer=self.tokenizer,
            #         model_hf_config=self.model_hf_config,
            #         tensor_parallel_size=self.tensor_parallel_size,
            #         dtype=self.config.dtype,
            #         enforce_eager=self.config.enforce_eager,
            #         gpu_memory_utilization=self.config.gpu_memory_utilization,
            #         skip_tokenizer_init=False,
            #         max_model_len=self.config.prompt_length +
            #         self.config.response_length,
            #         load_format=self.config.load_format)
            #     print("vLLM is ready to roll!")

            #     if attempt < max_retries - 1:
            #         continue















        
        



                
                
                    



                    




                        












            













        




            



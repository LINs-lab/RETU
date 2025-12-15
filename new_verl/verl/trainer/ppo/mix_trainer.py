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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    _compute_response_info
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from tensordict import TensorDict
import math
import gc
import psutil


WorkerType = type[Worker]

def create_rl_train_dataset_with_targets(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files. 数据文件路径列表
        data_config: The data config. 数据配置对象，包含数据集的各种设置
        tokenizer (Tokenizer): The tokenizer.  分词器，用于文本处理
        processor (Processor): The processor.  数据处理器，用于数据预处理
        is_train: 布尔值，指示是否为训练模式（默认为True）

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset_with_target_2 import RLHFDatasetWithTarget_2

    train_dataset = RLHFDatasetWithTarget_2(
        config = data_config,
        parquet_files = data_paths,
        tokenizer = tokenizer,
    )

    return train_dataset


def generate_off_sequences(prompts, actor_config, tokenizer):
    from verl.utils.torch_functional import get_response_mask
    def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids):
        # 移除左侧填充，只保留右侧填充的有效token
        non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
        if len(non_pad_index) == 0:
            return []
        else:
            # 找到最后一个非填充token的位置
            last_non_pad_index = non_pad_index[-1][0]
            # 保留从开始到这个位置的所有token
            token_ids = prompt_token_ids[:last_non_pad_index+1].tolist()
        return token_ids

    idx = prompts.batch['input_ids']
    attention_mask = prompts.batch['attention_mask']
    position_ids = prompts.batch['position_ids']
    # eos_token_id = prompts.meta_info['eos_token_id']
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    response_length = actor_config.rollout.response_length

    tgt_input_ids = prompts.batch['tgt_input_ids']  # [bsz, tgt_len]
    non_tensor_batch = prompts.non_tensor_batch
    batch_size = idx.size(0)
    # Process target input ids - add eos token if needed
    tgt_list = [
        _pre_process_inputs_right_pad(pad_token_id, tgt_input_ids[i]) for i in range(batch_size)
    ]
    tgt_list = [
        tgt_list[i] + [eos_token_id,] if len(tgt_list[i]) > 0 else tgt_list[i]
        for i in range(batch_size)
    ]
    
    # For off-policy data, prefix_ratio is always 1.0 (use all target data)
    # No repetition needed for off-policy data
    prefix_ratios = [1.0] * len(tgt_list)
    
    # Use entire target as response (prefix_ratio = 1.0)
    response_list = tgt_list
    
    # Prepare response tensor
    resp_max_len = max([len(resp) for resp in response_list]) if response_list else 0
    response = torch.ones(len(response_list), max(resp_max_len, response_length)).fill_(pad_token_id)
    
    # Fill response tensor and create prefix mask
    prefix_mask = torch.zeros([len(response_list), response_length], dtype=torch.bool).to(idx.device)
    
    for i in range(len(response_list)):
        resp_len = min(len(response_list[i]), response_length)
        if resp_len > 0:
            response[i][:resp_len] = torch.tensor(response_list[i][:resp_len])
            # All tokens are from off-policy data (prefix)
            prefix_mask[i, :resp_len] = True
    
    response = response.to(idx.device)[:, : response_length].to(idx.dtype)
    
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



def compute_partial_entropy(concernd_token_mask, entropys, response_masks, loss_agg_mode):
    if concernd_token_mask.any():
        concerned_entropys = entropys[concernd_token_mask]  # shape: [2, 6]
        concerned_masks = response_masks[concernd_token_mask]  # shape: [2, 6]
        # print("Correct entropys:", concerned_entropys)
        # print("Correct masks:", concerned_masks)
        concerned_entropy = agg_loss(loss_mat=concerned_entropys, loss_mask=concerned_masks, loss_agg_mode = loss_agg_mode)
        concerned_entropy=concerned_entropy.detach().item()
    else:
        concerned_entropy = float('nan')
    return concerned_entropy

def check_memory_usage(stage=""):
    """Monitor memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 / 1024 / 1024
    print(f"[{stage}] Memory usage: {memory_gb:.2f} GB")


def memory_cleanup():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# import torch_npu
# def memory_cleanup():
#     """Force memory cleanup for any available device (CUDA, NPU, CPU, etc.)."""
#     gc.collect()
#     torch_npu._C._npu_emptyCache()
#     print('clean npu memory')
        


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    
    elif adv_estimator == AdvantageEstimator.GRPO_SPLIT:
        from verl.trainer.ppo.mix_core_alg import compute_grpo_outcome_advantage_split
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        prefix_mask = data.batch['prefix_mask']
        on_policy_mask = ~prefix_mask.any(-1)
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            on_policy_mask=on_policy_mask,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns


    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class MixRayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()

        # 改成 upt 格式
        # print('collate_fn None?', (collate_fn is None))
        self._create_dataloader_upt(train_dataset, val_dataset, collate_fn, train_sampler)

    def select_on_off_ada_balance(self, on_solve_num: int):
        '''
            自适应平衡策略选择器, 这个函数根据当前"已解决数量"(on_solve_num)来动态调整三种操作的数量，用于实现某种自适应平衡策略。
            Args: 
                on_solve_num, 当前已解决的数量（可能是任务、问题或样本的数量）
            Return:
                on_remove_num: 要移除的on类型数量
                on_add_num: 要添加的on类型数量
                off_add_num: 要添加（或移除，如果为负）的off类型数量
        '''
        if self.config.trainer.unify_strategy == 'switch': # Switch（开关策略）
            on_add_num = 0 # on_add_num 总是0，表示不添加on类型
            if on_solve_num <= self.config.trainer.switch_gate:
                #  初级阶段 (on_solve_num <= switch_gate) ，
                on_remove_num = 8 # 大量移除on类型
                off_add_num = 1 # 少量添加off类型
            elif on_solve_num <= self.config.trainer.switch_gate_off:
                # 中级阶段
                on_remove_num = 8 # 大量移除on类型  
                off_add_num = -1 # 减少off类型（可能是移除）
            else:
                # 高级阶段， 保持现状，不进行操作
                on_remove_num = 0  # 不操作
                off_add_num = 0  # 不操作

            return on_remove_num, on_add_num, off_add_num

        if self.config.trainer.unify_strategy == 'soft':
            on_remove_num = 0 # 不移除on类型
            on_add_num = 0 # 不添加on类型  
            off_add_num = 1 # 总是添加1个off类型

            return on_remove_num, on_add_num, off_add_num
    
    
    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if config.actor_rollout_ref.actor.strategy == "megatron":
                model_parallel_size = (
                    config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                    * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
                )
                assert (
                    n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
                ), (
                    f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                    f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
                )
                megatron_dp = n_gpus // (
                    model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
                )
                minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
            else:
                minimal_bsz = n_gpus

            # 1. Check total batch size for data correctness
            real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
            assert real_train_batch_size % minimal_bsz == 0, (
                f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
                f"({minimal_bsz})"
            )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")
    
    # ----- upt dataloader --- #
    def _create_dataloader_upt(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler


        # 使用 create_rl_train_dataset_with_targets 构造train dataset
        if train_dataset is None:
            train_dataset = create_rl_train_dataset_with_targets(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        
        # 构造 train_sampler
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        
        
        # Import both collate functions
        from verl.utils.dataset.rl_dataset_with_target_2 import collate_fn as train_collate_fn
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
        train_collate = train_collate_fn  # Use target collate_fn for training
        val_collate = default_collate_fn   # Use default for validation
        # # Use the correct collate_fn for train and val
        # if collate_fn is None:
        #     train_collate = train_collate_fn  # Use target collate_fn for training
        #     val_collate = default_collate_fn   # Use default for validation
        # else:
        #     train_collate = collate_fn
        #     val_collate = collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        test_samples = [self.train_dataset[i] for i in range(min(2, len(self.train_dataset)))]
        test_batch = collate_fn(test_samples)
        print(f"[DEBUG] Test batch keys from collate_fn: {list(test_batch.keys())}")
        print(f"[DEBUG] 'tgt_input_ids' in test batch: {'tgt_input_ids' in test_batch}") # 这里是有tgt_input_ids 的
        # ['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids', 'data_source', 'ability', 'reward_model',
        #  'extra_info', 'DeepSeek-R1-Distill-Qwen-1.5B', 'DeepSeek-R1-Distill-Qwen-32B', 'DeepSeek-R1-Distill-Qwen-7B', 
        # 'answer', '__index_level_0__', 'raw_prompt_ids', 'index', 'tools_kwargs', 'interaction_kwargs']
        #

        
        # 获取 self.train_dataloader， self.val_dataloader 
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=train_collate,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=val_collate,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        # 总的训练steps
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
    
    # ----------------------- #




    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as parquet."""
        # def prompt_2(x):
        #     input=x['input']
        #     problem=input.split('system\nYou are a helpful assistant.\nuser\n')[-1].split('\nassistant\n')[0]
        #     problem_template = 'system\nYou are a helpful assistant.\nuser\n' +  problem + '\nassistant\n' 
        #     if problem_template != input:
        #         raise Exception
        #     else:
        #         prompt = [{'content': problem, 'role':'user'}]
        #     return prompt 
        import re
        def prompt_2(x):
            input_text = x['input']
            pattern = r'\nuser\n(.*?)\nassistant\n'
            match = re.search(pattern, input_text, re.DOTALL)
            if not match:
                raise Exception("Could not extract problem content between \\nuser\\n and \\nassistant\\n")
            problem = match.group(1)
            return [{'content': problem, 'role':'user'}]

        def add_pos_neg_responses_fast(df):
            """
            使用列表推导式，最高效的方式
            """
            # 一次性提取所有数据
            outputs = df['output'].tolist()
            scores = df['score'].tolist()
            
            # 使用列表推导式批量处理
            pos_res = [
                [out for out, sc in zip(output, score) if sc == 1]
                for output, score in zip(outputs, scores)
            ]
            
            neg_res = [
                [out for out, sc in zip(output, score) if sc == 0]
                for output, score in zip(outputs, scores)
            ]
            
            df['pos_res'] = pos_res
            df['neg_res'] = neg_res
            
            return df
        
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        df_filename = os.path.join(dump_path, f"{self.global_steps}.parquet") 

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            # "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        data_source=reward_extra_infos_dict.pop('data_source')
        ability=reward_extra_infos_dict.pop('ability')
        reward_model=reward_extra_infos_dict.pop('reward_model')
        extra_info=reward_extra_infos_dict.pop('extra_info')

        lines = []
        unique_prompt_set = dict()
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            unique_prompt=entry['input']
            # 初始化prompt 对应的信息
            if unique_prompt not in unique_prompt_set:
                unique_prompt_set[unique_prompt]={}
                data_dict = unique_prompt_set[unique_prompt]  
                data_dict['data_source'] = entry['data_source']  
                try:
                    data_dict['ability'] = entry['ability']  
                except:
                    data_dict['ability'] = 'math'
                data_dict['input'] = entry["input"]
                data_dict['prompt'] = prompt_2(entry)
                data_dict['answer'] = entry['reward_model']['ground_truth']
                data_dict['reward_model'] = entry["reward_model"]
                data_dict['extra_info'] = entry["extra_info"]
                data_dict['output'] = [entry["output"]]
                data_dict['score'] = [entry["score"]]
                try:
                    data_dict['reward'] = [entry["reward"]]
                except:
                    data_dict['reward'] = [entry["score"]]
                data_dict['step'] = [entry["step"]]
                data_dict['win_rate'] = sum(data_dict['reward']) / len(data_dict['reward'])
        
            else:
                # 找到 unique_prompt 对应的group
                data_dict = unique_prompt_set[unique_prompt]    
                data_dict['output'].append(entry["output"])
                data_dict['score'].append(entry["score"])
                try:
                    data_dict['reward'].append(entry["reward"])
                except:
                    data_dict['reward'].append(entry["score"])
                data_dict['step'].append(entry["step"])
                data_dict['win_rate'] = sum(data_dict['reward']) / len(data_dict['reward'])            
            # lines.append(json.dumps(entry, ensure_ascii=False))
        
        lines_combined_df = pd.DataFrame(list(unique_prompt_set.values()))
        
        lines_combined_df=add_pos_neg_responses_fast(lines_combined_df)
        filename_combined = filename.split('.')[0]+'_combine.parquet'
        # lines_combined_df.to_parquet(filename_combined)
        lines_combined_df.to_parquet(df_filename)
        print(f"Dumped generations to {df_filename}")


    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto, include_tgt_input_ids: bool = False) -> DataProto:
        # 创建一个集合 {"data_source", "reward_model", "extra_info", "uid"}
        # 与 batch.non_tensor_batch 中实际存在的键取交集
        # 目的是找出既在预设列表中又实际存在于批次中的键
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        if include_tgt_input_ids is False:
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"] 
        else:
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids", "tgt_input_ids"] 
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        non_tensor_batch_keys_to_pop = non_tensor_batch_keys_to_pop - set({'__index_level_0__', 'DeepSeek-R1-Distill-Qwen-7B', 'DeepSeek-R1-Distill-Qwen-1.5B', 'DeepSeek-R1-Distill-Qwen-32B', 'tools_kwargs', 'interaction_kwargs'})
        
        #  
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop, # ["input_ids", "attention_mask", "position_ids"] 
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop), # non_tensor_batch 去除 {"data_source", "reward_model", "extra_info", "uid"} 的部分
        )


        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    




    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            
            try:
                reward_extra_infos_dict['data_source'].extend(test_batch.non_tensor_batch["data_source"])
            except:
                print('no data_source')
            try:
                reward_extra_infos_dict['ability'].extend(test_batch.non_tensor_batch["ability"])
            except:
                print('no ability')
            try:
                reward_extra_infos_dict['reward_model'].extend(test_batch.non_tensor_batch["reward_model"])
            except:
                print('no reward_model')
            
            try:
                reward_extra_infos_dict['extra_info'].extend(test_batch.non_tensor_batch["extra_info"])
            except:
                print('no extra_info')
            
            
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        device_num_per_node = self.config.trainer.n_gpus_per_node
        node_num = self.config.trainer.nnodes
        self.device_total_num = device_num_per_node * node_num

        self.global_steps = 0
        pre_entropy = 0.0
        pre_pos_entropy = 0.0
        pre_neg_entropy = 0.0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # TODO: add add_tgt_with_acc/sfted_data_item_list to config
        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.

        if self.config.trainer.remove_sfted_data:
            sfted_data_item_list = []
        
        
        for epoch in range(self.config.trainer.total_epochs):

            # --- upt ----# 
            if self.config.trainer.remove_sfted_data:
                print('hi there')
                if len(sfted_data_item_list) > 0:
                    self.train_dataset.remove_data(sfted_data_item_list)

                    # Reconstruct train_dataloader
                    from torch.utils.data import DataLoader, SequentialSampler
                    # 构造 train_sampler
                    if train_sampler is None:
                        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
                    # 构造 分批器 collate_fn
                    if collate_fn is None:
                        from verl.utils.dataset.rl_dataset_with_target_2  import collate_fn as train_collate_fn
                        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
                        collate_fn = default_collate_fn
                    num_workers = self.config.data["dataloader_num_workers"]
                    # 更新 self.train_dataloader
                    self.train_dataloader = StatefulDataLoader(
                        dataset=self.train_dataset,
                        batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                        num_workers=num_workers,
                        drop_last=True,
                        collate_fn=train_collate_fn,
                        sampler=train_sampler,
                    )
                sfted_data_item_list = []
                memory_cleanup()
            
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                if 'tgt_input_ids' not in batch_dict:
                    print("[WARNING] tgt_input_ids not in batch_dict from dataloader!")
                    # Check first few elements to understand data structure
                    for key in list(batch_dict.keys())[:3]:
                        if hasattr(batch_dict[key], 'shape'):
                            print(f"  {key}: shape={batch_dict[key].shape}")


                # --- upt ----# 
                if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                    # Before popping, copy the required data first
                    batch.batch['raw_input_ids'] = batch.batch['input_ids'].clone()
                    batch.batch['raw_attention_mask'] = batch.batch['attention_mask'].clone()
                    batch.batch['raw_position_ids'] = batch.batch['position_ids'].clone()
                    # pop those keys for generation
                    # gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    gen_batch = self._get_gen_batch(batch)
                    # 将 on_num 添加到 gen_batch 的 meta_info 中
                    if not hasattr(gen_batch, 'meta_info'):
                        gen_batch.meta_info = {}
                    gen_batch.meta_info['on_num'] = self.config.actor_rollout_ref.rollout.n_verify
                    # meta_info = {
                    #     "eos_token_id": self.model_config.generation_config.eos_token_id
                    #     if self.model_config.generation_config is not None
                    #     else self.model_config.tokenizer.eos_token_id,
                    #     "pad_token_id": self.model_config.generation_config.pad_token_id
                    #     if self.model_config.generation_config is not None
                    #     else self.model_config.tokenizer.pad_token_id,
                    # }
                else:
                    # gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                    gen_batch = self._get_gen_batch(batch, include_tgt_input_ids=True)
                    print(f'[DEBUG] gen_batch.non_tensor_batch.keys():  {gen_batch.non_tensor_batch.keys()}')

                gen_batch.meta_info['global_steps'] = self.global_steps
                # ------------ #                
                # gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                # gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                
                
                with marked_timer("step", timing_raw): # 在这里计算per step 的用时
                    # 执行rollout, rollout 的结果存在了gen_batch_output 当中
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode: # 非 异步rollout 
                            if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                                ## TODO: 调整 rollout
                                ## 在这里添加调试代码
                                print("=== Debug gen_batch before generate_on_sequences ===")
                                print(f"gen_batch type: {type(gen_batch)}")
                                print(f"Is DataProto: {isinstance(gen_batch, DataProto)}")
                                gen_batch_output = self.actor_rollout_wg.generate_on_sequences(gen_batch)
                            else:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        print('finish rollout .......')
                        # timing_raw.update(gen_batch_output.meta_info["timing"])
                        # gen_batch_output.meta_info.pop("timing", None)
                    
                    # This code matches a prompt ID with its N responses. 此时 batch 还未repeating, 可以对每个batch 分配一个 uid
                    # uid 会放在non_tensor_batch 当中 
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )


                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    

                    
                    # ---- upt ---- # 
                    # 因为在rollout 环节，每个 prompts 复制了n次， 所以 为了要把 gen_batch_output 并入 batch 当中，batch 需要repeating
                    if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_verify, interleave=True)
                    else:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    print('batch len: ', len(batch.batch))
                    print('gen_batch_output len: ', len(gen_batch_output.batch))
                
                    batch = batch.union(gen_batch_output)

                    if self.config.trainer.add_full_target_when_none:
                        pass


                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)


                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # 计算rewards
                    with marked_timer("reward", timing_raw, color="yellow"): # 统计 reward 的计算耗时
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            # batch 里所有sample （tbz * n） 的rewards 都存在了reward_tensor当中 
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                    
                    # ---923，15:13 补充 --- #
                    # reward_tensor 要存到 batch.batch['token_level_scores'] 当中
                    batch.batch['token_level_scores'] = reward_tensor 
                    
                    
                    # calculate solve_none/solve_all/solve_partial
                    uids = batch.non_tensor_batch['uid']
                    unique_uids = np.unique(uids)
                    valid_mask = torch.ones(len(uids), dtype=torch.bool)
                    solve_none = 0
                    solve_all = 0

                    fail_value = 0
                    success_value = 1
                    format_value = -1

                    for i in range(0,9):
                        if f'batch/solve_{str(i)}' not in metrics:
                            metrics[f'batch/solve_{str(i)}'] = 0
                        else:
                            pass

                    # 遍历所有的 prompt index，
                    for uid in unique_uids:
                        # 获取 uid_mask, 目的为了筛出 属于同一个prompt的 responses
                        uid_mask = uids == uid # 
                        # 获取 prompt 对应的每个 trajectory 的reward
                        uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                        # print(len(uid_rewards))
                        # print('uid_rewards: ', uid_rewards)
                        correct_num = uid_rewards.sum()
                        metrics[f'batch/solve_{str(int(correct_num))}'] = metrics[f'batch/solve_{str(int(correct_num))}'] + 1

                        # Check if all rewards are 0 or all are 1 for this uid
                        if (uid_rewards == 0).all(): 
                            # 如果rewards 都是 0, 意味着 此prompt非常困难，解不出来
                            valid_mask[uid_mask] = False 
                            solve_none += 1
                        elif (uid_rewards == 1).all(): 
                            # 如果rewards 都是 1, 意味着  此prompt非常简单，全解出来
                            valid_mask[uid_mask] = False
                            solve_all += 1
                    # Log to metrics
                    
                    metrics['batch/solve_none'] = solve_none #  batch/solve_none -> 一个tbz中，全解不出来的prompts
                    metrics['batch/solve_all'] = solve_all # batch/solve_all -> 一个tbz中，全解出来的prompts
                    metrics['batch/solve_partial'] = len(unique_uids) - solve_none - solve_all # batch/solve_partial -> 一个tbz中, 部分解出的prompts                    
                    
                    
                    # upt， 只在特定策略模式下执行（排除'no'和'soft'策略）
                    if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                        # Collect all uid information that needs on-policy data generation
                        all_on_batches = []  # 存储所有新生成的on-policy数据批次
                        uid_balance = {}    # 存储每个uid的平衡策略参数
                        uid_raw_data = {}   # 存储每个uid的原始数据
                        for uid in unique_uids:
                            # 根据每个uid的成功次数决定要添加/移除的数据量
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)
                            # Count on_solve_num for this uid
                            on_solve_num = (uid_rewards == success_value).sum().item()
                            on_remove_num, on_add_num, off_add_num = self.select_on_off_ada_balance(on_solve_num)
                            uid_balance[uid] = (on_remove_num, on_add_num, off_add_num)

                            #  存储原始数据
                            uid_indices = np.where(uid_mask)[0]
                            first_idx = uid_indices[0]
                            uid_raw_data[uid] = {
                                    'input_ids': batch.batch['raw_input_ids'][first_idx:first_idx+1],
                                    'attention_mask': batch.batch['raw_attention_mask'][first_idx:first_idx+1],
                                    'position_ids': batch.batch['raw_position_ids'][first_idx:first_idx+1],
                                }
                            # 生成新数据（如果需要）
                            if on_add_num != 0:                                
                                # Extract data for this prompt and repeat on_add_num times
                                prompt_data = {}
                                prompt_data['input_ids'] = batch.batch['raw_input_ids'][first_idx:first_idx+1].repeat(on_add_num, 1)
                                prompt_data['attention_mask'] = batch.batch['raw_attention_mask'][first_idx:first_idx+1].repeat(on_add_num, 1)
                                prompt_data['position_ids'] = batch.batch['raw_position_ids'][first_idx:first_idx+1].repeat(on_add_num, 1)
                                prompt_data['tgt_input_ids'] = batch.batch['tgt_input_ids'][first_idx:first_idx+1].repeat(on_add_num, 1)
                                # Extract non_tensor_batch and meta_info from original batch
                                new_non_tensor_batch = {}
                                for key, value in batch.non_tensor_batch.items():
                                    if key == 'uid':
                                        new_non_tensor_batch[key] = np.array([uid] * on_add_num, dtype=object)
                                    else:
                                        # Copy the first sample's value to all new samples
                                        new_non_tensor_batch[key] = np.array([value[first_idx]] * on_add_num, dtype=value.dtype)
                                
                                # 创建新数据批次
                                on_batch = DataProto(
                                    batch=TensorDict(prompt_data, batch_size=[on_add_num]),
                                    non_tensor_batch=new_non_tensor_batch,
                                )
                                
                                all_on_batches.append(on_batch)
                                # Immediately clean up temporary data
                                del prompt_data, new_non_tensor_batch
                        # Clean up original data
                        gc.collect()

                        # If there's data to generate, process it uniformly
                        if all_on_batches:
                            # Merge all on-policy data
                            merged_on_batch_dict = {}
                            merged_on_non_tensor_dict = {}

                            # Merge tensor data
                            for key in all_on_batches[0].batch.keys():
                                if key != 'batch_size':
                                    merged_on_batch_dict[key] = torch.cat([b.batch[key] for b in all_on_batches], dim=0)
                            
                            # Merge non_tensor data
                            for key in all_on_batches[0].non_tensor_batch.keys():
                                merged_on_non_tensor_dict[key] = np.concatenate([b.non_tensor_batch[key] for b in all_on_batches], axis=0)
                            
                            total_on_batch_size = sum(b.batch.batch_size[0] for b in all_on_batches)
                            merged_on_batch_dict = TensorDict(merged_on_batch_dict, batch_size=[total_on_batch_size])
                            
                            combined_on_batch = DataProto(
                                batch=merged_on_batch_dict,
                                non_tensor_batch=merged_on_non_tensor_dict,
                                meta_info={'global_steps': self.global_steps}
                            )

                            on_gen_batch = combined_on_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                            on_gen_batch.meta_info['global_steps'] = self.global_steps

                            # Check if batch size is a multiple of 8, pad if not
                            
                            original_batch_size = on_gen_batch.batch.batch_size[0]
                            remainder = original_batch_size % self.device_total_num
                            if remainder != 0:
                                padding_size = self.device_total_num - remainder
                                
                                # Pad the three types of data by repeating the last sample
                                padded_batch_dict = {}
                                for key in ['input_ids', 'attention_mask', 'position_ids']:
                                    last_sample = on_gen_batch.batch[key][-1:].repeat(padding_size, 1)
                                    padded_batch_dict[key] = torch.cat([on_gen_batch.batch[key], last_sample], dim=0)
                                
                                # Create padded batch
                                padded_batch_size = original_batch_size + padding_size
                                padded_batch_dict = TensorDict(padded_batch_dict, batch_size=[padded_batch_size])
                                
                                on_gen_batch = DataProto(
                                    batch=padded_batch_dict,
                                    meta_info=on_gen_batch.meta_info
                                )
                                del padded_batch_dict
                            
                            # Uniformly generate on-policy sequences
                            on_gen_batch.meta_info['on_num'] = 1
                            on_gen_output = self.actor_rollout_wg.generate_on_sequences(on_gen_batch)

                            # If padding was done before, now need to remove padded data
                            if remainder != 0:
                                # Remove padded data, keep only original data
                                filtered_output_dict = {}
                                for key in on_gen_output.batch.keys():
                                    if key != 'batch_size':
                                        filtered_output_dict[key] = on_gen_output.batch[key][:original_batch_size]
                                
                                filtered_output_dict = TensorDict(filtered_output_dict, batch_size=[original_batch_size])
                                
                                on_gen_output = DataProto(
                                    batch=filtered_output_dict,
                                    meta_info=on_gen_output.meta_info
                                )
                                del filtered_output_dict
                            
                            combined_on_batch = combined_on_batch.union(on_gen_output)
                            # Calculate rewards
                            on_reward_tensor = self.reward_fn(combined_on_batch)
                            combined_on_batch.batch['token_level_scores'] = on_reward_tensor

                            batch.pop(batch_keys=['raw_input_ids', 'raw_attention_mask', 'raw_position_ids'])
                            
                            # Merge into original batch
                            merged_batch_dict = {}
                            merged_non_tensor_dict = {}
                            
                            # Merge tensor data
                            for key in batch.batch.keys():
                                if key != 'batch_size':
                                    merged_batch_dict[key] = torch.cat([batch.batch[key], combined_on_batch.batch[key]], dim=0)
                            
                            # Merge non_tensor data
                            for key in batch.non_tensor_batch.keys():
                                merged_non_tensor_dict[key] = np.concatenate([batch.non_tensor_batch[key], combined_on_batch.non_tensor_batch[key]], axis=0)
                            
                            new_batch_size = batch.batch.batch_size[0] + combined_on_batch.batch.batch_size[0]
                            merged_batch_dict = TensorDict(merged_batch_dict, batch_size=[new_batch_size])
                            
                            batch = DataProto(
                                batch=merged_batch_dict,
                                non_tensor_batch=merged_non_tensor_dict,
                                meta_info=batch.meta_info.copy()
                            )

                            del combined_on_batch, on_gen_output, merged_on_batch_dict, merged_on_non_tensor_dict
                            gc.collect()
                        
                        else:
                            batch.pop(batch_keys=['raw_input_ids', 'raw_attention_mask', 'raw_position_ids'])
                        
                        all_off_batches = []
                        for uid in unique_uids:
                            off_add_num = uid_balance[uid][2]
                            
                            if off_add_num != 0:
                                whether_off = False
                                if off_add_num < 0:
                                    off_add_num = -1 * off_add_num
                                    whether_off = True
                                    
                                uid_mask = uids == uid
                                uid_indices = np.where(uid_mask)[0]
                                first_idx = uid_indices[0]

                                prompt_data = {}
                                prompt_data['input_ids'] = uid_raw_data[uid]['input_ids']
                                prompt_data['attention_mask'] = uid_raw_data[uid]['attention_mask']
                                prompt_data['position_ids'] = uid_raw_data[uid]['position_ids']
                                prompt_data['tgt_input_ids'] = batch.batch['tgt_input_ids'][first_idx:first_idx+1]

                                other_off_add_num = off_add_num - 1
                                if other_off_add_num > 0:
                                    other_off_data = self.train_dataloader.dataset.random_get(num=other_off_add_num)
                                    prompt_data['input_ids'] = torch.cat([prompt_data['input_ids'], other_off_data['input_ids']], dim=0)
                                    prompt_data['attention_mask'] = torch.cat([prompt_data['attention_mask'], other_off_data['attention_mask']], dim=0)
                                    prompt_data['position_ids'] = torch.cat([prompt_data['position_ids'], other_off_data['position_ids']], dim=0)
                                    prompt_data['tgt_input_ids'] = torch.cat([prompt_data['tgt_input_ids'], other_off_data['tgt_input_ids']], dim=0)

                                    # Extract non_tensor_batch and meta_info from original batch
                                new_non_tensor_batch = {}
                                for key, value in batch.non_tensor_batch.items():
                                    if key == 'uid':
                                        new_non_tensor_batch[key] = np.array([uid] * off_add_num, dtype=object)
                                    else:
                                        # Copy the first sample's value to all new samples
                                        new_non_tensor_batch[key] = np.array([value[first_idx]] * off_add_num, dtype=value.dtype)

                                if whether_off:
                                    prompt_data['whether_off'] = torch.tensor([True] * off_add_num, dtype=torch.bool)
                                else:
                                    prompt_data['whether_off'] = torch.tensor([False] * off_add_num, dtype=torch.bool)
                                
                                # Create DataProto object
                                off_batch = DataProto(
                                    batch=TensorDict(prompt_data, batch_size=[off_add_num]),
                                    non_tensor_batch=new_non_tensor_batch,
                                )
                                all_off_batches.append(off_batch)
                
                        # If there's data to generate, process it uniformly
                        if all_off_batches:
                            # Merge all on-policy data
                            merged_off_batch_dict = {}
                            merged_off_non_tensor_dict = {}

                            # Merge tensor data
                            for key in all_off_batches[0].batch.keys():
                                if key != 'batch_size':
                                    merged_off_batch_dict[key] = torch.cat([b.batch[key] for b in all_off_batches], dim=0)
                            
                            # Merge non_tensor data
                            for key in all_off_batches[0].non_tensor_batch.keys():
                                merged_off_non_tensor_dict[key] = np.concatenate([b.non_tensor_batch[key] for b in all_off_batches], axis=0)
                            
                            total_off_batch_size = sum(b.batch.batch_size[0] for b in all_off_batches)
                            merged_off_batch_dict = TensorDict(merged_off_batch_dict, batch_size=[total_off_batch_size])

                            combined_off_batch = DataProto(
                                batch=merged_off_batch_dict,
                                non_tensor_batch=merged_off_non_tensor_dict,
                                meta_info={'global_steps': self.global_steps}
                            )

                            off_gen_batch = combined_off_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                            off_gen_batch.meta_info['global_steps'] = self.global_steps
                            # Check if batch size is a multiple of 8, pad if not
                            original_batch_size = off_gen_batch.batch.batch_size[0]
                            remainder = original_batch_size % self.device_total_num
                            if remainder != 0:
                                padding_size = self.device_total_num - remainder

                                # Pad the four types of data by repeating the last sample
                                padded_batch_dict = {}
                                for key in ['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids']:
                                    last_sample = off_gen_batch.batch[key][-1:].repeat(padding_size, 1)
                                    padded_batch_dict[key] = torch.cat([off_gen_batch.batch[key], last_sample], dim=0)
                                
                                # Create padded batch
                                padded_batch_size = original_batch_size + padding_size
                                padded_batch_dict = TensorDict(padded_batch_dict, batch_size=[padded_batch_size])

                                off_gen_batch = DataProto(
                                    batch=padded_batch_dict,
                                    meta_info=off_gen_batch.meta_info
                                )
                                del padded_batch_dict
                            # Uniformly generate off-policy sequences
                            # off_gen_output = self.actor_rollout_wg.generate_off_sequences(off_gen_batch)
                            off_gen_output = generate_off_sequences(off_gen_batch, self.config.actor_rollout_ref, self.tokenizer)
                                                        
                            # If padding was done before, now need to remove padded data
                            if remainder != 0:
                                # Remove padded data, keep only original data
                                filtered_output_dict = {}
                                for key in off_gen_output.batch.keys():
                                    if key != 'batch_size':
                                        filtered_output_dict[key] = off_gen_output.batch[key][:original_batch_size]
                                
                                filtered_output_dict = TensorDict(filtered_output_dict, batch_size=[original_batch_size])
                                
                                off_gen_output = DataProto(
                                    batch=filtered_output_dict,
                                    meta_info=off_gen_output.meta_info
                                )
                                del filtered_output_dict
                            
                            combined_off_batch = combined_off_batch.union(off_gen_output)

                            # Calculate rewards
                            off_reward_tensor = self.reward_fn(combined_off_batch)
                            combined_off_batch.batch['token_level_scores'] = off_reward_tensor

                            # Merge into original batch
                            if 'whether_off' in batch.batch:
                                batch.batch['whether_off'] = torch.tensor([False] * batch.batch.batch_size[0], dtype=torch.bool)
                            
                            merged_batch_dict = {}
                            merged_non_tensor_dict = {}

                            # Merge tensor data
                            # 在原来的代码之前添加调试
                            print(f"=== Debug batch keys at line 1946 ===")
                            print(f"batch.batch keys: {list(batch.batch.keys())}")
                            print(f"combined_off_batch.batch keys: {list(combined_off_batch.batch.keys())}")

                            if "response_mask" not in combined_off_batch.batch.keys():
                                combined_off_batch.batch["response_mask"] = compute_response_mask(combined_off_batch)

                            for key in batch.batch.keys():
                                if key != 'batch_size':
                                    merged_batch_dict[key] = torch.cat([batch.batch[key], combined_off_batch.batch[key]], dim=0)
                            
                            # Merge non_tensor data
                            for key in batch.non_tensor_batch.keys():
                                merged_non_tensor_dict[key] = np.concatenate([batch.non_tensor_batch[key], combined_off_batch.non_tensor_batch[key]], axis=0)
                            
                            new_batch_size = batch.batch.batch_size[0] + combined_off_batch.batch.batch_size[0]
                            merged_batch_dict = TensorDict(merged_batch_dict, batch_size=[new_batch_size])

                            batch = DataProto(
                                batch=merged_batch_dict,
                                non_tensor_batch=merged_non_tensor_dict,
                                meta_info=batch.meta_info.copy()
                            )

                            del combined_off_batch, off_gen_output, merged_off_batch_dict, merged_off_non_tensor_dict
                            gc.collect()

                        if self.config.trainer.remove_on or self.config.trainer.unify_strategy == 'switch':
                            if self.config.trainer.unify_strategy != 'switch':
                                # Calculate the amount of data to remove
                                remove_count = self.config.actor_rollout_ref.rollout.n_verify * self.config.data.train_batch_size
                                
                                # Remove tensor data from batch
                                filtered_batch_dict = {}
                                for key in batch.batch.keys():
                                    if key != 'batch_size':
                                        filtered_batch_dict[key] = batch.batch[key][remove_count:]
                                
                                # Remove data from non_tensor_batch
                                filtered_non_tensor_dict = {}
                                for key in batch.non_tensor_batch.keys():
                                    filtered_non_tensor_dict[key] = batch.non_tensor_batch[key][remove_count:]
                                
                                # Calculate new batch size
                                new_batch_size = batch.batch.batch_size[0] - remove_count
                                filtered_batch_dict = TensorDict(filtered_batch_dict, batch_size=[new_batch_size])

                                # Rebuild batch
                                batch = DataProto(
                                    batch=filtered_batch_dict,
                                    non_tensor_batch=filtered_non_tensor_dict,
                                    meta_info=batch.meta_info.copy()
                                )
                                # Immediately clean up temporary data
                                del filtered_batch_dict, filtered_non_tensor_dict
                                gc.collect()
                            else:
                                keep_mask = torch.ones(batch.batch.batch_size[0], dtype=torch.bool)

                                for uid in unique_uids:
                                    # 每个uid要移除的on-policy数据量
                                    on_remove_num = uid_balance[uid][0]
                                    if on_remove_num != 0:
                                        uid_mask = uids == uid
                                        uid_indices = np.where(uid_mask)[0]

                                        if self.config.trainer.remove_sfted_data:
                                            sfted_data_item = batch.non_tensor_batch['item'][uid_indices[0]]
                                            sfted_data_item_list.append(sfted_data_item)
                                            print('Removing SFTed data:', sfted_data_item)
                                            print(sfted_data_item_list)

                                        # Get prefix_mask for all data corresponding to this uid
                                        uid_prefix_masks = batch.batch['prefix_mask'][uid_indices]
                                        # Determine if it's on-policy data (on-policy data has no True in prefix_mask)
                                        is_on_policy = ~uid_prefix_masks.any(-1)
                                        on_policy_indices = uid_indices[is_on_policy.cpu().numpy()]

                                        keep_mask[on_policy_indices] = False

                                # Remove tensor data from batch
                                filtered_batch_dict = {}
                                for key in batch.batch.keys():
                                    if key != 'batch_size':
                                        filtered_batch_dict[key] = batch.batch[key][keep_mask]
                                
                                # Remove data from non_tensor_batch
                                filtered_non_tensor_dict = {}
                                for key in batch.non_tensor_batch.keys():
                                    filtered_non_tensor_dict[key] = batch.non_tensor_batch[key][keep_mask.cpu().numpy()]
                                
                                    # Calculate new batch size
                                new_batch_size = keep_mask.sum().item()
                                filtered_batch_dict = TensorDict(filtered_batch_dict, batch_size=[new_batch_size])

                                # Rebuild batch
                                batch = DataProto(
                                    batch=filtered_batch_dict,
                                    non_tensor_batch=filtered_non_tensor_dict,
                                    meta_info=batch.meta_info.copy()
                                )
                                # Immediately clean up temporary data  
                                del filtered_batch_dict, filtered_non_tensor_dict, keep_mask
                                gc.collect()
                    else: # "soft" 场景
                        # add on-policy metrics
                        prefix_mask = batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                        metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)

                    if self.config.trainer.unify_strategy == 'soft':
                        on_coef_list = torch.tensor([0.] * batch.batch.batch_size[0])
                        off_coef_list = torch.tensor([0.] * batch.batch.batch_size[0])
                        sft_coef_list = torch.tensor([0.] * batch.batch.batch_size[0])

                        if self.config.trainer.soft_type == 1:
                            coef_dict = {
                                0: (1., 1., 1.), # Should not occur, can be ignored
                                1: (0., 1., 1.),
                                2: (0.125, 1., 0.5),
                                3: (0.25, 1., 0.25),
                                4: (0.5, 1., 0.125),
                                5: (1., 1., 0.),
                                6: (1., 1., 0.),
                                7: (1., 1., 0.),
                                8: (1., 1., 0.),
                            }
                        elif self.config.trainer.soft_type == 2:
                                coef_dict = {
                                    0: (1., 1., 1.), # Should not occur, can be ignored
                                    1: (0., 0., 1.),
                                    2: (0.125, 0., 0.5),
                                    3: (0.25, 0., 0.25),
                                    4: (0.5, 0., 0.125),
                                    5: (1., 0., 0.),
                                    6: (1., 0., 0.),
                                    7: (1., 0., 0.),
                                    8: (1., 0., 0.),
                                }
                        else:
                            coef_dict = {
                                0: (1., 1., 1.), # Should not occur, can be ignored
                                1: (0., 0., 1.),
                                2: (0.125, 0.5, 0.5),
                                3: (0.25, 1., 0.25),
                                4: (0.5, 0.5, 0.125),
                                5: (1., 0.25, 0.),
                                6: (1., 0.125, 0.),
                                7: (1., 0., 0.),
                                8: (1., 0., 0.),
                            }
                        
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            reward_tensor = batch.batch['token_level_scores']
                            uid_rewards = reward_tensor[uid_mask].sum(-1)
                            on_solve_num = (uid_rewards == success_value).sum().item()

                            on_coef, off_coef, sft_coef = coef_dict[on_solve_num]
                            on_coef_list[uid_mask] = on_coef
                            off_coef_list[uid_mask] = off_coef
                            sft_coef_list[uid_mask] = sft_coef

                        batch.batch['on_coef'] = on_coef_list
                        batch.batch['off_coef'] = off_coef_list
                        batch.batch['sft_coef'] = sft_coef_list  
                    
                    # Check if batch size is a multiple of 8, pad if not
                    original_batch_size = batch.batch.batch_size[0]
                    remainder = original_batch_size % self.device_total_num
                    if remainder != 0:
                        padding_size = self.device_total_num - remainder
                        # Pad tensor data in batch.batch by repeating the last sample
                        padded_batch_dict = {}
                        for key in batch.batch.keys():
                            if key in ['on_coef', 'off_coef', 'sft_coef']:
                                last_sample = batch.batch[key][-1:].repeat(padding_size)
                                padded_batch_dict[key] = torch.cat([batch.batch[key], last_sample], dim=0)
                                continue
                            if key != 'batch_size':
                                last_sample = batch.batch[key][-1:].repeat(padding_size, 1)
                                padded_batch_dict[key] = torch.cat([batch.batch[key], last_sample], dim=0)
            
                        # Pad data in batch.non_tensor_batch
                        padded_non_tensor_dict = {}
                        for key in batch.non_tensor_batch.keys():
                            last_sample = np.array([batch.non_tensor_batch[key][-1]] * padding_size, dtype=batch.non_tensor_batch[key].dtype)
                            padded_non_tensor_dict[key] = np.concatenate([batch.non_tensor_batch[key], last_sample], axis=0)

                        # Create padded batch
                        padded_batch_size = original_batch_size + padding_size
                        padded_batch_dict = TensorDict(padded_batch_dict, batch_size=[padded_batch_size])
                
                        batch = DataProto(
                                batch=padded_batch_dict,
                                non_tensor_batch=padded_non_tensor_dict,
                                meta_info=batch.meta_info.copy()
                            )
                        # Immediately clean up temporary data
                        del padded_batch_dict, padded_non_tensor_dict
                        gc.collect()

                    
                    


                    # ---- upt end ---- #
                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        print('computing old_log_prob and entropys...')
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    
                    
                    # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                    # where it is subtracted directly from the policy loss
                            
                    
                    # 计算advantage， 统计耗时
                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # If padding was done before, now need to remove padded data
                        if remainder != 0:
                            # Remove padded data, keep only original data
                            filtered_batch_dict = {}
                            for key in batch.batch.keys():
                                if key != 'batch_size':
                                    filtered_batch_dict[key] = batch.batch[key][:original_batch_size]
                            
                            # Remove data from non_tensor_batch
                            filtered_non_tensor_dict = {}
                            for key in batch.non_tensor_batch.keys():
                                filtered_non_tensor_dict[key] = batch.non_tensor_batch[key][:original_batch_size]
                            filtered_batch_dict = TensorDict(filtered_batch_dict, batch_size=[original_batch_size])
                            
                            batch = DataProto(
                                batch=filtered_batch_dict,
                                non_tensor_batch=filtered_non_tensor_dict,
                                meta_info=batch.meta_info.copy()
                            )
                            # Immediately clean up temporary data
                            del filtered_batch_dict, filtered_non_tensor_dict
                            gc.collect()


                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        print('use reward rule for reward score...')   
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # compute alpha and beta for prefix reward weighting
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        assert prefix_mask.shape == advantages.shape

                        alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = alpha_weight + beta_weight
                        batch.batch['advantages'] = prefix_weight * advantages

                        if self.config.data.get('disable_truncation_advantage', False):
                            responses = batch.batch['responses']
                            responses_mask = responses != self.tokenizer.pad_token_id
                            response_length = responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            has_truncated = response_length >= max_len
                            no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                            truncated_mask = has_truncated & no_eos
                            batch.batch['advantages'][truncated_mask] = 0
                        
                        if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                            assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                            reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                            batch.batch['advantages'][prefix_mask] = reward_weight / n_samples

                    # --------------- 结束 advantage 的计算 ---------- #

                    # Check if batch size is a multiple of 8, pad if not
                    batch.batch['whether_pad'] = torch.tensor([False] * batch.batch.batch_size[0], dtype=torch.bool)
                    original_batch_size = batch.batch.batch_size[0]
                    remainder = original_batch_size % self.device_total_num
                    if remainder != 0:
                        padding_size = self.device_total_num - remainder
                        # Pad tensor data in batch.batch by repeating the last sample
                        padded_batch_dict = {}
                        for key in batch.batch.keys():
                            if key != 'batch_size':
                                if key == 'whether_pad':
                                    last_sample = torch.tensor([True] * padding_size, dtype=torch.bool)
                                else:
                                    last_sample = batch.batch[key][-1:].repeat(padding_size, 1)
                                padded_batch_dict[key] = torch.cat([batch.batch[key], last_sample], dim=0)
                        
                        # Pad data in batch.non_tensor_batch
                        padded_non_tensor_dict = {}
                        for key in batch.non_tensor_batch.keys():
                            last_sample = np.array([batch.non_tensor_batch[key][-1]] * padding_size, dtype=batch.non_tensor_batch[key].dtype)
                            padded_non_tensor_dict[key] = np.concatenate([batch.non_tensor_batch[key], last_sample], axis=0)
                        
                        # Create padded batch
                        padded_batch_size = original_batch_size + padding_size
                        padded_batch_dict = TensorDict(padded_batch_dict, batch_size=[padded_batch_size])
                        
                        batch = DataProto(
                            batch=padded_batch_dict,
                            non_tensor_batch=padded_non_tensor_dict,
                            meta_info=batch.meta_info.copy()
                        )
                        # Immediately clean up temporary data
                        del padded_batch_dict, padded_non_tensor_dict
                        gc.collect()
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)
                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        print('prepare to reduce metrics')
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)


                    
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        try:
                            reward_extra_infos_dict['data_source'].extend(batch.non_tensor_batch["data_source"])
                        except:
                            print('no data_source')
                        try:
                            reward_extra_infos_dict['ability'].extend(batch.non_tensor_batch["ability"])
                        except:
                            print('no ability')
                        try:
                            reward_extra_infos_dict['reward_model'].extend(batch.non_tensor_batch["reward_model"])
                        except:
                            print('no reward_model')
                        
                        try:
                            reward_extra_infos_dict['extra_info'].extend(batch.non_tensor_batch["extra_info"])
                        except:
                            print('no extra_info')

                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            # sample_gts = [
                            #     item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                            #     for item in batch
                            # ]
                            sample_gts = None

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                # Force memory cleanup
                memory_cleanup()
                # check_memory_usage("batch_end")


                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)




def compute_data_metrics_ours(batch, use_critic=True):
    # TODO: add response length
    whether_keep = ~batch.batch['whether_pad']
    # Remove tensor data from batch
    filtered_batch_dict = {}
    for key in batch.batch.keys():
        if key != 'batch_size':
            filtered_batch_dict[key] = batch.batch[key][whether_keep]
    
    # Remove data from non_tensor_batch
    filtered_non_tensor_dict = {}
    for key in batch.non_tensor_batch.keys():
        filtered_non_tensor_dict[key] = batch.non_tensor_batch[key][whether_keep.cpu().numpy()]
    
    # Calculate new batch size
    new_batch_size = whether_keep.sum().item()
    filtered_batch_dict = TensorDict(filtered_batch_dict, batch_size=[new_batch_size])
    
    # Rebuild batch
    batch = DataProto(
        batch=filtered_batch_dict,
        non_tensor_batch=filtered_non_tensor_dict,
        meta_info=batch.meta_info.copy()
    )
    del filtered_batch_dict
    del filtered_non_tensor_dict
    gc.collect()

    
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # compute on/off policy stats
    off_policy_mask = batch.batch['prefix_mask'].any(-1) # [bsz, ]
    on_policy_mask = ~off_policy_mask
    off_response_length = response_length[off_policy_mask]
    on_response_length = response_length[on_policy_mask]
    
    off_example_ratio = off_policy_mask.sum().item() / (off_policy_mask.sum().item() + on_policy_mask.sum().item())

    off_sequence_score = sequence_score[off_policy_mask]
    on_sequence_score = sequence_score[on_policy_mask]

    # on/off prompt score
    # batch_size = batch.batch.batch_size[0] / n_samples
    # on_prompt_score, off_prompt_score = [], []
    # sequence_score = sequence_score.reshape(batch_size, n_samples, sequence_score.shape[-1]) # [bsz, n, l]
    # for i in range(batch_size):
    #     on_prompt_score.append(sequence_score[i][on_policy_mask[i]].mean())
    #     off_prompt_score.append(sequence_score[i][off_policy_mask[i]].mean())

    # on_prompt_score = torch.cat(on_prompt_score, dim=0)
    # off_prompt_score = torch.cat(off_prompt_score, dim=0)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # on/off policy response length
        'on_off_metrics/on_response_length_mean':
            torch.mean(on_response_length).detach().item(),
        'on_off_metrics/off_response_length_mean':
            torch.mean(off_response_length).detach().item(),
        'on_off_metrics/on_score':
            torch.mean(on_sequence_score).detach().item(),
        'on_off_metrics/off_score':
            torch.mean(off_sequence_score).detach().item(),
        # 'on_off_metrics/on_prompt_score':
        #     torch.mean(on_prompt_score).detach().item(),
        # 'on_off_metrics/off_prompt_score':
        #     torch.mean(off_prompt_score).detach().item(),
        'on_off_metrics/off_example_ratio':
            off_example_ratio,
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    if 'whether_off' in batch.batch:
        on_data_ratio = on_policy_mask.sum().item() / (off_policy_mask.sum().item() + on_policy_mask.sum().item())
        off_data_ratio = batch.batch['whether_off'].sum().item() / (off_policy_mask.sum().item() + on_policy_mask.sum().item())
        sft_data_ratio = (off_policy_mask.sum().item() - batch.batch['whether_off'].sum().item()) / (off_policy_mask.sum().item() + on_policy_mask.sum().item())
        metrics['uni/on_data_ratio'] = on_data_ratio
        metrics['uni/off_data_ratio'] = off_data_ratio
        metrics['uni/sft_data_ratio'] = sft_data_ratio
    return metrics
from verl.utils.py_functional import append_to_dict
#  config.offline_loss_type == "sft" or use_sft
def pack_vanilla_rl_loss_inputs(old_log_prob, 
                                log_prob, 
                                advantages, 
                                response_mask, 
                                loss_agg_mode,
                                config):
    vanilla_elements = {
        'old_log_prob': old_log_prob,
        'log_prob': log_prob,
        'advantages': advantages,
        'response_mask': response_mask,
        'loss_agg_mode': loss_agg_mode,
        'config': config
    }
    return vanilla_elements
    


def seperate_on_off(model_inputs, 
                    log_prob, 
                    old_log_prob,
                    response_mask,
                    advantages,
                    ):
    prefix_mask = model_inputs['prefix_mask']
    off_policy_mask = prefix_mask.any(-1) # 指定off policy data的位置
    off_policy_logprob = log_prob[off_policy_mask] # 获得off_policy 的 log_prob
    off_policy_eos_mask = response_mask[off_policy_mask] # 获得off_policy data 的response mask

    # 指定on policy data的位置
    on_policy_mask = ~off_policy_mask  # 指定on policy data的位置
    on_policy_logprob = log_prob[on_policy_mask] # 获得on_policy 的 log_prob
    on_policy_old_logprob = old_log_prob[on_policy_mask] # 获得on_policy data 的old log_prob
    on_policy_advantages = advantages[on_policy_mask] # 获得on_policy data 的adv
    on_policy_eos_mask = response_mask[on_policy_mask] # 获得on_policy data 的response_mask

    off_elements = {
        'off_policy_mask':off_policy_mask,
        'off_policy_logprob':off_policy_logprob,
        'off_policy_eos_mask':off_policy_eos_mask
    }

    on_elements = {
        'prefix_mask': prefix_mask,
        'on_policy_mask':on_policy_mask,
        'on_policy_logprob':on_policy_logprob,
        'on_policy_old_logprob':on_policy_old_logprob,
        'on_policy_advantages':on_policy_advantages,
        'on_policy_eos_mask':on_policy_eos_mask
    }
    return off_elements, on_elements



import torch
from  verl.trainer.ppo.mix_core_algos import compute_sft_pure_loss
def get_pure_sft_loss(off_elements):
    # 获得off_policy 的 log_prob
    off_policy_logprob = off_elements['off_policy_logprob']
    # 获得off_policy data 的response mask
    off_policy_eos_mask = off_elements['off_policy_eos_mask']
    # 基于off-policy data的 log_prob 和 response mask, 可以计算出
    sft_loss = compute_sft_pure_loss(log_prob=off_policy_logprob,
                                    eos_mask=off_policy_eos_mask)
    
    return sft_loss

from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
def rl_loss_fn_router(loss_mode):
    if loss_mode == 'vanilla':
        from verl.trainer.ppo.mix_core_algos import compute_policy_loss_vanilla
        policy_loss_fn = compute_policy_loss_vanilla

    elif loss_mode == 'off_policy': # offline_loss_type, LUFFY 用这个
        from verl.trainer.ppo.mix_core_algos import compute_token_on_off_policy_loss
        policy_loss_fn = compute_token_on_off_policy_loss
    
    elif loss_mode == 'switch_off_sft':
        from verl.trainer.ppo.mix_core_algos import compute_token_on_off_policy_loss_weight 
        policy_loss_fn = compute_token_on_off_policy_loss_weight
    
    elif loss_mode == 'off_sft':
        from verl.trainer.ppo.mix_core_algos import compute_token_on_off_policy_mask_loss
        policy_loss_fn = compute_token_on_off_policy_mask_loss

    elif loss_mode == 'srft': # SRFT 用这个
        from verl.trainer.ppo.mix_core_algos import compute_token_on_off_policy_srft_loss
        policy_loss_fn = compute_token_on_off_policy_srft_loss
    return policy_loss_fn
    

    
def get_policy_loss(
        policy_loss_fn, 
        vanilla_elements, 
        on_elements,
        metrics
    ):
    old_log_prob = vanilla_elements['old_log_prob']
    log_prob = vanilla_elements['log_prob']
    advantages = vanilla_elements['advantages']
    response_mask = vanilla_elements['response_mask']
    loss_agg_mode = vanilla_elements['loss_agg_mode']
    config = vanilla_elements['config']
    if on_elements is None:
        print('Performing vanilla policy loss!') 
        info = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=config,
                            )
        pg_loss = info['pg_loss']
    
    elif 'on_coef' in on_elements: 
        # SRFT case， 需要 target_probs, on_coef, correct_answer_mask, srft_type
        clip_upper_bound=config.clip_upper_bound
        prefix_mask = on_elements['prefix_mask']
        target_probs = on_elements['target_probs'] 
        on_coef = on_elements['on_coef']
        correct_answer_mask = on_elements['correct_answer_mask']


        off_cliprange=config.off_policy_cliprange
        on_coef=on_coef
        correct_answer_mask=correct_answer_mask
        srft_type=config.srft_type
        off_normalize=config.off_policy_normalize
        off_max_clip=config.off_policy_max_clip if config.off_policy_max_clip != -1 else None
        off_min_clip=config.off_policy_min_clip if config.off_policy_min_clip != -1 else None
        all_max_clip=config.all_max_clip if config.all_max_clip != -1 else None
        off_policy_reshape=config.off_policy_reshape
        off_policy_reshape_weight=config.off_policy_reshape_weight
        off_policy_reshape_pow_exp=config.off_policy_reshape_pow_exp
        on_policy_reshape=config.on_policy_reshape
        on_policy_reshape_weight=config.on_policy_reshape_weight
        on_policy_reshape_pow_exp=config.on_policy_reshape_pow_exp
        loss_remove_token_mean=config.loss_remove_token_mean
        loss_remove_clip=config.loss_remove_clip
        ret_dict = policy_loss_fn(
                    old_log_prob=old_log_prob, 
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    clip_upper_bound=clip_upper_bound,
                    prefix_mask=prefix_mask,
                    off_cliprange=off_cliprange,
                    on_coef=on_coef,
                    correct_answer_mask=correct_answer_mask,
                    srft_type=srft_type,
                    off_normalize=off_normalize,
                    off_max_clip=off_max_clip,
                    off_min_clip=off_min_clip,
                    all_max_clip=all_max_clip,
                    off_policy_reshape=off_policy_reshape,
                    off_policy_reshape_weight=off_policy_reshape_weight,
                    off_policy_reshape_pow_exp=off_policy_reshape_pow_exp,
                    on_policy_reshape=on_policy_reshape,
                    on_policy_reshape_weight=on_policy_reshape_weight,
                    on_policy_reshape_pow_exp=on_policy_reshape_pow_exp,
                    target_probs=target_probs,
                    loss_remove_token_mean=loss_remove_token_mean,
                    loss_remove_clip=loss_remove_clip,
                    loss_agg_mode=loss_agg_mode,
                    config=config,
                )
        pg_loss = ret_dict['pg_loss']
        metric_data = dict()
        for ret_key in ['off_pg_loss', 'on_pg_loss', 'off_pg_clipfrac', 'on_pg_clipfrac', 'ppo_kl', 'off_policy_prob', 'on_policy_prob', 'off_ratio_mean', 'off_ratio_max_clip_frac', 'off_ratio_min_clip_frac']:
            if ret_key in ret_dict:
                metric_data[f'actor/{ret_key}'] = ret_dict[ret_key].detach().item()
        append_to_dict(metrics, metric_data)
                                  
    else: # LUFFY case， 需要 target_probs
        clip_upper_bound = config.clip_upper_bound
        prefix_mask = on_elements['prefix_mask']
        target_probs = on_elements['target_probs'] 

        off_cliprange= config.off_policy_cliprange
        off_normalize= config.off_policy_normalize
        off_max_clip= config.off_policy_max_clip if config.off_policy_max_clip != -1 else None
        off_min_clip= config.off_policy_min_clip if config.off_policy_min_clip != -1 else None
        all_max_clip= config.all_max_clip if  config.all_max_clip != -1 else None
        off_policy_reshape= config.off_policy_reshape
        off_policy_reshape_weight= config.off_policy_reshape_weight
        off_policy_reshape_pow_exp= config.off_policy_reshape_pow_exp
        on_policy_reshape= config.on_policy_reshape
        on_policy_reshape_weight= config.on_policy_reshape_weight
        on_policy_reshape_pow_exp= config.on_policy_reshape_pow_exp
        loss_remove_token_mean= config.loss_remove_token_mean,
        loss_remove_clip= config.loss_remove_clip,
        
        ret_dict = policy_loss_fn(
                old_log_prob=old_log_prob, 
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                clip_upper_bound= clip_upper_bound,
                prefix_mask= prefix_mask,
                off_cliprange= off_cliprange,
                off_normalize= off_normalize,
                off_max_clip= off_max_clip, 
                off_min_clip= off_min_clip, 
                all_max_clip= all_max_clip, 
                off_policy_reshape = off_policy_reshape, 
                off_policy_reshape_weight = off_policy_reshape_weight,
                off_policy_reshape_pow_exp = off_policy_reshape_pow_exp,
                on_policy_reshape= on_policy_reshape,
                on_policy_reshape_weight= on_policy_reshape_weight,
                on_policy_reshape_pow_exp= on_policy_reshape_pow_exp,
                target_probs = target_probs,
                loss_remove_token_mean = loss_remove_token_mean,
                loss_remove_clip= loss_remove_clip,
                loss_agg_mode= loss_agg_mode,
                config = config,
            )
        pg_loss = ret_dict['pg_loss']
        metric_data = dict()
        for ret_key in ['off_pg_loss', 'on_pg_loss', 'off_pg_clipfrac', 'on_pg_clipfrac', 'ppo_kl', 'off_policy_prob', 'on_policy_prob', 'off_ratio_mean', 'off_ratio_max_clip_frac', 'off_ratio_min_clip_frac']:
            if ret_key in ret_dict:
                metric_data[f'actor/{ret_key}'] = ret_dict[ret_key].detach().item()
        append_to_dict(metrics, metric_data)
    return pg_loss, metrics
        

    







def append_ppo_actor_raw_metrics(metrics, micro_batch_metrics, pg_loss, loss_scale_factor, pg_clipfrac,  ppo_kl, pg_clipfrac_lower, entropy_loss):
    try:
        pg_loss = pg_loss.detach().item() * loss_scale_factor
        micro_batch_metrics.update({"actor/pg_loss":pg_loss})
    except:
        pass

    try:
        pg_clipfrac = pg_clipfrac.detach().item()
        micro_batch_metrics.update({"actor/pg_clipfrac":pg_clipfrac})
    except:
        pass
    
    try:
        ppo_kl = ppo_kl.detach().item()
        micro_batch_metrics.update({"actor/ppo_kl":ppo_kl})
    except:
        pass

    try:
        pg_clipfrac_lower = pg_clipfrac_lower.detach().item()
        micro_batch_metrics.update({"actor/pg_clipfrac_lower":pg_clipfrac_lower})
    except:
        pass
    
    try:
        entropy_loss = entropy_loss.detach().item()
        micro_batch_metrics.update({"actor/entropy_loss":entropy_loss})
    except:
        pass
    # micro_batch_metrics.update(
    #     {
    #         "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
    #         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    #         "actor/ppo_kl": ppo_kl.detach().item(),
    #         "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    #         'actor/entropy_loss': entropy_loss.detach().item(),
    #     }
    # )
    append_to_dict(metrics, micro_batch_metrics)
    return metrics
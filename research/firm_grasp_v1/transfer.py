"""显式迁移冻结v1.6 actor；不复用其critic、优化器或任务身份。"""
import json
import hashlib
import math
from pathlib import Path
import torch

EXPECTED_WEIGHT_SHA256 = '0650ac4b35e742430f910319492d8ada621ee1c25f954e8aba6bd8163e59ddf6'
EXPECTED_SOURCE_SHA256 = '4e7b7df5c4dbf5802c076c024a16f690f096afa708b36045ae97fae72b5299d3'
EXPECTED_ACTION_DIM = 14


def _state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode('utf-8'))
        if hasattr(value, 'detach'):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode('ascii'))
            digest.update(json.dumps(list(tensor.shape), separators=(',', ':')).encode('utf-8'))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    return digest.hexdigest()


def _validate_action_offset(action_offset):
    if not isinstance(action_offset, (list, tuple)) or len(action_offset) != EXPECTED_ACTION_DIM:
        raise ValueError(f'warmstart动作偏移必须恰有{EXPECTED_ACTION_DIM}个数值')
    values = []
    for index, value in enumerate(action_offset):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'warmstart动作偏移第{index}项不是数值')
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f'warmstart动作偏移第{index}项不是有限数值') from exc
        if not math.isfinite(number) or not -1.0 <= number <= 1.0:
            raise ValueError(f'warmstart动作偏移第{index}项必须在[-1,1]内且为有限值')
        values.append(number)
    return values


def _mean_output_bias(actor):
    mlp = getattr(actor, 'mlp', None)
    if not isinstance(mlp, torch.nn.Module):
        raise TypeError('RSL actor缺少MLP均值网络')
    distribution = getattr(actor, 'distribution', None)
    if distribution is None or type(distribution).__name__ != 'GaussianDistribution':
        raise TypeError('动作偏移只支持RSL GaussianDistribution均值输出')
    if getattr(distribution, 'input_dim', None) != EXPECTED_ACTION_DIM:
        raise ValueError('Gaussian均值输出维度不是14D')
    linear_layers = [module for module in mlp.modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise TypeError('RSL actor均值网络没有Linear层')
    final_linear = linear_layers[-1]
    if final_linear.out_features != EXPECTED_ACTION_DIM or final_linear.bias is None:
        raise ValueError('RSL actor均值网络末层必须是带bias的14D Linear')
    module_name = next(name for name, module in actor.named_modules() if module is final_linear)
    return final_linear, module_name + '.bias'


def transfer_actor(runner, path, action_offset=None):
    if action_offset is not None:
        action_offset = _validate_action_offset(action_offset)
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_WEIGHT_SHA256:
        raise ValueError('actor迁移源不是本轮受保护的v1.6权重')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    info = payload.get('infos', {}).get('thumb_pad_grasp_v1', {})
    if info.get('critical_source_sha256') != EXPECTED_SOURCE_SHA256:
        raise ValueError('源checkpoint核心源码身份不匹配')
    if info.get('observation_dimensions') != {'actor_dim':161,'critic_dim':222,'action_dim':14}:
        raise ValueError('源actor接口不匹配')
    actor = runner.alg.get_policy()
    actor_sha256_before_copy = _state_dict_sha256(actor.state_dict())
    source_actor_sha256 = _state_dict_sha256(payload['actor_state_dict'])
    actor.load_state_dict(payload['actor_state_dict'], strict=True)
    for name, tensor in actor.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), payload['actor_state_dict'][name].cpu()):
            raise RuntimeError('actor及normalizer未被逐位迁移：'+name)
    actor_sha256_after_copy = _state_dict_sha256(actor.state_dict())
    if actor_sha256_after_copy != source_actor_sha256:
        raise RuntimeError('actor复制前后SHA256不一致')
    if runner.alg.optimizer.state:
        raise RuntimeError('迁移时优化器应为全新空状态')
    action_offset_applied = None
    mean_bias_state_key = None
    if action_offset is not None:
        final_linear, mean_bias_state_key = _mean_output_bias(actor)
        copied_state = {name: tensor.detach().clone() for name, tensor in actor.state_dict().items()}
        delta = torch.as_tensor(action_offset, dtype=final_linear.bias.dtype, device=final_linear.bias.device)
        with torch.no_grad():
            final_linear.bias.add_(delta)
        if not bool(torch.isfinite(final_linear.bias).all()):
            raise ValueError('应用warmstart动作偏移后actor均值bias出现非有限值')
        action_offset_applied = delta.detach().cpu().tolist()
        updated_state = actor.state_dict()
        for name, tensor in updated_state.items():
            expected = copied_state[name] + delta if name == mean_bias_state_key else copied_state[name]
            if not torch.equal(tensor.detach().cpu(), expected.detach().cpu()):
                raise RuntimeError('动作偏移修改了actor均值末层bias以外的状态：' + name)
    new_initialization_sha256 = _state_dict_sha256(actor.state_dict())
    return {'kind':'explicit_actor_only_transfer_from_frozen_v1_6',
            'source_checkpoint':str(path),'source_checkpoint_sha256':digest,
            'source_core_sha256':EXPECTED_SOURCE_SHA256,
            'actor_and_observation_normalizer_copied_exactly':True,
            'actor_sha256_before_copy':actor_sha256_before_copy,
            'source_actor_state_sha256':source_actor_sha256,
            'actor_sha256_after_copy':actor_sha256_after_copy,
            'actor_copy_sha256_matched':actor_sha256_after_copy == source_actor_sha256,
            'warmstart_action_offset_requested':action_offset,
            'warmstart_action_offset_applied':action_offset_applied,
            'actor_mean_bias_state_key':mean_bias_state_key,
            'actor_sha256_after_offset':new_initialization_sha256,
            'new_initialization_actor_sha256':new_initialization_sha256,
            'critic_reset':True,'optimizer_reset':True,
            'prior_goal_success_is_not_new_learning':True}

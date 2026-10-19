"""将冻结 v1.6/161D actor 严格迁移为 162D clock actor。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch

from .transfer import (
    EXPECTED_ACTION_DIM,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_WEIGHT_SHA256,
    _mean_output_bias,
    _state_dict_sha256,
    _validate_action_offset,
)


EXPECTED_SOURCE_ACTOR_DIM = 161
EXPECTED_SOURCE_CRITIC_DIM = 222
TARGET_ACTOR_DIM = 162
TARGET_CRITIC_DIM = 223
APPENDED_CLOCK_MEAN = 0.0
APPENDED_CLOCK_VAR = 1.0
APPENDED_CLOCK_STD = 1.0

EXPECTED_SOURCE_SHAPES = {
    'obs_normalizer._mean': (1, 161),
    'obs_normalizer._var': (1, 161),
    'obs_normalizer._std': (1, 161),
    'obs_normalizer.count': (),
    'distribution.log_std_param': (14,),
    'mlp.0.weight': (512, 161),
    'mlp.0.bias': (512,),
    'mlp.2.weight': (256, 512),
    'mlp.2.bias': (256,),
    'mlp.4.weight': (128, 256),
    'mlp.4.bias': (128,),
    'mlp.6.weight': (14, 128),
    'mlp.6.bias': (14,),
}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(repr(tuple(value.shape)).encode('ascii'))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_source(payload):
    if not isinstance(payload, Mapping):
        raise ValueError('checkpoint 顶层必须为 mapping')
    infos = payload.get('infos')
    info = infos.get('thumb_pad_grasp_v1') if isinstance(infos, Mapping) else None
    if not isinstance(info, Mapping):
        raise ValueError('源 checkpoint 缺少 thumb_pad_grasp_v1 信息')
    if info.get('critical_source_sha256') != EXPECTED_SOURCE_SHA256:
        raise ValueError('源 checkpoint 核心源码身份不匹配')
    if info.get('observation_dimensions') != {
        'actor_dim': EXPECTED_SOURCE_ACTOR_DIM,
        'critic_dim': EXPECTED_SOURCE_CRITIC_DIM,
        'action_dim': EXPECTED_ACTION_DIM,
    }:
        raise ValueError('源 actor 接口不匹配：要求 161/222/14')
    state = payload.get('actor_state_dict')
    if not isinstance(state, Mapping):
        raise ValueError('源 checkpoint 缺少 actor_state_dict')
    if set(state) != set(EXPECTED_SOURCE_SHAPES):
        unknown = sorted(set(state) - set(EXPECTED_SOURCE_SHAPES))
        missing = sorted(set(EXPECTED_SOURCE_SHAPES) - set(state))
        raise ValueError(f'源 actor key/schema 不匹配：unknown={unknown}, missing={missing}')
    for name, expected_shape in EXPECTED_SOURCE_SHAPES.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'源 actor 张量不是 Tensor：{name}')
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f'源 actor 张量形状不匹配：{name} 应为 {expected_shape}，得到 {tuple(value.shape)}'
            )
    return state, info


def _adapt_source_state(source) -> tuple[dict[str, torch.Tensor], list[str]]:
    adapted: dict[str, torch.Tensor] = {}
    changed: list[str] = []
    for name, value in source.items():
        if name == 'obs_normalizer._mean':
            appended = value.new_full((value.shape[0], 1), APPENDED_CLOCK_MEAN)
            adapted[name] = torch.cat((value.detach().clone(), appended), dim=1)
            changed.append(name)
        elif name == 'obs_normalizer._var':
            appended = value.new_full((value.shape[0], 1), APPENDED_CLOCK_VAR)
            adapted[name] = torch.cat((value.detach().clone(), appended), dim=1)
            changed.append(name)
        elif name == 'obs_normalizer._std':
            appended = value.new_full((value.shape[0], 1), APPENDED_CLOCK_STD)
            adapted[name] = torch.cat((value.detach().clone(), appended), dim=1)
            changed.append(name)
        elif name == 'mlp.0.weight':
            appended = value.new_zeros((value.shape[0], 1))
            adapted[name] = torch.cat((value.detach().clone(), appended), dim=1)
            changed.append(name)
        else:
            adapted[name] = value.detach().clone()
    return adapted, changed


def _change_kind(name: str) -> str:
    if name == 'obs_normalizer._mean':
        return 'append_clock_mean_0'
    if name == 'obs_normalizer._var':
        return 'append_clock_var_1'
    if name == 'obs_normalizer._std':
        return 'append_clock_std_1'
    if name == 'mlp.0.weight':
        return 'append_zero_clock_input_column'
    return 'copy_exactly'


def _verify_adapted(source, actor_state, changed: list[str]) -> None:
    for name, source_value in source.items():
        target_value = actor_state[name]
        if name in changed:
            if target_value.shape[:-1] != source_value.shape[:-1]:
                raise RuntimeError(f'迁移张量前缀批维不一致：{name}')
            if target_value.shape[-1] != source_value.shape[-1] + 1:
                raise RuntimeError(f'迁移张量未恰追加一列：{name}')
            if not torch.equal(target_value[..., :-1].detach().cpu(), source_value.detach().cpu()):
                raise RuntimeError(f'迁移张量旧列未逐位保留：{name}')
            appended = target_value[..., -1].detach().cpu()
            if name == 'obs_normalizer._mean':
                expected, label = APPENDED_CLOCK_MEAN, 'mean0'
            elif name == 'obs_normalizer._var':
                expected, label = APPENDED_CLOCK_VAR, 'var1'
            elif name == 'obs_normalizer._std':
                expected, label = APPENDED_CLOCK_STD, 'std1'
            else:
                expected, label = 0.0, 'zero input weight'
            if not torch.equal(appended, torch.full_like(appended, expected)):
                raise RuntimeError(f'迁移新 clock 列不是 {label}：{name}')
        elif not torch.equal(target_value.detach().cpu(), source_value.detach().cpu()):
            raise RuntimeError(f'迁移未逐位复制 actor 张量：{name}')


def transfer_actor(runner, path, action_offset=None):
    """同旧 API 迁移受保护源 actor；clock 列与 normalizer prior 显式初始化。"""

    if action_offset is not None:
        action_offset = _validate_action_offset(action_offset)
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_WEIGHT_SHA256:
        raise ValueError('actor迁移源不是本轮受保护的v1.6权重')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    source, source_info = _validate_source(payload)

    actor = runner.alg.get_policy()
    if int(getattr(actor, 'obs_dim', -1)) != TARGET_ACTOR_DIM:
        raise ValueError(f'目标 actor 必须是 {TARGET_ACTOR_DIM}D，得到 {getattr(actor, "obs_dim", None)}')
    destination = actor.state_dict()
    if set(destination) != set(source):
        raise ValueError('源与目标 actor state_dict key/schema 不一致，拒绝 strict=False')

    actor_sha256_before_copy = _state_dict_sha256(destination)
    source_actor_sha256 = _state_dict_sha256(source)
    adapted, changed = _adapt_source_state(source)
    for name, value in adapted.items():
        expected_shape = destination[name].shape
        if tuple(value.shape) != tuple(expected_shape):
            raise ValueError(
                f'适配后 actor 张量形状不匹配：{name} 应为 {tuple(expected_shape)}，得到 {tuple(value.shape)}'
            )
    actor.load_state_dict(adapted, strict=True)
    _verify_adapted(source, actor.state_dict(), changed)
    actor_sha256_after_adaptation = _state_dict_sha256(actor.state_dict())

    if runner.alg.optimizer.state:
        raise RuntimeError('迁移时优化器应为全新空状态')
    action_offset_applied = None
    mean_bias_state_key = None
    adapted_copy = {
        name: tensor.detach().clone() for name, tensor in actor.state_dict().items()
    }
    if action_offset is not None:
        final_linear, mean_bias_state_key = _mean_output_bias(actor)
        delta = torch.as_tensor(
            action_offset, dtype=final_linear.bias.dtype, device=final_linear.bias.device
        )
        with torch.no_grad():
            final_linear.bias.add_(delta)
        if not bool(torch.isfinite(final_linear.bias).all()):
            raise ValueError('应用warmstart动作偏移后actor均值bias出现非有限值')
        action_offset_applied = delta.detach().cpu().tolist()
        for name, tensor in actor.state_dict().items():
            expected = adapted_copy[name] + delta if name == mean_bias_state_key else adapted_copy[name]
            if not torch.equal(tensor.detach().cpu(), expected.detach().cpu()):
                raise RuntimeError('动作偏移修改了actor均值末层bias以外的状态：' + name)
    final_state = actor.state_dict()
    new_initialization_sha256 = _state_dict_sha256(final_state)

    tensor_changes = {}
    for name, source_value in source.items():
        target_value = final_state[name]
        tensor_changes[name] = {
            'change': _change_kind(name),
            'source_shape': list(source_value.shape),
            'target_shape': list(target_value.shape),
            'source_sha256': _tensor_sha256(source_value),
            'target_sha256': _tensor_sha256(target_value),
            'legacy_prefix_exact': True,
        }
        if name in changed:
            tensor_changes[name]['appended_clock_column'] = (
                APPENDED_CLOCK_MEAN if name == 'obs_normalizer._mean'
                else APPENDED_CLOCK_VAR if name == 'obs_normalizer._var'
                else APPENDED_CLOCK_STD if name == 'obs_normalizer._std'
                else 0.0
            )
    if mean_bias_state_key is not None:
        tensor_changes[mean_bias_state_key]['change'] = 'copy_then_add_14d_action_offset_to_bias'
        tensor_changes[mean_bias_state_key]['action_offset_applied'] = action_offset_applied
        tensor_changes[mean_bias_state_key]['legacy_prefix_exact'] = bool(torch.equal(
            final_state[mean_bias_state_key].detach().cpu(),
            source[mean_bias_state_key].detach().cpu()))

    count = final_state['obs_normalizer.count']
    if tuple(count.shape) != () or not torch.equal(
        count.detach().cpu(), source['obs_normalizer.count'].detach().cpu()
    ):
        raise RuntimeError('normalizer count 未逐位复制')

    return {
        'kind': 'explicit_actor_prefix_transfer_from_frozen_v1_6_plus_clock',
        'source_checkpoint': str(path),
        'source_checkpoint_sha256': digest,
        'source_core_sha256': EXPECTED_SOURCE_SHA256,
        'source_observation_dimensions': dict(source_info['observation_dimensions']),
        'target_observation_dimensions': {
            'actor_dim': TARGET_ACTOR_DIM,
            'critic_dim': TARGET_CRITIC_DIM,
            'action_dim': EXPECTED_ACTION_DIM,
        },
        'strict_load': True,
        'actor_and_observation_normalizer_copied_exactly': False,
        'legacy_prefix_preserved': True,
        'legacy_actor_prefix_dim': EXPECTED_SOURCE_ACTOR_DIM,
        'legacy_critic_prefix_dim': EXPECTED_SOURCE_CRITIC_DIM,
        'appended_clock_dim': 1,
        'appended_clock_position': {'actor': 161, 'critic': 222},
        'appended_clock_initialized_zero': True,
        'observation_normalizer_clock_prior': {
            'mean': APPENDED_CLOCK_MEAN,
            'var': APPENDED_CLOCK_VAR,
            'std': APPENDED_CLOCK_STD,
            'kind': 'unit_variance_prior_not_observed_clock_samples',
        },
        'observation_normalizer_count_copied_from_source': True,
        'observation_normalizer_count_value': int(count.item()),
        'observation_normalizer_count_semantics': (
            'shared legacy global count; it is not a count of observed clock samples '
            'and the clock normalizer uses an explicit unit-variance prior'
        ),
        'actor_sha256_before_copy': actor_sha256_before_copy,
        'source_actor_state_sha256': source_actor_sha256,
        'actor_sha256_after_adaptation': actor_sha256_after_adaptation,
        'actor_sha256_after_offset': new_initialization_sha256,
        'new_initialization_actor_sha256': new_initialization_sha256,
        'changed_keys': list(changed) + (
            [mean_bias_state_key] if mean_bias_state_key is not None else []
        ),
        'tensor_changes': tensor_changes,
        'warmstart_action_offset_requested': action_offset,
        'warmstart_action_offset_applied': action_offset_applied,
        'actor_mean_bias_state_key': mean_bias_state_key,
        'action_semantics_unchanged': True,
        'critic_reset': True,
        'optimizer_reset': True,
        'optimizer_state_empty': True,
        'prior_goal_success_is_not_new_learning': True,
    }

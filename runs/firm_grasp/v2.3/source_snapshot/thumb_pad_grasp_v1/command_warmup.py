"""显式的低冲击策略命令起步；不改写关节/果实状态或接触判据。"""
WARMUP_POLICY_STEPS = 35


def apply_command_warmup(actions, episode_policy_step):
    if actions.ndim != 2 or episode_policy_step.ndim != 1 or actions.shape[0] != episode_policy_step.shape[0]:
        raise ValueError('缓起步需要逐环境二维动作及同长度一维回合年龄，禁止广播错误输入')
    gain = (episode_policy_step.to(dtype=actions.dtype) / WARMUP_POLICY_STEPS).clamp(0, 1)
    return actions * gain[:, None]

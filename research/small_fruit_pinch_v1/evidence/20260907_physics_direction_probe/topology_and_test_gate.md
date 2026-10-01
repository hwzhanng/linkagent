# 单左手拓扑与代码门（2026-09-07）

## 场景拓扑

- USD：`research/small_fruit_pinch_v1/assets/left_hand_urdf/left_hand_l20_physx.usd`
- 唯一 articulation：固定基座左手
- 21 个关节：16 个主动轴 + 5 个软件跟随轴
- 不含机械臂关节、右臂、右手或 CAN/网络接口

## 运行门

```text
native_physx_smoke policy=(2,114) critic=(2,175) info_keys=['log','step']
pytest: 4 passed, 50 warnings
python -m py_compile: passed
git diff --check: passed
```

这些是结构和代码门，不替代三指真实接触及后续断梗/保持评估。

相较此前 51 条告警，已消除 `object.__new__` 测试夹具触发的
`DirectRLEnv.__del__` 非阻断资源释放告警；剩余 50 条均来自上游 Isaac Lab/PyTorch 的
deprecated API，不在项目代码中绕过或修改上游源码。

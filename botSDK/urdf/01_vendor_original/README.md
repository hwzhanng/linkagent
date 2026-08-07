# 厂商原始资产

该目录保存未经项目拼接的上游资料。

## 目录

- `01_download_archives/`
  - `L20_8_Urdf文件.zip`：用户从厂商官网下载的原始外层压缩包。
  - `L20_8_l_s.zip`、`L20_8_r_s.zip`：从外层包原样取出的内层压缩包。
- `02_extracted_packages/`
  - `L20_8_l_s/`、`L20_8_r_s1/`：两个内层压缩包的完整原样解压结果，包含 URDF、STL、CSV、launch、配置和导出日志。
- `03_delivered_sdk_docs/`
  - 当前工作目录中随设备交付的机械臂 API 与控制平台 PDF。
- `04_delivered_sdk_models/`
  - 随控制平台交付、当前页面使用的 `lkls73_i1` 双机械臂原始模型。
- `05_online_official/`
  - `linkerhand-urdf_0b2a80ff68b0/`：官方 LinkerHand URDF 仓库 2026-07-21 快照中的标准 L20 左右手。
  - `linker-sim_ed1b8ea26bb1/`：官方仿真仓库 2026-07-01 快照中的 LKLS73 URDF/MJCF、workstation 元数据和 Isaac 后端关键文件。

## 关键校验值

| 文件 | SHA-256 |
|---|---|
| `L20_8_Urdf文件.zip` | `d50b1c2a0f48954301189f72c68908e8723f9671f443dc5f1058c88715b9f487` |
| `L20_8_l_s.zip` | `39d959067acdb3995a7b51c6eb0428f47c78bd15514339b929985903f3e90158` |
| `L20_8_r_s.zip` | `09dc3d8bfc5183178491b64b813afb58e2e341694514174183ca9657171f85c6` |
| 交付 SDK 机械臂 PDF | `7214942fa1bad9b954c45abc923006b94f4a227869f7a6d6e5d3f5628b01f7d3` |
| 交付 SDK 控制平台 PDF | `7538c98ff45e66aa4188c073982d27dd7de8a12972fa8d24cbc947a9106e0ef3` |
| 交付 LKLS73 双臂 URDF | `732b7f41003d339e0ae67d8b9b9d04d266a89e32dd58f9f50037e02b86ffa804` |
| 在线官方 L20 左手 URDF | `2c29f77a892d026ad92dfba6a61b3c5b70e90784c4a6dabdc09503f7f0209e3e` |
| 在线官方 L20 右手 URDF | `171c28116fe6f1e7b885fd5cf046044c4367c7d0b8036ab4743946cc8e619b9a` |

上游地址：

- https://github.com/linker-bot/linkerhand-urdf
- https://github.com/linker-bot/linker-sim
- https://www.linkerbot.cn/

# 上游快照

- 仓库：https://github.com/linker-bot/linker-sim
- commit：`ed1b8ea26bb1161326916934e2ceb596ea01f7a6`
- commit 时间：2026-07-01T23:42:46+08:00
- commit 说明：`fix(hand-decoder): permute SDK cols to manifest order via decoder.yaml`
- 许可证：Apache-2.0

本地保留 LKLS73 机械臂/基座组件、三个 LKLS73 workstation 的配方和生成结果、
Isaac 加载器关键文件及相关文档。LKLS73 的 Git LFS STL 已解析为真实二进制文件。

注意：workstation 中所引用的 L6/O6/L25 手组件没有一并复制，因此这里主要用于
机械臂参数和 Isaac 配置溯源，不是完整可直接运行的 `linker-sim` checkout。

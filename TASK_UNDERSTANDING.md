# 通用昇腾 NPU 仿真适配：首版开发范围

用户要求：将 `Mutinifni/splitwise-sim` fork 到个人 GitHub 账号，继续开发昇腾 NPU 适配。已确认优先完成通用适配，尚未指定 NPU 型号、模型和推理框架。

- 个人 fork：`https://github.com/wushanchi/splitwise-sim`
- 上游基线：`main@8f99e7dc9b407f4ce2488d03dd44c0b8b946dab0`
- 开发分支：`feat/ascend-npu-simulator`
- 开发方式：独立 worktree；保留上游 GPU 配置和性能库。

## 本次交付

1. 增加通用 NPU 处理器与可覆盖的硬件/初始部署配置。
2. 增加独立的 profile 驱动迭代时延模型。CSV 明确区分阶段、batch 大小、本轮新 token 数和 Decode 总 context token 数，JSON 元数据记录来源及单位；默认拒绝未经许可的合成数据，不将 GPU 曲线自动用于 NPU。
3. 精确形状查表为默认；nearest 近邻查表需要显式启用并记录使用次数。未知模型、硬件、TP、阶段和非法时延立即报错。不将近邻值宣称为标定精度。
4. 新模型依赖 context 长度，每轮重新查询，禁止沿用旧模型的多轮事件合并。
5. 增加带固定启动延迟、明确 GiB/s 单位的 KV 链路模型；同一有向实例对复用链路并串行传输，表达这一层的争用。它不等同于 HCCL 协议、共享 NIC/交换机或逐层通信重叠仿真。
6. 提供 CPU 可运行的 NPU 聚合/P-D 分离合成示例、测量数据接入文档、必要的回归测试和结果元数据。
7. 对现代模型允许明确设置 KV 结构或每 token KV 字节数，保留现有 MHA 默认口径。

## 数据接口契约

新增性能模型类：`profiled_performance_model.ProfiledPerformanceModel`。

- 配置参数：`profile_path`, `metadata_path`, `allow_synthetic=False`, `lookup_mode="exact"`。
- CSV：`model,hardware,tensor_parallel,phase,batch_size,batch_tokens,context_tokens,latency_ms`。
- phase：`prefill`, `decode`, `mixed`。
- batch_tokens：本轮所有 Prompt 的输入长度之和 + Decode 任务数。
- context_tokens：本轮 Decode 任务的 `request.processed_tokens` 之和，纯 Prefill 为 0。
- 重复形状取时延中位数；有效时延为严格正的有限数。
- 元数据至少包含 `schema_version=1`, `provenance`（`measured` 或 `synthetic`）, `description`, `latency_unit="ms"`；measured 必须带明确采样环境和来源说明。
- `supports_iteration_coalescing=False`，防止变化的 context 共用旧的每轮时长。
- 示例模型 `synthetic-llm`、硬件 `ascend-generic`、TP=1；数据明确为合成算术示例。

新增通信调度类：`scheduler.CalibratedKVScheduler`，参数 `transfer_bandwidth_gib_s` 和 `transfer_latency_ms`；可复用 `KVScheduler` 的分池与接口，默认不采用 overlap 倍率。专用 Link 明确复用范围，并校验带宽、延迟、内存等待和串行完成逻辑。

新增入口 `python run.py --config-name config_ascend ...`，使用 basic launcher；不改默认 GPU 入口。提供聚合与两池部署配置，输出配置和 profile 来源元数据。

## 验证和证据边界

验证包含性能表校验/形状分派/来源门禁、context 每轮更新、KV 容量公式、链路串行与内存等待、两类端到端运行、原 GPU 示例回归、零时刻到达初始化顺序。

本机未获授权访问任何指定 NPU 服务器，本次不伪造 CANN/torch_npu/HCCL 实测。交付表述为“通用适配基础和合成路径已验证；真实昇腾标定与硬件校准待提供目标环境后完成”。当前仍为 Prefill/Decode 仿真，不宣称已实现 Attention/FFN 分离。

用户已授权个人 fork 下的开发；完成检查后提交并推送开发分支，核对远端 SHA。暂不向上游提交 PR，也不自动合入个人 main。

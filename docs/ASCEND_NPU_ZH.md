# 通用昇腾 NPU 仿真适配指南

这份适配把 NPU 资源、按形状查表的迭代时延和带固定延迟的 KV 传输接入 SplitwiseSim。仿真在 CPU 上执行，不调用 CANN、torch_npu 或 HCCL，也不需要 NPU。它模拟的是 **Prefill/Decode 分离**，不是 Attention/FFN 分离。

目前没有绑定具体昇腾型号、模型、框架和机器拓扑。仓库中的 `ascend-generic`、`synthetic-llm`、64 GiB 设备容量、16 GiB 模型权重、时延与网络参数都是示例假设，不是任何真实型号的规格或实测结果。合成数据可以验证软件路径，不能预测真实 NPU 的吞吐、延迟、功耗或节省成本。

## 1. 五分钟运行合成示例

建议创建 Python 3.11 或 3.12 独立环境；本次端到端验证使用 Windows / Python 3.12。最小依赖不包含 Ray、Notebook 绘图库或设备软件栈。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-sim.txt
python scripts/run_ascend_examples.py
```

请在仓库根目录安装依赖。示例脚本可以从其他目录启动，它会把仿真子进程的工作目录设为仓库根目录。

脚本明确启用 `allow_synthetic=True` 和 `lookup_mode=nearest`，运行两组相同设备数量的配置：

| 项目 | 聚合 `aggregate` | P/D 分离 `pd` |
|---|---|---|
| 服务器 / 设备数 | 2 台 × 每台 1 卡 | 2 台 × 每台 1 卡 |
| 模型 / TP / PP | synthetic-llm / 1 / 1 | synthetic-llm / 1 / 1 |
| 实例 | 2 个可同时执行 P 与 D 的实例 | 1 个 P 实例 + 1 个 D 实例 |
| 调度器 | TokenJSQScheduler | CalibratedKVScheduler |
| KV 跨实例传输 | 无 | 同一有向实例对共用串行链路 |
| 功率 / 能耗 | 禁用 | 禁用 |

输入为 `examples/ascend_trace.csv`，8 个确定性请求，首请求在 0 秒到达，prompt 长度循环 16、32、64、128，输出均为 8 token。正常结果是两组各完成 8/8 请求，聚合无跨实例 KV flow，分离产生 8 个 KV flow。

结果默认写到 `results/ascend_examples/`：

```text
demo_manifest.json             两组命令、完成数及合成指标摘要
aggregate.log / pd.log         子进程完整日志
aggregate/ / pd/
  .hydra/                     本次实际配置和覆盖参数
  simulation_metadata.json   来源、文件哈希、查表次数、完成数与链路边界
  summary.csv                完成请求的指标统计
  detailed/0.csv             每个完成请求的指标
  request_nodes.csv          Prompt/Token/Flow 的位置和时间戳
  instances/0/               debug 模式下的逐轮执行记录
```

可指定目录或只运行一组：

```bash
python scripts/run_ascend_examples.py --output-dir results/my_ascend_demo --case pd
```

相同设备数只是控制变量之一。合成时延、稀疏形状近邻匹配和 8 请求输入都不足以支持“P/D 在某型号上提升多少”的结论；示例脚本不计算或宣传加速比。

## 2. 新增入口与安全默认值

原 `python run.py`、GPU 硬件配置和 `data/perf_model.csv` 保留。新入口是 `config_ascend`，使用 Hydra basic launcher，无 Ray 地址配置。

下面的命令**默认会拒绝运行**，因为当前配置指向明确标注为 synthetic 的示例性能文件：

```bash
python run.py --config-name config_ascend
```

这不是安装失败。要运行演示，需明确选择合成来源和近邻近似：

```bash
# 聚合
python run.py --config-name config_ascend performance_model.allow_synthetic=True performance_model.lookup_mode=nearest

# P/D 分离
python run.py --config-name config_ascend performance_model.allow_synthetic=True performance_model.lookup_mode=nearest start_state=ascend_pd applications.0.scheduler=ascend_kv
```

真实 profile 的默认使用方式是 `allow_synthetic=False`、`lookup_mode=exact`。不要为了绕过校验，把合成文件的 `provenance` 改成 measured；读取器校验的是元数据声明和文件合法性，并不能替代测量来源审计。

配置文件分工：

| 文件 | 应修改的实验量 |
|---|---|
| `configs/config_ascend.yaml` | 组装新入口、仿真时长、输出目录 |
| `configs/cluster/ascend_two_servers.yaml` | 服务器 SKU 和台数 |
| `configs/hardware_repo/processors/ascend-generic.yaml` | NPU 名称和每卡可用容量字节数 |
| `configs/hardware_repo/skus/ascend-example-server.yaml` | 每台服务器卡数；示例 tdp=0 不是实测功耗 |
| `configs/applications/ascend_synthetic.yaml` | 模型架构、权重大小和调度器 |
| `configs/start_state/ascend_aggregate.yaml` | 聚合实例 TP、batch 数及 token 预算 |
| `configs/start_state/ascend_pd.yaml` | P/D 数量及两池各自 TP、batch/token 预算 |
| `configs/performance_model/ascend_profiled.yaml` | profile、metadata 路径与查询策略 |
| `configs/orchestrator_repo/schedulers/ascend_kv.yaml` | 有效传输带宽 GiB/s、固定启动延迟 ms |
| `configs/power_model/disabled.yaml` | 明确禁用功率和能耗分析 |

当前 `start_state` 的 homogeneous 分池先选择若干服务器，再按 TP 分组实例。默认每台只有 1 卡，因此 1+1 正好对应 1P+1D；扩展为多卡/多实例前，应核对真正创建的实例数量，不能只看配置字段名 `num_instances`。聚合初始化则为每台服务器创建一个实例；本模型要求该实例的设备数等于 TP。

## 3. 迭代时延 CSV 契约

模型类是 `profiled_performance_model.ProfiledPerformanceModel`，不是旧的 GPU 插值数据库。必需字段如下：

```csv
model,hardware,tensor_parallel,phase,batch_size,batch_tokens,context_tokens,latency_ms
```

| 字段 | 单位与严格含义 |
|---|---|
| model | 模型架构名称，必须匹配实例 `model.name` |
| hardware | 处理器配置名称，必须匹配实例设备名称 |
| tensor_parallel | 正整数，整个实例的 TP 卡数 |
| phase | `prefill`、`decode` 或 `mixed` |
| batch_size | 本轮参与计算的任务数，正整数；不是请求输出长度 |
| batch_tokens | 本轮所有完整 prompt 长度之和，加 Decode 任务数 |
| context_tokens | **仅 Decode 任务**的 `request.processed_tokens` 总和；纯 Prefill 为 0 |
| latency_ms | 整个实例完成本轮迭代的时长，毫秒，必须为有限正数 |

例如同轮包含两个 Decode 请求，上下文分别为 100 和 200，则 `phase=decode,batch_size=2,batch_tokens=2,context_tokens=300`。同轮若增加一条长度 64 的完整 prompt，则变为 `phase=mixed,batch_size=3,batch_tokens=66,context_tokens=300`。Prefill 自身的 prompt 不再加到 `context_tokens`，避免重复编码。

这里的 context 是框架接入时必须对齐的状态变量。模拟器在完成一轮后更新 processed_tokens，下一轮重新查询；测量侧应按同样的逻辑记录 Decode **轮开始前**已处理 token 数，不要用最大序列长度、当前 batch 最大 context 或所有 prompt+decode 输入的总和替代它。

查表行为：

- `(model,hardware,TP,phase)` 必须存在；未知组直接报错，绝不回退到 H100/A100 或另一个模型。
- 默认 `exact` 要求三个形状维度完全一致；缺少形状会报错，提示补充测量覆盖。
- 同一完整形状的重复样本取 `latency_ms` 中位数；保留原重复行有助于复核采样次数。
- 显式 `nearest` 仅在同组内选择最近的已记录形状。距离是三个形状维度 `abs(candidate−query)/max(query,1)` 之和；并列时按形状排序稳定选取。
- nearest 是形状近似，不是插值、外推精度保证或自动标定。首次发生会警告，结果元数据记录 exact/nearest 查询次数。
- CSV 时延从毫秒转换为模拟器秒数。不要预先转换为秒后仍标 `latency_ms`。

模型每轮依赖当前 context，因此声明 `supports_iteration_coalescing=False`。运行器不得沿用旧性能模型的“一次计算时延后合并多个 Decode 轮”快捷路径。旧 GPU 数据库仍可使用其原行为。

当前接口只支持 PP=1，以及 trace 中 `batch_size=1` 的单请求输入；多个请求由实例组成 batch。PP>1、预打包请求 `batch_size>1` 或实例设备数与 TP 不一致都会被拒绝。接口不支持 prompt chunking；所有 Prefill 长度是本轮完整 prompt。

## 4. JSON 来源元数据

所有 profile 都需要 JSON 元数据，最少包括：

```json
{
  "schema_version": 1,
  "provenance": "synthetic",
  "description": "Explicit arithmetic example, not measured hardware performance",
  "latency_unit": "ms"
}
```

当 `provenance="measured"` 时，读取器额外要求非空字符串 `source` 和非空对象 `environment`。下面是需要实际填完的**测量记录模板**，不是本仓库已经获得的实测证据：

```json
{
  "schema_version": 1,
  "provenance": "measured",
  "description": "<实际模型与目标部署的完整迭代测量>",
  "latency_unit": "ms",
  "source": "<原始日志/性能采集文件的归档路径、记录编号或可核对地址>",
  "environment": {
    "measurement_date": "<日期>",
    "device_model": "<具体NPU型号和芯片版本>",
    "devices_per_instance": "<实际TP卡数>",
    "server_and_topology": "<服务器、卡间互连与NUMA布局>",
    "driver_firmware": "<驱动与固件版本>",
    "cann_version": "<CANN版本>",
    "framework_and_commit": "<推理框架及完整commit>",
    "torch_and_torch_npu": "<实际版本，若适用>",
    "model_revision": "<权重与模型配置版本/校验值>",
    "weight_compute_kv_dtypes": "<权重、计算与KV dtype及量化策略>",
    "execution_mode": "<eager/graph、编译、融合及批处理策略>",
    "power_and_clock_settings": "<电源/频率模式，禁止未记录的混用>"
  },
  "measurement_protocol": {
    "iteration_boundary": "<开始/结束事件、同步边界、是否含Host开销>",
    "warmup_and_repetitions": "<各shape预热次数、重复次数>",
    "shape_generation": "<各请求context长度分布与输入构造>",
    "tp_communication": "included in the whole-instance iteration latency",
    "cross_instance_kv_transfer": "excluded; calibrated separately",
    "raw_samples": "<逐样本日志及哈希/版本信息>"
  }
}
```

模板中的详细环境项属于复现建议；代码当前仅校验基础字段以及 measured 的 source/environment 存在且非空。附加 JSON 字段原样保存到输出，便于审阅，但填写字段本身不证明测量真实或准确。

## 5. 如何测量并接入真实昇腾数据

### 5.1 先固定环境和语义

固定目标 NPU、拓扑、模型权重和配置、框架完整版本、精度/量化模式、batch 策略与 TP。对同一硬件名不要混入不同功率模式、不同软件优化开关和模型版本；若需要比较，应使用独立 profile 与元数据文件。

确认框架与模拟器的 token 记账一致：Prefill 生成首 token；Decode 只处理后续 token；`context_tokens` 使用 Decode 轮开始前已处理长度。记录每条请求的原始 context 长度，CSV 中的总和不是完整分布。同为总 context=300 的 `[150,150]` 与 `[1,299]`，实际 kernel 效率可能不同，现有特征无法区分这种差异。

### 5.2 采样整轮迭代，不是孤立算子

1. 用目标框架已经验证的运行方式完成模型加载与功能/精度确认。预热到编译、图捕获、缓存分配等一次性成本稳定；是否计入这些成本应单独记录。
2. 对目标 workload 的 Prefill、Decode、mixed 分别构造形状网格。聚合配置可能产生 mixed，必须实测这一阶段或明确拒绝未覆盖 mixed；不能凭一个固定倍率自动当成实测。
3. 以整个实例一次迭代作为计时边界。在 TP>1 时，一行时延应表示所有 rank 都完成该轮所需的时间，包含该轮内部 HCCL 集合通信及其与计算的实际依赖。不要把每卡时延相加，也不要在仿真中再次叠加已经计入采样的 TP 通信。
4. 使用目标框架或 profiler 的可信同步事件。只测 Host 异步提交耗时会漏掉设备执行；只测某一个 kernel/某一条流又会漏掉通信或其他关键流。明确同步范围，记录测量是否改变了自然重叠；不能把刻意串行采样的结果无条件用于异步流水部署。
5. 每个完整形状保留多次原始样本，检查异常、热状态和背景负载。本模型对重复行取中位数；若需要尾延迟模型，必须另外扩展接口，不能把确定性中位数查表称为硬件 p99 模型。
6. 留出未参与拟合/取表的数据和完整短 trace，比较迭代、TTFT、E2E 和资源占用误差。通过后再扩大工作负载扫描。nearest 查询比例高说明覆盖不足，不是模型自动纠正了缺失数据。

本仓库不提供未经目标 NPU 验证的“一键硬件 benchmark”脚本。测量应在获得目标机器和实际框架后实现，并保留原始采集结果；本次开发完成的是数据接口与 CPU 仿真验证。

### 5.3 跨实例 KV 独立测量

跨实例传输使用：

\[
t_{KV}=\alpha_{ms}/1000+S_{bytes}/(B_{GiB/s}\cdot 1024^3).
\]

按真实 TP、发送/接收机器和链路，测量多个 KV 载荷大小的单向完成时间，拟合固定启动项与有效带宽。时间边界应从 KV 可发送/提交开始，到目标端数据可被 Decode 使用；如果打包、拷贝或同步包含在这个边界里，需明确写进测量协议。它们不能同时又藏进 Prefill/Decode 整轮 profile 而被重复计算。

模拟器传输的 `S` 是整个实例的 KV 字节数，不是一个 rank 的局部分片。多卡并行传输时应使用相应实例级聚合有效带宽，不能直接把单网口标称速率或单 rank 带宽套入。GiB/s 使用 1024³，Gb/s 使用 10⁹ bit；单位换算必须先除以 8 再统一字节基数。

`CalibratedKVScheduler` 为相同有向 `(src_instance,dest_instance)` 复用一个 `CalibratedLink`，该链路同一时刻只运行一个 flow，按 FIFO 传输；不同有向实例对可并行。目标内存不足时先等待，以 1 秒仿真间隔重试。它没有模拟共享 NIC、交换机拥塞、HCCL 协议细节、逐层 KV 流水或跨实例计算通信重叠，不应据此推导这些机制的收益。

### 5.4 接入配置

为真实模型和硬件增加独立 YAML，避免覆盖示例名：

1. 在 `configs/hardware_repo/processors/` 增加目标 NPU，配置 `processor.NPU` 和实测/可核对的可用内存字节数；增加相应 SKU 和 cluster。
2. 在 model_repo 增加真实模型架构/权重配置，并配置准确 KV 布局。
3. 创建应用及 start_state，保证单实例设备数=TP，PP=1。
4. 导出严格符合 CSV schema 的性能数据和实际 filled metadata。
5. 指向新文件，保持 exact 和 synthetic 禁用；运行小 trace，检查完成数、OOM、查询次数和结果误差。

示例命令结构如下，`my_*` 指你已经新增并核验的配置：

```bash
python run.py --config-name config_ascend cluster=my_npu_cluster applications=my_model start_state=my_aggregate performance_model.profile_path=data/my_measured_profile.csv performance_model.metadata_path=data/my_measured_profile.metadata.json performance_model.allow_synthetic=False performance_model.lookup_mode=exact
```

## 6. KV 容量及现代注意力结构

`LLMArchitecture` 支持两种互斥方式，均表示**整个模型、单序列每 token 的 KV 字节数**：

1. 显式 `num_kv_heads`、`head_dim`、`kv_dtype_size`，得到 `2 × layers × KV_heads × head_dim × KV_dtype_bytes`。没有提供时回退到原 MHA 结构和模型 dtype。
2. 显式 `kv_cache_bytes_per_token`，用于已经校准的压缩/MLA 等布局容量。不能同时填写上述 KV heads/dimension/dtype 字段。

请求 KV 大小为 `request.batch_size × token_count × bytes_per_token`。示例 GQA 架构为 8 层、2 KV heads、head_dim=64、dtype=2，每 token 总 KV=4096 字节。默认 trace 的 batch_size=1。

容量公式只描述逻辑缓存字节数，不能自动覆盖分页块尾部浪费、对齐、量化 scale、工作区和临时 buffer。真实可用容量应预留这些开销，或者使用明确校准后的每 token 字节值。设置 MLA 的字节数并不等于实现了 MLA kernel、MLA 通信或它的压缩/解压耗时；这些仍需包含在适当 profile/传输测量边界内。

当前 TP 内存分摊继承了均匀分布假设，不模拟不均匀切分、复制 KV 或特定专家并行。不要将未验证的 EP、PP 或 Attention/FFN 拓扑配置解释为已经支持。

## 7. 输出解释与可追溯性

`simulation_metadata.json` 保存：CPU 离散事件标记、运行状态、输入/完成/未完成请求数、模拟结束时刻、profile 来源与路径/SHA256、原元数据、exact/nearest 查询次数、功率禁用状态和链路争用范围。运行阶段异常也会尽可能通过 finally 留存这些信息。审阅时至少确认：

- `all_requests_completed=true`；否则 summary 只覆盖已完成请求，可能有幸存者偏差。
- `run_status="completed"` 只代表执行没有抛出异常，不代表所有输入请求均完成；请求完整性以 counts 和 `all_requests_completed` 为准。
- `performance_model.provenance` 与报告证据层级一致。
- exact 模式没有把缺失形状暗中替换；nearest 模式的次数和用途有明确说明。
- `oom.csv` 不存在或已被定位；不能忽略 OOM 标记后继续比较吞吐。
- 仿真时间是模拟时钟；Python 实际运行秒数是工具执行成本，不能作为 NPU 模型推理延迟。

保留原仓库指标定义：TTFT 是 PromptTask 完成相对请求到达的时间；queue_time 是 Prefill 开始前等待；TBT 仍为 `(E2E−TTFT)/输出总token数`，不同于常见除以 `输出token数−1` 的 TPOT。`nth_token_overheads` 可帮助观察 Prefill 完成至 Decode 启动间隙。不要把 TBT 当成逐 token ITL 尾分布。

如果仿真终止时没有请求完成，`summary.csv` 的相关统计为空值，而不是零延迟或零排队。应以元数据中的输入/完成/未完成 counts 判断运行结果；不能把缺失统计填成 0 后参加性能比较。

继承的事件引擎在取出下一个事件之前检查当前时钟，因此可能执行首个超过 `end_time` 的事件；`end_time` 不能视为严格统计截断点。短时运行应同时检查 `simulated_time_seconds` 和请求完成数，固定窗口评估需按事件实际时间筛选。本轮适配未修改该引擎行为。

吞吐应从指定稳态时间窗口内完成的请求数或输出 token 数计算，明确分母、窗口、未完成请求处理与总设备数。Notebook 里由 trace 名称解析的请求率是提供负载，不天然等于完成吞吐。比较 P/D 与聚合时，两者都需独立调优，并保持相同总卡数/成本、workload 和 SLO，且使用真实标定后的 profile。

## 8. 验证与常见问题

```bash
python -m unittest discover -s tests -v
```

集成测试覆盖：两种 NPU 合成部署完整完成、0 秒到达、P/D 依赖和同链路串行、来源门禁与近邻记录、原 GPU 数据库/原 trace 的 baseline 回归。另用 exact 小表验证单请求 Prefill 10 ms、三个 Decode context 分别为 2/3/4 且耗时 1/2/4 ms，最终 E2E 必须为 17 ms、nearest 查询为 0，防止旧的多轮合并复用首轮时延。单元测试另覆盖 profile 合法性、context 每轮更新、GQA/显式容量和链路内存等待等。

| 现象 | 处理 |
|---|---|
| Synthetic profile refused | 演示可明确 allow_synthetic=True；真实评估应接入 measured 文件 |
| exact shape not found | 按错误中的 phase/batch/context 补采样；仅探索时显式选 nearest |
| unknown model/hardware/TP/phase | 核对配置名称和 CSV 分组；不要回退 GPU profile |
| PP 或预打包 batch 不支持 | 本适配 PP=1、trace batch_size=1；先转换 workload 或另行扩展模型 |
| 新配置缺少 mixed 数据 | 聚合会产生混合轮；增加实测 mixed 覆盖，不能假设纯 P/D 相加正确 |
| OOM 或持续内存等待 | 检查权重、KV、TP 分布、可用容量、目标队列和预留空间 |
| 需要真实功率/能耗结果 | 当前 disabled 模型不能提供；需独立采样并开发相应功率与积分模型 |

开发与验证范围以 `TASK_UNDERSTANDING.md` 为准。真实昇腾标定和硬件校准需要目标环境后才能完成。

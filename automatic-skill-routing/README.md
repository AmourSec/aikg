# Automatic Skill Routing

自动发现、选择并使用 Skills 的子系统——端到端落地,平台无关。

本文件夹是**自包含交付物**,与仓库其余部分解耦。可整目录取走或替换,不影响知识库本体。

---

## 1. 这是什么

解决一个问题:用户用自然语言描述任务,系统自动完成——

1. 发现候选 Skills(不要求用户知道 Skill 名称);
2. 语义判断适合的 Skills(可零个、一个、多个);
3. 使用前向用户知会真实名称与用途,必要时等待确认;
4. 加载选中 Skills 的完整说明并继续执行。

系统分两条相互独立的链路:

- **构建期目录**:`build_catalog.py` 扫描 `sources.yaml` 声明的本地来源，并可在维护者显式运行时同步启用的 Git 来源；它不在用户任务路由时联网。当前 Skill 数量与来源数以 `skills-router/catalog.json` 为准;
- **运行时远程提供方**:可选的 Ascend KG 提供方(`ascend-kg`,https://ascend.wiki),在用户逐任务授予网络同意后才联网检索,详见第 6 节。

协议平台无关,大模型自适应执行。

---

## 2. 文件清单

```text
automatic-skill-routing/
├── README.md                              ← 本文件,总说明与使用指南
├── OVERVIEW.md                            ← 快速理解:架构、流程、决策点
├── AUTOMATIC_SKILL_ROUTING_DESIGN.md      ← 设计方案(逐条对应需求)
└── skills-router/
    ├── README.md                          ← 维护文档(新增/更新/验证/故障恢复)
    ├── ROUTING_PROTOCOL.md                ← 平台无关路由协议(Step 0–5)
    ├── opencode-adapter.md                ← opencode 平台适配参考(含远程提供方映射)
    ├── config/
    │   └── sources.yaml                   ← 构建期来源配置(人维护,唯一来源真相)
    ├── scripts/
    │   ├── build_catalog.py               ← 通用扫描器,生成 catalog.json
    │   ├── validate_catalog.py            ← 完整性校验
    │   ├── generate_router_context.py     ← 生成可注入的 router-context.md
    │   └── test_routing.py                ← 回归测试
    ├── runtime/                           ← 运行时远程提供方(与构建期独立)
    │   ├── contracts.py                   ← 类型与协议契约
    │   ├── coordinator.py                 ← 同意门与降级协调
    │   ├── ascend_kg.py                   ← Ascend KG 提供方
    │   ├── ascend_kg_parsing.py           ← 远程响应边界解析
    │   ├── http_transport.py              ← urllib 传输层
    │   ├── local_catalog.py               ← 本地候选加载与类型化降级
    │   ├── token_registry.py              ← 任务级不透明响应令牌
    │   ├── rendering.py                   ← 远程正文定界渲染
    │   ├── facade.py + wire.py            ← 生产门面 RouterTask 与外部契约
    │   └── ndjson.py + __main__.py 等     ← 可执行 NDJSON 入口(python3 -m runtime)
    ├── tests/                             ← 运行时单元测试(假传输层,不联网)
    ├── catalog.json                       ← 自动生成,不要手改
    └── router-context.md                  ← 自动生成,注入模型上下文用
```

**阅读顺序建议:**

| 角色 | 先读 |
| --- | --- |
| 想快速理解方案 | `OVERVIEW.md` |
| 要评审设计 | `AUTOMATIC_SKILL_ROUTING_DESIGN.md` |
| 要落地实现 | `skills-router/ROUTING_PROTOCOL.md` + `skills-router/README.md` |
| 要在 opencode 中用 | `skills-router/opencode-adapter.md` |
| 要理解运行时远程提供方 | 本文件第 6 节 + `AUTOMATIC_SKILL_ROUTING_DESIGN.md` 第 7 节 |
| 要新增/维护 Skill | `skills-router/README.md` 第 3–7 节 |

---

## 3. 与仓库其余部分的关系

| 依赖方向 | 说明 |
| --- | --- |
| 本文件夹 → 仓库 `skills/` | `sources.yaml` 的 local source 指向仓库根 `skills/`,扫描其中的 `SKILL.md` |
| 本文件夹 → 仓库 `docs/` | 设计文档引用 `docs/11-knowledge-index/skills-authoring-guide.md` 作为 Skill 编写规范 |
| 仓库 → 本文件夹 | 无反向依赖。本文件夹可整目录移除而不影响知识库 |

**Skill 编写规范**仍遵循仓库既有的 `docs/11-knowledge-index/skills-authoring-guide.md`。本系统只额外建议 frontmatter 可选字段(`tags` / `confirm` / `triggers`),完全向后兼容。

---

## 4. 快速验证

前置:已安装 `pyyaml`(`pip install pyyaml`),且有网络访问 gitcode.com(首次同步 GitCode 仓库)。

```bash
# 1. 同步来源并生成 catalog(扫描 local + 各 GitCode 仓库)
python3 automatic-skill-routing/skills-router/scripts/build_catalog.py

# 2. 校验完整性
python3 automatic-skill-routing/skills-router/scripts/validate_catalog.py

# 3. 生成可注入的路由上下文
python3 automatic-skill-routing/skills-router/scripts/generate_router_context.py

# 4. 跑回归测试
python3 automatic-skill-routing/skills-router/scripts/test_routing.py

# 5. 跑运行时远程提供方单元测试(假传输层,不联网)
cd automatic-skill-routing/skills-router && python3 -m unittest discover -s tests -t .

# 6. 可执行入口自检(无网络即可成功;纯本地会话见 skills-router/opencode-adapter.md 第 6.9 节)
cd automatic-skill-routing/skills-router && python3 -m runtime --help
```

预期输出:

```text
catalog: N skills, ...
wrote automatic-skill-routing/skills-router/catalog.json
OK: N skills, 0 problems
wrote automatic-skill-routing/skills-router/router-context.md
PASS catalog_valid
PASS no_duplicates
PASS recall_matching
PASS recall_irrelevant
PASS load_skill
PASS confirm_logic
OK: all 6 tests passed (N skills)
OK (unittest 全部通过)
usage: python3 -m runtime ...
```

N 与冲突/错误数量以当次 `catalog.json` 生成为准;远程仓库内部的同名冲突和格式问题会被正确识别并排除,不影响可用 Skills。步骤 1、2、4 依赖 `.skills-cache` 同步完整;步骤 3、5、6 完全离线可用。

**如何使用路由上下文**:把 `router-context.md` 内容注入大模型系统提示,模型即可按协议自动路由。详见 `skills-router/ROUTING_PROTOCOL.md`。

---

## 5. 核心工作流程

构建期与协议执行(维护命令可同步 Git 来源；任务路由不因此联网):

```text
sources.yaml (人维护)
      ↓
build_catalog.py 扫描所有 source 的 **/SKILL.md
      ↓
catalog.json (自动生成,轻量元数据,不含正文;数量以当次生成为准)
      ↓
Agent 按 ROUTING_PROTOCOL.md 执行:
  Step 1 RECALL   召回候选(关键词+语义,≤10 条元数据)
  Step 2 SELECT   大模型语义判断,返回 0/1/N 个,带 order
  Step 3 NOTIFY   展示真实名称与用途
         CONFIRM  若任一 confirm=true,整组等待用户同意
  Step 4 LOAD     按需读取选中 SKILL.md 全文
  Step 5 EXECUTE  按 Skill 自身说明继续完成任务
```

运行时远程提供方(可选,仅用户同意后联网,见第 6 节):

```text
本地候选 + 远程候选
      ↓
每任务一次网络/隐私同意 → Ascend KG 检索(https://ascend.wiki)
      ↓
选择后含远程候选 → 单独的激活确认 → 加载远程内容
      ↓
执行目标:已装本地走原生 skill 工具 / 未装本地读校验路径 / 远程仅定界内联
```

---

## 6. 运行时远程提供方(Ascend KG)

`skills-router/runtime/` 实现可选的运行时远程检索,与构建期目录相互独立:catalog 仍是本地扫描产物,不因远程检索而改变。要点:

- **开启条件**:环境变量 `ASCEND_KG_API_KEY`(未设置或空白即不出网);每任务一次网络/隐私同意;加载远程内容前单独的激活确认;
- **可执行入口**:`python3 -m runtime --catalog catalog.json --workspace-root <root> --native-skill <name>` 提供 NDJSON、一进程一任务的生产门面;宿主适配器按 `opencode-adapter.md` 第 6.9 节接入;
- **出站请求**:检索 `POST https://ascend.wiki/search`,JSON body 仅含当前任务文本、`top_k: 10`、`with_neighbors: false`,以规范 UTF-8 字节精确发送,同意前向用户逐字披露;加载 `GET https://ascend.wiki/skill/<id>`;均带 `X-API-Key` 请求头;
- **有界**:超时 10 秒;检索响应上限 1 MiB;Skill 内容上限 256 KiB;仅 HTTP 429 重试(退避 0.5/1.0/2.0 秒,最多 4 次尝试);
- **本地回退**:无密钥、同意被拒、无匹配、含糊、401/403、429、503 及其他非 200、超时、响应非法、超限,任一情形都回退为仅用本地 catalog,不中断任务;本地候选不可加载(如 `.skills-cache` 缺失)同样以类型化降级报告,原生已装 Skill 不受影响;
- **信任与隐私**:远程内容是不可信文本,无策略权威;引用的额外资源一律不抓取;不持久化、不上传、不比较分数;响应令牌是任务级不透明句柄,不含候选 ID,任务结束即失效;远程正文仅以固定定界包络(`<<<REMOTE_SKILL_CONTENT>>>` ... `<<<END_REMOTE_SKILL_CONTENT>>>`)内联;候选展示保留来源(提供方、source_repo、source_file);
- **opencode 映射**:已安装本地 skill 用原生 `skill(name=...)`;未安装本地 skill 按校验路径读取;远程 skill 仅以定界内联内容进入对话,绝不调用原生 skill 工具,绝不安装、复制或软链;
- **上游关系**:仅参照 [`agent0/kg-tools`](https://gitcode.com/agent0/kg-tools),固定 commit 为 [`5568d8eedc70eebf155cd4e2728aee93ea02962d`](https://gitcode.com/agent0/kg-tools/commit/5568d8eedc70eebf155cd4e2728aee93ea02962d);其编排引擎未集成,未 vendor 任何上游代码。

完整设计见 `AUTOMATIC_SKILL_ROUTING_DESIGN.md` 第 7 节;opencode 落地细节见 `skills-router/opencode-adapter.md`;维护与故障排查见 `skills-router/README.md` 第 8 节。

---

## 7. 关键设计决策

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 路由是否硬编码 Skill 名 | 否,只扫 `**/SKILL.md` | 新增 Skill 无需改代码(需求第 6 节) |
| 来源如何接入 | sources.yaml 配置 | 新增来源无需专用代码分支(需求第 6 节) |
| 两个 GitCode 仓库如何接入 | 同一种 type=git 机制 | 需求第 9 节第 8 条 |
| catalog 是否含正文 | 否,只含元数据 | 控制上下文体积(需求第 6 节) |
| 召回是否用向量数据库 | 否,关键词+大模型语义 | 需求第 3 节不引入不必要复杂度 |
| 协议是否绑定平台 | 否,平台无关 | 需求第 3 节不绑定 Agent App |
| confirm 机制 | 协议层“模型停止等待用户” | 不依赖平台原生 confirm |
| Skill 元数据扩展 | 全部可选字段 | 向后兼容现有 SKILL.md |
| 远程检索是否默认开启 | 否,需密钥+逐任务同意 | 网络访问必须由用户显式授权 |
| 远程内容如何进入上下文 | 仅定界内联,不落盘不安装 | 远程内容不可信,不得获得持久影响力 |
| 是否集成 kg-tools 编排引擎 | 否,仅作设计参照 | 保持本系统轻量与自主可控 |

---

## 8. 实施状态

本轮已端到端落地,以下决策点已确认:

| 决策点 | 落地结果 | 备注 |
| --- | --- | --- |
| 交付范围 | 端到端落地 | 同步、catalog、路由上下文、回归测试全部完成 |
| 同步远程仓库 | 已启用 | GitCode 仓库已实际 clone 并扫描 |
| 默认 confirm 类别 | destructive / write-fs / network / exec-shell | 可在 sources.yaml 调整 |
| 召回机制 | 关键词 + 大模型语义 | 当前规模下有效,后续可引入向量索引 |
| 平台绑定 | 平台无关 | 大模型自适应执行协议 |
| 交付目录 | `automatic-skill-routing/` | 自包含 |
| 运行时远程提供方 | Ascend KG 已实现 | 同意门、有界请求、本地回退、单元测试全部落地;上游 kg-tools 未集成、未 vendor;未对线上 API 做过实测 |

### 当前规模

构建期指标以 `skills-router/catalog.json` 当前生成为准(来源数、可选 Skills、conflicts/errors 数量、文件大小均可从中直接读取)。运行时远程提供方不改变 catalog,其行为由 `skills-router/tests/` 的单元测试保障。

---

## 9. 后续优化方向

以下非本轮必须,但值得后续考虑:

1. **召回精度**:catalog 达数百 skills(当前数量以 catalog.json 生成为准)时关键词召回可能过多假阳性,可引入向量索引提升 Step 1 精度;
2. **catalog 分页**:若 skills 增至 500+,router-context.md 可能过大,可拆为 name-only 索引 + 按需读取 description;
3. **conflicts 处理**:同名冲突来自远程仓库内部重复(当前数量见 catalog.json 的 conflicts 数组),可向上游反馈或配置排除目录;
4. **errors 修复**:格式错误位于远程仓库中(当前数量见 catalog.json 的 errors 数组),可向上游反馈补全 frontmatter;
5. **confirm 覆盖**:当前远程 skills 均未声明 confirm/tags,可按需为高风险类 skill 补充。

不变量(任何后续修改都必须遵守,见 `skills-router/ROUTING_PROTOCOL.md` 第 7 节):

- 不硬编码 Skill 名称;
- selected 必须在 candidates 之内,candidates 必须在 catalog 之内;
- 正文按需加载,不全量载入;
- confirm 整组等待;
- 无匹配允许返回空;
- 知会前置。

---

## 10. 更多文档

| 想了解 | 看 |
| --- | --- |
| 完整设计与需求对应 | `AUTOMATIC_SKILL_ROUTING_DESIGN.md` |
| 路由协议细节 | `skills-router/ROUTING_PROTOCOL.md` |
| 运行时远程提供方设计 | `AUTOMATIC_SKILL_ROUTING_DESIGN.md` 第 7 节 |
| 如何新增/修改 Skill 或来源 | `skills-router/README.md` |
| 如何排查故障 | `skills-router/README.md` 第 9 节 |
| AI 修改本系统的守则 | `skills-router/README.md` 第 12 节 |
| opencode 适配(含远程提供方映射) | `skills-router/opencode-adapter.md` |

需求由用户提出,原始需求文件未纳入仓库;当前设计权威为本文件夹的 `AUTOMATIC_SKILL_ROUTING_DESIGN.md` 与 `skills-router/ROUTING_PROTOCOL.md`(历史设计文档见 `docs/superpowers/`,已标注被取代)。

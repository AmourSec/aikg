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

已接入三个来源,共 386 个 Skills。协议平台无关,大模型自适应执行。

---

## 2. 文件清单

```text
automatic-skill-routing/
├── README.md                              ← 本文件,总说明与使用指南
├── OVERVIEW.md                            ← 快速理解:架构、流程、决策点
├── AUTOMATIC_SKILL_ROUTING_DESIGN.md      ← 设计方案(逐条对应需求)
├── skills-router/
│   ├── README.md                          ← 维护文档(新增/更新/验证/故障恢复)
│   ├── ROUTING_PROTOCOL.md                ← 平台无关路由协议(Step 1–5)
│   ├── opencode-adapter.md                ← opencode 平台适配参考
│   ├── config/
│   │   └── sources.yaml                   ← 来源配置(人维护,唯一来源真相)
│   ├── scripts/
│   │   ├── build_catalog.py               ← 通用扫描器,生成 catalog.json
│   │   ├── validate_catalog.py            ← 完整性校验
│   │   ├── generate_router_context.py     ← 生成可注入的 router-context.md
│   │   └── test_routing.py                ← 回归测试
│   ├── catalog.json                       ← 自动生成,不要手改
│   └── router-context.md                  ← 自动生成,注入模型上下文用
```

**阅读顺序建议:**

| 角色 | 先读 |
| --- | --- |
| 想快速理解方案 | `OVERVIEW.md` |
| 要评审设计 | `AUTOMATIC_SKILL_ROUTING_DESIGN.md` |
| 要落地实现 | `skills-router/ROUTING_PROTOCOL.md` + `skills-router/README.md` |
| 要在 opencode 中用 | `skills-router/opencode-adapter.md` |
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
# 1. 同步来源并生成 catalog(扫描 local + 2 个 GitCode 仓库)
python3 automatic-skill-routing/skills-router/scripts/build_catalog.py

# 2. 校验完整性
python3 automatic-skill-routing/skills-router/scripts/validate_catalog.py

# 3. 生成可注入的路由上下文
python3 automatic-skill-routing/skills-router/scripts/generate_router_context.py

# 4. 跑回归测试
python3 automatic-skill-routing/skills-router/scripts/test_routing.py
```

预期输出:

```text
catalog: 386 skills, 5 conflicts, 9 errors
wrote automatic-skill-routing/skills-router/catalog.json
OK: 386 skills, 0 problems
wrote automatic-skill-routing/skills-router/router-context.md (165 KB, 386 skills)
PASS catalog_valid
PASS no_duplicates
PASS recall_matching
PASS recall_irrelevant
PASS load_skill
PASS confirm_logic
OK: all 6 tests passed (386 skills)
```

5 个 conflicts 和 9 个 errors 来自远程仓库内部的同名重复和格式问题,系统已正确识别并排除,不影响可用 Skills。

**如何使用路由上下文**:把 `router-context.md` 内容注入大模型系统提示,模型即可按协议自动路由。详见 `skills-router/ROUTING_PROTOCOL.md`。

---

## 5. 核心工作流程

```text
sources.yaml (人维护)
      ↓
build_catalog.py 扫描所有 source 的 **/SKILL.md
      ↓
catalog.json (自动生成,轻量元数据,不含正文)
      ↓
Agent 按 ROUTING_PROTOCOL.md 执行:
  Step 1 RECALL   召回候选(关键词+语义,≤10 条元数据)
  Step 2 SELECT   大模型语义判断,返回 0/1/N 个,带 order
  Step 3 NOTIFY   展示真实名称与用途
         CONFIRM  若任一 confirm=true,整组等待用户同意
  Step 4 LOAD     按需读取选中 SKILL.md 全文
  Step 5 EXECUTE  按 Skill 自身说明继续完成任务
```

---

## 6. 关键设计决策

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

---

## 7. 实施状态

本轮已端到端落地,以下决策点已确认:

| 决策点 | 落地结果 | 备注 |
| --- | --- | --- |
| 交付范围 | 端到端落地 | 同步、catalog、路由上下文、回归测试全部完成 |
| 同步远程仓库 | 已启用 | 两个 GitCode 仓库已实际 clone 并扫描 |
| 默认 confirm 类别 | destructive / write-fs / network / exec-shell | 可在 sources.yaml 调整 |
| 召回机制 | 关键词 + 大模型语义 | 386 skills 规模下有效,后续可引入向量索引 |
| 平台绑定 | 平台无关 | 大模型自适应执行协议 |
| 交付目录 | `automatic-skill-routing/` | 自包含 |

### 当前规模

| 指标 | 值 |
| --- | --- |
| 来源数 | 3(local + 2 GitCode) |
| 可选 Skills | 386 |
| 同名冲突(已排除) | 5 |
| 格式错误(已排除) | 9 |
| catalog.json | 284 KB |
| router-context.md | 165 KB |

---

## 8. 后续优化方向

以下非本轮必须,但值得后续考虑:

1. **召回精度**:386 skills 时关键词召回可能过多假阳性,可引入向量索引提升 Step 1 精度;
2. **catalog 分页**:若 skills 增至 500+,router-context.md 可能过大,可拆为 name-only 索引 + 按需读取 description;
3. **conflicts 处理**:5 个同名冲突来自远程仓库内部重复,可向上游反馈或配置排除目录;
4. **errors 修复**:9 个格式错误在远程仓库中,可向上游反馈补全 frontmatter;
5. **confirm 覆盖**:当前远程 skills 均未声明 confirm/tags,可按需为高风险类 skill 补充。

不变量(任何后续修改都必须遵守,见 `skills-router/ROUTING_PROTOCOL.md` 第 7 节):

- 不硬编码 Skill 名称;
- selected 必须在 candidates 之内,candidates 必须在 catalog 之内;
- 正文按需加载,不全量载入;
- confirm 整组等待;
- 无匹配允许返回空;
- 知会前置。

---

## 9. 更多文档

| 想了解 | 看 |
| --- | --- |
| 完整设计与需求对应 | `AUTOMATIC_SKILL_ROUTING_DESIGN.md` |
| 路由协议细节 | `skills-router/ROUTING_PROTOCOL.md` |
| 如何新增/修改 Skill 或来源 | `skills-router/README.md` |
| 如何排查故障 | `skills-router/README.md` 第 8 节 |
| AI 修改本系统的守则 | `skills-router/README.md` 第 11 节 |
| opencode 适配 | `skills-router/opencode-adapter.md` |

需求原文见仓库根 `AUTOMATIC_SKILL_ROUTING_TASK.md`(用户维护,非本文件夹产出)。

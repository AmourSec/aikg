# 快速理解 Automatic Skill Routing

一分钟看懂本子系统是什么、怎么跑、当前状态。

---

## 要解决的问题

用户不知道有哪些 Skill,也不想记名字。用户只描述任务,系统自动找到合适的 Skill,告知用户要用哪些,经同意后加载使用。

---

## 当前状态

**已端到端落地。** 运行时远程提供方的完整离线单元测试通过；构建期回归需本地 `.skills-cache` 完整时执行。

| 指标 | 值 |
| --- | --- |
| 构建期来源 | local + GitCode 仓库(明细见 catalog.json 的 sources) |
| 可选 Skills | 以 catalog.json 当前生成为准 |
| 路由上下文 | router-context.md(生成产物,大小随 catalog 变化) |
| 平台绑定 | 无,大模型自适应 |
| 运行时远程提供方 | Ascend KG(https://ascend.wiki),可选;需 ASCEND_KG_API_KEY + 每任务网络同意 + 激活确认 |

---

## 一图看懂

构建期(维护命令可同步启用的 Git 来源；不在用户任务路由时联网):

```text
sources.yaml          人维护,声明 Skill 来源(本地目录 / GitCode 仓库)
      ↓
build_catalog.py      通用扫描器,同步 git 仓库 + 扫 **/SKILL.md frontmatter
      ↓
catalog.json          轻量元数据,不含正文,数量以当次生成为准
      ↓
generate_router_context.py
      ↓
router-context.md     可注入的路由上下文(协议 + 精简 catalog)
      ↓
大模型按协议 6 个阶段执行
  0. CONSENT  仅远程检索需要逐任务网络/隐私同意
  1. RECALL   从 catalog 召回 ≤10 条候选(关键词+语义)
  2. SELECT   大模型判断,返回 0/1/N 个,带顺序
  3. NOTIFY   告知用户要用哪些 Skill
     CONFIRM  若有 confirm=true 的,整组等待用户同意
  4. LOAD     按需读取选中 SKILL.md 全文
  5. EXECUTE  按 Skill 说明继续完成任务
```

运行期(可选,仅用户同意后联网,与构建期相互独立):

```text
本地候选 + 远程候选
      ↓
每任务一次网络/隐私同意 → 检索 Ascend KG(POST https://ascend.wiki/search)
      ↓                        出站载荷仅:当前任务文本 + top_k + with_neighbors
选择含远程候选 → 单独的激活确认 → 加载远程内容(GET /skill/<id>)
      ↓
执行目标:已装本地走原生 skill 工具 / 未装本地读校验路径 / 远程仅定界内联
```

---

## 四个关键不变量

1. **路由代码不认 Skill 名字**——只扫 `**/SKILL.md`,加新 Skill 不改代码;
2. **catalog 只存元数据**——正文 Step 4 才按需加载,省上下文;
3. **协议不绑定平台**——任何支持“工具调用+多轮对话”的 Agent 都能适配;
4. **远程内容永远不可信**——运行时远程提供方须经两道用户同意,内容只做定界内联注入,无策略权威、不落盘、不安装、不抓取额外资源,任何远程失败都回退本地。

---

## 跑一遍

```bash
python3 automatic-skill-routing/skills-router/scripts/build_catalog.py
python3 automatic-skill-routing/skills-router/scripts/validate_catalog.py
python3 automatic-skill-routing/skills-router/scripts/generate_router_context.py
python3 automatic-skill-routing/skills-router/scripts/test_routing.py

# 运行时远程提供方单元测试(假传输层 + socket 封禁,不联网)
cd automatic-skill-routing/skills-router && python3 -m unittest discover -s tests -t .

# 可执行入口自检(无网络即可成功)
cd automatic-skill-routing/skills-router && python3 -m runtime --help
```

前三步与 `test_routing.py` 依赖 `.skills-cache` 同步完整(见上);单元测试与入口自检完全离线可用。

---

## 如何使用

把 `skills-router/router-context.md` 内容注入大模型系统提示。模型读到此文件后,会自动按协议路由用户任务。

---

## 运行时远程提供方(Ascend KG)

构建期 catalog 之外,`skills-router/runtime/` 提供可选的运行时远程检索,并附一个可执行生产门面(`python3 -m runtime`,NDJSON、一进程一任务),要点:

- 环境变量 `ASCEND_KG_API_KEY` 未设置或为空白时完全本地,不出网;
- 每任务一次网络/隐私同意;选中远程 Skill 后、加载内容前还有一次单独的激活确认;
- 出站载荷只有当前任务文本 + `top_k` + `with_neighbors`;超时 10 秒;响应有字节上限;仅 429 限次重试;
- 无密钥、被拒、无匹配、含糊、401/403、429、503、超时、非法、超限,任一情形都回退本地 catalog;
- 远程内容不可信、无策略权威、不落盘、不上传、不比较分数;
- 上游 [`agent0/kg-tools`](https://gitcode.com/agent0/kg-tools) 仅作设计参照,固定 commit 为 [`5568d8eedc70eebf155cd4e2728aee93ea02962d`](https://gitcode.com/agent0/kg-tools/commit/5568d8eedc70eebf155cd4e2728aee93ea02962d),编排引擎未集成,未 vendor 任何上游代码。

设计见 `AUTOMATIC_SKILL_ROUTING_DESIGN.md` 第 7 节;opencode 映射见 `skills-router/opencode-adapter.md`。

---

## 文件入口

| 看 | 了解 |
| --- | --- |
| `README.md` | 总说明 |
| `AUTOMATIC_SKILL_ROUTING_DESIGN.md` | 完整设计 |
| `skills-router/ROUTING_PROTOCOL.md` | 路由协议 |
| `skills-router/README.md` | 维护操作手册 |
| `skills-router/opencode-adapter.md` | opencode 适配与远程提供方映射 |

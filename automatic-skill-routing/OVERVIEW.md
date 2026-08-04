# 快速理解 Automatic Skill Routing

一分钟看懂本子系统是什么、怎么跑、当前状态。

---

## 要解决的问题

用户不知道有哪些 Skill,也不想记名字。用户只描述任务,系统自动找到合适的 Skill,告知用户要用哪些,经同意后加载使用。

---

## 当前状态

**已端到端落地。** 3 个来源、386 个 Skills、6 项回归测试全部通过。

| 指标 | 值 |
| --- | --- |
| 来源 | local + Ascend/agent-skills + cann/cannbot-skills |
| 可选 Skills | 386 |
| 路由上下文 | 165 KB(router-context.md) |
| 平台绑定 | 无,大模型自适应 |

---

## 一图看懂

```text
sources.yaml          人维护,声明 Skill 来源(本地目录 / GitCode 仓库)
      ↓
build_catalog.py      通用扫描器,同步 git 仓库 + 扫 **/SKILL.md frontmatter
      ↓
catalog.json          轻量元数据(284K,386 skills),不含正文
      ↓
generate_router_context.py
      ↓
router-context.md     可注入的路由上下文(协议 + 精简 catalog,165K)
      ↓
大模型按协议 5 步执行
  1. RECALL   从 catalog 召回 ≤10 条候选(关键词+语义)
  2. SELECT   大模型判断,返回 0/1/N 个,带顺序
  3. NOTIFY   告知用户要用哪些 Skill
     CONFIRM  若有 confirm=true 的,整组等待用户同意
  4. LOAD     按需读取选中 SKILL.md 全文
  5. EXECUTE  按 Skill 说明继续完成任务
```

---

## 三个关键不变量

1. **路由代码不认 Skill 名字**——只扫 `**/SKILL.md`,加新 Skill 不改代码;
2. **catalog 只存元数据**——正文 Step 4 才按需加载,省上下文;
3. **协议不绑定平台**——任何支持“工具调用+多轮对话”的 Agent 都能适配。

---

## 跑一遍

```bash
python3 automatic-skill-routing/skills-router/scripts/build_catalog.py
python3 automatic-skill-routing/skills-router/scripts/validate_catalog.py
python3 automatic-skill-routing/skills-router/scripts/generate_router_context.py
python3 automatic-skill-routing/skills-router/scripts/test_routing.py
```

---

## 如何使用

把 `skills-router/router-context.md` 内容注入大模型系统提示。模型读到此文件后,会自动按协议路由用户任务。

---

## 文件入口

| 看 | 了解 |
| --- | --- |
| `README.md` | 总说明 |
| `AUTOMATIC_SKILL_ROUTING_DESIGN.md` | 完整设计 |
| `skills-router/ROUTING_PROTOCOL.md` | 路由协议 |
| `skills-router/README.md` | 维护操作手册 |

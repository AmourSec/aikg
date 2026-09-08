## 榜单维护说明

榜单由仓库内的浏览量快照和文章作者信息自动生成，不需要手动维护排行行数或顺序。

### 给文章标注作者

在文章开头的 YAML 元数据区域添加 `authors`，使用稳定的姓名或账号。同一个人在所有文章中使用完全相同的名字，才能正确合并统计。

```yaml
---
title: 文章标题
authors:
  - alan
  - contributor-name
owner: maintainers
license: CC-BY-4.0
---
```

单人文章也推荐使用列表。已有单作者字段 `author: alan` 同样可用；同时存在时以 `authors` 为准。`authors` 必须是非空字符串列表，空姓名或错误格式会让生成任务报错，避免错误归属悄悄上线。

`owner` 继续表示维护责任，`sources` 或论文的作者表示参考来源，它们都不用于文章贡献者排行。正文代码块里的 `authors` 示例也不会被当作文章署名。

### 历史文章如何归属

没有 `authors` 或 `author` 的文章，采用 Git 历史中首次新增该文件的提交作者；重命名会追溯原始文件，后续修改者不会替换这个归属。Git 首次提交者是仓库贡献记录，若与实际作者不同，请直接补充文章署名。

Git 历史中的同一人若用了不同姓名或邮箱，可通过仓库根目录的 `.mailmap` 归一化；显示时只使用姓名，不展示邮箱。显式 `authors` 则始终使用填写的姓名，需要由维护者保持一致。两位同名贡献者应使用不同账号作为署名，避免合并。

无署名且无可追溯创建记录的文章会显示“作者未标注”，其浏览量仍进入文章浏览量榜，但不进入贡献者榜。

### 排行与统计范围

- 贡献者浏览量榜：先汇总该贡献者名下所有文章的累计浏览量，再取 Top 10；不是只汇总文章榜上的 10 篇。
- 贡献文章数榜：按该贡献者名下的知识文章数量取 Top 10；多作者文章分别计入每位作者。
- 文章浏览量榜：按文章累计浏览量取 Top 10，同时显示文章标题、全部作者和浏览量。
- 多作者文章的完整浏览量计入每位作者，每位作者在同篇文章里只计一次。贡献者榜总和可能大于文章浏览量总和。
- 只纳入 `mkdocs.yml` 导航中的知识文章。首页、知识地图、主题 `index.md` 概览、模板、内部计划与榜单自身不参与统计。
- 未访问文章按 0 计入。各榜不足 10 项时显示实际数量。

浏览量与文章页面共用 `docs/assets/data/pageviews.json`，由 GoatCounter 提供接入以来的累计次数，不是独立访客数，也不是当天增量。无效访问量会中止生成，保留上次页面。

### 每日自动更新

现有 `Deploy Docs` 工作流每天 23:17（UTC+8）触发，依次获取浏览量、生成榜单、运行测试、构建站点、提交统计文件并部署 GitHub Pages。GitHub Actions 调度可能延迟，实际完成时间以工作流为准。

推送文章或作者修改时也会使用已有快照重新生成榜单。没有新的浏览量变化时，快照的“最近变更”时间保持不变，但榜单更新日期会在下一次每日任务后更新。生成日期使用 UTC+8。

任务需要完整 Git 历史，因此 checkout 设置 `fetch-depth: 0`。继续使用已有的 `GOATCOUNTER_API_KEY` 仓库 Secret，无需增加密钥或本地定时任务。API、作者解析、测试或构建失败时任务中止，线上保留上次成功部署。

### 本地生成与检查

在仓库根目录使用 Python 3.12 或以上版本，安装 `requirements.txt` 中的项目依赖后运行：

```bash
python -m scripts.update_leaderboard
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/*.test.cjs
mkdocs build --strict
```

本地生成只读取现有快照，不调用 GoatCounter API。`python -m scripts.update_leaderboard --help` 可查看命令帮助。生成的 `docs/contribution-leaderboard.md` 会被覆盖；修改榜单说明请编辑 `scripts/leaderboard_maintenance.md`，修改统计与生成逻辑请编辑 `scripts/update_leaderboard.py`，修改作者读取规则请编辑 `scripts/leaderboard_data.py`。

需要立刻同步最新浏览量并部署时，可在 GitHub Actions 的 `Deploy Docs` 页面执行 `Run workflow`。

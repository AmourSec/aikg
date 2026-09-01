# Umami Cloud 文章浏览量设计

## 背景

AI Knowledge Graph 是通过 Material for MkDocs 构建并部署到 GitHub Pages 的纯静态站点。GitHub Pages 不提供可供站点使用的逐页面访问日志，也不能在请求期间写入数据库，因此文章浏览量需要由外部统计服务采集，再转换成站点可读取的静态快照。

本设计使用 Umami Cloud 采集页面访问，由 GitHub Actions 每晚把累计浏览量同步到仓库，并在文章标题下展示最近一次成功同步的结果。整个流程不依赖用户电脑常驻运行。

## 目标

- 为每个 MkDocs 页面采集累计 pageview。
- 在文章标题下展示截至最近一次成功同步的浏览量。
- 每天 23:17（Asia/Taipei）自动同步统计数据。
- 仅在统计数据发生变化时提交生成文件。
- API 密钥只存在于 GitHub Actions Secret，不进入仓库或浏览器。
- 同步失败时保留上一版数据和已部署站点。
- 保持现有 GitHub Pages 构建和部署方式。

## 非目标

- 不实现实时更新的公开计数器；公开数字最多滞后一天。
- 不修改每篇 Markdown 的正文或 front matter。
- 不在仓库中保存访客级事件、IP、User-Agent 或会话信息。
- 不构建自定义统计后台；详细分析继续使用 Umami Cloud 控制台。
- 不迁移文章 URL 之间的历史数据；URL 改名时需另行维护别名映射。

## 架构

系统包含三个相互独立的部分：

1. **浏览器采集**：所有生成页面加载 Umami Cloud 提供的跟踪脚本，并使用公开的 Website ID 上报页面访问。Website ID 不是密钥，可以出现在生成的 HTML 中。
2. **每日同步**：GitHub Actions 使用私密 API Key 调用 Umami Cloud API，将逐路径累计 pageview 转换为稳定、排序后的 JSON 快照。
3. **页面展示**：本地 JavaScript 读取随站点一起部署的 JSON，根据当前规范化路径查找并渲染浏览量。

数据流：

```text
访客浏览文章
  -> Umami Cloud 记录访问
  -> GitHub Actions 每晚读取逐路径累计 pageview
  -> 更新 docs/assets/data/pageviews.json
  -> 同一次工作流构建并部署 GitHub Pages
  -> 页面从本地 JSON 展示最近一次快照
```

## 组件与文件

### `mkdocs.yml`

- 注册 Umami 跟踪脚本和页面浏览量展示脚本。
- 继续使用现有 Material 主题和构建配置。
- 不包含 API Key。

### `docs/assets/javascripts/analytics.js`

- 配置或加载 Umami Cloud 官方跟踪脚本。
- 使用用户创建站点后获得的公开 Website ID。
- 每次完整页面加载只上报一次。
- 当前站点未启用 `navigation.instant`；如果未来启用，需要改为监听 Material 的导航 observable，避免漏报或重复上报。

### `scripts/update_pageviews.py`

- 从环境变量读取 `UMAMI_API_KEY` 和 `UMAMI_WEBSITE_ID`。
- 先调用 Umami 的 date-range API 获得当前可用统计区间。
- 调用 expanded metrics API，以 `path` 为维度获取 `pageviews`，每页最多 500 条，并处理 `offset` 分页。
- 规范化路径、过滤非页面记录、按路径排序。
- 先在内存中验证完整响应，再以确定性格式写入目标文件。
- 网络、鉴权、JSON 结构或分页异常时以非零状态退出，不覆盖旧快照。
- 日志不得输出 API Key 或完整请求头。

使用的 Umami Cloud API 根地址为 `https://api.umami.is/v1`。统计接口以 Umami 官方 Cloud API Key 认证，使用 Bearer header。

### `docs/assets/data/pageviews.json`

生成文件采用以下结构：

```json
{
  "schema_version": 1,
  "updated_at": "2026-09-01T23:17:00+08:00",
  "pages": {
    "/03-inference-systems/kv-cache/": 926,
    "/03-inference-systems/vllm/": 1382
  }
}
```

- `updated_at` 只在页面计数发生变化时更新，因此无访问变化时不会制造空提交。
- `pages` 的键为规范化后的站内绝对路径，值为非负整数 pageview。
- 初始文件使用 `{"schema_version": 1, "updated_at": null, "pages": {}}`，使首次部署不依赖 Umami 已有数据。

### `docs/assets/javascripts/pageviews.js`

- 使用 `fetch` 读取站内 `assets/data/pageviews.json`，不直接调用 Umami API。
- 从 `location.pathname` 移除查询字符串和片段，并统一目录尾部 `/`。
- 找到 `.md-content__inner > h1` 后插入浏览量文本。
- 没有对应计数、JSON 不可用或 DOM 结构不匹配时静默隐藏，不影响正文。
- 使用 `textContent` 渲染数字，不拼接不可信 HTML。
- 数字通过 `Intl.NumberFormat("zh-CN")` 格式化。

### `docs/assets/stylesheets/extra.css`

- 为浏览量文本增加使用 Material 现有颜色变量的低强调样式。
- 不引入新的字体、颜色体系或布局组件。

### `.github/workflows/deploy-pages.yml`

- 保留现有 `push` 和 `workflow_dispatch` 入口，增加每天 23:17、`Asia/Taipei` 的 `schedule`。
- `schedule` 和手动运行在构建前同步浏览量；普通内容 push 不访问 Umami API。
- 将 `contents` 权限调整为 `write`，保留 `pages: write` 和 `id-token: write`。
- 快照变化时，以 `github-actions[bot]` 身份提交唯一生成文件。
- 无变化时跳过提交。
- 无论是否生成提交，当前工作流继续构建当前工作树并部署；不能依赖机器人提交再次触发 Pages 工作流。
- 同步失败时终止构建和部署，线上继续保留上一版成功产物。

## 路径规则

- 统计键只使用 URL pathname，不含 scheme、host、query 或 fragment。
- 空路径规范化为 `/`。
- 非根目录路径统一以 `/` 结尾，与 MkDocs 默认目录 URL 对齐。
- 同一个页面的 `/path` 与 `/path/` 合并到 `/path/`。
- 第一版同步所有 Umami 返回的路径，不增加一次预构建来过滤 sitemap。已删除页面的历史键可能继续存在于 JSON 中，但没有页面会读取它们，不影响显示；未来仅在文件规模成为实际问题时再增加清理策略。

## 安全与隐私

- `UMAMI_API_KEY` 存放在 GitHub Repository Actions Secret。
- `UMAMI_WEBSITE_ID` 是公开标识，可放在配置中；为保持工作流配置一致，也可以作为 Repository Variable 保存。
- Action 权限遵循最小化原则：仅需要仓库内容写入、Pages 部署和 OIDC。
- 浏览器永远不获得 Umami API Key。
- 仓库只保存聚合后的路径与数字，不保存用户级记录。
- 所有外部请求必须使用 HTTPS，并设置有限超时。

## 失败处理

- Umami API 返回非 2xx：同步脚本失败，旧快照不变。
- API 返回未知结构或非整数 pageview：同步脚本失败，不发布部分数据。
- 分页中途失败：整次同步失败，不写入目标文件。
- Git 提交或推送失败：工作流失败，不部署未被仓库记录的快照。
- 页面加载 JSON 失败：不显示浏览量，文章其余内容正常工作。
- 某路径尚无数据：不显示 `0`，避免把“尚未同步”误解成真实零访问。

## 验证方案

### 自动验证

- 为同步脚本提供标准库单元测试，覆盖分页、路径规范化、无变化、错误响应和确定性 JSON 输出。
- 对展示脚本至少执行语法检查；如仓库没有现成 JavaScript 测试框架，不为该小功能引入新的前端工具链。
- 执行 `mkdocs build --strict`，确保资源路径和配置有效。

### 手动验证

1. 本地启动 MkDocs，确认文章页在空快照下正常显示且无控制台错误。
2. 使用测试快照确认标题下显示正确的格式化数字，并验证不存在路径时静默隐藏。
3. 在 GitHub 手动运行部署工作流，确认 Umami API 拉取成功并只修改快照文件。
4. 访问线上文章产生页面浏览，再次手动同步，确认 Umami、仓库 JSON 和页面显示形成完整闭环。
5. 临时使用无效凭据运行受控测试，确认旧快照未被覆盖且线上站点不受影响。

## 验收标准

- Umami Cloud 控制台能看到线上文章访问。
- 每晚定时工作流能在无本地电脑参与的情况下执行。
- 有浏览量变化时，仅生成预期的快照提交并完成 Pages 部署。
- 没有变化时不生成提交。
- 每个已有统计记录的文章页显示对应累计 pageview。
- API Key 不出现在 Git 历史、构建产物、页面源码或工作流日志中。
- 任一外部依赖失败都不会破坏现有文章访问。

# 收据识别 & Quartzy 提交助手

这个小工具是用来处理实验室采购收据的。上传图片后，它会自动识别出商品信息，支持手动校对，然后一键提交到 Quartzy 的 request 系统。

## 结构说明

### `app.py`
主逻辑入口，基于 Streamlit 搭了个网页界面，流程大概是这样：

1. **上传图片**
   - 支持 jpg/png/webp
   - 支持一次上传多张图片
   - 图片会以 gallery 形式显示，点击选择后校对对应表格

2. **点击“识别收据”**
   - 先用 MinerU 精准解析接口把图片解析成 Markdown
   - 再用 DeepSeek 从 Markdown 抽取出结构化数据
   - 支持批量识别，最多 5 张收据同时处理
   - 每张图片都有独立的可编辑商品表格（名称、货号、数量、单价等）

3. **点击“提交至 Quartzy”**
   - 把表格中的数据逐条转成 request 并发给 Quartzy API

---

### 模型提示词（`receipt_markdown_prompt`）
在代码里定义了一个提示词，告诉 DeepSeek 怎么从 MinerU 解析出的 Markdown 里抽取收据、遇到不确定信息怎么处理、一些品牌缩写怎么写等。

你可以直接改 `receipt_markdown_prompt` 里的内容来适配你自己的需求，或者以后扩展支持英文收据。

---


你还需要配置：
- MinerU API Key
- DeepSeek API Key
- Quartzy API Token
- 你的实验室 ID 和 type ID

这些都放在 `.env` 里，python dotenv会处理读取。

---

### 数据格式转换

识别出来的东西会显示在界面上，可以修改，改完了点提交到 Quartzy 就会自动提交修改后的数据上去。

---

### 特性


- 表格 UI 可校对修改
- 一键提交到 Quartzy（支持批量）

---
### TO-DO

- 操作逻辑优化，现在每次识别完图片需要自己点x然后再拖下一张图进来
- 显示逻辑优化，识别图片的时候希望不要在图片上显示正在加载
- 程序运行的时候有个 warning, 有空也可以解决下，目前是还不影响使用
- 收据自动按时间归档
- 写个Dockerfile跑在实验室服务器上

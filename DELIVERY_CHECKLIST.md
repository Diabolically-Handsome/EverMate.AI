# EverMate 交付检查清单（最小版）

## 1) 环境准备
1. Python 版本 `>=3.10`
2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 可选：本地 Ollama 运行中（默认 `http://localhost:11434`）

## 2) 启动检查
1. 启动应用：
```bash
python app.py
```
2. Welcome 页面正常显示
3. 点击“开始使用”后进入聊天页

## 3) GUI 功能检查
1. 切换语言/主题不报错
2. 顶部“记忆状态”可显示：
   - 记忆目录
   - Chunks/Terms/Uploads 计数
   - 最近分析时间
3. “查看记忆”面板可打开/关闭
4. 拖拽 `.txt/.docx` 后点击“构建/重建记忆”成功

## 4) 记忆链路检查
1. 连续对话 2-3 轮后，`Chunks` 计数增长
2. 点击“分析记忆”后，最近分析时间更新
3. Memory Root 下存在：
   - `index.sqlite`
   - `chunks/`
   - `01_core.md`
   - `02_persona.md`
   - `03_vault.md`

## 5) 退出自动保存检查
1. 调整窗口大小和位置，切换语言/主题/人格/模型
2. 输入框输入未发送文本，关闭应用
3. 重新启动后确认以下状态恢复：
   - 窗口尺寸/位置（以及最大化状态）
   - 当前页面（Welcome 或 Chat）
   - 语言/主题/人格/模型选择
   - 聊天区内容、输入框文本、待导入文件列表、记忆面板显示状态

## 6) 交付产物建议
1. 发布包内包含：`assets/`, `views/`, `app.py`, `memory_manager.py`, `requirements.txt`
2. 发布说明写明：
   - 首次运行会自动创建记忆目录
   - 源码运行默认使用 `./memory`
   - 打包后的 macOS 应用默认使用 `~/Library/Application Support/EverMate/memory`

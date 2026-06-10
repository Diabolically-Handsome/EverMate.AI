# EverMate 交付检查清单（最小版）

## 1) 环境准备
1. Python 版本 `>=3.10`
2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 可选：本地 Ollama 运行中（默认 `http://localhost:11434`）

## 2) 自动化检查
```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q   # 全部通过
```

## 3) 启动检查
1. 启动应用：
```bash
python app.py
```
2. Welcome 页面正常显示
3. 点击“开始使用”后进入聊天页
4. Ollama 未启动时发送消息：应弹出安装/启动引导（而不是“模型未安装”）

## 4) GUI 功能检查
1. 切换语言/主题不报错；英文模式下错误弹窗也是英文
2. 侧栏“记忆状态”可显示：记忆目录、Chunks/Terms/Uploads 计数、最近分析时间
3. “查看记忆”面板可打开/关闭
4. 拖拽 `.txt/.docx`（或包含它们的文件夹）后点击“构建/重建记忆”成功；
   拖拽悬停时虚线框高亮
5. 发送消息：回复**流式逐字出现**，期间窗口可拖动、不冻结
6. 记忆管理三项可用：清除聊天记忆 / 管理已导入文档（可删除）/ 清空全部记忆

## 5) 记忆链路检查
1. 连续对话 2-3 轮后，`Chunks` 计数增长
2. 点击“分析记忆”后，最近分析时间更新（在后台执行，UI 不冻结）
3. 重复导入同一文件：提示“重复内容已自动跳过”
4. Memory Root（`~/Library/Application Support/EverMate/memory`）下存在：
   - `index.sqlite`
   - `chunks/`
   - `01_core.md`
   - `02_persona.md`
   - `03_vault.md`

## 6) 退出自动保存检查
1. 调整窗口大小和位置，切换语言/主题/人格/模型
2. 输入框输入未发送文本，关闭应用
3. 重新启动后确认以下状态恢复：
   - 窗口尺寸/位置（以及最大化状态）
   - 当前页面（Welcome 或 Chat）
   - 语言/主题/人格/模型选择
   - 聊天区内容、输入框文本、待导入文件列表、记忆面板显示状态
4. 再开第二个实例：应提示“EverMate 已在运行”并退出

## 7) 交付产物清单
发布包（或源码发布）必须包含：
- `app.py`
- `engine/`（整个包）
- `views/`
- `assets/`
- `memory_manager.py`（兼容垫片）
- `ollama_client.py`
- `models_config.py`
- `runtime_paths.py`
- `i18n_qt.py`
- `requirements.txt`
- `LICENSE`

发布说明写明：
- 首次运行会自动创建记忆目录
- 记忆目录统一位于 `~/Library/Application Support/EverMate/memory`
  （源码运行与打包运行一致；可用 `MEMORY_DIR` 覆盖）
- Ollama 需单独安装

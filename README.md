# 🐾 EverMate.AI

### Your Local AI Pet & Friend
本地 AI 宠物 / AI 朋友 · Privacy-first · Offline whenever possible

[English](#english-version) · [中文说明](#中文说明)

---

## 中文说明

### ✨ 项目简介
**EverMate.AI** 是一个运行在您电脑上的 AI 宠物 / AI 朋友。  
通过 [Ollama](https://ollama.com/) 调用本地大语言模型进行对话，默认不上传数据，尽量离线运行，让AI像朋友一样长期陪伴您。

### 🧩 计划功能（Roadmap）
- [ ] 本地聊天窗口（支持上下文记忆）
- [ ] 多模型选择（Qwen、DeepSeek、GPT-OSS-20B 等）
- [ ] 人格设定（可编辑 System Prompt）
- [ ] 本地长期记忆（sqlite / json）
- [ ] 语音输入与语音播报（Whisper + 本地 TTS）
- [ ] 全局热键 / 托盘菜单 / 开机自启
- [ ] 浏览器侧边栏扩展（Page Assist 风格）
- [ ] 桌面端封装（Tauri / Electron）
- [ ] 多模态（看图 / 读文件，硬件允许时）

**开发日志**
- 2025-08-08：项目创建，占坑阶段

### 🚀 快速开始（占坑阶段）
> 代码即将开源，以下为预期最小运行方式

1. 安装 **Ollama**（macOS / Windows / Linux）  
2. 拉取基础模型：
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
3. 启动本地应用（未来示例）：
   ```bash
   python app.py
   ```

### 🏗️ 技术栈（规划）
- 推理：Ollama（本地 LLM，REST API）
- 界面：Gradio（MVP）→ WebExtension → Tauri / Electron
- 存储：sqlite / json（长期记忆与偏好）
- 语音：ffmpeg + Whisper / 本地 TTS
- 协议：MIT

### 🔒 隐私声明
- 默认本地推理，不上传对话内容  

---

## English Version

### ✨ Overview
**EverMate.AI** is your **local AI pet & friend**.  
Powered by [Ollama](https://ollama.com/) and open LLMs, it prefers offline inference and keeps your data on device.

### 🧩 Planned Features (Roadmap)
- [ ] Local chat UI with context memory
- [ ] Model picker (Qwen, DeepSeek, GPT-OSS-20B, …)
- [ ] Editable persona (System Prompt)
- [ ] Long-term local memory (sqlite / json)
- [ ] Voice input & TTS (Whisper + local TTS)
- [ ] Global hotkey / tray / autostart
- [ ] Browser side panel (Page-Assist style)
- [ ] Desktop app packaging (Tauri / Electron)
- [ ] Multimodal (vision / files, when hardware allows)

**Dev Log**
- 2025-08-08: Repository created (placeholder)

### 🚀 Quick Start (Placeholder)
> Code coming soon. Expected minimal run:

1. Install **Ollama** (macOS / Windows / Linux)  
2. Pull a base model:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ```
3. Start local app (future example):
   ```bash
   python app.py
   ```

### 🏗️ Tech Stack (Planned)
- Inference: Ollama (local LLM via REST)
- UI: Gradio (MVP) → WebExtension → Tauri / Electron
- Storage: sqlite / json
- Voice: ffmpeg + Whisper / local TTS
- License: MIT

### 🔒 Privacy
- Prefer local inference; no conversation uploads  

---

## 📜 License
MIT License — see [LICENSE](LICENSE)

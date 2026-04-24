# Pi Mono - Python Port

Python版本的pi-mono monorepo，用于构建AI编码代理。

## 包结构

本项目包含三个pip包，采用统一版本号：

| 包 | 功能 |
|---|------|
| `pi-ai` | LLM API抽象层，支持阿里云百炼和OpenAI |
| `pi-agent-core` | Agent运行时，状态管理、事件系统、工具执行 |
| `pi-coding-agent` | 编码代理CLI，会话管理、TUI界面 |

## 安装

```bash
pip install pi-coding-agent
```

这将自动安装 `pi-ai` 和 `pi-agent-core` 作为依赖。

## 使用

### 命令行

```bash
# 交互模式（默认）
pi --workspace /path/to/project

# 单次提示
pi "帮我修复这个bug" --workspace /path/to/project --mode print

# RPC模式（程序化控制）
pi --mode rpc --workspace /path/to/project

# 指定模型
pi --model gpt-4o --workspace /path/to/project

# 继续已有会话
pi --session session_abc123 --workspace /path/to/project
```

### SDK嵌入

```python
import asyncio
from pi_coding_agent import create_sdk_session, run_sdk_prompt

async def main():
    session = create_sdk_session("/path/to/project")
    
    def on_event(event):
        if event["type"] == "message_end":
            print("收到响应")
    
    output = await run_sdk_prompt(session, "分析这个项目的架构")
    print(output)

asyncio.run(main())
```

## 功能特性

### AgentSession
- 封装Agent核心，管理会话状态
- 7个编码工具（read、write、edit、bash、grep、find、ls）
- 消息队列（steer、follow_up）
- 事件订阅系统

### SessionManager
- JSONL格式持久化
- 树形分支结构
- 版本迁移支持（v1→v2→v3）
- 压缩机制（自动压缩长对话）

### 工具
- **read** - 读取文件，支持offset/limit分页
- **write** - 创建或覆盖文件
- **edit** - 精确字符串替换
- **bash** - 执行shell命令，支持超时和中断
- **grep** - 搜索文件内容（正则）
- **find** - 搜索文件路径（glob）
- **ls** - 列出目录内容

### TUI界面
- Textual富终端界面
- 消息显示（用户/助手角色样式）
- 多行输入编辑器
- 状态栏显示运行状态
- 快捷键：Ctrl+Enter提交、Ctrl+C退出、Ctrl+L清空

### 扩展系统
- 本地扩展：`.pi/extensions/*.py`
- 全局扩展：`~/.pi/extensions/*.py`
- 每个扩展定义 `extension_factory()` 函数

## 支持的LLM提供商

- **阿里云百炼** - qwen-plus、qwen-max等模型
- **OpenAI** - gpt-4o、gpt-4-turbo等模型

## 开发

### 安装开发依赖

```bash
cd pi-mono
pip install -e ./pi-ai
pip install -e ./pi-agent-core
pip install -e ./pi_coding_agent
```

### 运行测试

```bash
pytest tests/ -v
```

当前测试状态：
- pi-ai: 230测试通过
- pi-agent-core: 230测试通过
- pi-coding-agent: 511测试通过
- **总计: 971测试通过**

### 目录结构

```
pi-mono/
  pi_ai/                    # LLM API包
    types.py                # 消息类型定义
    stream.py               # 事件流
    providers/
      bailian.py            # 百炼提供商
      openai.py             # OpenAI提供商
      faux.py               # 测试mock
  
  pi_agent_core/            # Agent核心包
    agent.py                # Agent类
    agent_loop.py           # Agent循环
    _abort.py               # AbortController
    _event_stream.py        # 事件流
    types.py                # 类型定义
  
  pi_coding_agent/          # 编码代理包
    cli.py                  # CLI入口
    core/
      agent_session.py      # AgentSession
      session_manager.py    # 会话管理
      compaction.py         # 压缩
      tools/                # 7个工具
      extensions/           # 扩展系统
    modes/
      print_mode.py         # Print模式
      rpc.py                # RPC模式
      sdk.py                # SDK模式
    tui/                    # Textual界面
  
  tests/                    # 测试文件
```

## 命令行选项

```
用法: pi [提示词] [选项]

选项:
  --workspace, -w    工作区目录（默认当前目录）
  --model, -m        模型ID
  --mode             运行模式: interactive, print, rpc
  --no-tui           禁用TUI（使用基础输入）
  --session, -s      继续已有会话
  --version, -v      显示版本
```

## 与TypeScript版本的对应

| TypeScript | Python | 说明 |
|------------|--------|------|
| `@mariozechner/pi-ai` | `pi-ai` | LLM API层 |
| `@mariozechner/pi-agent-core` | `pi-agent-core` | Agent运行时 |
| `@mariozechner/pi-coding-agent` | `pi-coding-agent` | 编码代理CLI |
| `@mariozechner/pi-tui` | Textual | 使用Textual替代自定义TUI |

## 版本历史

### 0.1.0 (2026-04)
- Python移植初始版本
- 完成三个核心包
- 971测试通过

## 许可证

MIT
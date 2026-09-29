# 图片分拣 / LUT 调色工具

在自己电脑上运行的网页工具，批量处理图片，图片不会上传到任何服务器。

| 功能 | 做什么 |
|---|---|
| 文字分拣 | 按文字面积占比分类，把含文字 / 不含文字的图片挪走 |
| 色彩分析 | 找出过曝、欠曝、纯白纯黑、高饱和纯色、主色搭配不合格的图片 |
| 构图分析 | 用视觉大模型判断构图，挑出构图不合格的图片 |
| 去边裁剪 | 检测并裁掉图片四周的纯色边框 |
| LUT 调色 | 从参考图或文字描述生成 LUT，批量调色，拖动滑杆实时预览 |

---

## 下载与安装

### Mac（推荐）

1. 点本页右上方绿色的 **Code → Download ZIP**，或到 [Releases](../../releases) 下载压缩包，解压。
2. 双击文件夹里的 **`启动.command`**。
   - 提示「无法验证开发者」：在文件上点右键 → 打开 → 再点「打开」。
   - 还是打不开：打开「终端」，输入 `bash `（后面有空格），把 `启动.command` 拖进去，按回车。
3. 第一次打开会自动安装运行环境（需联网，约 3~10 分钟，只需一次）。
   电脑上没有 Python 时会自动打开 Python 官网，安装 Python 3.12 后再双击一次。
4. 装好后浏览器自动打开 <http://127.0.0.1:8765>。使用期间不要关掉终端窗口。

以后每次使用，双击 `启动.command` 即可。

### Windows

1. 安装 [Python 3.12](https://www.python.org/downloads/windows/)，安装时勾选「Add python.exe to PATH」。
2. 在解压后的文件夹地址栏输入 `cmd` 回车，执行（第一次需要几分钟）：

   ```bat
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. 以后每次启动：

   ```bat
   .venv\Scripts\python app.py
   ```

   然后用浏览器打开 <http://127.0.0.1:8765>

### 手动启动（任意系统）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_samples.py        # 可选：生成示例图
python app.py                     # 或 python app.py --folder /path/to/images
```

支持 Python 3.9 ~ 3.13。

---

## 使用说明

页面左侧「图片文件夹」填完整路径。Mac 上在访达里选中文件夹，按 `Option + Command + C` 拷贝路径。

> ⚠️ 文字分拣、色彩分析、构图分析的「移出…」按钮会把图片从原文件夹**直接挪走**（不是复制），建议先用副本试。
> 去边裁剪和 LUT 调色导出为新文件，不改动原图。

### 文字分拣

- 拖动阈值实时重分类，不必重新扫描；柱状图显示文字面积分布
- 点图片放大，可看到检测到的文字框
- 「移出含文字 / 不含文字图片」挪到 `原名_with_text` / `原名_no_text`
- 文字占比 = 所有文字框的并集面积 / 整图面积

### 色彩分析

- 过曝 / 纯白 / 纯黑 / 欠曝面积，默认阈值 10%
- 高饱和高明度纯色面积，默认阈值 20%
- 主色是否属于对比色、互补色、邻近色、相邻色、同色系
- 任一条不达标即进入移出名单；移出到 `原名_color_fail`，每张附 `{文件名}.json`，另有总表 `color_fail.json`

### 构图分析

需要能看图的大模型，二选一：

- 本机安装 [Ollama](https://ollama.com) 并下载视觉模型（如 `ollama pull llava`）
- 设置环境变量 `OPENAI_API_KEY`（可选 `COMPOSITION_LLM_BASE` 指定兼容接口地址）

### 去边裁剪

自动检测四周纯色边框并裁掉。导出 JPEG 需要 `jpegtran`（Mac：`brew install jpeg-turbo`）。

### LUT 调色

1. 填 LUT 名称，再填「参考图文件夹」或「特性描述」，点「生成 LUT」
2. 点「载入待处理图」
3. 拖动滑杆调整，右侧效果用 GPU 实时渲染；停手后自动换成精确结果
4. 导出调好色的图片（打包为 zip）

---

## 常见问题

- **安装依赖失败**：多半是网络问题，换个网络后重新双击 `启动.command`，会自动重装。
- **提示端口被占用**（address already in use）：工具可能已在运行，直接打开 <http://127.0.0.1:8765>。
- **换端口**：`python app.py --port 8800`

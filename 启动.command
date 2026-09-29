#!/bin/bash
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$HOME/.local/bin:$PATH"
URL="http://127.0.0.1:8765"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

pause_exit() {
  echo
  read -r -p "按回车键关闭窗口…" _
  exit "${1:-1}"
}

pick_python() {
  for cmd in python3.12 python3.11 python3.10 python3.13 python3.9 python3; do
    path=$(command -v "$cmd" 2>/dev/null) || continue
    if "$path" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

if curl -s -o /dev/null "$URL/api/defaults"; then
  echo "工具已经在运行，直接打开浏览器：$URL"
  open "$URL"
  exit 0
fi

if [ ! -x .venv/bin/python ] || [ ! -f .venv/.ready ]; then
  if ! PY=$(pick_python); then
    echo "没有找到可用的 Python（需要 3.9 ~ 3.13）。"
    echo "即将打开 Python 官网，请下载安装 Python 3.12，装好后重新双击「启动.command」。"
    open "https://www.python.org/downloads/macos/"
    pause_exit
  fi
  echo "首次使用，正在准备运行环境（约 3~10 分钟，只需要一次）…"
  echo "使用 Python：$PY"
  rm -rf .venv
  "$PY" -m venv .venv || { echo "创建运行环境失败。"; pause_exit; }
  .venv/bin/python -m pip install -q --upgrade pip -i "$MIRROR" --timeout 60 || true
  if ! .venv/bin/python -m pip install -r requirements.txt -i "$MIRROR" --timeout 60; then
    echo "国内镜像安装失败，改用官方源重试…"
    if ! .venv/bin/python -m pip install -r requirements.txt --timeout 60; then
      echo "依赖安装失败，请检查网络后重新双击「启动.command」。"
      pause_exit
    fi
  fi
  touch .venv/.ready
  echo "运行环境准备完成。"
fi

(
  for _ in $(seq 1 120); do
    if curl -s -o /dev/null "$URL/api/defaults"; then
      open "$URL"
      exit 0
    fi
    sleep 1
  done
) &

echo "正在启动，浏览器会自动打开 $URL"
echo "使用期间请不要关闭这个窗口；关闭窗口即停止工具。"
.venv/bin/python app.py "$@" || pause_exit

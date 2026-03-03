#!/bin/bash
# 即時英翻中字幕系統 - 啟動腳本
# Author: Jason Cheng (Jason Tools)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# 24-bit 真彩色
C_TITLE='\033[38;2;100;180;255m'   # 藍色
C_OK='\033[38;2;80;255;120m'       # 綠色
C_WARN='\033[38;2;255;220;80m'     # 黃色
C_ERR='\033[38;2;255;100;100m'     # 紅色
C_DIM='\033[38;2;100;100;100m'     # 暗灰
C_WHITE='\033[38;2;255;255;255m'   # 白色
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${C_TITLE}============================================================${NC}"
echo -e "${C_TITLE}${BOLD}  jt-live-whisper v1.7.4 - 即時英翻中字幕系統${NC}"
echo -e "${C_TITLE}  by Jason Cheng (Jason Tools)${NC}"
echo -e "${C_TITLE}============================================================${NC}"
echo ""

# --input 和 --summarize 模式不需要 BlackHole
SKIP_BLACKHOLE=0
for arg in "$@"; do
    if [ "$arg" = "--input" ] || [ "$arg" = "--summarize" ] || [ "$arg" = "--diarize" ]; then
        SKIP_BLACKHOLE=1
        break
    fi
done

# 檢查 BlackHole
if [ "$SKIP_BLACKHOLE" -eq 0 ] && ! system_profiler SPAudioDataType 2>/dev/null | grep -qi "blackhole"; then
    echo -e "${C_WARN}[提醒] 未偵測到 BlackHole 2ch 虛擬音訊裝置${NC}"
    echo ""
    echo -e "${C_WHITE}請先安裝 BlackHole：${NC}"
    echo -e "  ${C_DIM}brew install --cask blackhole-2ch${NC}"
    echo ""
    echo -e "${C_WHITE}安裝後需要重新啟動電腦。${NC}"
    echo ""
    echo -e "${C_WHITE}然後設定 macOS 多重輸出裝置：${NC}"
    echo -e "  ${C_DIM}1. 開啟「音訊 MIDI 設定」(Audio MIDI Setup)${NC}"
    echo -e "  ${C_DIM}2. 點左下角 + → 建立「多重輸出裝置」${NC}"
    echo -e "  ${C_DIM}3. 勾選你的喇叭/耳機 + BlackHole 2ch${NC}"
    echo -e "  ${C_DIM}4. 在音訊設定中，將喇叭設為此多重輸出裝置${NC}"
    echo ""
    read -p "是否仍然繼續？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# 檢查 venv
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${C_ERR}[錯誤] 找不到 Python 虛擬環境: $VENV_DIR${NC}"
    echo "請先執行安裝步驟。"
    exit 1
fi

# 啟用 venv 並執行
source "$VENV_DIR/bin/activate"

echo -e "${C_OK}Python 環境已啟用${NC}"
echo ""

python3 "$SCRIPT_DIR/translate_meeting.py" "$@"

# 安全網：確保終端機恢復正常（防止 Ctrl+S raw mode 殘留）
stty sane 2>/dev/null

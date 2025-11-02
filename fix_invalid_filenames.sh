#!/bin/bash
# 🛠️ 安全修复含 Windows 非法字符的文件名
# 仅改动真正非法字符，不动目录结构
# 作者: ChatGPT (for Zhenxing Liu)

# Windows 不允许的字符集合
INVALID_PATTERN='[<>:"\\|?*]'

echo "🔍 正在扫描含非法字符的文件名（忽略目录结构）..."

# 统计计数
COUNT=0

# 遍历仓库所有受管文件
git ls-files | while read -r FILE; do
    # 取出目录和文件名
    DIR=$(dirname "$FILE")
    BASENAME=$(basename "$FILE")

    # 检查文件名中是否有非法字符
    if [[ "$BASENAME" =~ $INVALID_PATTERN ]]; then
        # 替换非法字符为下划线
        NEWNAME=$(echo "$BASENAME" | sed 's/[<>:"\\|?*]/_/g')

        # 如果新旧文件名不同
        if [ "$BASENAME" != "$NEWNAME" ]; then
            COUNT=$((COUNT + 1))
            echo "⚠️  重命名: $FILE → $DIR/$NEWNAME"
            git mv "$FILE" "$DIR/$NEWNAME"
        fi
    fi
done

if [ $COUNT -eq 0 ]; then
    echo "✅ 没有发现非法文件名，一切正常。"
else
    echo
    echo "✅ 共修复 $COUNT 个非法文件名。"
    echo "📦 请执行以下命令以提交："
    echo "    git commit -m 'rename invalid filenames (safe version)'"
fi

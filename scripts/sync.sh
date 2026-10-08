#!/bin/bash
# cv-paper-study 로컬 동기화 스크립트 (macOS 기본 bash/curl/git만 사용)
#   1) GitHub에서 최신 내용(새 주차 요약·코드)을 받아오고
#   2) 완료된 논문의 원문 PDF를 각 논문 폴더에 paper.pdf로 내려받습니다.
# PDF는 .gitignore에 의해 GitHub에 올라가지 않고 이 컴퓨터에만 남습니다.
#
# 사용법:  ./scripts/sync.sh   (저장소 안 어디서 실행해도 됨)

set -e
cd "$(dirname "$0")/.."

echo "1) GitHub에서 최신 내용 받는 중..."
git pull --ff-only

echo ""
echo "2) 완료된 논문의 원문 PDF 확인 중..."

# arXiv에 없는 논문의 PDF 직접 링크
pdf_override() {
  case "$1" in
    alexnet) echo "https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf" ;;
    *) echo "" ;;
  esac
}

awk '
  function val(line) { sub(/^[^:]*: */, "", line); gsub(/"/, "", line); return line }
  function flush() { if (week != "") print week "|" slug "|" status "|" arxiv "|" ref }
  /^- week:/          { flush(); week = val($0); slug = status = arxiv = ref = ""; next }
  /^  slug:/          { slug = val($0) }
  /^  status:/        { status = val($0) }
  /^  arxiv_id:/      { arxiv = val($0) }
  /^  reference_url:/ { ref = val($0) }
  END { flush() }
' curriculum.yml | while IFS='|' read -r week slug status arxiv ref; do
  [ "$status" = "done" ] || continue
  dir=$(printf "papers/week%02d/%s" "$week" "$slug")
  [ -d "$dir" ] || continue

  if ls "$dir"/*.pdf >/dev/null 2>&1; then
    echo "  - $slug: 이미 있음"
    continue
  fi

  url=""
  if [ -n "$arxiv" ] && [ "$arxiv" != "null" ]; then
    url="https://arxiv.org/pdf/$arxiv"
  else
    case "$ref" in *.pdf) url="$ref" ;; esac
    [ -n "$url" ] || url=$(pdf_override "$slug")
  fi

  if [ -z "$url" ]; then
    echo "  - $slug: 자동으로 받을 수 있는 PDF 링크가 없어요. 직접 받아주세요 → $ref"
    continue
  fi

  echo "  - $slug: 내려받는 중... ($url)"
  if curl -fsSL -A "Mozilla/5.0 (cv-paper-study sync)" -o "$dir/paper.pdf" "$url" \
     && [ "$(head -c 4 "$dir/paper.pdf")" = "%PDF" ]; then
    echo "    완료 → $dir/paper.pdf"
  else
    rm -f "$dir/paper.pdf"
    echo "    실패했어요. 직접 받아주세요 → $url"
  fi
  sleep 3  # arXiv 서버 예의상 간격 두기
done

echo ""
echo "동기화 끝!"

#!/usr/bin/env bash
# Download MedQuAD and TREC 2017 LiveQA-Med at pinned commits (both CC BY 4.0).
set -euo pipefail
mkdir -p data && cd data
fetch() {  # repo dir commit
  if [ ! -d "$2" ]; then
    git clone -q "https://github.com/abachaa/$1.git" "$2"
    git -C "$2" checkout -q "$3"
  fi
}
fetch MedQuAD MedQuAD 577bd37b96c02d1833b2c9eed2de9f96964e96cb
fetch LiveQA_MedicalTask_TREC2017 LiveQA 725a2d914e9794afbb897ac2006e3b0c6a7834c4
# LiveQA relevance judgments against MedQuAD answers ship as a zip inside MedQuAD.
unzip -o -q MedQuAD/QA-TestSet-LiveQA-Med-Qrels-2479-Answers.zip -d qrels && rm -rf qrels/__MACOSX
echo "data ready: $(pwd)"

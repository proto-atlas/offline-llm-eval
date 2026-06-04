# Offline LLM Eval

[![検証](https://github.com/proto-atlas/offline-llm-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/proto-atlas/offline-llm-eval/actions/workflows/ci.yml)

Offline LLM Eval は、RAG/LLM 機能の回答を、外部 API に依存せずローカルで確認するための Python 製ツールです。

引用整合性、回答しないべきケース、応答 JSON 形式、応答時間、外部サービス相当のエラーを、fixture（再現用データ）と明示的な判定条件で確認します。評価ケース、実行結果、差分、検証証跡を後から読み返せる形に残します。

## 主な流れ

- 評価ケースと実行結果を SQLite に保存する。
- 引用整合性、no-answer、応答 JSON 形式、応答時間、外部サービス相当のエラーを判定する。
- localhost 向け API で run、case、差分、検証証跡を確認する。
- 既存 run を CLI で判定し、CI で扱える終了コードを返す。
- secret 値に似た文字列を伏せた検証証跡 Markdown を出力する。

## 用語

- fixture: 評価用に固定した入力、期待値、実行結果。外部 API の現在状態に左右されず、同じ条件を再確認するために使います。
- run: 複数の評価ケースを実行した単位。
- no-answer: 根拠が足りない場合など、回答しないことを期待するケース。
- 検証証跡 Markdown: 判定結果、失敗理由、差分、確認対象外の項目を読み返せる形にした Markdown 出力。

## 使い方

Python 3.12 以上で確認しています。

```powershell
python -m pip install -e ".[dev]"
python -m alembic upgrade head
python -m offline_llm_eval.main
```

別の PowerShell で API と CLI を確認します。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/api/runs
python -m offline_llm_eval.cli.check --help
```

品質確認は次のコマンドで再実行できます。

```powershell
python -m ruff check alembic src tests
python -m ruff format --check alembic src tests
python -m mypy src tests
python -m pytest
```

## ドキュメント

- [公開仕様](./docs/spec.md)
- [検証記録](./docs/evidence/verification-2026-06-02.md)

## 評価で確認できる例

- 引用 ID が期待した根拠に含まれているか。
- 回答しないべきケースで、no-answer として扱えているか。
- 応答 JSON が期待する形式を満たしているか。
- 応答時間が判定条件内に収まっているか。
- 外部サービス相当のエラーを、評価結果として記録できているか。

## 検証証跡 Markdown

検証証跡 Markdown には、判定結果と差分に加えて、この検証からは判断しない項目を `not_claimed` として出力します。既定では次の項目を対象にしています。

- 運用環境でそのまま使えるか
- あらゆる LLM 出力品質を評価できるか
- 外部 LLM サービスの費用を制御できるか
- 長時間運用に耐えるか

出力例は [docs/evidence/verification-2026-06-02.md](docs/evidence/verification-2026-06-02.md) にあります。

## 技術スタック

- Python 3.12+
- FastAPI
- SQLAlchemy 2.0 async
- Pydantic
- Alembic
- SQLite
- pytest
- ruff
- mypy

v0.1.0 の自動検証は SQLite 接続を対象にしています。PostgreSQL は接続 URL を受け付ける構成までを扱います。

## 現在入れていないもの

- 本番環境向けの認証/認可
- あらゆる LLM 出力品質の評価
- 外部 LLM サービスの品質や費用の評価
- 長時間稼働の評価
- 監視、障害対応、SLA を含む運用一式
- データセット取り込みから評価実行と結果保存までを一括で起動する CLI/API

## ライセンス

MIT License。詳細は [LICENSE](LICENSE) を参照してください。

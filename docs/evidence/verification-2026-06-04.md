# 検証結果 2026-06-04

## 対象

- ローカル検証実行コミット: `926f6d0bbda489201ff9d1ce590d809dacb977d9`
- ローカル検証環境: Windows / Python 3.13.13 / Node.js 24.15.0
- `src/`
- `tests/`
- `alembic/`
- `.github/workflows/ci.yml`
- `Dockerfile`
- `README.md`
- `docs/`

## 方法と結果

| 方法 | 対象 | 結果 |
|---|---|---|
| `python -m ruff check alembic src tests` | Python実装、migration、テスト | `All checks passed!` |
| `python -m ruff format --check alembic src tests` | Python実装、migration、テスト | `125 files already formatted` |
| `python -m mypy src tests alembic` | Python実装、migration、テスト | `Success: no issues found in 125 source files` |
| `python -m pytest` | 全テスト | `400 passed`、4 warnings |
| `python -m pip install -e . --no-deps` | package metadata と editable install | 成功 |
| `docker build .` | Docker image build | Docker daemon未起動のため未確認 |

## warning

`python -m pytest` の warning 件数は、依存を再解決した環境では変わる場合があります。
今回の環境で確認した内訳は、Python 3.12 で追加された datetime adapter の非推奨警告が、Python 3.13.13 実行時にも `aiosqlite/core.py:63` 経由で表示されたものです。

## 検証境界

- 既定DB接続は SQLite で検証しています。
- PostgreSQL は SQLAlchemy async engine に接続URLを渡せる構成ですが、この検証では PostgreSQL サーバーへの接続テストは実施していません。
- 外部LLMサービスの品質、費用、長時間運用はこの検証結果の対象外です。
- Docker build は、今回のローカル環境ではDocker daemon未起動のため未確認です。

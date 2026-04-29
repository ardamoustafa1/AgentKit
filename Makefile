.PHONY: format lint test coverage build publish

# Tüm kodu ruff ile düzenler
format:
	poetry run ruff format agentkit tests

# Kod kalitesini denetler ve hataları düzeltir
lint:
	poetry run ruff check --fix agentkit tests
	poetry run mypy --strict agentkit tests

# Birim testleri çalıştırır
test:
	poetry run pytest -v

# Test kapsam (coverage) raporu çıkarır
coverage:
	poetry run pytest --cov=agentkit --cov-report=term-missing

# PyPI için tekerlek (wheel) ve kaynak (sdist) dosyalarını oluşturur
build:
	poetry build

# Projeyi PyPI'a gönderir
publish: build
	poetry publish

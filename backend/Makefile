.PHONY: test test-seasons help

help:
	@echo "Available commands:"
	@echo "  make test          - Run all tests (pytest)"
	@echo "  make test-seasons  - Run season matching tests"
	@echo "  make test-fast     - Run tests with minimal output"

test:
	pytest scripts/testing/test_season_matching.py -v

test-seasons:
	python scripts/testing/test_season_matching.py

test-fast:
	pytest scripts/testing/test_season_matching.py -q



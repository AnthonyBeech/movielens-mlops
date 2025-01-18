DIR := .

lint:
	ruff check ${DIR} --fix

format:
	ruff format ${DIR}

check:
	lint format
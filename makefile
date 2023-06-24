install: requirements.txt
	python -m venv venv
	source ./venv/bin/activate
	pip install -r requirements.txt

readme: README.md
	cat ./README.md

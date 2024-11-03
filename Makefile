image:
	docker build -t detection-engine:local .

run_app_pipenv:
	pipenv run python app.py

run_test_pipenv:
	pipenv run python test_model.py

run_app:
	python app.py

run_test:
	python test_model.py

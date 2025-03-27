obtain_data:
	@echo Obtaining raw data...
	mkdir -p src
	curl -L -o src/data.zip https://www.kaggle.com/api/v1/datasets/download/cdminix/us-drought-meteorological-data
	curl -L -o src/counties.zip https://gist.github.com/sdwfrost/d1c73f91dd9d175998ed166eb216994a/archive/e89c35f308cee7e2e5a784e1d3afc5d449e9e4bb.zip
	unzip src/data.zip -d src
	unzip -j src/counties.zip -d src
	rm src/data.zip src/counties.zip
	@echo Raw data obtained and saved in src/ directory!

reset_data:
	@echo Removing src/ directory...
	rm -rf src
	mkdir -p src
	@echo src/ directory reset!
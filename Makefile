.PHONY: all
all:
	mkdir -p models
	cd models; \
		wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip; \
		unzip inception5h.zip
	mkdir -p images
	@echo '-----------------------------------------------------------------'
	@echo ''
	@echo ' DONE-zo!'
	@echo ''
	@echo '-----------------------------------------------------------------'
	@echo ' Now add images you want to DeepDream-ize to the images/ folder!'
	@echo '-----------------------------------------------------------------'

.PHONY: clean
clean:
		rm -rf models
		rm -rf images

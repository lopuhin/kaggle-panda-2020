IMAGE_NAME=kostia_panda
CONTAINER_NAME=kostia_panda

# HELP
.PHONY: help

help: ## This help.
	@awk 'BEGIN (FS = ":.*?## ") /^[a-zA-Z_-]+:.*?## / (printf "\033[36m%-30s\033[0m %s\n", $$1, $$2)' $(MAKEFILE_LIST)

build:  ## Build the container
	nvidia-docker build -t $(IMAGE_NAME) .

run-dl: ## Run container at dl
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd)/panda:/panda \
		-v /mnt/ssd1/kaggle/dataset/panda:/data \
		-v /mnt/hdd1/learning_dumps/kostia_panda:/dumps/_runs \
		-v /home/n01z3/.cache/torch/checkpoints:/root/.cache/torch/checkpoints \
		$(IMAGE_NAME)

run-rtx: ## Run container at dl
	nvidia-docker run \
		-itd \
		--ipc=host \
		--name=$(CONTAINER_NAME) \
		-v $(shell pwd)/panda:/panda \
		-v /data1/n01z3/dataset/spacenet6:/dataset \
		-v /data2/dumps/kostia_panda:/dumps/_runs \
		-v /data2/torch_checkpoints:/root/.cache/torch/checkpoints \
		$(IMAGE_NAME)


exec: ## Run a bash in a running container
	nvidia-docker exec -it $(CONTAINER_NAME) bash

stop: ## Stop and remove a running container
	docker stop $(CONTAINER_NAME); docker rm $(CONTAINER_NAME)
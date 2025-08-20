# Makefile for State - Focused on essentials and build speed
# Optimized for H100 systems with maximum core utilization

.PHONY: help build run test clean
.DEFAULT_GOAL := help

# Configuration
DOCKER_IMAGE := state:latest
CORES := $(shell nproc)  # Use all available cores on H100 system

# Colors for output
GREEN := \033[0;32m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show available commands
	@echo "$(BLUE)State - Essential Commands$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(GREEN)%-12s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

build: ## Build Docker image with maximum speed optimization
	@echo "$(BLUE)Building with BuildKit ($(CORES) cores)...$(NC)"
	DOCKER_BUILDKIT=1 sudo docker build \
		--progress=plain \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg MAX_JOBS=$(CORES) \
		-t $(DOCKER_IMAGE) .
	@echo "$(GREEN)Build completed$(NC)"

run: ## Run State container with GPU support
	@echo "$(BLUE)Starting State container...$(NC)"
	sudo docker run --rm -it --gpus all \
		-v $(PWD)/examples:/app/examples:ro \
		-v $(PWD)/data:/app/data:rw \
		$(DOCKER_IMAGE)

test: ## Quick functionality test
	@echo "$(BLUE)Testing State...$(NC)"
	sudo docker run --rm --gpus all $(DOCKER_IMAGE) --help
	@echo "$(GREEN)Test completed$(NC)"

clean: ## Clean up Docker resources
	@echo "$(BLUE)Cleaning up...$(NC)"
	sudo docker system prune -f
	@echo "$(GREEN)Cleanup completed$(NC)"
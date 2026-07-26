# Configuration
REPO_DIR = .
BACKEND ?=

# Detect if running inside WSL
IS_WSL := $(shell grep -qi microsoft /proc/version && echo 1 || echo 0)

# Auto-detect backend if not explicitly provided
ifeq ($(BACKEND),)
    ifeq ($(IS_WSL),1)
        # In WSL, check if NVIDIA CUDA driver bridge is present
        ifneq ($(wildcard /usr/lib/wsl/lib/libcuda.so*),)
            BACKEND = cuda
        else
            BACKEND = cpu
        endif
    else
        # Native Linux detection fallback
        ifneq ($(wildcard /usr/local/cuda*),)
            BACKEND = cuda
        else
            BACKEND = cpu
        endif
    endif
endif

# Determine CMake flags based on backend selection
ifeq ($(BACKEND),cuda)
    CMAKE_FLAGS = -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
else ifeq ($(BACKEND),vulkan)
    CMAKE_FLAGS = -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
else
    CMAKE_FLAGS = -DCMAKE_BUILD_TYPE=Release
endif

.PHONY: check-config setup clone build install run clean help

check-config: ## Check current configuration before building
	@echo "========================================"
	@echo "          BUILD CONFIGURATION           "
	@echo "========================================"
	@echo "  Environment : $(if $(filter 1,$(IS_WSL)),WSL (Windows Subsystem for Linux),Native Linux)"
	@echo "  Backend     : $(BACKEND)"
	@echo "  CMake Flags : $(CMAKE_FLAGS)"
	@echo "  Repository  : $(REPO_DIR)"
	@echo "========================================"

setup: ## Install system dependencies
	@echo "==> Installing system dependencies..."
	sudo apt update && sudo apt install -y build-essential cmake git npm

build: ## Build the C++ engine
	@echo "==> Building C++ engine ($(BACKEND) backend)..."
	cd $(REPO_DIR)/engine && mkdir -p build && cd build && \
	cmake .. $(CMAKE_FLAGS) && \
	cmake --build . -j $$(nproc)

install: ## Install Node.js dependencies for server and UI
	@echo "==> Installing Node.js dependencies for server..."
	cd $(REPO_DIR)/server && npm install
	@echo "==> Installing Node.js dependencies for UI..."
	cd $(REPO_DIR)/ui && npm install

run: ## Launch the application
	@echo "==> Launching application..."
	cd $(REPO_DIR) && ./launch.sh

clean: ## Clean build artifacts
	@echo "==> Cleaning build files..."
	rm -rf $(REPO_DIR)/engine/build
	rm -rf $(REPO_DIR)/server/node_modules
	rm -rf $(REPO_DIR)/ui/node_modules

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Convenience wrapper for Linux / WSL development.
# Windows-native development uses dev.bat / LAUNCH.bat / dev-rebuild.bat instead.

# Configuration
REPO_DIR = .
BACKEND ?=

# Detect if running inside WSL
IS_WSL := $(shell grep -qi microsoft /proc/version 2>/dev/null && echo 1 || echo 0)

# Auto-detect backend if not explicitly provided: CUDA if a driver is visible,
# else Vulkan if the SDK's shader compiler is installed, else CPU.
ifeq ($(BACKEND),)
    ifeq ($(IS_WSL),1)
        # In WSL, check if NVIDIA CUDA driver bridge is present
        ifneq ($(wildcard /usr/lib/wsl/lib/libcuda.so*),)
            BACKEND = cuda
        endif
    else
        # Native Linux detection
        ifneq ($(wildcard /usr/local/cuda*),)
            BACKEND = cuda
        endif
    endif
endif
ifeq ($(BACKEND),)
    ifneq ($(shell command -v glslc 2>/dev/null),)
        BACKEND = vulkan
    else
        BACKEND = cpu
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

NODE_MAJOR := $(shell node -v 2>/dev/null | sed 's/^v\([0-9]*\).*/\1/')

.PHONY: check-config check-node setup submodules build install run clean help

check-config: ## Check current configuration before building
	@echo "========================================"
	@echo "          BUILD CONFIGURATION           "
	@echo "========================================"
	@echo "  Environment : $(if $(filter 1,$(IS_WSL)),WSL (Windows Subsystem for Linux),Native Linux)"
	@echo "  Backend     : $(BACKEND)"
	@echo "  CMake Flags : $(CMAKE_FLAGS)"
	@echo "  Node.js     : $(if $(NODE_MAJOR),v$(NODE_MAJOR) (need 20-24),not found)"
	@echo "  Repository  : $(REPO_DIR)"
	@echo "========================================"

check-node: ## Verify Node.js 20-24 LTS is installed
	@if [ -z "$(NODE_MAJOR)" ]; then \
		echo "ERROR: Node.js not found. Install Node 20-24 LTS, e.g. with nvm:"; \
		echo "       https://github.com/nvm-sh/nvm  then: nvm install 24 && nvm use 24"; \
		exit 1; \
	fi
	@if [ "$(NODE_MAJOR)" -lt 20 ] || [ "$(NODE_MAJOR)" -gt 24 ]; then \
		echo "ERROR: Node v$(NODE_MAJOR) detected, but HOT-Step requires Node 20-24 LTS"; \
		echo "       With nvm: nvm install 24 && nvm use 24"; \
		exit 1; \
	fi

setup: ## Install system dependencies (except Node - use nvm for that)
	@echo "==> Installing system dependencies..."
	sudo apt update && sudo apt install -y build-essential cmake git
	@echo ""
	@echo "==> NOTE: Node.js 20-24 LTS is also required (apt's version is often wrong)."
	@echo "    Recommended: install via nvm - https://github.com/nvm-sh/nvm"
	@echo "    Then run 'make check-node' to verify."

submodules: ## Initialise git submodules (required before first build)
	@echo "==> Initialising git submodules..."
	git submodule update --init --recursive

build: submodules ## Build the C++ engine
ifeq ($(BACKEND),vulkan)
	@command -v glslc >/dev/null 2>&1 || { \
		echo "ERROR: Vulkan backend selected but glslc not found."; \
		echo "       Install the Vulkan SDK: https://vulkan.lunarg.com/sdk/home"; \
		exit 1; }
endif
	@echo "==> Building C++ engine ($(BACKEND) backend)..."
	cd $(REPO_DIR)/engine && mkdir -p build && cd build && \
	cmake .. $(CMAKE_FLAGS) && \
	cmake --build . -j $$(nproc)

install: check-node ## Install Node.js dependencies for server and UI
	@echo "==> Installing Node.js dependencies for server..."
	cd $(REPO_DIR)/server && npm install
	@echo "==> Installing Node.js dependencies for UI..."
	cd $(REPO_DIR)/ui && npm install

run: ## Launch the application
	@echo "==> Launching application..."
	cd $(REPO_DIR) && ./launch.sh

clean: ## Clean build artifacts (requires CONFIRM=1 - CUDA rebuild takes 20+ min)
ifneq ($(CONFIRM),1)
	@echo "This deletes engine/build and all node_modules."
	@echo "A CUDA engine rebuild from scratch takes 20+ minutes."
	@echo "Run 'make clean CONFIRM=1' if you really want this."
	@exit 1
else
	@echo "==> Cleaning build files..."
	rm -rf $(REPO_DIR)/engine/build
	rm -rf $(REPO_DIR)/server/node_modules
	rm -rf $(REPO_DIR)/ui/node_modules
endif

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

#!/usr/bin/env python3
"""
Build script to compile C++ physics engine to WebAssembly
"""

import os
import subprocess
import sys
from pathlib import Path

def build_wasm():
    """Compile C++ to WebAssembly using Emscripten"""
    
    # Get project directories
    project_root = Path(__file__).parent.parent
    core_engine_dir = project_root / "core_engine"
    frontend_dir = project_root / "frontend"
    
    # Source files
    source_files = [
        core_engine_dir / "physics_engine.cpp",
        core_engine_dir / "collision_detection.cpp",
        core_engine_dir / "game_objects.cpp"
    ]
    
    # Output file
    output_file = frontend_dir / "libs" / "physics_engine.wasm"
    js_output_file = frontend_dir / "libs" / "physics_engine.js"
    
    # Ensure output directory exists
    (frontend_dir / "libs").mkdir(exist_ok=True)
    
    # Emscripten compiler command
    emcc_cmd = [
        "emcc",
        *[str(f) for f in source_files if f.exists()],
        "-o", str(js_output_file),
        "-O3",  # Optimization level 3
        "-s", "WASM=1",
        "-s", "MODULARIZE=1",
        "-s", "EXPORT_ES6=1",
        "-s", "USE_ES6_IMPORT_META=0",
        "-s", "EXPORTED_RUNTIME_METHODS=['ccall','cwrap']",
        "-s", "EXPORTED_FUNCTIONS=['_malloc','_free']",
        "-s", "ALLOW_MEMORY_GROWTH=1",
        "-s", "MAXIMUM_MEMORY=256MB",
        "-s", "ENVIRONMENT='web'",
        "--bind",  # For Embind
        "-std=c++17",
        "-I" + str(core_engine_dir)
    ]
    
    print("Building WebAssembly module...")
    print("Command:", " ".join(emcc_cmd))
    
    try:
        result = subprocess.run(emcc_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Build successful!")
            print(f"WASM file: {output_file}")
            print(f"JS loader: {js_output_file}")
            
            # Create a simple loader for the WASM module
            create_loader_js(frontend_dir / "libs")
            
        else:
            print("❌ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return 1
            
    except FileNotFoundError:
        print("❌ Emscripten (emcc) not found!")
        print("Please install Emscripten from: https://emscripten.org/docs/getting_started/downloads.html")
        return 1
    
    return 0

def create_loader_js(libs_dir):
    """Create a JavaScript loader for the WASM module"""
    
    loader_content = """
// Physics Engine WASM Loader
class PhysicsEngineLoader {
    constructor() {
        this.engine = null;
        this.isLoading = false;
        this.loadCallbacks = [];
    }
    
    async load() {
        if (this.engine) {
            return this.engine;
        }
        
        if (this.isLoading) {
            return new Promise(resolve => {
                this.loadCallbacks.push(resolve);
            });
        }
        
        this.isLoading = true;
        
        try {
            // Import the WASM module
            const module = await import('./physics_engine.js');
            
            // Initialize the physics engine
            this.engine = await new module.PhysicsEngine();
            
            console.log('✅ Physics Engine (WASM) loaded successfully');
            
            // Resolve all waiting callbacks
            this.loadCallbacks.forEach(callback => callback(this.engine));
            this.loadCallbacks = [];
            
            return this.engine;
            
        } catch (error) {
            console.error('❌ Failed to load Physics Engine:', error);
            throw error;
        } finally {
            this.isLoading = false;
        }
    }
    
    getEngine() {
        return this.engine;
    }
    
    isLoaded() {
        return this.engine !== null;
    }
}

// Create global instance
window.physicsEngineLoader = new PhysicsEngineLoader();

// Auto-load in development
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    window.addEventListener('DOMContentLoaded', () => {
        window.physicsEngineLoader.load().catch(console.error);
    });
}
"""
    
    loader_file = libs_dir / "physics_loader.js"
    loader_file.write_text(loader_content)
    print(f"✅ Created loader: {loader_file}")

def build_native():
    """Build native C++ library for testing"""
    
    core_engine_dir = Path(__file__).parent
    build_dir = core_engine_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Simple Makefile content
    makefile_content = """
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -fPIC
TARGET = libbubblephysics.so
SRC = physics_engine.cpp collision_detection.cpp game_objects.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -shared -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

test: all
	./test_runner

.PHONY: all clean test
"""
    
    makefile_path = core_engine_dir / "Makefile"
    makefile_path.write_text(makefile_content)
    
    print("✅ Created Makefile for native build")
    print("Run 'make' in core_engine directory to build native library")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "native":
        build_native()
    else:
        sys.exit(build_wasm())

# whisper-rs - Modifications for Shared GGML

## Overview

This document describes how to modify `whisper-rs` to use the shared GGML library from `ggml-sys` instead of building its own embedded GGML. This allows `whisper-rs` and `llama-cpp-2` to share the same GGML library, avoiding duplicate symbol conflicts.

## Prerequisites

- Fork of `whisper-rs` repository (or `whisper-rs-sys` if it's a separate crate)
- `ggml-sys` crate set up and available (see `GGML_SYS_SETUP.md`)
- Understanding of Rust build scripts and CMake

## Repository Structure

```
whisper-rs/
├── Cargo.toml
├── sys/
│   ├── Cargo.toml
│   ├── build.rs
│   ├── wrapper.h
│   └── whisper.cpp/  # whisper.cpp source code
└── src/
    └── lib.rs
```

## Step 1: Add `use-shared-ggml` Feature to `whisper-rs/Cargo.toml`

Add the feature to the main crate's `Cargo.toml`:

```toml
[features]
# ... existing features ...
# Use shared GGML backend to avoid duplicate symbol conflicts
use-shared-ggml = ["whisper-rs-sys/use-shared-ggml"]
```

## Step 2: Modify `whisper-rs/sys/Cargo.toml`

### 2.1 Add `ggml-sys` as Optional Dependency

Add `ggml-sys` to the `[dependencies]` section:

```toml
[dependencies]
# ... existing dependencies ...
ggml-sys = { git = "https://github.com/your-username/ggml-sys.git", branch = "main", optional = true }
# OR if using path dependency:
# ggml-sys = { path = "../../ggml-sys", optional = true }
```

### 2.2 Add `use-shared-ggml` Feature

Add the feature to the `[features]` section:

```toml
[features]
# ... existing features ...
use-shared-ggml = ["ggml-sys"]
```

## Step 3: Modify `whisper-rs/sys/build.rs`

### 3.1 Add Code to Get GGML Paths from `ggml-sys`

At the top of the `main()` function, after getting the manifest directory:

```rust
// Get ggml-sys paths if available (when use-shared-ggml is enabled)
let ggml_lib_dir = env::var("DEP_GGML_SYS_ROOT")
    .map(|root| PathBuf::from(root).join("lib"))
    .ok();
let ggml_include_dir = env::var("DEP_GGML_SYS_INCLUDE")
    .map(PathBuf::from)
    .ok();
let ggml_prefix = ggml_lib_dir.as_ref()
    .and_then(|lib_dir| lib_dir.parent().map(|p| p.to_path_buf()));
```

### 3.2 Modify the Build Logic to Handle `use-shared-ggml`

Find the section where GGML is built (usually after bindgen generation) and replace it with:

```rust
// If use-shared-ggml feature is enabled, skip building ggml and link to shared library
if cfg!(feature = "use-shared-ggml") {
    // Link to shared ggml libraries from ggml-sys
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");
    
    if cfg!(target_os = "macos") || cfg!(feature = "openblas") {
        println!("cargo:rustc-link-lib=dylib=ggml-blas");
    }
    
    if cfg!(feature = "vulkan") {
        println!("cargo:rustc-link-lib=dylib=ggml-vulkan");
    }
    
    if cfg!(feature = "hipblas") {
        println!("cargo:rustc-link-lib=dylib=ggml-hip");
    }
    
    if cfg!(feature = "metal") {
        println!("cargo:rustc-link-lib=dylib=ggml-metal");
    }
    
    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=dylib=ggml-cuda");
    }
    
    if cfg!(feature = "openblas") {
        println!("cargo:rustc-link-lib=dylib=ggml-blas");
    }
    
    if cfg!(feature = "intel-sycl") {
        println!("cargo:rustc-link-lib=dylib=ggml-sycl");
    }
    
    // Build only whisper (not ggml)
    let mut config = Config::new(&whisper_root);
    
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_ALL_WARNINGS", "OFF")
        .define("WHISPER_ALL_WARNINGS_3RD_PARTY", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_USE_SYSTEM_GGML", "ON")  // Use system ggml (shared library)
        .very_verbose(true)
        .pic(true);
    
    // CRITICAL: Tell CMake where to find ggml
    if let Some(ref prefix) = ggml_prefix {
        // Set CMAKE_PREFIX_PATH to where ggml-sys installed ggml
        config.define("CMAKE_PREFIX_PATH", prefix.to_str().unwrap());
        // Set ggml_DIR to the cmake config directory
        let ggml_cmake_dir = prefix.join("lib").join("cmake").join("ggml");
        if ggml_cmake_dir.exists() {
            config.define("ggml_DIR", ggml_cmake_dir.to_str().unwrap());
        }
    }
    
    // Alternative: If CMake config files aren't in the expected location,
    // you may need to set additional paths
    if let Some(ref lib_dir) = ggml_lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    if let Some(ref include_dir) = ggml_include_dir {
        // Add include directory for CMake
        config.define("GGML_INCLUDE_DIR", include_dir.to_str().unwrap());
    }
    
    if cfg!(target_os = "windows") {
        config.cxxflag("/utf-8");
        println!("cargo:rustc-link-lib=advapi32");
    }
    
    if cfg!(feature = "coreml") {
        config.define("WHISPER_COREML", "ON");
        config.define("WHISPER_COREML_ALLOW_FALLBACK", "1");
    }
    
    // Configure whisper-specific features (not ggml)
    if cfg!(debug_assertions) || cfg!(feature = "force-debug") {
        config.define("CMAKE_BUILD_TYPE", "RelWithDebInfo");
        config.cxxflag("-DWHISPER_DEBUG");
    } else {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }
    
    // Allow passing any WHISPER or CMAKE compile flags
    for (key, value) in env::vars() {
        let is_whisper_flag =
            key.starts_with("WHISPER_") && key != "WHISPER_DONT_GENERATE_BINDINGS";
        let is_cmake_flag = key.starts_with("CMAKE_");
        if is_whisper_flag || is_cmake_flag {
            config.define(&key, &value);
        }
    }
    
    let destination = config.build();
    add_link_search_path(&out.join("build")).unwrap();
    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-lib=static=whisper");
    
    // ... rest of whisper-specific build logic ...
    
} else {
    // Original code: build whisper with embedded ggml
    // ... existing build logic ...
}
```

### 3.3 Update Bindgen Includes

When `use-shared-ggml` is enabled, update bindgen to use `ggml-sys` headers:

```rust
let builder = bindgen::Builder::default().header("wrapper.h");

#[cfg(feature = "metal")]
let builder = builder.header(whisper_src.join("ggml/include/ggml-metal.h").to_str().unwrap());

#[cfg(feature = "vulkan")]
let builder = builder
    .header(whisper_src.join("ggml/include/ggml-vulkan.h").to_str().unwrap())
    .clang_arg("-DGGML_USE_VULKAN=1");

// When use-shared-ggml is enabled, use ggml-sys headers
if cfg!(feature = "use-shared-ggml") {
    if let Some(ref include_dir) = ggml_include_dir {
        builder = builder.clang_arg(format!("-I{}", include_dir.display()));
    }
} else {
    // Use embedded ggml headers
    builder = builder
        .clang_arg(format!("-I{}", whisper_src.join("ggml/include").display()));
}

let bindings = builder
    .clang_arg(format!("-I{}", whisper_src.display()))
    .clang_arg(format!("-I{}", whisper_src.join("include").display()))
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .generate();
```

## Step 4: Update `wrapper.h` (if needed)

Ensure `wrapper.h` includes the correct path:

```c
#include "whisper.h"  // Not <include/whisper.h>
```

## Step 5: Test the Build

1. Build with the feature enabled:
   ```bash
   cargo build --features use-shared-ggml
   ```

2. Verify that:
   - No GGML source is compiled (only whisper.cpp)
   - Links to `dylib=ggml` libraries
   - CMake finds the shared GGML library

## Step 6: Update Documentation

Update your crate's README to document the new feature:

```markdown
## Features

- `use-shared-ggml`: Use a shared GGML library from `ggml-sys` instead of building embedded GGML.
  This is useful when using both `whisper-rs` and `llama-cpp-2` together to avoid duplicate symbol conflicts.

  ```toml
  [dependencies]
  whisper-rs = { git = "...", features = ["use-shared-ggml"] }
  ggml-sys = { git = "..." }
  ```
```

## Critical Points

1. **`WHISPER_USE_SYSTEM_GGML=ON`**: Must be set when using shared GGML
2. **`CMAKE_PREFIX_PATH`**: Must point to where `ggml-sys` installed GGML
3. **`ggml_DIR`**: Should point to the CMake config directory if available
4. **Link to `dylib=ggml`**: Not static libraries
5. **Export paths**: Use `DEP_GGML_SYS_ROOT` and `DEP_GGML_SYS_INCLUDE` from `ggml-sys`

## Verification Checklist

- [ ] `use-shared-ggml` feature is defined in both `whisper-rs` and `whisper-rs-sys` Cargo.toml
- [ ] `ggml-sys` is added as optional dependency
- [ ] Build script checks for `use-shared-ggml` feature
- [ ] `WHISPER_USE_SYSTEM_GGML=ON` is set when using shared GGML
- [ ] `CMAKE_PREFIX_PATH` or `ggml_DIR` points to where `ggml-sys` installed GGML
- [ ] Links to `dylib=ggml` libraries (not static)
- [ ] Build succeeds with `--features use-shared-ggml`

## Troubleshooting

### CMake can't find ggml package

**Error**: `Could not find a package configuration file provided by "ggml"`

**Solution**: 
- Ensure `CMAKE_PREFIX_PATH` is set correctly
- Check that `ggml-sys` exports `DEP_GGML_SYS_ROOT`
- Verify that CMake config files exist in `$DEP_GGML_SYS_ROOT/lib/cmake/ggml/`

### Duplicate symbol errors

**Error**: Multiple definition of `ggml_*` symbols

**Solution**:
- Ensure you're linking to `dylib=ggml`, not `static=ggml`
- Verify that `ggml-sys` has `links = "ggml"` in its Cargo.toml
- Check that only one crate is building GGML

### Build fails with missing headers

**Error**: `fatal error: 'ggml.h' file not found`

**Solution**:
- Ensure `ggml_include_dir` is set correctly
- Check that `DEP_GGML_SYS_INCLUDE` is exported by `ggml-sys`
- Verify bindgen includes the correct paths


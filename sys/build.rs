#![allow(clippy::uninlined_format_args)]

extern crate bindgen;

use cmake::Config;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap();
    // Link C++ standard library
    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={}", cpp_stdlib);
    }
    // Link macOS Accelerate framework for matrix calculations
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        #[cfg(feature = "coreml")]
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=CoreML");
        }
        #[cfg(feature = "metal")]
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    }

    #[cfg(feature = "coreml")]
    println!("cargo:rustc-link-lib=static=whisper.coreml");

    #[cfg(feature = "openblas")]
    {
        if let Ok(openblas_path) = env::var("OPENBLAS_PATH") {
            println!(
                "cargo::rustc-link-search={}",
                PathBuf::from(openblas_path).join("lib").display()
            );
        }
        if cfg!(windows) {
            println!("cargo:rustc-link-lib=libopenblas");
        } else {
            println!("cargo:rustc-link-lib=openblas");
        }
    }
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");
        cfg_if::cfg_if! {
            if #[cfg(target_os = "windows")] {
                let cuda_path = PathBuf::from(env::var("CUDA_PATH").unwrap()).join("lib/x64");
                println!("cargo:rustc-link-search={}", cuda_path.display());
            } else {
                println!("cargo:rustc-link-lib=culibos");
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
                println!("cargo:rustc-link-search=/opt/cuda/lib64");
                println!("cargo:rustc-link-search=/opt/cuda/lib64/stubs");
            }
        }
    }
    #[cfg(feature = "hipblas")]
    {
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=amdhip64");

        cfg_if::cfg_if! {
            if #[cfg(target_os = "windows")] {
                panic!("Due to a problem with the last revision of the ROCm 5.7 library, it is not possible to compile the library for the windows environment.\nSee https://github.com/ggerganov/whisper.cpp/issues/2202 for more details.")
            } else {
                println!("cargo:rerun-if-env-changed=HIP_PATH");

                let hip_path = match env::var("HIP_PATH") {
                    Ok(path) =>PathBuf::from(path),
                    Err(_) => PathBuf::from("/opt/rocm"),
                };
                let hip_lib_path = hip_path.join("lib");

                println!("cargo:rustc-link-search={}",hip_lib_path.display());
            }
        }
    }

    #[cfg(feature = "openmp")]
    {
        if target.contains("gnu") {
            println!("cargo:rustc-link-lib=gomp");
        } else if target.contains("apple") {
            println!("cargo:rustc-link-lib=omp");
            println!("cargo:rustc-link-search=/opt/homebrew/opt/libomp/lib");
        }
    }

    println!("cargo:rerun-if-changed=wrapper.h");

    // Get ggml-rs paths if available (when use-shared-ggml is enabled)
    // Use new whisper-specific environment variables from ggml-rs
    // ggml-rs now exports: DEP_GGML_RS_GGML_WHISPER_LIB_DIR, DEP_GGML_RS_GGML_WHISPER_BIN_DIR, DEP_GGML_RS_GGML_WHISPER_BASENAME
    let ggml_lib_dir = env::var("DEP_GGML_RS_GGML_WHISPER_LIB_DIR")
        .map(|v| {
            println!("cargo:warning=[GGML] Found DEP_GGML_RS_GGML_WHISPER_LIB_DIR: {}", v);
            PathBuf::from(v)
        })
        .or_else(|e| {
            println!("cargo:warning=[GGML] DEP_GGML_RS_GGML_WHISPER_LIB_DIR not set: {:?}", e);
            Err(e)
        })
        .ok();
    let ggml_bin_dir = env::var("DEP_GGML_RS_GGML_WHISPER_BIN_DIR")
        .map(|v| {
            println!("cargo:warning=[GGML] Found DEP_GGML_RS_GGML_WHISPER_BIN_DIR: {}", v);
            PathBuf::from(v)
        })
        .or_else(|e| {
            println!("cargo:warning=[GGML] DEP_GGML_RS_GGML_WHISPER_BIN_DIR not set: {:?}", e);
            Err(e)
        })
        .ok();
    let ggml_lib_basename = env::var("DEP_GGML_RS_GGML_WHISPER_BASENAME")
        .map(|v| {
            println!("cargo:warning=[GGML] Found DEP_GGML_RS_GGML_WHISPER_BASENAME: {}", v);
            v
        })
        .unwrap_or_else(|e| {
            println!("cargo:warning=[GGML] DEP_GGML_RS_GGML_WHISPER_BASENAME not set: {:?}, using fallback", e);
            "ggml_whisper".to_string()
        });
    let ggml_include_dir = if let Ok(include) = env::var("DEP_GGML_RS_GGML_WHISPER_INCLUDE") {
        Some(PathBuf::from(include))
    } else if let Ok(include) = env::var("DEP_GGML_RS_INCLUDE") {
        Some(PathBuf::from(include))
    } else if let Ok(include) = env::var("DEP_GGML_INCLUDE") {
        Some(PathBuf::from(include))
    } else {
        ggml_lib_dir.as_ref().and_then(|lib_dir| {
            lib_dir.parent().map(|p| PathBuf::from(format!("{}/include", p.display())))
        })
    };
    // ggml_prefix will be recalculated later when needed for CMake

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let whisper_root = out.join("whisper.cpp");

    // Get manifest directory (where build.rs is located)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let whisper_cpp_source = manifest_dir.join("whisper.cpp");

    // Helper function to check if directory has contents
    let dir_has_contents = |path: &PathBuf| -> bool {
        path.read_dir()
            .map(|entries| entries.count() > 0)
            .unwrap_or(false)
    };
    
    // If whisper.cpp doesn't exist locally, download it
    let whisper_exists = whisper_cpp_source.exists() && dir_has_contents(&whisper_cpp_source);
    
    if !whisper_exists {
        println!("cargo:warning=whisper.cpp not found, downloading from GitHub...");
        
        // Try to initialize submodule first (if in a git repo)
        let git_result = Command::new("git")
            .args(&["submodule", "update", "--init", "--recursive", "whisper.cpp"])
            .current_dir(&manifest_dir)
            .output();
        
        // Check if submodule init worked
        let submodule_success = git_result.is_ok() && 
            git_result.as_ref().unwrap().status.success() &&
            whisper_cpp_source.exists() &&
            dir_has_contents(&whisper_cpp_source);
        
        // If submodule init failed or whisper.cpp still doesn't exist, clone directly
        if !submodule_success {
            println!("cargo:warning=Submodule init failed, cloning whisper.cpp directly...");
            
            // Clone whisper.cpp to a temp location
            let temp_whisper = out.join("whisper.cpp-temp");
            if temp_whisper.exists() {
                std::fs::remove_dir_all(&temp_whisper).unwrap_or_default();
            }
            
            let clone_result = Command::new("git")
                .args(&["clone", "--depth", "1", "https://github.com/ggerganov/whisper.cpp.git", temp_whisper.to_str().unwrap()])
                .output();
            
            match clone_result {
                Ok(output) if output.status.success() => {
                    // Move the cloned directory to the expected location
                    if whisper_cpp_source.exists() {
                        std::fs::remove_dir_all(&whisper_cpp_source).ok();
                    }
                    std::fs::create_dir_all(whisper_cpp_source.parent().unwrap()).unwrap();
                    
                    // Try rename first, fall back to copy if rename fails (cross-filesystem)
                    if std::fs::rename(&temp_whisper, &whisper_cpp_source).is_err() {
                        fs_extra::dir::copy(&temp_whisper, &manifest_dir, &Default::default()).unwrap_or_else(|e| {
                            panic!("Failed to copy cloned whisper.cpp: {}", e);
                        });
                        std::fs::remove_dir_all(&temp_whisper).ok();
                    }
                    println!("cargo:warning=Successfully downloaded whisper.cpp");
                }
                Ok(output) => {
                    eprintln!("Failed to clone whisper.cpp: {}", String::from_utf8_lossy(&output.stderr));
                    panic!("Failed to download whisper.cpp. Please ensure git is installed and you have internet access.");
                }
                Err(e) => {
                    panic!("Failed to run git clone: {}. Please ensure git is installed.", e);
                }
            }
        } else {
            println!("cargo:warning=Successfully initialized whisper.cpp submodule");
        }
    }

    // Now copy whisper.cpp to the build directory
    if !whisper_root.exists() || !whisper_root.join("CMakeLists.txt").exists() {
        if whisper_root.exists() {
            std::fs::remove_dir_all(&whisper_root).unwrap_or_default();
        }
        std::fs::create_dir_all(&whisper_root).unwrap();
        fs_extra::dir::copy(&whisper_cpp_source, &out, &Default::default()).unwrap_or_else(|e| {
            panic!(
                "Failed to copy whisper sources from {} to {}: {}",
                whisper_cpp_source.display(),
                whisper_root.display(),
                e
            )
        });
        
        // Verify CMakeLists.txt exists after copy
        if !whisper_root.join("CMakeLists.txt").exists() {
            panic!(
                "CMakeLists.txt not found in {} after copy. Source: {}",
                whisper_root.display(),
                whisper_cpp_source.display()
            );
        }
    }

    if env::var("WHISPER_DONT_GENERATE_BINDINGS").is_ok() {
        let _: u64 = std::fs::copy("src/bindings.rs", out.join("bindings.rs"))
            .expect("Failed to copy bindings.rs");
    } else {
        // Get absolute path to wrapper.h
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let wrapper_h = manifest_dir.join("wrapper.h");
        let mut bindings = bindgen::Builder::default().header(wrapper_h.to_str().unwrap());

        // When use-shared-ggml is enabled, use ggml-rs headers
        if cfg!(feature = "use-shared-ggml") {
            #[cfg(feature = "metal")]
            {
                if let Some(ref include_dir) = ggml_include_dir {
                    bindings = bindings.header(include_dir.join("ggml-metal.h").to_str().unwrap());
                }
            }
            #[cfg(feature = "vulkan")]
            {
                if let Some(ref include_dir) = ggml_include_dir {
                    bindings = bindings
                        .header(include_dir.join("ggml-vulkan.h").to_str().unwrap())
                        .clang_arg("-DGGML_USE_VULKAN=1");
                }
            }
        } else {
            // Use embedded ggml headers
            #[cfg(feature = "metal")]
            {
                bindings = bindings.header(whisper_cpp_source.join("ggml/include/ggml-metal.h").to_str().unwrap());
            }
            #[cfg(feature = "vulkan")]
            {
                bindings = bindings
                    .header(whisper_cpp_source.join("ggml/include/ggml-vulkan.h").to_str().unwrap())
                    .clang_arg("-DGGML_USE_VULKAN=1");
            }
        }

        // Use whisper_cpp_source for include paths since that's where the files are
        // (they get copied to whisper_root later for CMake)
        // IMPORTANT: Add GGML include path FIRST so whisper.h can find ggml.h
        let mut bindings_builder = bindings;
        
        // Add GGML include path
        if cfg!(feature = "use-shared-ggml") {
            if let Some(ref include_dir) = ggml_include_dir {
                bindings_builder = bindings_builder.clang_arg(format!("-I{}", include_dir.display()));
            } else {
                panic!("use-shared-ggml feature is enabled but DEP_GGML_RS_GGML_WHISPER_LIB_DIR is not set. Make sure ggml-rs is properly configured and built.");
            }
        } else {
            bindings_builder = bindings_builder.clang_arg(format!("-I{}", whisper_cpp_source.join("ggml/include").display()));
        }
        
        // Now add whisper include paths
        bindings_builder = bindings_builder
            .clang_arg(format!("-I{}", whisper_cpp_source.display()))
            .clang_arg(format!("-I{}", whisper_cpp_source.join("include").display()));

        let bindings = bindings_builder
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate();

        match bindings {
            Ok(b) => {
                let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
                b.write_to_file(out_path.join("bindings.rs"))
                    .expect("Couldn't write bindings!");
            }
            Err(e) => {
                println!("cargo:warning=Unable to generate bindings: {}", e);
                println!("cargo:warning=Using bundled bindings.rs, which may be out of date");
                // copy src/bindings.rs to OUT_DIR
                std::fs::copy("src/bindings.rs", out.join("bindings.rs"))
                    .expect("Unable to copy bindings.rs");
            }
        }
    };

    // stop if we're on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // If use-shared-ggml feature is enabled, skip building ggml and link to shared library
    if cfg!(feature = "use-shared-ggml") {
        // IMPORTANT: We need to link to the whisper-specific GGML libraries explicitly
        // ggml-rs now builds both variants (whisper and llama) unconditionally
        // Libraries are named using the basename from DEP_GGML_RS_GGML_WHISPER_BASENAME
        // (typically "ggml_whisper", "ggml_whisper-base", "ggml_whisper-cpu", "ggml_whisper-cuda", etc.)
        // When CUDA is enabled on ggml-rs, it builds ggml_whisper-cuda and places it in LIB_DIR
        // ggml-rs builds these libraries but doesn't link them for dependent crates
        // We link them here based on what's available in the LIB_DIR
        
        // Verify that ggml-rs build script ran and exported the required variables
        if ggml_lib_dir.is_none() {
            panic!(
                "use-shared-ggml feature is enabled but DEP_GGML_RS_GGML_WHISPER_LIB_DIR is not set.\n\
                This means ggml-rs's build script did not run or did not export the required variables.\n\
                Please verify:\n\
                1. ggml-rs is in [build-dependencies] of this crate's Cargo.toml\n\
                2. ggml-rs's build.rs exports: DEP_GGML_RS_GGML_WHISPER_LIB_DIR, DEP_GGML_RS_GGML_WHISPER_BIN_DIR, DEP_GGML_RS_GGML_WHISPER_BASENAME\n\
                3. The variable names match exactly (case-sensitive)"
            );
        }
        
        let lib_base_name = &ggml_lib_basename;
        
        println!("cargo:warning=[GGML] Using whisper-specific GGML libraries with basename: {}", lib_base_name);
        
        // Add library search path (ggml-rs already links the libraries)
        if let Some(ref lib_dir) = ggml_lib_dir {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
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
        // Construct prefix from lib_dir (ggml-rs installs to separate directories per variant)
        let ggml_prefix = ggml_lib_dir.as_ref().and_then(|lib_dir| lib_dir.parent().map(|p| p.to_path_buf()));
        
        if let Some(ref prefix) = ggml_prefix {
            // Set CMAKE_PREFIX_PATH to where ggml-rs installed ggml
            config.define("CMAKE_PREFIX_PATH", prefix.to_str().unwrap());
            // Set ggml_DIR to the cmake config directory
            let ggml_cmake_dir = prefix.join("lib").join("cmake").join("ggml");
            if ggml_cmake_dir.exists() {
                config.define("ggml_DIR", ggml_cmake_dir.to_str().unwrap());
            }
        }
        
        // Alternative: If CMake config files aren't in the expected location,
        // you may need to set additional paths
        if let Some(ref include_dir) = ggml_include_dir {
            // Add include directory for CMake
            config.define("GGML_INCLUDE_DIR", include_dir.to_str().unwrap());
        }
        
        // CRITICAL: Do NOT patch ggml-config.cmake - let ggml-rs handle all patching
        // ggml-rs builds both variants and patches the config files when building
        // If we also patch it, we create duplicate add_library calls
        // ggml-rs should handle ALL patching, including:
        // - Replacing library names with namespaced versions (ggml_whisper, ggml_llama)
        // - Adding add_library calls with proper guards
        // - Setting up CMake targets correctly
        // We only need to ensure CMake can find the patched config file
        if let Some(ref lib_dir) = ggml_lib_dir {
            // Also set GGML_LIBRARY directly as a fallback
            let lib_file = if cfg!(target_os = "windows") {
                format!("{}.lib", lib_base_name)
            } else {
                format!("lib{}.a", lib_base_name)
            };
            let namespaced_lib = lib_dir.join(&lib_file);
            
            if namespaced_lib.exists() {
                config.define("GGML_LIBRARY", namespaced_lib.to_str().unwrap());
                println!("cargo:warning=[GGML] Setting GGML_LIBRARY to whisper-specific library: {}", namespaced_lib.display());
            } else {
                config.define("GGML_LIB_DIR", lib_dir.to_str().unwrap());
                println!("cargo:warning=[GGML] Whisper-specific library not found at {}, using GGML_LIB_DIR", namespaced_lib.display());
            }
        }
        
        if cfg!(target_os = "windows") {
            config.cxxflag("/utf-8");
            println!("cargo:rustc-link-lib=advapi32");
        }
        
        if cfg!(feature = "coreml") {
            config.define("WHISPER_COREML", "ON");
            config.define("WHISPER_COREML_ALLOW_FALLBACK", "1");
        }
        
        // Note: GGML features (cuda, hipblas, vulkan, metal, openblas, intel-sycl) are handled
        // by the shared ggml-rs library, so we don't configure them here.
        // We only configure whisper-specific features.
        
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
        
        if cfg!(not(feature = "openmp")) {
            config.define("GGML_OPENMP", "OFF");
        }
        
        let destination = config.build();
        add_link_search_path(&out.join("build")).unwrap();
        println!("cargo:rustc-link-search=native={}", destination.display());
        println!("cargo:rustc-link-lib=static=whisper");
        
        // CRITICAL: Link to the namespaced GGML libraries (whisper-specific: ggml_whisper)
        // ggml-rs builds the libraries but doesn't link them for dependent crates
        // All library names use lib_base_name which is "ggml_whisper" (whisper-specific)
        if let Some(ref lib_dir) = ggml_lib_dir {
            // Ensure the library search path is set right before linking (whisper-specific libraries)
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            
            // Always link the base libraries (whisper-specific: ggml_whisper, ggml_whisper-base, ggml_whisper-cpu)
            println!("cargo:rustc-link-lib=dylib={}", lib_base_name);
            println!("cargo:rustc-link-lib=dylib={}-base", lib_base_name);
            println!("cargo:rustc-link-lib=dylib={}-cpu", lib_base_name);
            
            // Link feature-specific libraries if they exist (all whisper-specific: ggml_whisper-*)
            // These libraries are built by ggml-rs when the corresponding features are enabled
            // For example, if ggml-rs is built with "cuda" feature, it will build ggml_whisper-cuda
            // and place it in DEP_GGML_RS_GGML_WHISPER_LIB_DIR
            // Check for CUDA (whisper-specific: ggml_whisper-cuda)
            // Note: CUDA runtime libraries (cudart, cublas, etc.) are linked separately above
            let cuda_lib = if cfg!(target_os = "windows") {
                lib_dir.join(format!("{}-cuda.lib", lib_base_name))
            } else if cfg!(target_os = "macos") {
                lib_dir.join(format!("lib{}-cuda.dylib", lib_base_name))
            } else {
                lib_dir.join(format!("lib{}-cuda.so", lib_base_name))
            };
            if cuda_lib.exists() || lib_dir.join(format!("{}-cuda.dll", lib_base_name)).exists() {
                println!("cargo:rustc-link-lib=dylib={}-cuda", lib_base_name);
                println!("cargo:warning=[GGML] Linked whisper-specific CUDA library: {}-cuda", lib_base_name);
            }
            
            // Check for Vulkan (whisper-specific: ggml_whisper-vulkan)
            let vulkan_lib = if cfg!(target_os = "windows") {
                lib_dir.join(format!("{}-vulkan.lib", lib_base_name))
            } else if cfg!(target_os = "macos") {
                lib_dir.join(format!("lib{}-vulkan.dylib", lib_base_name))
            } else {
                lib_dir.join(format!("lib{}-vulkan.so", lib_base_name))
            };
            if vulkan_lib.exists() || lib_dir.join(format!("{}-vulkan.dll", lib_base_name)).exists() {
                println!("cargo:rustc-link-lib=dylib={}-vulkan", lib_base_name);
            }
            
            // Check for Metal (macOS) (whisper-specific: ggml_whisper-metal)
            if cfg!(target_os = "macos") {
                let metal_lib = lib_dir.join(format!("lib{}-metal.dylib", lib_base_name));
                let metal_static = lib_dir.join(format!("lib{}-metal.a", lib_base_name));
                if metal_lib.exists() || metal_static.exists() {
                    println!("cargo:rustc-link-lib=dylib={}-metal", lib_base_name);
                }
            }
            
            // Check for BLAS (whisper-specific: ggml_whisper-blas)
            if cfg!(target_os = "macos") || cfg!(feature = "openblas") {
                let blas_lib = if cfg!(target_os = "windows") {
                    lib_dir.join(format!("{}-blas.lib", lib_base_name))
                } else if cfg!(target_os = "macos") {
                    lib_dir.join(format!("lib{}-blas.dylib", lib_base_name))
                } else {
                    lib_dir.join(format!("lib{}-blas.so", lib_base_name))
                };
                if blas_lib.exists() || lib_dir.join(format!("{}-blas.dll", lib_base_name)).exists() ||
                   lib_dir.join(format!("lib{}-blas.a", lib_base_name)).exists() {
                    println!("cargo:rustc-link-lib=dylib={}-blas", lib_base_name);
                }
            }
            
            // Check for HIP (whisper-specific: ggml_whisper-hip)
            if cfg!(feature = "hipblas") {
                let hip_lib = if cfg!(target_os = "windows") {
                    lib_dir.join(format!("{}-hip.lib", lib_base_name))
                } else if cfg!(target_os = "macos") {
                    lib_dir.join(format!("lib{}-hip.dylib", lib_base_name))
                } else {
                    lib_dir.join(format!("lib{}-hip.so", lib_base_name))
                };
                if hip_lib.exists() || lib_dir.join(format!("{}-hip.dll", lib_base_name)).exists() {
                    println!("cargo:rustc-link-lib=dylib={}-hip", lib_base_name);
                }
            }
            
            // Check for SYCL (whisper-specific: ggml_whisper-sycl)
            if cfg!(feature = "intel-sycl") {
                let sycl_lib = if cfg!(target_os = "windows") {
                    lib_dir.join(format!("{}-sycl.lib", lib_base_name))
                } else if cfg!(target_os = "macos") {
                    lib_dir.join(format!("lib{}-sycl.dylib", lib_base_name))
                } else {
                    lib_dir.join(format!("lib{}-sycl.so", lib_base_name))
                };
                if sycl_lib.exists() || lib_dir.join(format!("{}-sycl.dll", lib_base_name)).exists() {
                    println!("cargo:rustc-link-lib=dylib={}-sycl", lib_base_name);
                }
            }
        }
        
        // On Windows, copy whisper-specific GGML DLLs to the target directory for runtime
        // All DLLs are whisper-specific: ggml_whisper.dll, ggml_whisper-base.dll, etc.
        // Use BIN_DIR if available (from DEP_GGML_RS_GGML_WHISPER_BIN_DIR), otherwise fall back to LIB_DIR
        if cfg!(target_os = "windows") && cfg!(feature = "use-shared-ggml") {
            let dll_source_dir = ggml_bin_dir.as_ref().or(ggml_lib_dir.as_ref());
            if let Some(ref dll_dir) = dll_source_dir {
                copy_namespace_dlls_to_target(dll_dir, lib_base_name);
            }
        }
        
    } else {
        // Original code: build whisper with embedded ggml
        let mut config = Config::new(&whisper_root);

        config
            .profile("Release")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("WHISPER_ALL_WARNINGS", "OFF")
            .define("WHISPER_ALL_WARNINGS_3RD_PARTY", "OFF")
            .define("WHISPER_BUILD_TESTS", "OFF")
            .define("WHISPER_BUILD_EXAMPLES", "OFF")
            .very_verbose(true)
            .pic(true);

        if cfg!(target_os = "windows") {
            config.cxxflag("/utf-8");
            println!("cargo:rustc-link-lib=advapi32");
        }

        if cfg!(feature = "coreml") {
            config.define("WHISPER_COREML", "ON");
            config.define("WHISPER_COREML_ALLOW_FALLBACK", "1");
        }

        if cfg!(feature = "cuda") {
            config.define("GGML_CUDA", "ON");
            config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
            config.define("CMAKE_CUDA_FLAGS", "-Xcompiler=-fPIC");
        }

        if cfg!(feature = "hipblas") {
            config.define("GGML_HIP", "ON");
            config.define("CMAKE_C_COMPILER", "hipcc");
            config.define("CMAKE_CXX_COMPILER", "hipcc");
            println!("cargo:rerun-if-env-changed=AMDGPU_TARGETS");
            if let Ok(gpu_targets) = env::var("AMDGPU_TARGETS") {
                config.define("AMDGPU_TARGETS", gpu_targets);
            }
        }

        if cfg!(feature = "vulkan") {
            config.define("GGML_VULKAN", "ON");
            if cfg!(windows) {
                println!("cargo:rerun-if-env-changed=VULKAN_SDK");
                println!("cargo:rustc-link-lib=vulkan-1");
                let vulkan_path = match env::var("VULKAN_SDK") {
                    Ok(path) => PathBuf::from(path),
                    Err(_) => panic!(
                        "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                    ),
                };
                let vulkan_lib_path = vulkan_path.join("Lib");
                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            } else if cfg!(target_os = "macos") {
                println!("cargo:rerun-if-env-changed=VULKAN_SDK");
                println!("cargo:rustc-link-lib=vulkan");
                let vulkan_path = match env::var("VULKAN_SDK") {
                    Ok(path) => PathBuf::from(path),
                    Err(_) => panic!(
                        "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                    ),
                };
                let vulkan_lib_path = vulkan_path.join("lib");
                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            } else {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }

        if cfg!(feature = "openblas") {
            config.define("GGML_BLAS", "ON");
            config.define("GGML_BLAS_VENDOR", "OpenBLAS");
            if env::var("BLAS_INCLUDE_DIRS").is_err() {
                panic!("BLAS_INCLUDE_DIRS environment variable must be set when using OpenBLAS");
            }
            config.define("BLAS_INCLUDE_DIRS", env::var("BLAS_INCLUDE_DIRS").unwrap());
            println!("cargo:rerun-if-env-changed=BLAS_INCLUDE_DIRS");
        }

        if cfg!(feature = "metal") {
            config.define("GGML_METAL", "ON");
            config.define("GGML_METAL_NDEBUG", "ON");
            config.define("GGML_METAL_EMBED_LIBRARY", "ON");
        } else {
            // Metal is enabled by default, so we need to explicitly disable it
            config.define("GGML_METAL", "OFF");
        }

        if cfg!(debug_assertions) || cfg!(feature = "force-debug") {
            // debug builds are too slow to even remotely be usable,
            // so we build with optimizations even in debug mode
            config.define("CMAKE_BUILD_TYPE", "RelWithDebInfo");
            config.cxxflag("-DWHISPER_DEBUG");
        } else {
            // we're in release mode, explicitly set to release mode
            // see also https://codeberg.org/tazz4843/whisper-rs/issues/226
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

        if cfg!(not(feature = "openmp")) {
            config.define("GGML_OPENMP", "OFF");
        }

        if cfg!(feature = "intel-sycl") {
            config.define("BUILD_SHARED_LIBS", "ON");
            config.define("GGML_SYCL", "ON");
            config.define("GGML_SYCL_TARGET", "INTEL");
            config.define("CMAKE_C_COMPILER", "icx");
            config.define("CMAKE_CXX_COMPILER", "icpx");
        }

        let destination = config.build();

        add_link_search_path(&out.join("build")).unwrap();

        println!("cargo:rustc-link-search=native={}", destination.display());
        if cfg!(feature = "intel-sycl") {
            println!("cargo:rustc-link-lib=whisper");
            println!("cargo:rustc-link-lib=ggml");
            println!("cargo:rustc-link-lib=ggml-base");
            println!("cargo:rustc-link-lib=ggml-cpu");
        } else {
            println!("cargo:rustc-link-lib=static=whisper");
            println!("cargo:rustc-link-lib=static=ggml");
            println!("cargo:rustc-link-lib=static=ggml-base");
            println!("cargo:rustc-link-lib=static=ggml-cpu");
        }
        if cfg!(target_os = "macos") || cfg!(feature = "openblas") {
            println!("cargo:rustc-link-lib=static=ggml-blas");
        }
        if cfg!(feature = "vulkan") {
            if cfg!(feature = "intel-sycl") {
                println!("cargo:rustc-link-lib=ggml-vulkan");
            } else {
                println!("cargo:rustc-link-lib=static=ggml-vulkan");
            }
        }

        if cfg!(feature = "hipblas") {
            println!("cargo:rustc-link-lib=static=ggml-hip");
        }

        if cfg!(feature = "metal") {
            println!("cargo:rustc-link-lib=static=ggml-metal");
        }

        if cfg!(feature = "cuda") {
            println!("cargo:rustc-link-lib=static=ggml-cuda");
        }

        if cfg!(feature = "openblas") {
            println!("cargo:rustc-link-lib=static=ggml-blas");
        }

        if cfg!(feature = "intel-sycl") {
            println!("cargo:rustc-link-lib=ggml-sycl");
        }
    }

    println!(
        "cargo:WHISPER_CPP_VERSION={}",
        get_whisper_cpp_version(&whisper_root)
            .expect("Failed to read whisper.cpp CMake config")
            .expect("Could not find whisper.cpp version declaration"),
    );

    // for whatever reason this file is generated during build and triggers cargo complaining
    _ = std::fs::remove_file("bindings/javascript/package.json");
}

// From https://github.com/alexcrichton/cc-rs/blob/fba7feded71ee4f63cfe885673ead6d7b4f2f454/src/lib.rs#L2462
fn get_cpp_link_stdlib(target: &str) -> Option<&'static str> {
    if target.contains("msvc") {
        None
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        Some("c++")
    } else if target.contains("android") {
        Some("c++_shared")
    } else {
        Some("stdc++")
    }
}

fn add_link_search_path(dir: &std::path::Path) -> std::io::Result<()> {
    if dir.is_dir() {
        println!("cargo:rustc-link-search={}", dir.display());
        for entry in std::fs::read_dir(dir)? {
            add_link_search_path(&entry?.path())?;
        }
    }
    Ok(())
}

fn get_whisper_cpp_version(whisper_root: &std::path::Path) -> std::io::Result<Option<String>> {
    let cmake_lists = BufReader::new(File::open(whisper_root.join("CMakeLists.txt"))?);

    for line in cmake_lists.lines() {
        let line = line?;

        if let Some(suffix) = line.strip_prefix(r#"project("whisper.cpp" VERSION "#) {
            let whisper_cpp_version = suffix.trim_end_matches(')');
            return Ok(Some(whisper_cpp_version.into()));
        }
    }

    Ok(None)
}

/// Copy whisper-specific GGML DLLs to the target directory for runtime on Windows
/// lib_base_name comes from DEP_GGML_RS_GGML_WHISPER_BASENAME (typically "ggml_whisper")
/// This copies DLLs like: ggml_whisper.dll, ggml_whisper-base.dll, ggml_whisper-cpu.dll, etc.
/// dll_dir should be from DEP_GGML_RS_GGML_WHISPER_BIN_DIR (or LIB_DIR as fallback)
fn copy_namespace_dlls_to_target(dll_dir: &PathBuf, lib_base_name: &str) {
    let target_dir = env::var("OUT_DIR")
        .ok()
        .and_then(|out| {
            PathBuf::from(&out)
                .ancestors()
                .nth(3) // Go up from build/.../out to target/debug or target/release
                .map(|p| p.to_path_buf())
        });
    
    if let Some(target) = target_dir {
        // List of namespace-specific libraries to copy
        let libraries = vec![
            lib_base_name.to_string(),
            format!("{}-base", lib_base_name),
            format!("{}-cpu", lib_base_name),
            format!("{}-cuda", lib_base_name),
            format!("{}-vulkan", lib_base_name),
            format!("{}-metal", lib_base_name),
            format!("{}-hip", lib_base_name),
            format!("{}-blas", lib_base_name),
            format!("{}-sycl", lib_base_name),
        ];
        
        let mut copied_count = 0;
        
        for lib_name in &libraries {
            let dll_name = format!("{}.dll", lib_name);
            let src = dll_dir.join(&dll_name);
            if src.exists() {
                let dst = target.join(&dll_name);
                if let Err(e) = std::fs::copy(&src, &dst) {
                    println!("cargo:warning=[GGML] Failed to copy {} to {}: {}", 
                        src.display(), dst.display(), e);
                } else {
                    println!("cargo:warning=[GGML] Copying namespace-specific library: {}", dll_name);
                    copied_count += 1;
                }
            }
        }
        
        if copied_count > 0 {
            println!("cargo:warning=[GGML] Copied {} namespace-specific GGML libraries", copied_count);
        }
    }
}

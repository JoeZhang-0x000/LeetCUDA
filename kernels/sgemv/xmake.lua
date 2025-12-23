add_rules("mode.debug", "mode.release")
add_requires("cmake::Torch", {
    alias = "libtorch",
    system = true,
    configs = {
        shared = false,
        envs = {
            CMAKE_PREFIX_PATH = "/usr/local/lib/python3.12/dist-packages/torch/share/cmake"
        }
    }
})

add_requires("pybind11")
target("sgemv")
    add_rules("python.module") 
    add_cxflags("-std=c++17")
    add_rules("cuda.build")
    set_kind("shared")
    add_linkdirs("/usr/local/lib/python3.12/dist-packages/torch/lib")
    add_links("torch", "torch_python", "c10")
    add_files("sgemv.cu")
    add_packages("libtorch", "pybind11", "python")
    set_installdir("./")


target("tune")
    add_rules("cuda.build")
    set_kind("binary")
    add_files("tune.cu")
    
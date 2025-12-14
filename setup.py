from setuptools import setup, find_packages, Extension

try:
    import pybind11
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
    import pybind11

ext_modules = [
    Extension(
        "eagle.model.fast_prefix_tree",
        ["eagle/model/fast_prefix_tree.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17"]
    ),
    Extension(
        "ngram_data_structures.prefix_tree_cpp",
        ["ngram_data_structures/prefix_tree_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17"]
    ),
    Extension(
        "ngram_data_structures.suffix_tree_idx_cpp",
        ["ngram_data_structures/suffix_tree_idx_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17"]
    ),
    Extension(
        "ngram_data_structures.suffix_tree_partial_result_cpp",
        ["ngram_data_structures/suffix_tree_partial_result_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17"]
    ),
    Extension(
        "ngram_data_structures.suffix_tree_full_cpp",
        ["ngram_data_structures/suffix_tree_full_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17"]
    ),
    Extension(
        "ngram_data_structures.non_tree_implementations.hashmap_cpp",
        ["ngram_data_structures/non_tree_implementations/hashmap_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17"]
    ),
    Extension(
        "ngram_data_structures.prefix_tree_speedy_cpp",
        ["ngram_data_structures/prefix_tree_speedy.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17"]
    ),
]

setup(
    name='eagle-llm',
    version='3.0.0',
    description='Accelerating LLMs by 3x with No Quality Loss',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='yuhui.li@stu.pku.edu.cn',
    url='https://github.com/SafeAILab/EAGLE',
    packages=find_packages(),
    install_requires=[
        'torch==2.0.1',
        'transformers==4.46.2',
        'accelerate==1.11.0',
        'fschat==0.2.31',
        'gradio==3.50.2',
        'openai==0.28.0',
        'anthropic==0.5.0',
        'sentencepiece==0.1.99',
        'protobuf==3.19.0',
        'wandb',
        'pybind11'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    ext_modules=ext_modules,
)


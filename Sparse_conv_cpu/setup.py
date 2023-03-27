from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='my_sparse_conv_cpu',
    ext_modules=[
        CppExtension('my_sparse_conv_cpu', [
            'sparse_conv_cpu.cpp',
            'sparse_conv_cpu_kernel.cpp',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
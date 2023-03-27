from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_conv',
    ext_modules=[
        CUDAExtension('sparse_conv', 
        extra_compile_args={'cxx': [],"nvcc":["--extended-lambda"]},
        sources=[
            'sparse_conv_cuda.cpp',
            'sparse_conv_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
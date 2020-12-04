from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='catCuda',
    ext_modules=[
        CUDAExtension(
            name='catCuda',
            sources=[
                'src/kernels/cat/catKernels.cu'
            ],
            depends=[
                'src/kernels/cat/spikeKernels.h',
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-arch=sm_60', '-O2', '-use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name='catCpp',
    ext_modules=[
        CppExtension(
            name='catCpp',
            sources=[
                'src/kernels/cat/catCpp.cpp'
            ],
            depends=[
                'src/kernels/cat/catCpp.hpp',
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2', '-use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

setup(
    name='catSNN',
    packages = ['catSNN'],
    package_dir = {'catSNN': 'src'},
)

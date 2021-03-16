import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

if torch.cuda.is_available():
    print('Including CUDA code.')
    setup(
        name='trilinear',
        ext_modules=[
            CUDAExtension('trilinear', [
                'src/trilinear_cuda.cpp',
                'src/trilinear_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    print('NO CUDA is found. Fall back to CPU.')

    # here compling the module as `trilinear`
    # just use `import trilinear` in source code.
    setup(name='trilinear',
          ext_modules=[CppExtension('trilinear', ['src/trilinear.cpp'])],
          cmdclass={'build_ext': BuildExtension})

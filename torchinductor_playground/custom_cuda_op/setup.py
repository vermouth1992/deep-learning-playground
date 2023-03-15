from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_sigmoid',
    ext_modules=[
        CUDAExtension('my_sigmoid', [
            'custom_op.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
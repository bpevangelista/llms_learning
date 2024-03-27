from setuptools import setup
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CUDAExtension(
        name='mha',
        sources=[
            'cstr/mha.cpp',
            'cstr/mha_kernel.cu',
        ],
        include_dirs=[
            #'/home/bpevangelista/projects/cutlass/include',
            '../../../cutlass/include',
        ],
    )
]
setup(
    name='mha',
    version='0.0.1',
    description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)

from setuptools import setup
from torch.utils import cpp_extension

setup(name='nn_api',
      ext_modules=[cpp_extension.CppExtension('nn_api', ['cstr/nn_api.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

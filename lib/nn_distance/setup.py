from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='nn_distance',
    ext_modules=[
        CUDAExtension('nn_distance', [
            'src/nn_distance.cpp',
            'src/nn_distance_cuda.cu', ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],

    cmdclass={
        'build_ext': BuildExtension
    })

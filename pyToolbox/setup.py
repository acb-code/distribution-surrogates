from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext

compile_args = {
    'msvc': ['/openmp', '/O2'],         # '/fp:fast'
    'mingw32': ['-fopenmp', '-O3'],     # '-ffast-math'
    'gcc': ['-fopenmp', '-O3'],         # '-ffast-math'
}
link_args = {
    'mingw32': ['-fopenmp'],
    'gcc': ['-fopenmp'],
}

extensions = [
    Extension(
        name='pyToolBox.geom.grid_functions',
        sources=['pyToolBox/geom/grid_functions.pyx']
    ),
]


class BuildExtSubclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in compile_args:
            for ext in self.extensions:
                ext.extra_compile_args = compile_args[c]
        if c in link_args:
            for ext in self.extensions:
                ext.extra_link_args = link_args[c]
        build_ext.build_extensions(self)


setup(
    name='pyToolBox',
    version='0.1',
    packages=[
        'pyToolBox',
        'pyToolBox.dr',
        'pyToolBox.geom',
        'pyToolBox.io',
        'pyToolBox.rom',
        'pyToolBox.sampling',
        'pyToolBox.surrogate'
    ],
    package_data={
        'pyToolBox.sampling': [
            'joe-kuo-old.1111',
            'new-joe-kuo-6.21201',
        ]
    },
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': BuildExtSubclass},
    url='',
    license='',
    author='cperron7',
    author_email='',
    description='',
    install_requires=[
        'numpy',
        'scipy',
        'cython',
        'vtk',
    ],
    extras_require={
        'scripts': [
            'matplotlib',
            'pandas',
            'mkl-service',
            'h5py',
        ]
    },
    python_requires='>=3.5',
    zip_safe=False,
)

from setuptools import setup, find_packages
from glob import glob

setup(
    name='mppi',
    version='0.1.0',
    py_modules=[
        'mppi_node',
        'mppi_tracking',
        'infer_env',
        'vis_node',
    ],
    packages=['utils', 'dynamics_models'],
    package_dir={'': '.'},  # tells setuptools to look in current dir
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/mppi']),
        ('share/mppi', ['package.xml', 'config.yaml']),
        ('share/mppi/waypoints', glob('waypoints/**/*', recursive=True)),
    ],
    install_requires=[
        'setuptools',
        'jax',
        'jaxlib',
        'numpy',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='MPPI controller using JAX and ROS 2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mppi_node = mppi_node:main',
            'vis_node = vis_node:main',
        ],
    },
)


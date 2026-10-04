import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))

setup(
    name="mini_torchrec",
    version="0.1.0",
    packages=find_packages(include=["mini_torchrec"]),
    ext_modules=[
        CUDAExtension(
            name="mini_torchrec._C",
            sources=["csrc/ops.cpp", "csrc/fused_tbe.cu", "csrc/shard_lookup.cu"],
            include_dirs=[os.path.join(here, "csrc")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

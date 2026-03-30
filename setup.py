from setuptools import setup

setup(
    name="hangzhou-tunneling",
    version="0.1.0",
    py_modules=["hangzhou_tunneling"],
    install_requires=[
        "qwen-agent[mcp]>=0.1.0",
    ],
    entry_points={
        "console_scripts": [
            "hangzhou-tunneling=hangzhou_tunneling:main",
        ],
    },
    python_requires=">=3.10",
)

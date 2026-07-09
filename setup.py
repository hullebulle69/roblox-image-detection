from setuptools import find_packages, setup

setup(
    name="roblox-character-detector",
    version="1.0.0",
    description="Advanced computer-vision detector for Roblox characters in images.",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24", "pillow>=9.0", "opencv-python>=4.8"],
    entry_points={
        "console_scripts": [
            "roblox-detect=roblox_detector.cli:main",
            "roblox-detect-gui=roblox_detector.gui:main",
        ],
    },
)

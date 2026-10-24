from glob import glob
from setuptools import find_packages, setup

setup(
    name="edgehand",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/edgehand"]),
        ("share/edgehand", ["package.xml"]),
        ("share/edgehand/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "intent_node = edgehand.nodes.intent_node:main",
        "vision_node = edgehand.nodes.vision_node:main",
        "handover_node = edgehand.nodes.handover_node:main",
        "pi_bridge = edgehand.nodes.pi_bridge:main",
    ]},
)

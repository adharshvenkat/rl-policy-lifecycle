from setuptools import find_packages, setup

package_name = 'go1_policy_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adharshvenkat',
    maintainer_email='adharshvenkat@users.noreply.github.com',
    description='ROS2 bridge exposing a trained Go1 RL locomotion policy as a Nav2-compatible robot driver.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'policy_bridge_node = go1_policy_bridge.policy_bridge_node:main'
        ],
    },
)

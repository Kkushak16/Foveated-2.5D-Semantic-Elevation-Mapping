from setuptools import setup

package_name = 'foveated_grid_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kushak',
    maintainer_email='kushak16@example.com',
    description='ROS 2 LiDAR bridge for the Foveated 2.5D Grid Engine',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lidar_bridge_node = foveated_grid_bridge.lidar_bridge_node:main',
        ],
    },
)

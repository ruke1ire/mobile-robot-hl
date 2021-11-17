from setuptools import setup
import os
from glob import glob

package_name = 'mobile_robot_hl'

submodule_names = ['model', 'episode_data', 'joystick', 'logger']
submodule_paths = [f"{package_name}/{name}" for name in submodule_names]


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name] + submodule_paths,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*_launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rom',
    maintainer_email='rom.parnichkun@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'supervisor_node = mobile_robot_hl.supervisor_node:main',
            'agent_node = mobile_robot_hl.agent_node:main',
            'trainer_node = mobile_robot_hl.trainer_node:main',
        ],
    },
)

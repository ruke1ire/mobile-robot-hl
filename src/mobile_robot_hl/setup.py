from setuptools import setup

package_name = 'mobile_robot_hl'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'joystick_node = mobile_robot_hl.joystick_node:main',
        ],
    },
)

from setuptools import setup

setup(
	name='OpenRV',
	url='https://github.com/OpenRoboVision/OpenRV',
	author='kivicode',
	author_email='kivicode@yandex.com',
	packages=['openrv'],
	install_requires=['numpy>=1.6', 'opencv-python<4.0', 'scikit-image>=0.16', 'imutils>=0.5.3', 'scipy>=1.3.1', 'sklearn'],
	version='0.1.6.6.7',
	license='GNU GPLv3',
	description='',
	long_description=open('README.md').read(),
	long_description_content_type="text/markdown",
	classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
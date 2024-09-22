from setuptools import setup, find_packages

install_requires = open('requirements.txt').read().splitlines()

setup(
    name='mgamdata',
    version='1.0.0',
    packages=find_packages(),
    license='GNU General Public License v3.0',
    description="mgam's data toolkit for a better world!",
    author='Yiqin Zhang',
    author_email='312065559@qq.com',
    include_package_data=True,
    install_requires=install_requires,
)
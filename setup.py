from setuptools import setup, find_packages

setup(
    name='ir',  
    version='0.1',  

    # Description
    description='have',

    # Author information
    author='Quincy',
    author_email='austaining@gmail.com',

    # License
    license='RPI',  

    # Packages to install
    packages=find_packages(),

    # Dependencies
    install_requires=[
        'numpy>=1.0',  # Example dependency
        'matplotlib>=3.0',  # Example dependency
        # Add other dependencies here
    ],

    # Optional: include additional package data, like non-Python files
    # include_package_data=True,

    # Optional: specify entry points for command-line scripts
    # entry_points={
    #     'console_scripts': [
    #         'your_script_name = your_package.module:function',
    #     ],
    # },

    # Optional: classifiers for your package
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        # Add more classifiers as needed
    ],
)

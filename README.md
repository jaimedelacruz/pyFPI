# pyFPI
Tools for characterization of Fabry Perot instruments from observational data
Developed by J. de la Cruz Rodriguez, G. B. Scharmer & P. Suetterlin.

References: de la Cruz Rodriguez, Scharmer, Suetterlin (2025, in prep.)

## Support
The code is distributed "as it is".

Although we encourage people to submit bug-reports, dedicated support will only be provided within a scientific collaboration with us, as we have very limited man power.

We include a commented example that reads HRE and LRE scans over a small FOV, performs the data fits and plots the results. Simply run the example.py file located in example_CRISP after installing the C++ module.


## Dependencies
We have tested pyFPI with the GCC and CLANG compilers, both in Linux and OSX (with Apple M-processors)

The code includes source written in C++ with a Python interface.
It makes use of the following C/C++ libraries: Eigen-3, FFTW-3 & OpenMP.
It will also require the following Python packages: NumPy, Cython, setuptools, matplotlib (to run the example).

The code has been extensively run in Linux and OSX.

## Compilation instructions in Linux
Under Linux, one can easily install the Eigen-3 and FFTW-3 libraries, the python dependencies and compile the code out of the box with the followin command:
```
python3 setup.py build_ext install  --user
```

Alternatively, if the installation folder is not writeable by the user, one can compile the module and copy it manually to the working folder or to the PYTHONPATH location:
```
python3 setup.py build_ext --inplace
mv pyFPI.*.so /destination/folder/
```

## Compilation instructions in OSX
In OSX, getting the dependencies in place is less standarized. I personally use MacPorts, as both Gcc and Clang are available, and it provides quite a lot of control over different python versions.

With MacPorts, one can install the required packages for the corresponding version of python (e.g., 3.13):
```
sudo port install py313-numpy py313-cython py313-matplotlib eigen3 fftw-3 py313-scipy py313-setuptools
```

And then compile and install the module with the same command line as in Linux:
```
python3 setup.py build_ext install  --user
```

Alternatively, if the installation folder is not writeable by the user, one can compile the module and copy it manually to the working folder or to the PYTHONPATH location:
```
python3 setup.py build_ext --inplace
mv pyFPI.*.so /destination/folder/
```

It must be possible to install all the packages with anaconda python (use miniforge and mamba in that case), but I have not done it myself.


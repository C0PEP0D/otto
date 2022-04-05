Installation
============

Requirements
------------

OTTO requires Python 3.8 or greater.
Dependencies are listed in `requirements.txt <https://github.com/C0PEP0D/otto/blob/main/requirements.txt>`_,
missing dependencies will be installed automatically.

Optional: OTTO requires `FFmpeg <https://www.ffmpeg.org/>`_ to make videos.
If FFmpeg is not installed, OTTO will save video frames as images instead.

Conda users
-----------

If you use conda to manage your Python environments, you can install OTTO in a dedicated environment ``ottoenv``::

    conda create --name ottoenv python=3.8
    conda activate ottoenv

Installing
----------
First download the package or clone the git repository with::

    git clone https://github.com/C0PEP0D/otto.git

Then go to the ``otto`` directory and install OTTO using::

    python3 setup.py install

Testing
-------
You can test your installation with following command::

    python3 -m pytest tests

This will execute the test functions located in the folder ``tests``.
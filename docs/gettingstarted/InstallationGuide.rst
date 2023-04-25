.. SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
..
.. SPDX-License-Identifier: GPL-3.0-or-later

.. _installationguide:

Installation Guide
==================

SaQC is written in Python, so the easiest way to set up your system to use SaQC
for your needs is using the Python Package Index (PyPI). It might be a good idea
to install SaQC into its own virtual environment. 


System Requirements
-------------------
SaQC is tested to run with Python version from 3.7 to 3.9 on 64-bit operating systems (Linux and Windows).


Set up a virtual environment
-----------------------------

It is good practice to create new virtual environments for different projects. This
helps keeping dependencies separated and avoids issues with conflicting versions of
a single module. The exact process to setup such an environment depends on your operating
system and python version/distribution. The following sections should get you started on
UNIX-like Systems and Windows.


On Unix/Mac-systems
"""""""""""""""""""

On Unix-like systems the process is usually rather easy. Open up a terminal window and
copy-paste the following commands

.. code-block:: sh

   # create virtual environment called "saqc-env"
   python -m venv saqc-env

   # activate the virtual environment
   source saqc-env/bin/activate

On Windows-systems
""""""""""""""""""

On windows, things are a bit more evolved however. The first hurdle to take is usually an
installation of Python itself. There are many options available, one popular solution is
the `Conda package management system <https://docs.conda.io/en/latest/>`_. After its
`installation <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
type in the following:

.. code-block:: sh

   # create virtual environment called "saqc-env"
   conda create -n saqc-enc

   # activate the virtual environment
   conda activate saqc-env

   # install pip, we will need it later
   conda install pip


Install SaQC
------------

We currently distribute SaQC via the `Python Package Index (PyPI) <https://pypi.org/>`_
or through our `GitLab-repository <https://git.ufz.de/rdm-software/saqc>`_.

The latest stable versions are available with 

.. code-block:: sh

   python -m pip install saqc

or

.. code-block:: sh

   pip install git+https://git.ufz.de/rdm-software/saqc@master


If you feel more adventurous, feel free to use the latest development version from our
`GitLab-repository <https://git.ufz.de/rdm-software/saqc>`_. We try to keep the
develop branch in a workable state, but sill don't make any guarantees here.

.. code-block:: sh

   pip install git+https://git.ufz.de/rdm-software/saqc@develop


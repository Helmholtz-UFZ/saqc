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

The exact process depends on your operating system and python version/distribution.
The following sections should get you started on UNIX-like Systems and Windows.


On Unix/Mac-systems
"""""""""""""""""""

.. code-block:: sh

   # create virtual environment called "env_saqc"
   python3 -m venv env_saqc

   # activate the virtual environment
   source env_saqc/bin/activate

On Windows-systems
""""""""""""""""""

.. code-block:: sh

   # create virtual environment called "env_saqc"
   py -3 -m venv env_saqc

   # move to the Scripts directory in "env_saqc"
   cd env_saqc/Scripts

   # activate the virtual environment
   ./activate


Get SaQC
--------

We currently distribute SaQC via the `Python Package Index (PyPI) <https://pypi.org/>`_
or through our `GitLab-repository <https://git.ufz.de/rdm-software/saqc>`_.

From PyPI
"""""""""

The latest stable versions are available from PyPI:

.. code-block:: sh

   python3 -m pip install saqc

From the Gitlab repository
""""""""""""""""""""""""""

If you feel more adventurous feel free to use the latest development version from our
`GitLab-repository <https://git.ufz.de/rdm-software/saqc>`_. While we try to keep the
develop branch in a workable state, we sill won't make any guarantees here.

.. code-block:: sh

   pip install git+https://git.ufz.de/rdm-software/saqc@develop


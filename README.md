SysBio_Modeling
===============

A simple, small toolbox for estimating parameters in ODE-based models.  The main usecase is to estimate parameters in models where parameters depend heavily on experiment-specific settings.

Features:
---------
 - Assimulo (http://www.jmodelica.org/assimulo, in progress) and SciPy for numerical integration of differential equations.

 - SymPy for automatic derivation of forward sensitivity equations, useful when doing gradient-based optimization.

 - SciPy (http://www.scipy.org/), nlopt (http://ab-initio.mit.edu/wiki/index.php/NLopt), and geo_lsqt (http://arxiv.org/abs/1201.5885) for optimization and parameter estimation.

 - matplotlib for plotting and visualization.

Heavily inspired from sloppycell (Sethna group).

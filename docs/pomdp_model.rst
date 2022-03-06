.. _sec-model:

Observation model
=================

The observation model is based on a physical modeling of dispersion and detection in a turbulent medium proposed in
[Vergassola2007]_ and generalized to an arbitrary number of dimensions in [Loisy2022]_.

Observations ("hits") are drawn randomly according to a Poisson distribution

.. math::
   \begin{equation}
   \text{Pr}(h | {\bf x}^a,{\bf x}^s) = \frac{\mu^h \exp(-\mu)}{h!} \qquad \text{with} \, \mu = \mu(\lVert {\bf x}^s - {\bf x}^a \rVert_2)
   \end{equation}

which mean :math:`\mu` is a function of the Euclidean distance :math:`d=\lVert {\bf x}^s - {\bf x}^a \rVert_2`
between the agent and the source (measured in number of grid cells).

The expression of :math:`\mu(d)` for an arbitrary number of dimensions :math:`n` is

.. math::
   \begin{align}
   & n=1: && \displaystyle \mu(d) = I \frac{2 L}{2 L-1} \exp(-d/L) \\
   & n=2: && \displaystyle \mu(d) = I \frac{1}{\ln(2 L)} K_{0} (d/L) \\
   & n=3: && \displaystyle \mu(d) = I \frac{1}{2 d} \exp(-d/L) \\
   & n=4: && \displaystyle \mu(d) = I \left( \frac{1}{2 L} \right)^{2} \frac{L}{d} K_{1} (d/L)
   \end{align}

and more generally for :math:`n\geqslant 3`

.. math::
   \begin{equation}
   \mu(d) = I \left( \frac{1}{2 L} \right)^{n-2} \left( \frac{L}{d} \right)^{n/2-1} \frac{(n-2)}{\Gamma(n/2)} \frac{K_{n/2-1} (d/L)}{2^{n/2-1}}
   \end{equation}

where
:math:`L` is a dimensionless dispersion lengthscale that determines the size of the search domain,
:math:`I` is a dimensionless source intensity,
:math:`\Gamma` is the gamma function, and
:math:`K_{\nu}` is the modified Bessel function of the second kind of order :math:`\nu`.

In the code

  - `n` is called ``N_DIMS``,
  - :math:`L` is called ``LAMBDA_OVER_DX``,
  - :math:`I` is called ``R_DT``.


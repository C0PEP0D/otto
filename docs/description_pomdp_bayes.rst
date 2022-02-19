.. _sec-bayes:

Bayesian inference
==================

Each observation provides some information about the source location, which can be accounted for using Bayesian inference.
In the case of a sequential process such as the source-tracking POMDP, Bayes' rule can be applied after each observation
to maintain an up-to-date belief :math:`p({\bf x})` which encompasses all information gathered so far.

The update after observing :math:`o_t` in :math:`{\bf x}^a_{t+1}` reads

.. math::
   \begin{equation}
    p_{t+1}({\bf x}) = \text{Bayes}(p_t({\bf x}), {\bf x}^a_{t+1}, o_{t})
   \end{equation}

where :math:`\text{Bayes}(p({\bf x}), {\bf x}^a, o)` is the operator that maps the prior :math:`p_t`
to the posterior :math:`p_{t+1}` through Bayes's rule

.. math::
   \begin{equation}
    \text{Bayes}(p({\bf x}), {\bf x}^a, o) = \frac{\Pr(o | {\bf x}^a,{\bf x}) p({\bf x})}{\sum_{{\bf x}'} \Pr(o | {\bf x}^a,{\bf x}') p({\bf x}')}
   \end{equation}

and where :math:`\Pr(o | {\bf x}^a,{\bf x})` is called the evidence in Bayesian terminology.

Let us now go through the update rule for each observation.

If :math:`o=F`, the source has been found in :math:`{\bf x}^a`, and the posterior distribution is simply a Dirac distribution

.. math::
   \begin{equation}
    \text{Bayes}(p({\bf x}), {\bf x}^a, F) = \delta({\bf x} - {\bf x}^a).
   \end{equation}


Otherwise, :math:`o=(\bar{F}, h)`, meaning that the source has not been found and that :math:`h` hits were perceived.
The posterior distribution after not finding the source is a simple renormalization

.. math::
    \begin{equation}
     \text{Bayes}(p({\bf x}), {\bf x}^a, \bar{F}) =
     \begin{cases}
        0 & \text{if ${\bf x} = {\bf x}^a$,} \\
        \dfrac{p({\bf x})}{\sum_{{\bf x} \neq {\bf x}^a} p({\bf x})} & \text{otherwise.}
    \end{cases}
    \end{equation}

The posterior after a hit :math:`h` is

.. math::
   \begin{equation}
    \text{Bayes}(p({\bf x}), {\bf x}^a, h) = \frac{\Pr(h | {\bf x}^a,{\bf x}) \, p({\bf x})}{\sum_{{\bf x}'} \Pr(h | {\bf x}^a,{\bf x}') \, p({\bf x}')}.
   \end{equation}

The full update after observing :math:`o=(\bar{F}, h)` is therefore given by the successive application of each partial update

.. math::
   \begin{equation}
    \text{Bayes}(p({\bf x}), {\bf x}^a, o) = \text{Bayes} ( \text{Bayes}(p({\bf x}), {\bf x}^a, \bar{F}) , {\bf x}^a, h ).
    \end{equation}

:ref:`sec-stepbystep` shows how Bayesian updates are computed through an example.

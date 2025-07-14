DynODE Framework Documentation
==============================

This documentation is intended to provide a slightly more in-depth overview of
the DynODE framework, which is a flexible and modular framework for
simulating infectious disease dynamics. The framework is designed to be
customizable and extensible, allowing users to easily modify and adapt
it to their specific needs.

For disease specific implementations of this framework, please refer to
`DynODE-Models <https://github.com/cdcent/DynODE-Models>`__. Which may
still be private but may become public in the future.

This documentation aims to cover the following topics:

1. The 4 major DynODE modules

   1. `Configuration <markdown/configuration.html>`__
   2. `Simulation <markdown/simulation.html>`__
   3. `Inference <markdown/inference.html>`__
   4. `Utils <markdown/utils.html>`__

2. The library backend of DynODE

   1. `JAX <markdown/backend-libraries.html#jax>`__
   2. `Chex <markdown/backend-libraries.html#chex>`__
   3. `Numpyro <markdown/backend-libraries.html#numpyro>`__
   4. `Diffrax <markdown/backend-libraries.html#diffrax>`__

Anticpated Applications of models build with DynODE include: - Peak
hospitalization incidence and timing - Impact of a new variant be
(particularly for COVID) - Impact of H1N1/H3N2/B dominance and/or an
antigenic shift - Regional differences in burden - Impact of vaccination
timing/reformulation/increased uptake

After reading this documentation, I encourage you to check out the ``examples/``
directory, which contains a set of example models of increasing
complexity. Slowly building from a simple SIR model, adding seasonality
modules, adding more compartments, and even inference capabilities.

After understanding the examples, I encourage you to write your own and
put a PR up! We can always use more examples and its a good learning
exercise to write your own model using the DynODE framework.

To really jump into the weeds, you can also check out the
`DynODE-Models <https://github.com/cdcent/DynODE-Models>`__ where you
can find disease specific implementations of the DynODE framework. These
models are more complex and may require a deeper understanding of the
framework, but they provide a good reference for how to implement more
advanced features and functionalities.

Contact Us
==========

For code specific questions reach out to:

Tom Hladish (utx5@cdc.gov) or Ben Kok Toh (tjk3@cdc.gov) or Michael
Batista(upi8@cdc.gov)

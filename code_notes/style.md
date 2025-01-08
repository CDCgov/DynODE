# Review of Style Continuity and Guideline Adherence  

DynODE appears to be using the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#style-guide)
style guidelines. This can be a very readable standard, but it is also
important when selecting a style guide that, when used, it is used consistently.
One thing that he developers may want to consider is a CI documentation style
compliance check using `pydocstyle`. This will help catch early deviations from
the standard selected.

In this document, I will highlight a few deviations I see from the standard
that I feel are important to fix and understand the "why" behind the reason the
standard.

## Summary text outside of summary

This typically happens when drafting docstrings and there is more information
necessary to provide about args/parameters, or returns. In numpydocs, summaries
are meant to be short and extended in a `Notes` section. This can also appear
to happen when the other style guidance is not well understood.

### Observed examples

[def _get_upstream_parameters(self) -> dict:](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/abstract_parameters.py#L106-L107)

This is a hidden method, so perhaps style adherence is of lower priority, but
a docstring was provided so it should follow the selected style. Here the
Returns section starts off OK, bu then the return description isn't indented
and contains a lot of text. The summary text is also takes on a non conforming
structure. Say what is does, not what it returns, that is what the `Returns`
section is for. Start sentences with capital letters and end them with periods.
You may also find that `pydocstyle` will throw an error if you don't start a
summary after the `"""` as opposed to adding a return first.

```python
    def _get_upstream_parameters(self) -> dict:
        """
        returns a dictionary containing self.UPSTREAM_PARAMETERS, erroring if any of the parameters
        within are not found within self.config.

        Samples any parameters which are of type(numpyro.distribution).

        Returns
        ------------
        dict[str: Any]

        returns a dictionary where keys map to parameters within self.UPSTREAM_PARAMETERS and the values
        are the value of that parameter within self.config, distributions are sampled and replaced
        with a jax ArrayLike representing that value in the JIT compilation scheme used by jax/numpyro.
        """
        # multiple chains of MCMC calling get_parameters()
        # should not share references, deep copy, GH issue for this created
        freeze_params = copy.deepcopy(self.config)
        ...
```

[def generate_downstream_parameters(self, parameters: dict) -> dict:](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/abstract_parameters.py#L139)
If the first line is a sentence, capitalize it. Also, `Raises` gets it's own
section.

[def vaccination_rate(self, t: ArrayLike) -> jax.Array:](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/abstract_parameters.py#L324)
Many linters will complain if your line lengths don't stay to some maximum.
the PEP 8 max docstring line length is 72. This can be configured in linters,
but all files should comply to the max defined in the linter configuration.

["""_summary_](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/dynode_runner.py#L401)
Summary missing.

## Examples not following the python standard

This is important for the following reasons:

 - New users learning a codebase for the first time will learn faster if they
 are not forced to parse a new example syntax for each piece of functional code
 they come across.
 - Following the proper example structure allows for the future use of doctests.
 This is whn you can configure your dev testing environment to run examples as
 unit tests.
 - Your docstrings will not pass the style compliance test and be a headache
 to fix.
 - It will help you think more like a software developer and discover
 anti-patterns before you have api lock-in into something you regret.

### Observed examples

[def seasonal_vaccination_reset(self, t: ArrayLike) -> ArrayLike:](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/abstract_parameters.py#L608)

This is clearly not the structure described in
[the style guide](https://numpydoc.readthedocs.io/en/latest/format.html#examples).
If you can't follow this the structure in the style guide you may be in one of
the following situations:

 - This function/method is not meant for public exposure.
 - The patterns you have used make this code inaccessible as an api.

```python
        """
        if model implements seasonal vaccination, returns evaluation of
        a continuously differentiable function at time `t` to outflow
        individuals from the top most vaccination bin (functionally the
        seasonal tier) into the second highest bin.

        Example
        ----------
        if self.config.SEASONAL_VACCINATION == True

        at `t=utils.date_to_sim_day(self.config.VACCINATION_SEASON_CHANGE)`
        returns 1 else returns near 0 for t far from
        self.config.VACCINATION_SEASON_CHANGE.

        This value of 1 is used by model ODES to outflow individuals
        from the top vaccination bin into the one below it, indicating a
        new vaccination season.
        """
```

[def all_immune_states_without(strain: int, num_strains: int):](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/utils.py#L436)
Wrong example structure.

[def combined_strains_mapping(](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/utils.py#L498) Same, and
why is Examples before Parameters and Returns?

[def convert_hist(strains: str, STRAIN_IDX: IntEnum) -> int:](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/utils.py#L791) Same.


## Other observations

[def load_initial_state(](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/covid_sero_initializer.py#L55)
`Requires` is not a section described in the numpydoc guidelines

[Raises](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/mechanistic_inferer.py#L179)
Places where optional sections are used but incomplete or wrong (e.g., `Raises`
and one error described, but not all other errors that can arise). In this
example, and `AssertionError` can also occur, but is not listed.

[NOTE](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/mechanistic_runner.py#L53)
`Notes`, not `Note`

[def convert_strain(strain: str, STRAIN_IDX: IntEnum) -> int:](https://github.com/CDCgov/DynODE/blob/36908651183ea27e3b8816cdf79da9fa0ee515a8/src/dynode/utils.py#L821)
Even simple utils need attention to detail. The `Returns` section here is not
structured quite right. it also is an example of when to use CAPITALIZED
assignments and when not to. There is a whole world to dive into with *Naming*
and DynODE shows some inconsistencies with best these best practices. One
reason to adopt the Google Python style guide is that there are excellent
resources to these deep topics in that style guide, where much is missing from
the numpydocs style guide, leaving the user to go digging elsewhere.
[ref](https://google.github.io/styleguide/pyguide.html#316-naming)

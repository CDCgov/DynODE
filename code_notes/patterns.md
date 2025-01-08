# Patterns

In this document we will explore patterns and anti-patterns that DynODE is
using. As with most modeling code, defining (hyper-)parameters, data
verification, data manipulation, data validation, and model version tracking
are activities outside of the actual model definition and operation but take
up an outsized portion of time to define and maintain. Because many data
scientists come from an academic background where math and statistical rigor
are places at the highest value level, most of us view the model as the
focal point of our code. Unfortunately the real world outside of a mathematical
construct is much messier. For this reason, most modeling code is written
inside out, where the model is defined with good abstraction and adherence
to uniform styling and mathematical patterns, and everything around the model
is an afterthought meant only to swat the bugs away from making their way to
the precious model. I am not saying that this exactly describes DynODE, but it
is worth some reflection.

There are several useful patterns to follow for resilient and maintainable ETL
code, and data verification and validation. What must be done for many modeling
codebases after a model is developed, is to put a much higher value on all the
software surrounding the model. This is an art and a practice that we develop
over time.

## ETL

DynODE appears to parameterize and operate a great deal of ETL and data
manipulation via a combination of custom configuration files and very specific
utilities. Abstraction is your friend with this type of work.

The `utils.py` file is where most of this abstraction can occur. For example:
There are many functions that appear to be thin wrappers around the datetime
library. First it is worth mentioning that this pattern makes for a crowded
namespace with a lot of arbitrary function names. This will only be useful for
code authors. There are a few ways to handle something like this: 1) abandon
these helpers and just use the datetime library in your code; 2) create a 
`SimDatetime` Class and these helpers become methods to the object. It
appears that there are probably a few variables (e.g., `start_time`)
and operations performed on those ona regular basis.

Another observed patter in the `utils.py` file is where a function is design
for one specific feature/independent variable being parsed from some input data.

[generate_yearly_age_bins_from_limits](https://github.com/CDCgov/DynODE/blob/809de958aee5ed4c08d772a3c13daddbf8ded1fd/src/dynode/utils.py#L1111)
```python
def generate_yearly_age_bins_from_limits(age_limits: list) -> list[list[int]]:
    """
    given age limits, generates age bins with each year contained in that bin up to 85 years old exclusive

    Example
    ----------
    age_limits = [0, 5, 10, 15 ... 80]
    returns [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]... [80, 81, 82, 83, 84]]

    Parameters
    ----------
    age_limits: list(int):
        beginning with minimum age inclusive, boundary of each age bin exclusive. Not including last age bin.
        do not include implicit 85 in age_limits, this function appends that bin automatically.
    """
    age_groups = []
    for age_idx in range(1, len(age_limits)):
        age_groups.append(
            list(range(age_limits[age_idx - 1], age_limits[age_idx]))
        )
    age_groups.append(list(range(age_limits[-1], 85)))
    return age_groups
```

Might be better defined as:

```python
def split_range(start: int, stop: int, breaks: List[int]) -> List[List[int]]:
    """Creates a nested list of integers given a start, stop, and breaks.

    Parameters
    ----------
    start: int
        The starting integer value (inclusive) to the initial range.
    stop: int
        The range maximum (exclusive) integer value.
    breaks: List[int]
        The breaks to make in the initial range to create the nested
        list
    
    Returns
    -------
    List[List[int]]
        The split_range nested list
    
    Raises
    ------
    ValueError
        When breaks are not all within the initial range

    Examples
    --------
    >>> split_range(0, 12, [5, 7])
    [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11]]

    >>> split_range(0, 12, [0, 5, 7])
    [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11]]
    """
    from itertools import islice # python standard library
    if max(breaks) > stop or min(breaks) < start:
        raise ValueError('breaks must be within the start and stop range')
    if not start in breaks: breaks = [start] + breaks
    if not stop in breaks: breaks = [stop] + breaks
    breaks = sorted(list(set(breaks))) # ensure no repeats and sorted
    return [
        [*islice(range(start, stop), breaks[bi], breaks[bi + 1])]
        for bi in range(len(breaks)-1)
    ]
```
This example shows a general purpose utility for creating split integer ranges,
doesn't pre set a max value, has the right format for examples in the doc string,
and can easily be later used with more specificity in a class object. When
possible the standard library with list comprehension. The function
`generate_yearly_age_bins_from_limits` also silently fails if age limits
outside of 85 are used.

## Configuration

DynODE uses configurations as an approach to parameterize models. This is
a common place practice in Model DevOps. What is noticeable is that the
configurations for DynODE is that these configurations can get rather large
and lack a formal validation schema defined in the code. Since Python 3.10
case matching has been made possible for this as a native approach in python
but I also like [json-schema](https://json-schema.org/) for this for a lot of
reasons.

Configurations should also not live in code. For more on this see the following
model devops diagram:

Lastly, CAPITALIZATION in Python means some specific things and I believe it is
being misused for the config files and in much of the code. See the
[style document](./style.md) for more on this.

## Metadata

Currently only open questions relative to DynODE:

 - How is it captured?
 - Where is it stored?
 - What type of QA is being performed with it? 
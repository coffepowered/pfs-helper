# Packaging (temp notes)
- Writing [pyproject.toml](https://martin-thoma.com/pyproject-toml/)
- specifying [dependencies](https://www.python.org/dev/peps/pep-0631/)

 
Now run this command from the same directory where pyproject.toml is located:

> python3 -m build

To upload:
> python3 -m twine upload dist/*

# Developing
`.devcontainer` folder is provided for convenience.

## Notes on type hints
* [Typechecking with a Python Library That Has No Type Hints](https://skeptric.com/python-type-stubs/)

> mypy program.py

This command makes mypy type check your program.py file and print out any errors it finds. Mypy will type check your code statically: this means that it will check for errors without ever running your code, just like a linter.



# TODO
For better documentation 

- add a figure like this: https://www.researchgate.net/publication/327905081_Relative_Importance_of_Climatic_and_Anthropogenic_Drivers_on_the_Dynamics_of_Aboveground_Biomass_across_Agro-Ecological_Zones_on_the_Mongolian_Plateau/figures?lo=1
- link something like this: https://towardsdatascience.com/a-guide-to-panel-data-regression-theoretics-and-implementation-with-python-4c84c5055cf8
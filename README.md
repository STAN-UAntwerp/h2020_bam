# H2020-BAM

## Poetry

Dependencies in this project are managed by poetry.
If you don't have poetry installed yet, you can follow the [documentation](https://python-poetry.org/docs/) for intstallation.
You can check whether the installion was successful by running `poetry --version`.

Unfortunately, poetry requires you to have python pre-installed somewhere. In order to get started you can follow these steps:
1. Create a new environment with the required python version (see [pyproject.toml](/pyproject.toml))
   This is easiest with conda: `conda create --name python310 python=3.10`
   Alternatively, you can download the required python version from [here](https://www.python.org/downloads/).
2. Make sure you're in the project repository and run `poetry env use <path_to_python>`
   If you got the required python version using conda, `<path_to_python>` is `<path_to_conda_env>/bin/python`.
3. You should be able to run `poetry install` now, which will install all dependencies in a new poetry managed virtual environment.
4. If you want to run any code from this repository, prepend your command with `poetry run`, e.g. `poetry run python main.py`, 
   Alternatively, you can also check which environment was created by poetry by running `poetry env info` and activating this environment with `source <path_to_environment>/bin/activate`

To make a jupyter kernel for this environment (for jupyter notebooks), run the following command:
`poetry run python -m ipykernel install --user --name BAM --display-name "BAM (Python 3.10)"`
You can check installed kernels with `jupyter kernelspec list` (or just open a notebook and check available kernels)


## Running the script

[notebooks/target_prediction.ipynb](notebooks/target_prediction.ipynb) contains the script to train all available estimators 
and write evaluation metrics to the output folder.

[notebooks/calculate_shap.ipynb](notebooks/calculate_shap.ipynb) contains the script to calculate the shap values of the 
previously saved models and stores them in the ouput folder.


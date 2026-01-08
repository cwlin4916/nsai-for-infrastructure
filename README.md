# `nsai_experiments`
Here, we present a de novo, extensible, modular, research-oriented, single-player reimplementation of the "AlphaZero" class of algorithms originally described in [1]. We also present a novel task called the Zoning Game, designed as a toy problem to test the applicability of various AlphaZero-based paradigms to infrastructure planning problems, along with several other games. We implement specializations of AlphaZero to play several games. We also include code that serves as a starting point to explore language generation with AlphaZero, as well as documentation and other related materials.

## Installation for Development
After cloning the repository, create a virtual environment in the repository root, activate it, and install the project in editable mode with its development dependencies:
```bash
python -m venv .venv  # using Python >= 3.9, ideally 3.12
source .venv/bin/activate
pip install -e '.[dev]'
```

If you have had to install an old version of PyTorch (e.g., 2.2), you may need to manually downgrade to NumPy 1:
```bash
pip install 'numpy<2'
```

If possible, prefer a newer version of PyTorch (e.g., >= 2.6).

## Running Tests
Simply invoke `pytest`:
```bash
python -m pytest
```

## Inspiration
Our single-player AlphaZero implementation is entirely novel; however, we were particularly inspired by these existing implementations:

  * Surag Nair, ["A Simple Alpha(Go) Zero Tutorial"](https://suragnair.github.io/posts/alphazero.html).
  * Thomas Moerland, ["A Single-Player Alpha Zero Implementation in 250 Lines of Python"](https://tmoer.github.io/AlphaZero/). Certain code from this implementation remains in our repository as we leveraged it to help develop several of our "games," though this is not incorporated into our core AlphaZero implementation.

## References
[1] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G., Graepel, T., & Hassabis, D. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354–359. https://doi.org/10.1038/nature24270

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

## Demos
A few entry points to quickly get a feel for what this codebase can do:

  * `python -m nsai_experiments.general_az_1p.cartpole.cartpole_easy_demo`: run a few iterations of single-player AlphaZero on [the Cart Pole game](https://gymnasium.farama.org/environments/classic_control/cart_pole/), declaring victory if the pole is still up after 100 moves. You should observe that (a) rewards generally increase over iterations, quickly approaching 1 for this relatively easy game; (b) "new" rewards are generally greater than "old" rewards; and (c) "network+MCTS" rewards are generally greater than "bare network" rewards.
  * `python -m nsai_experiments.general_az_1p.zoning_game.zgaz_easy_demo`: run some iterations of single-player AlphaZero on our novel Zoning Game. It's a little noisy here (see `zgaz_demo.py` for a version that does more self-play per neural network training session), and 30 iterations does not begin to approach peak performance, but if you take a moving average you should see definite upwards progress. This demo takes about 27 minutes on a system with 64 AMD Genoa CPU cores and 1 NVIDIA H100 GPU.

## Showcase
By running `python -m nsai_experiments.general_az_1p.zoning_game.zgaz_demo` for 800 iterations, this is obtained:

![Zoning Game AlphaZero 800 Iterations](/src/nsai_experiments/general_az_1p/zoning_game/zgaz_perf_analysis_run_800.png)

## Inspiration
Our single-player AlphaZero implementation is entirely novel; however, we were particularly inspired by these existing implementations:

  * Surag Nair, ["A Simple Alpha(Go) Zero Tutorial"](https://suragnair.github.io/posts/alphazero.html).
  * Thomas Moerland, ["A Single-Player Alpha Zero Implementation in 250 Lines of Python"](https://tmoer.github.io/AlphaZero/). Certain code from this implementation remains in our repository as we leveraged it to help develop several of our "games," though this is not incorporated into our core AlphaZero implementation.

## References
[1] Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G., Graepel, T., & Hassabis, D. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354–359. https://doi.org/10.1038/nature24270

## Citation
This codebase was authored by Gabriel Konar-Steenberg and Peter Graf and is published by the Alliance for Energy Innovation, LLC, which operates the US Department of Energy's National Laboratory of the Rockies, under software record SWR-26-016. It has been assigned the following DOI: [https://doi.org/10.11578/dc.20260108.1](https://doi.org/10.11578/dc.20260108.1).

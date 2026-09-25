from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


@pytest.fixture()
def world(cfg):
    from hpg.tools import World

    return World.from_config(cfg, ROOT)


@pytest.fixture(scope="session")
def graph(cfg):
    from hpg.graph_loader import load_graph

    return load_graph(ROOT / cfg["paths"]["graph"])

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# Import types to parse configs
from config.config_types import AllConfig

config_store = ConfigStore.instance()
config_store.store(name="config", node=AllConfig)

# Use Compose API to be able to use both fire and hydra
with initialize(version_base="1.2", config_path="."):
    all_config = compose(config_name="all_config")


def main():
    print(OmegaConf.to_yaml(all_config))


if __name__ == "__main__":
    main()

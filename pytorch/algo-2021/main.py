import os

import comet_ml
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

from module import module_dict
from hydra.utils import get_original_cwd


@hydra.main(config_path="./config", config_name="config")
def main(config):
    # !! hydra will change the working directory, so change it back to original one
    os.chdir(get_original_cwd())

    assert config.model in module_dict.keys()
    print(f"[INFO] Use '{config.model}' model.")

    trainer_config = dict(config.trainer)
    trainer_config["default_root_dir"] = os.path.join(
        trainer_config["default_root_dir"], config.model
    )

    # ------------------
    # run model according to args
    # ------------------
    if config.train:
        comet_config = dict(config.logger)
        comet_config["experiment_name"] = config.model
        comet_logger = CometLogger(**comet_config)
        trainer_config["logger"] = comet_logger

        # delete logger config, or it will be uploaded to comet
        del config.logger

        if config.restore_path:
            model = module_dict[config.model].load_from_checkpoint(
                config.restore_path, config=config
            )
        else:
            model = module_dict[config.model](config)
        if model.callbacks:
            trainer_config["callbacks"] = model.callbacks

        trainer = pl.Trainer(**trainer_config)

        trainer.tune(model)
        trainer.fit(model)

    if config.test:
        assert config.restore_path

        trainer_config["logger"] = None
        trainer = pl.Trainer(**trainer_config)

        model = module_dict[config.model].load_from_checkpoint(
            config.restore_path, config=config
        )
        trainer.test(model)

    if config.report:
        # from sklearn.metrics import classification_report
        # import pandas as pd
        assert config.restore_path

        trainer_config["logger"] = None
        trainer = pl.Trainer(**trainer_config)

        model = module_dict[config.model].load_from_checkpoint(config.restore_path)
        # TODO: save the validation results as report
        trainer.validate(model)


if __name__ == "__main__":
    main()  # notice: `config` will be passed by hydra, no need to specify

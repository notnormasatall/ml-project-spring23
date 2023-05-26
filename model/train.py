import warnings
warnings.filterwarnings('ignore')

from model import *
from dataset import *
import os
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl

WAV_WORK_DIR = "../../../extra_space2/solodzhuk_rodzin/wav/"
MODELS_DIR = "../../../extra_space2/solodzhuk_rodzin/models/"
FILES = ["".join(i.split(".")[:-1]) for i in os.listdir(WAV_WORK_DIR)]


def name_from_config(config: dict, extra_info=""):
    """
    Creates a name for a model given its config.
    """
    id = "".join([str(i) for i in np.random.randint(0, 10, 5)])
    keys = ["model_type", "num_heads", "num_layers"]
    name = "_".join([str(config[keys[0]])] + [f"{k}-{config[k]}" for k in keys[1:]])
    return id + name + "_" + extra_info

def run(train, validation, test):
    # logger
    wandb_logger = WandbLogger(project="ml-project", 
                            name = name_from_config(T5Config.from_json_file("config.json").to_dict(), "experience_model"), save_dir = MODELS_DIR)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min")

    trainer = pl.Trainer(logger=wandb_logger, 
                        max_epochs=50000, 
                        callbacks=[early_stop_callback], 
                        log_every_n_steps=3,
                        accelerator = "gpu",
                        devices=[2],
                        default_root_dir = MODELS_DIR
    )

    model = SimpleModel()
    model.fill_train_data(train[0], train[1], train[2], train[3])
    model.fill_validation_data(validation[0], validation[1], validation[2], validation[3])
    model.fill_test_data(test[0], test[1], test[2], test[3])

    # Train the model
    trainer.fit(model)
    trainer.save_checkpoint(pjoin(MODELS_DIR, 'improved_model.ckpt'))

if __name__=="__main__":
    train_output, train_mask, train_input, train_output_mask = get_data(FILES[:300])
    train = (train_output, train_mask, train_input, train_output_mask)
    print("train data extraction finished")
    val_output, val_mask, val_input, val_output_mask = get_data(FILES[300:400])
    validation = (val_output, val_mask, val_input, val_output_mask)
    print("val data extraction finished")
    test_output, test_mask, test_input, test_output_mask = get_data(FILES[400:420])
    test = (test_output, test_mask, test_input, test_output_mask)
    print("test data extraction finished")
    run(train, validation, test)
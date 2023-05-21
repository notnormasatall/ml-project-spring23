# NN
import lightning.pytorch as pl
import torch
import torch.nn as nn

# T5
from transformers import T5ForConditionalGeneration
from transformers import T5Config
from transformers import get_linear_schedule_with_warmup, AdamW, Adafactor

# Dataset
from data_utils import MyDataset
from torch.utils.data import DataLoader

EPOCHS = 1000


class T5LinearModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        config = T5Config.from_json_file("config.json")
        self.t5_model = T5ForConditionalGeneration(config)
        self.linear = nn.Linear(512, 512)

    def forward(self, inputs_embeds, attention_mask, decoder_input_ids, decoder_attention_mask):

        inputs_embeds = self.linear(inputs_embeds)
        raw_output = self.t5_model(inputs_embeds=inputs_embeds,
                                   attention_mask=attention_mask,

                                   decoder_attention_mask=decoder_attention_mask,
                                   labels=decoder_input_ids
                                   )

        return raw_output.loss, raw_output.logits


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.batch_size = 14
        self.model = T5LinearModel()

    def forward(self, decoder_input_ids, attention_mask, inputs_embeds, decoder_attention_mask):
        x = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                       decoder_attention_mask=decoder_attention_mask, decoder_input_ids=decoder_input_ids)
        return x

    def fill_train_data(self, decoder_input_ids, attention_mask, inputs_embeds, decoder_attention_mask):
        self.train_ds = MyDataset(decoder_input_ids=decoder_input_ids, attention_mask=attention_mask,
                                  inputs_embeds=inputs_embeds, decoder_attention_mask=decoder_attention_mask)

    def fill_validation_data(self, decoder_input_ids, attention_mask, inputs_embeds, decoder_attention_mask):
        self.validation_ds = MyDataset(decoder_input_ids=decoder_input_ids, attention_mask=attention_mask,
                                       inputs_embeds=inputs_embeds, decoder_attention_mask=decoder_attention_mask)

    def fill_test_data(self, decoder_input_ids, attention_mask, inputs_embeds, decoder_attention_mask):
        self.test_ds = MyDataset(decoder_input_ids=decoder_input_ids, attention_mask=attention_mask,
                                 inputs_embeds=inputs_embeds, decoder_attention_mask=decoder_attention_mask)

    def shared_step(self, batch, mode="train"):
        inputs_embeds = batch["inputs_embeds"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        loss, _ = self(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                       decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        self.log(f'{mode}_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = Adafactor(self.model.parameters(),
                              lr=0.001, relative_step=False, warmup_init=False)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=EPOCHS*47)
        return [optimizer], [{"scheduler": scheduler}]

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.validation_ds, batch_size=self.batch_size, num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=24)

    def predict_step(self, batch, batch_idx):
        inputs_embeds = batch["inputs_embeds"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        loss, logits = self(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        pred = torch.argmax(logits, axis=-1)
        return pred


def predict_tokens(trained_model, output, mask, input, output_mask):
    ds = DataLoader(MyDataset(decoder_input_ids=output, attention_mask=mask,
                    inputs_embeds=input, decoder_attention_mask=output_mask))
    trainer = pl.Trainer()
    return trainer.predict(trained_model, ds)

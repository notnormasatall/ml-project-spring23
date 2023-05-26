# NN
import lightning.pytorch as pl
import torch
import torch.nn as nn

# T5
from transformers import T5ForConditionalGeneration
from transformers import T5Config

# Dataset
from data_utils import MyDataset
from torch.utils.data import DataLoader

EPOCHS = 1000

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.batch_size = 14
        config = T5Config.from_json_file("config.json")
        self.t5_model = T5ForConditionalGeneration(config)
        self.linear = nn.Linear(512, 512)

    def forward(self, inputs_embeds, attention_mask, decoder_input_ids, decoder_attention_mask):
        inputs_embeds = self.linear(inputs_embeds)
        output = self.t5_model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, decoder_attention_mask = decoder_attention_mask, labels = decoder_input_ids)
        return output.loss, output.logits

    def fill_train_data(self, decoder_input_ids, attention_mask, inputs_embeds, decoder_attention_mask):
        self.train_ds = MyDataset(decoder_input_ids = decoder_input_ids, attention_mask = attention_mask, inputs_embeds = inputs_embeds, decoder_attention_mask = decoder_attention_mask)

    def fill_validation_data(self, decoder_input_ids, attention_mask, inputs_embeds, decoder_attention_mask):
        self.validation_ds = MyDataset(decoder_input_ids = decoder_input_ids, attention_mask = attention_mask, inputs_embeds = inputs_embeds, decoder_attention_mask = decoder_attention_mask)

    def fill_test_data(self, decoder_input_ids, attention_mask, inputs_embeds, decoder_attention_mask):
        self.test_ds = MyDataset(decoder_input_ids = decoder_input_ids, attention_mask = attention_mask, inputs_embeds = inputs_embeds, decoder_attention_mask = decoder_attention_mask)

    def shared_step(self, batch, mode="train"):
        inputs_embeds = batch["inputs_embeds"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        loss, _ = self(inputs_embeds = inputs_embeds, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask)
        self.log(f'{mode}_loss', loss)
        return loss    
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
        
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.validation_ds, batch_size=self.batch_size, num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=24)
    
    def predict(self, inputs_embeds, attention_mask):
        inputs_embeds = torch.tensor(inputs_embeds)
        attention_mask = torch.tensor(attention_mask)
        torch.manual_seed(0)
        with torch.no_grad():
            inputs_embeds = self.linear(inputs_embeds)
            pred = self.t5_model.generate(inputs_embeds = inputs_embeds, attention_mask = attention_mask, max_length = 512) 
        return pred

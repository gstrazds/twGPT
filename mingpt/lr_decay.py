import math
import pytorch_lightning as pl


class LearningRateDecayCallback(pl.Callback):

    def __init__(self, learning_rate, warmup_tokens=375e6, decay_tokens=260e9, lr_decay=True, pad_value=-1):
        super().__init__()
        self.learning_rate = learning_rate
        self._tokens_seen = 0
        self.lr_decay = lr_decay
        self.pad_value = pad_value
        self.warmup_tokens = warmup_tokens
        self.decay_tokens = decay_tokens

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # NOTE: uses y > pad_value to count tokens processed (assumes pad values are less than any actual token)

        optimizer = trainer.optimizers[0]
        # _, y = batch
        if len(batch) == 3:
            _, y, _unused__pad_len = batch
        else:
            assert len(batch) == 2, "Expecting each training batch to be a tuple of x,y "+int(len(batch))
            _, y = batch

        if self.lr_decay:   # if False, this callback does nothing
            self._tokens_seen += (y > self.pad_value).sum()  # number of tokens processed this step (i.e. label is not -100)
            if self._tokens_seen < self.warmup_tokens:
                # linear warmup
                lr_mult = float(self._tokens_seen) / float(max(1, self.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(self._tokens_seen - self.warmup_tokens) / float(
                    max(1, self.decay_tokens - self.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            pl_module.log('learning_rate', lr, on_step=True, on_epoch=False, prog_bar=True)

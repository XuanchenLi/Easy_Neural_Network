from .trainer import Trainer


class SimpleTrainer(Trainer):
    def __init__(self, *args, **kargs):
        Trainer.__init__(*args, **kargs)

    def variable_weights_init(self):
        pass

    def optimizer_update(self):
        self.optimizer.update()

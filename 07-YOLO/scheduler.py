from torch.optim.lr_scheduler import _LRScheduler

# scheduler according to paper configurations.
# copied from https://github.com/EclipseR33/yolo_v1_pytorch
# 对pytorch的scheduler类还不是很了解，我想还是要再看看

class Scheduler(_LRScheduler):
    def __init__(self, optimizer, step_warm_ep, lr_start, step_1_lr, step_1_ep,
                 step_2_lr, step_2_ep, step_3_lr, step_3_ep, last_epoch=-1):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.step_warm_ep = step_warm_ep
        self.step_1_lr = step_1_lr
        self.step_1_ep = step_1_ep
        self.step_2_lr = step_2_lr
        self.step_2_ep = step_2_ep
        self.step_3_lr = step_3_lr
        self.step_3_ep = step_3_ep
        self.last_epoch = last_epoch

        super(Scheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.lr_start for _ in self.optimizer.param_groups]
        lr = self._compute_lr_from_epoch()

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs

    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.step_warm_ep:
            lr = ((self.step_1_lr - self.lr_start)/self.step_warm_ep) * self.last_epoch + self.lr_start
        elif self.last_epoch < self.step_warm_ep + self.step_1_ep:
            lr = self.step_1_lr
        elif self.last_epoch < self.step_warm_ep + self.step_1_ep + self.step_2_ep:
            lr = self.step_2_lr
        elif self.last_epoch < self.step_warm_ep + self.step_1_ep + self.step_2_ep + self.step_3_ep:
            lr = self.step_3_lr
        else:
            lr = self.step_3_lr
        return lr
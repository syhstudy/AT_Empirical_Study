import torch
class FreeAT:
    """
        实例
        free_at = FreeAT(model=model)
        for i,(trains,labels) in enumerate(train_iter):
            # 正常训练
            outputs = model(trains)
            loss = F.cross_entropy(outputs,labels)
            loss.backward() # 反向传播得到正常的grad
            # 对抗训练
            m = 5
            for _t in range(m):
                free_at.attack(is_first_attack=(_t == 0))# 在embedding上添加对抗扰动, first attack时备份param.data
                if _t != pgd_k - 1:
                    model.zero_grad()
                else:
                    free_at.restore_grad()
                outputs = model(trains)
                loss_adv = F.cross_entropy(outputs,labels)
                loss_adv.backward()# 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            free_at.restore()# 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()
        """

    def __init__(self, model, eps=0.1):
        self.model = model
        self.eps = eps
        self.emb_backup = {}
        self.grad_backup = {}
        self.last_r_at = 0

    def attack(self, emb_name='embedding', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                param.data.add_(self.last_r_at)
                param.data = self.project(name, param.data)
                self.last_r_at = self.last_r_at + self.eps * param.grad.sign()

    def restore(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
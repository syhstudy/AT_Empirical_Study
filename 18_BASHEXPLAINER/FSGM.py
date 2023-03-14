
FGSM
官方实现
class FGSM:
    def __init__(self, model, eps=1):
        self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():

            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                r_at = self.eps * param.grad.sign()
                param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}
实例
fgsm = FGSM(model=model)
for i,(trains,labels) in enumerate(train_iter):
    # 正常训练
    outputs = model(trains)
    loss = F.cross_entropy(outputs,labels)
    loss.backward() # 反向传播得到正常的grad
    # 对抗训练
    fgsm.attack() # 在embedding上添加对抗扰动
    outputs = model(trains)
    loss_adv = F.cross_entropy(outputs,labels)
    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    fgsm.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()


FGM
官方实现
class FGM:
    def __init__(self, model, eps=1):
        self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}
实例
fgm = FGM(model=model)
for i,(trains,labels) in enumerate(train_iter):
    # 正常训练
    outputs = model(trains)
    loss = F.cross_entropy(outputs,labels)
    loss.backward() # 反向传播得到正常的grad
    # 对抗训练
    fgm.attack() # 在embedding上添加对抗扰动
    outputs = model(trains)
    loss_adv = F.cross_entropy(outputs,labels)
    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    fgm.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()


PGD
官方实现
# PGD
class PGD:
    def __init__(self, model, eps=1, alpha=0.3):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='embedding', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

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
实例
pgd = PGD(model=model)
for i,(trains,labels) in enumerate(train_iter):
    # 正常训练
    outputs = model(trains)
    loss = F.cross_entropy(outputs,labels)
    loss.backward() # 反向传播得到正常的grad
    # 对抗训练
    pgd_k = 3
    for _t in range(pgd_k):
        pgd.attack(is_first_attack=(_t == 0))# 在embedding上添加对抗扰动, first attack时备份param.data
        if _t != pgd_k - 1:
            model.zero_grad()
        else:
            pgd.restore_grad()
        outputs = model(trains)
        loss_adv = F.cross_entropy(outputs,labels)
        loss_adv.backward()# 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    pgd.restore()# 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()

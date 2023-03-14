import torch

class FreeLB(object):
    """
    Example
    model =
    loss_fun =
    freelb = FreeLB(loss_fun,adv_K=3,adv_lr=1e-2,adv_init_mag=2e-2)
    for batch_input, batch_label in data:
        inputs = {'input_ids':...,...,'labels':batch_label}
        #freelb.attack中进行了多次loss.backward()
        loss = freelb.attack(model,inputs)
        loss.backward()
        optimizer.step()
        model.zero_grad()
    """

    def __init__(self, loss_fun, adv_K=3, adv_lr=1e-2, adv_init_mag=2e-2, adv_max_norm=0., adv_norm_type='l2',
                 base_model='bert'):
        """
        初始化
        :param loss_fun: 任务适配的损失函数
        :param adv_K: 每次扰动对抗的小步数，最少是1 一般是3
        :param adv_lr: 扰动的学习率1e-2
        :param adv_init_mag: 初始扰动的参数 2e-2
        :param adv_max_norm:0  set to 0 to be unlimited 扰动的大小限制 torch.clamp()等来实现
        :param adv_norm_type: ["l2", "linf"]
        :param base_model: 默认的bert
        """
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag  # adv-training initialize with what magnitude, 即我们用多大的数值初始化delta
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model
        self.loss_fun = loss_fun

    def attack(self, model, inputs, labels, gradient_accumulation_steps=1):
        # model 可以放在初始化中

        input_ids = inputs['input_ids']

        # 得到初始化的embedding
        # 从bert模型中拿出embeddings层中的word_embeddings来进行input_ids到embedding的变换
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)
            # embeds_init = model.encoder.embeddings.word_embeddings(input_ids)

        if self.adv_init_mag > 0:  # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
            # 类型和设备转换
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)  # 扰动初始化

        for astep in range(self.adv_K):
            delta.requires_grad_()
            # bert transformer类模型在输入的时候inputs_embeds 和 input_ids 二选一 不然会报错。。。。。。源码
            inputs['inputs_embeds'] = delta + embeds_init  # 累积一次扰动delta
            inputs['input_ids'] = None

            # 下游任务的模型，我这里在模型输出没有给出loss 要自己计算原始loss
            logits = model(inputs)
            loss = self.loss_fun(logits, labels)
            loss = loss / self.adv_K

            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if astep == self.adv_K - 1:
                # further updates on delta
                break

            delta_grad = delta.grad.clone().detach()  # 备份扰动的grad

            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                         1)  # p='inf',无穷范数，获取绝对值最大者
                denorm = torch.clamp(denorm, min=1e-8)  # 类似np.clip，将数值夹逼到(min, max)之间
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()  # 计算该步的delta，然后累加到原delta值上(梯度上升)
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = getattr(model.module, self.base_model).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(input_ids)

        return loss
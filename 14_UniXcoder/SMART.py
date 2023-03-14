import torch
import torch.nn.functional as F

class SmartPerturbation():
    """
    step_size noise扰动学习率
    epsilon 梯度scale时防止分母为0
    norm_p 梯度scale采用的范式
    noise_var 扰动初始化系数
    loss_map 字典，loss函数的类型{"0":mse(),....}
    使用方法
    optimizer =
    model =
    loss_func =
    loss_map = {"0":loss_fun0,"1":loss_fun1,...}
    smart_adv = SmartPerturbation(model,epsilon,step_size,noise_var,loss_map)
    for batch_input, batch_label in data:
        inputs = {'input_ids':...,...,'labels':batch_label}
        logits = model(**inputs)
        loss = loss_func(logits,batch_label)
        loss_adv = smart_adv.forward(logits,input_ids,token_type_ids,attention_mask,)
        loss = loss + adv_alpha*loss_adv
        loss.backward()
        optimizer.step()
        model.zero_grad()
    """

    def __init__(self,
                 model,
                 epsilon=1e-6,
                 multi_gpu_on=False,
                 step_size=1e-3,
                 noise_var=1e-5,
                 norm_p='inf',
                 k=1,
                 fp16=False,
                 loss_map={},
                 norm_level=0):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.multi_gpu_on = multi_gpu_on
        self.fp16 = fp16
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.model = model
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0

    # 梯度scale
    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == 'l2':
            if sentence_level:
                direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon)
            else:
                direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon)
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction, eff_direction

    # 初始noise扰动
    def generate_noise(self, embed, mask, epsilon=1e-5):
        noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
        noise.detach()
        noise.requires_grad_()
        return noise

    # 对称散度loss
    def stable_kl(self, logit, target, epsilon=1e-6, reduce=True):
        logit = logit.view(-1, logit.size(-1)).float()
        target = target.view(-1, target.size(-1)).float()
        bs = logit.size(0)
        p = F.log_softmax(logit, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
        ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
        if reduce:
            return (p * (rp - ry) * 2).sum() / bs
        else:
            return (p * (rp - ry) * 2).sum()

    # 对抗loss输出
    def forward(self,
                logits,
                input_ids,
                token_type_ids,
                attention_mask,
                task_id=0,
                task_type="Classification",
                pairwise=1):
        # adv training
        assert task_type in set(['Classification', 'Ranking', 'Regression']), 'Donot support {} yet'.format(task_type)
        vat_args = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        # init delta
        embed = self.model(**vat_args)  # embed [B,S,h_dim] h_dim=768
        # embed生成noise
        noise = self.generate_noise(embed, attention_mask, epsilon=self.noise_vaokkr)
        # noise更新K轮
        for step in range(0, self.K):
            vat_args = {'inputs_embeds': embed + noise}
            # noise+embed得到对抗样本的输出logits
            adv_logits = self.model(**vat_args)
            if task_type == 'Regression':
                adv_loss = F.mse_loss(adv_logits, logits.detach(), reduction='sum')
            else:
                if task_type == 'Ranking':
                    adv_logits = adv_logits.view(-1, pairwise)
                adv_loss = self.stable_kl(adv_logits, logits.detach(), reduce=False)

            # 得到noise的梯度
            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return 0
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            # 得到新的scale的noise
            noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()
        vat_args = {'inputs_embeds': embed + noise}
        adv_logits = self.model(**vat_args)
        if task_type == 'Ranking':
            adv_logits = adv_logits.view(-1, pairwise)
        adv_lc = self.loss_map[task_id]
        # 计算对抗样本的对抗损失
        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        return adv_loss
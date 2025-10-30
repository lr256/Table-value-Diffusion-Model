import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 位置嵌入模块（用于时间步编码）
class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb


class ConditionalDiffusion_CycleGAN_tequan_Model(nn.Module):
    def __init__(self, input_dim=25, output_dim=2, hidden_dim=512, time_emb_dim=128):
        super().__init__()
        # 时间步嵌入层
        self.time_embed = nn.Sequential(
            PositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 输入处理层
        self.cond_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 中间处理层
        self.mid_layers = nn.Sequential(
            nn.Linear(hidden_dim + output_dim+ time_emb_dim, hidden_dim +output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim +output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_layer_ni= nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.hallucinator = nn.Sequential(  # 基于特权信息生成亚型
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )


    def forward(self, noise, cond, t,train_falg):#cond代表输入val，noise代表与输出相关的噪声
        # 时间步嵌入t的大小16，1
        t_emb = self.time_embed(t)
        # A-B处理条件输入
        cond_proj = self.cond_proj(cond)
        noise_fake=self.hallucinator(cond_proj)
        # 处理噪声输入
        noise_proj = self.noise_proj(noise)
        noise_fake_proj = self.noise_proj(noise_fake)


        if train_falg==0:#训练阶段

            #A-B
            # 对抗损失计算-特权信息
            validity_real = self.discriminator(noise)
            validity_fake = self.discriminator(noise_fake.detach())  # #
            # 特征融合（真实特权信息）
            x_real = torch.cat([cond_proj, noise, t_emb], dim=-1)
            x_real = self.mid_layers(x_real)
            x_real=self.output_layer(x_real)

            x_fake = torch.cat([cond_proj, noise_fake, t_emb], dim=-1)
            x_fake = self.mid_layers(x_fake)
            x_fake = self.output_layer(x_fake)

            #B-A real
            x_ni_real=torch.cat([noise_proj, noise, t_emb], dim=-1)
            x_ni_real= self.mid_layers(x_ni_real)
            x_ni_real = self.output_layer_ni(x_ni_real)

            # B-A fake
            x_ni_fake = torch.cat([noise_fake_proj, noise_fake, t_emb], dim=-1)
            x_ni_fake = self.mid_layers(x_ni_fake)
            x_ni_fake = self.output_layer_ni(x_ni_fake)


            return {
                'output_real': x_real,
                'output_fake': x_fake,
                'validity_real': validity_real,
                'validity_fake': validity_fake,
                'hallucinated': noise_fake,  # 'cavs':cavs,
                'output_ni_real':x_ni_real,
                'output_ni_fake':x_ni_fake,
            }

        else:
            validity_fake = self.discriminator(noise_fake.detach())  # #

            x_fake = torch.cat([cond_proj, noise_fake, t_emb], dim=-1)
            x_fake = self.mid_layers(x_fake)
            x_fake = self.output_layer(x_fake)

            x_real = torch.cat([cond_proj, noise, t_emb], dim=-1)
            x_real = self.mid_layers(x_real)
            x_real = self.output_layer(x_real)

            return {
                'output_fake': x_fake,#x_fake,
                'validity_fake': validity_fake,
            }



# 扩散过程工具类
class DiffusionProcess(nn.Module):
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02, device='cpu'):

        super().__init__()
        self.T = T
        self.device = device
        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)

        # 计算alpha相关参数
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)

        # 反向扩散参数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas[:-1] * (1. - self.alpha_bars[:-1]) / (1. - self.alpha_bars[1:])
        self.posterior_variance = torch.cat([self.posterior_variance[0:1], self.posterior_variance])



    def forward_diffusion(self, x0, t, noise=None):
        #前向扩散过程
        # 确保索引在范围内
        if noise is None:
            noise = torch.randn_like(x0)
        t = torch.clamp(t, 0, self.T - 1)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def reverse_diffusion(self, model, cond, batch_size=32, clip_value=3.0, progress_bar=False):
        #反向扩散生成样本
        # 从随机噪声开始
        x = torch.randn((batch_size, 2), device=self.device)

        # 逐步去噪
        steps = range(self.T - 1, -1, -1)
        if progress_bar:
            steps = tqdm(steps, desc="Reverse Diffusion")

        for t_step in steps:
            t = torch.full((batch_size,), t_step, device=self.device, dtype=torch.long)
            # 预测噪声
            with torch.no_grad():
                pred_noise = model(x, cond, t)

            # 计算当前时间步的参数
            alpha_t = self.alphas[t].view(-1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1)
            beta_t = self.betas[t].view(-1, 1)

            if t_step > 0:
                noise = torch.randn_like(x)
                sqrt_var = torch.sqrt(self.posterior_variance[t]).view(-1, 1)#self.
            else:
                noise = 0
                sqrt_var = 0

            # 计算去噪结果 - 使用改进的公式
            x = self.sqrt_recip_alphas[t].view(-1, 1) * (x - beta_t * pred_noise / torch.sqrt(1 - alpha_bar_t))
            x = x + sqrt_var * noise

            # 可选：裁剪值以防止极端值
            if clip_value is not None:
                x = torch.clamp(x, -clip_value, clip_value)

        return x

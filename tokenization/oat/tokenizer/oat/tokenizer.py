import torch
import torch.nn.functional as F
from typing import Tuple, Union, List, Optional

from oat.model.common.normalizer import LinearNormalizer 
from oat.tokenizer.base_tokenizer import BaseTokenizer 
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ

def pad_token_seq(token_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """
    Pad the token sequence to the maximum length.

    Args:
        token_ids: The token id sequence of shape [B, l].
        max_seq_len: The maximum sequence length L.

    Returns:
        Padded token id sequence.
    """
    device, dtype = token_ids.device, token_ids.dtype
    pad_len = max_seq_len - token_ids.shape[1]
    pad_seq = torch.zeros((token_ids.shape[0], pad_len), device=device, dtype=dtype)
    return torch.cat([token_ids, pad_seq], dim=1)  # [B, L]


class OATTok(BaseTokenizer):
    def __init__(self,
        encoder: RegisterEncoder,
        decoder: SinglePassDecoder,
        quantizer: FSQ,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.normalizer = LinearNormalizer()
        self.latent_horizon = self.decoder.latent_horizon

        self.vocab_size = self.quantizer.codebook_size


    def get_optimizer(
        self, 
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch):
        """Training forward pass.

        Returns:
            dict with recon_loss, vq_loss, latents (B, T', 256) pre-FSQ,
            codes (B, T') FSQ indices,
            fsq_codes (B, T', fsq_dim) post-round 4-d with STE.
        """
        samples = batch['action']
        action_real_lens = batch.get('action_real_lens', None)

        # normalize
        nsamples = self.normalizer['action'].normalize(samples)

        # encode: get both pre-FSQ (256-d) and post-head (4-d) representations
        pre_fsq = self.encoder._encode_transformer(nsamples)  # (B, T', 256)
        post_head = self.encoder.head(pre_fsq)                # (B, T', 4)

        # quantize
        quant, tokens = self.quantizer(post_head)

        # decode
        recons = self.decoder(quant)

        # Masked recon loss: only on real (non-padded) timesteps
        if action_real_lens is not None:
            B, T, D = nsamples.shape
            mask = torch.arange(T, device=nsamples.device).unsqueeze(0) < action_real_lens.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(nsamples)
            n_real = mask.sum()
            recon_loss = (((recons - nsamples) ** 2) * mask).sum() / n_real if n_real > 0 else F.mse_loss(recons, nsamples)
        else:
            recon_loss = F.mse_loss(recons, nsamples)

        return {
            'recon_loss': recon_loss,
            'vq_loss': torch.tensor(0.0, device=samples.device),
            'latents': pre_fsq,        # 256-d for aux heads
            'codes': tokens.detach(),   # FSQ indices
            'fsq_codes': quant,         # 4-d post-round with STE
        }

    def encode_pre_fsq(self, samples: torch.Tensor) -> torch.Tensor:
        """Return pre-FSQ 256-d register embeddings (before head projection + FSQ).

        Returns (B, num_registers, emb_dim=256) with full gradient flow.
        """
        nsamples = self.normalizer['action'].normalize(samples)
        return self.encoder._encode_transformer(nsamples)

    def encode(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # samples: (B, T, sample_dim)

        # normalize
        nsamples = self.normalizer['action'].normalize(samples)
        # encode & quantize
        latents = self.encoder(nsamples)
        latents, tokens = self.quantizer(latents)
        return latents, tokens

    def decode(self, 
        latents: torch.Tensor, 
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # latents: (B, T', encoder_emb_dim)
        if eval_keep_k is None:
            eval_keep_k = [latents.shape[1]] * latents.shape[0]

        assert all([k <= self.latent_horizon for k in eval_keep_k]), \
            f"All eval_keep_k must be <= {self.latent_horizon}"

        # decode - returns normalized samples
        nsamples = self.decoder(latents, eval_keep_k=eval_keep_k)

        # unnormalize
        samples = self.normalizer['action'].unnormalize(nsamples)
        return samples

    def autoencode(self, 
        samples: torch.Tensor, 
        eval_keep_k: Optional[List[int]] = None, 
    ) -> torch.Tensor:
        # samples: (B, T, sample_dim)
        latents, _ = self.encode(samples)
        recons = self.decode(latents, eval_keep_k=eval_keep_k)
        return recons

    def tokenize(self, samples: torch.Tensor) -> torch.Tensor:
        # samples: (B, T, sample_dim)
        _, tokens = self.encode(samples)
        return tokens

    def detokenize(self, 
        tokens: Union[torch.Tensor, List[List[int]]],
    ) -> torch.Tensor:
        # tokens: (B, T') or list of list of int
        
        # standardize
        if isinstance(tokens, list):
            token_lens = [t.shape[1] for t in tokens]
            tokens = torch.cat([
                pad_token_seq(t, self.latent_horizon)
                for t in tokens
            ], dim=0)
        elif isinstance(tokens, torch.Tensor):
            token_lens = [tokens.shape[1]] * tokens.shape[0]
            if tokens.shape[-1] < self.latent_horizon:
                tokens = pad_token_seq(tokens, self.latent_horizon)
        else:
            raise ValueError(f'Unknown token type {type(tokens)}')

        # codebook lookup & decode
        latents = self.quantizer.indices_to_embedding(tokens)
        samples = self.decode(latents, eval_keep_k=token_lens)
        return samples

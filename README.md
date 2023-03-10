
# XAI Custom Summary Process

![Untitled](https://user-images.githubusercontent.com/76906638/182199451-a657839c-28d5-4b00-8caf-8cba8b9354d4.png)

## Transformer Defalut Model  

PLM 에서 주로 사용되는 Transformer 의 Encoder 및 Decoder 구조 기초 구현 

## Using Xai Module on Cross Attention


Xai Module을 활용하여 중요 감성에 집중한 Seq2Seq 모델로서 
전체 프로세스 3번에서 Encoder Decoder Attention에서 전체 어텐션이 아닌 xai 밸류 기준으로 Attention을 진행 


        class Decoder(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_hidn)
                sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_hidn))
                self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

                self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])

            def forward(self, dec_inputs, enc_inputs, enc_outputs,shap_inputs=None,random_mask=None): # If Using Random masking Module, Encoder input replace Shap input 
                positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
                pos_mask = dec_inputs.eq(self.config.i_pad)
                positions.masked_fill_(pos_mask, 0)

                # (bs, n_dec_seq, d_hidn)
                dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions)

                # (bs, n_dec_seq, n_dec_seq)
                dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
                # (bs, n_dec_seq, n_dec_seq)
                dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
                # (bs, n_dec_seq, n_dec_seq)
                dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_decoder_mask), 0)


                if random_mask:

                    dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, shap_inputs,self.config.i_pad)
                    self_attn_probs, dec_enc_attn_probs = [], []

                    for layer in self.layers:

                        dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
                        self_attn_probs.append(self_attn_prob)
                        dec_enc_attn_probs.append(dec_enc_attn_prob)

                    return dec_outputs, self_attn_probs, dec_enc_attn_probs,dec_enc_attn_mask

                else:


                    dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)
                    self_attn_probs, dec_enc_attn_probs = [], []

                    for layer in self.layers:
                        dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
                        self_attn_probs.append(self_attn_prob)
                        dec_enc_attn_probs.append(dec_enc_attn_prob)

                    return dec_outputs, self_attn_probs, dec_enc_attn_probs,dec_enc_attn_mask



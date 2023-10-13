
def get_weighted_single_eval_pos_sampler(max_len):
    """
    This gives a sampler that can be used for `single_eval_pos` which yields good performance for all positions p,
    where p <= `max_len`. At most `max_len` - 1 examples are shown to the Transformer.
    :return: Sampler that can be fed to `train()` as `single_eval_pos_gen`.
    """
    return lambda: random.choices(range(max_len), [1 / (max_len - i) for i in range(max_len)])[0]


def get_normalized_uniform_encoder(encoder_creator):
    """
    This can be used to wrap an encoder that is fed uniform samples in [0,1] and normalizes these to 0 mean and 1 std.
    For example, it can be used as `encoder_creator = get_normalized_uniform_encoder(encoders.Linear)`, now this can
    be initialized with `encoder_creator(feature_dim, in_dim)`.
    :param encoder:
    :return:
    """
    return lambda in_dim, out_dim: nn.Sequential(Normalize(.5, math.sqrt(1/12)), encoder_creator(in_dim, out_dim))


class ScaledDecoder(nn.Module):
    def __init__(self, ninp, nhid, nout):
        super().__init__()
        self.linear = nn.Linear(ninp, nhid)
        self.linear1 = nn.Linear(nhid, nout)
        self.linear2 = nn.Linear(nhid, 10)

    def forward(self, x):
        #return torch.cat([self.linear1(x), self.linear2(x)], -1)
        x = self.linear(x)
        x = nn.GELU()(x)
        temps = self.linear2(x).softmax(-1) @ torch.tensor([1.,1.4,1.7,2.,5.,10.,20.,40.,80.,160.], device=x.device)
        if random.random() > .99:
            print(temps.shape,temps[:,:2])
        return self.linear1(x) / temps.unsqueeze(-1)

    def get_model(model_generator, config, should_train=True, device='cuda'):
        epochs = 0 if not should_train else config['epochs']

        model = train(priors.pyro.DataLoader
                      , Losses.bce
                      , encoders.Linear
                      , emsize=config['emsize']
                      , nhead=config['nhead']
                      , y_encoder_generator=encoders.Linear
                      , pos_encoder_generator=None
                      , batch_size=config['batch_size']
                      , nlayers=config['nlayers']
                      , nhid=config['emsize'] * config['nhid_factor']
                      , epochs=epochs
                      , warmup_epochs=config['epochs'] // 4
                      , bptt=config['seq_len']
                      , gpu_device=device
                      , dropout=config['dropout']
                      , steps_per_epoch=config['steps_per_epoch']
                      , single_eval_pos_gen=get_weighted_single_eval_pos_sampler(100)
                      , extra_prior_kwargs_dict={
                'num_outputs': config['num_outputs']
                , 'num_features': config['num_features']
                , 'canonical_args': None
                , 'fuse_x_y': False
                , 'model': model_generator
            }
                      , lr=config['lr']
                      , verbose=True)

        return model

def get_default_model_spec(size):
    bptt = 300

    if size == 'big':
        num_features = 8
        embed = 64
        nlayers = 2
    elif size == 'small':
        num_features = 3
        embed = 5
        nlayers = 2
    else:
        num_features = int(size.split("_")[0])
        embed = int(size.split("_")[1])
        nlayers = int(size.split("_")[2])

    return {'nlayers': nlayers, 'embed': embed, 'num_features': num_features, "seq_len": bptt}




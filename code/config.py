class Config(object):
    name = ''
    epoch_num = 50
    batch_size = 16
    step_size = 20
    train_batch_size = batch_size
    train_step_size = step_size
    valid_batch_size = batch_size
    valid_step_size = step_size
    test_batch_size = batch_size
    test_step_size = step_size

    word_embedding_dim = 300
    lstm_layers = 1  # 1
    lstm_size = 512
    lstm_forget_bias = 0.0
    max_grad_norm = 0.25
    max_vocab = 50000
    init_scale = 0.05
    learning_rate = 0.2
    decay = 0.5
    decay_when = 1.0
    dropout_prob = 0.5
    adagrad_eps = 1e-5
    use_embeddings = True
    update_embeddings = True


# Conditioned LM Models
class Seq2SeqConfig(Config):
    def __init__(self):
        super().__init__()


class RSTSeq2SeqConfig(Config):
    def __init__(self):
        super().__init__()
        self.use_crf = True
        self.rst_size = 45


# LM Models
class HRNNConfig(Config):
    def __init__(self):
        super().__init__()
        self.name = 'HRNN'


class RNNConfig(Config):
    def __init__(self):
        super().__init__()
        self.name = 'RNN'


class HREDConfig(Config):
    def __init__(self):
        super().__init__()
        self.name = 'HRED'


class DHRNNConfig(object):
    def __init__(self):
        super().__init__()
        self.name = 'DHRNN'

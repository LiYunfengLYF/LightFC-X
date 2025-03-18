class EnvironmentSettings:
    def __init__(self, env_num):
        if env_num == 101:
            self.workspace_dir = r''
            self.tensorboard_dir = r''
            self.pretrained_networks =r''

            # SOT to tir
            self.got10k_dir = r''
            self.got10k_tir_dir = r''

            # RGBT
            self.lasher_train_dir = r''
            self.lasher_test_dir =r''
            self.rgbt234_dir = r''
        else:
            print(env_num)
            raise f'env_num:{env_num}'

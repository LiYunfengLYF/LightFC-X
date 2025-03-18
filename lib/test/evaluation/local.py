from lib.test.evaluation.environment import EnvSettings


def local_env_settings(env_num):
    settings = EnvSettings()
    if env_num == 0:
        settings.prj_dir = r''
        settings.result_plot_path = r''
        settings.results_path = r''
        settings.save_dir = r''

        settings.lasher_path = r''
        # settings.gtot_path = r''
        # settings.rgbt234_path = r''
    else:

        print(f'env_num not match, got to lib/test/evaluation/local.py to check it. your env_num is:', env_num)
        raise
    return settings

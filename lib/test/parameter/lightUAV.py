from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.lightUAV.config import cfg, update_config_from_file


def parameters(yaml_name: str, run_epoch, pl_produce=False):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # save_dir = prj_dir + "/output_" + yaml_name[-4:]
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/lightUAV/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if pl_produce:
        params.checkpoint = "/home/ysy/PycharmProject/Light-UAV-Track-Dual_0702/pretrained_models/LightUAV_ep0293.pth.tar"
    else:
        # params.checkpoint = os.path.join(save_dir, "checkpoints/train/lightUAV/%s/LightUAV_ep%04d.pth.tar" %
        #                              (yaml_name, run_epoch))
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/lightUAV/%s/LightUAV_extreme_ep%04d.pth.tar" %
                                     (yaml_name, run_epoch))
        # params.checkpoint = "/home/wzq/Light-UAV-Track-Dual_0702_prompt/output2/checkpoints/train/lightUAV/vit_256_ep300_rainy/LightUAV_extreme_ep0300.pth.tar"
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params

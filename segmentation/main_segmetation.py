import os
from pprint import pformat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from stardist.models import StarDist2D

from hil.utils.general_helper import configure_paths, load_config, create_directory
from hil.utils.logger import Logger
from segmentation.core.seg_logic import SegProcessorLogic


def main():
    configure_paths()

    # Load experiment configuration from config file
    cfg_path = "../configs/segmentation_cfg.yml"
    config = load_config(cfg_path)

    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    save_path = os.path.join(config["general_paths"]["path_to_results_folder"], f"{config['case']}_results")
    create_directory(save_path)

    # Setup logger
    log_file_path = os.path.join(save_path, "Segmentation_logfile.log")
    logger = Logger(log_file_path=log_file_path)
    logger.info(f"Logger configured to write to {log_file_path}.")
    logger.debug(f"Initializing SegProcessorLogic with config: {pformat(config)}")

    segmentation_processor = SegProcessorLogic(
        case=config["case"],
        cartridge_path=config["general_paths"]["path_to_cartridge_folder"],
        save_path=save_path,
        model=model,
        img_size=config["cropped_img_parameter"]["img_size"],
        logger=logger
    )

    # Segment cells
    segmentation_processor.start_segmentation()


if __name__ == '__main__':
    main()

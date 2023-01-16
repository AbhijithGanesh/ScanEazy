import cv2
import os
from api import constants
from api.config import DEFAULT_CONFIG as config
from api.logger import logger
from api.core.utilities import (
    ImageUtils,
    MainOperations,
)
from .augmentors.manager import ProcessorManager
from .template import Organizer
from pathlib import Path

PROCESSOR_MANAGER = ProcessorManager()


def process_omr(template, omr_resp):
    csv_resp = {}
    unmarked_symbol = ""
    for q_no, resp_keys in template.concatenations.items():
        csv_resp[q_no] = "".join([omr_resp.get(k, unmarked_symbol) for k in resp_keys])
    for q_no in template.singles:
        csv_resp[q_no] = omr_resp.get(q_no, unmarked_symbol)
    return csv_resp


def process_file(file_path: str, json_obj: dict, args) -> tuple:
    curr_dir = Path(os.path.dirname(Path(file_path)))
    template = Organizer(json_obj, curr_dir, PROCESSOR_MANAGER.processors)

    in_omr = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    for i in range(ImageUtils.save_image_level):
        ImageUtils.reset_save_img(i + 1)
    ImageUtils.append_save_img(1, in_omr)
    in_omr = ImageUtils.resize_util(
        in_omr,
        config.dimensions.processing_width,
        config.dimensions.processing_height,
    )
    for pre_processor in template.pre_processors:
        in_omr = pre_processor.apply_filter(in_omr, args)

    if in_omr is None:
        logger.critical("No OMR File detected")

    if args.get("autoAlign") == None:
        args.update({"autoAlign": False})

    (
        response_dict,
        final_marked,
        multi_marked,
        multi_roll,
    ) = MainOperations.read_response(
        template,
        image=in_omr,
        name="test",
        auto_align=args.get("autoAlign"),
    )
    resp = process_omr(template, response_dict)
    return (resp, multi_marked)

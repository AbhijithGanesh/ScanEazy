import numpy as np
from api.logger import logger
import json
from api.core.file import validate_json
from api.core.merger_object import OVERRIDE_MERGER
from api.constants import SCHEMA_DEFAULTS_PATH, QTYPE_DATA


def parse_json(json_obj, template_path=SCHEMA_DEFAULTS_PATH):
    validate, _msg = validate_json(json_obj, template_path)

    if validate:
        logger.info(_msg)
        return json.dumps(json_obj, sort_keys=True, ensure_ascii=True)
    else:
        logger.critical(f"Something critically went wrong with parsing {_msg}")
        return Exception("Invalid JSON")


class Point:
    def __init__(self, pt, q_no, q_type, val):
        self.x = round(pt[0])
        self.y = round(pt[1])
        self.q_no = q_no
        self.q_type = q_type
        self.val = val


class QuestionBlock:
    def __init__(self, dimensions, key, orig, traverse_pts, empty_val):
        self.dimensions = tuple(round(x) for x in dimensions)
        self.key = key
        self.orig = orig
        self.traverse_pts = traverse_pts
        self.empty_val = empty_val
        self.shift = 0


class Organizer:
    def __init__(self, json_obj: dict, template_path: str, extensions):
        if (validate_json(json_obj, template_path)[0]):
            json_obj = json_obj
        self.path = template_path
        self.q_blocks = []
        self.dimensions = json_obj.get("dimensions")
        self.global_empty_val = json_obj.get("emptyVal")
        self.bubble_dimensions = json_obj.get("bubbleDimensions")
        self.concatenations = json_obj.get("concatenations")
        self.singles = json_obj.get("singles")

        if "qTypes" in json_obj:
            QTYPE_DATA.update(json_obj["qTypes"])

        self.pre_processors = [
            extensions[p["name"]](p["options"], template_path.parent)
            for p in json_obj.get("preProcessors", [])
        ]

        self.options = json_obj.get("options", {})

        for name, block in json_obj["qBlocks"].items():
            self.add_q_blocks(name, block)

    def add_q_blocks(self, key, rect):
        assert self.bubble_dimensions != [-1, -1]
        # For q_type defined in q_blocks
        if "qType" in rect:
            rect.update(**QTYPE_DATA[rect["qType"]])
        else:
            rect.update(**{"vals": rect["vals"], "orient": rect["orient"]})

        self.q_blocks += generate_grid(
            self.bubble_dimensions, self.global_empty_val, key, rect
        )

    def __str__(self):
        return str(self.path)


def generate_question_block(
    bubble_dimensions,
    q_block_dims,
    key,
    orig,
    q_nos,
    gaps,
    vals,
    q_type,
    orient,
    col_orient,
    empty_val,
):
    _h, _v = (0, 1) if (orient == "H") else (1, 0)
    traverse_pts = []
    o = [float(i) for i in orig]

    if col_orient == orient:
        for (q, _) in enumerate(q_nos):
            pt = o.copy()
            pts = []
            for (v, _) in enumerate(vals):
                pts.append(Point(pt.copy(), q_nos[q], q_type, vals[v]))
                pt[_h] += gaps[_h]
            # For diagonal endpoint of QuestionBlock
            pt[_h] += bubble_dimensions[_h] - gaps[_h]
            pt[_v] += bubble_dimensions[_v]
            # TODO- make a mini object for this
            traverse_pts.append(([o.copy(), pt.copy()], pts))
            o[_v] += gaps[_v]
    else:
        for (v, _) in enumerate(vals):
            pt = o.copy()
            pts = []
            for (q, _) in enumerate(q_nos):
                pts.append(Point(pt.copy(), q_nos[q], q_type, vals[v]))
                pt[_v] += gaps[_v]
            # For diagonal endpoint of QuestionBlock
            pt[_v] += bubble_dimensions[_v] - gaps[_v]
            pt[_h] += bubble_dimensions[_h]
            traverse_pts.append(([o.coOVERRIDE_MERGERpy(), pt.copy()], pts))
            o[_h] += gaps[_h]
    return QuestionBlock(q_block_dims, key, orig, traverse_pts, empty_val)


def generate_grid(bubble_dimensions, global_empty_val, key, rectParams):
    rect = OVERRIDE_MERGER.merge(
        {"orient": "V", "col_orient": "V", "emptyVal": global_empty_val}, rectParams
    )
    (q_type, orig, big_gaps, gaps, q_nos, vals, orient, col_orient, empty_val) = map(
        rect.get,
        [
            "qType",
            "orig",
            "bigGaps",
            "gaps",
            "qNos",
            "vals",
            "orient",
            "col_orient",  # todo: consume this
            "emptyVal",
        ],
    )

    grid_data = np.array(q_nos)
    if 0 and len(grid_data.shape) != 3 or grid_data.size == 0:
        logger.error(
            "Error(generate_grid): Invalid q_nos array given:",
            grid_data.shape,
            grid_data,
        )
        exit(32)

    orig = np.array(orig)

    num_qs_max = max([max([len(qb) for qb in row]) for row in grid_data])

    num_dims = [num_qs_max, len(vals)]

    q_blocks = []

    _h, _v = (0, 1) if (orient == "H") else (1, 0)
    q_start = orig.copy()
    orig_gap = [0, 0]
    for row in grid_data:
        q_start[_v] = orig[_v]
        for q_tuple in row:
            num_dims[0] = len(q_tuple)
            orig_gap[0] = big_gaps[0] + (num_dims[_v] - 1) * gaps[_h]
            orig_gap[1] = big_gaps[1] + (num_dims[_h] - 1) * gaps[_v]
            q_block_dims = [
                gaps[0] * (num_dims[_v] - 1) + bubble_dimensions[_h],
                gaps[1] * (num_dims[_h] - 1) + bubble_dimensions[_v],
            ]
            q_blocks.append(
                generate_question_block(
                    bubble_dimensions,
                    q_block_dims,
                    key,
                    q_start.copy(),
                    q_tuple,
                    gaps,
                    vals,
                    q_type,
                    orient,
                    col_orient,
                    empty_val,
                )
            )
            q_start[_v] += orig_gap[_v]
        q_start[_h] += orig_gap[_h]
    return q_blocks

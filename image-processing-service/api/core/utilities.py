import os
from dataclasses import dataclass
import cv2
import numpy as np
import api.constants as constants
from api.config import DEFAULT_CONFIG as config
from api.logger import logger


@dataclass
class ImageMetrics:
    resetpos = [0, 0]
    window_x, window_y = 0, 0
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))


class ImageUtils:
    save_image_level = config.outputs.save_image_level
    save_img_list = {}

    @staticmethod
    def reset_save_img(key):
        ImageUtils.save_img_list[key] = []

    # TODO: why is this static
    @staticmethod
    def append_save_img(key, img):
        if ImageUtils.save_image_level >= int(key):
            if key not in ImageUtils.save_img_list:
                ImageUtils.save_img_list[key] = []
            ImageUtils.save_img_list[key].append(img.copy())

    @staticmethod
    def save_img(path, final_marked):
        logger.info("Saving Image to " + path)
        cv2.imwrite(path, final_marked)

    @staticmethod
    def save_or_show_stacks(key, filename, save_dir=None, pause=1):
        if (
            ImageUtils.save_image_level >= int(key)
            and ImageUtils.save_img_list[key] != []
        ):
            name = os.path.splitext(filename)[0]
            result = np.hstack(
                tuple(
                    [
                        ImageUtils.resize_util_h(img, config.dimensions.display_height)
                        for img in ImageUtils.save_img_list[key]
                    ]
                )
            )
            result = ImageUtils.resize_util(
                result,
                min(
                    len(ImageUtils.save_img_list[key])
                    * config.dimensions.display_width
                    // 3,
                    int(config.dimensions.display_width * 2.5),
                ),
            )
            if save_dir is not None:
                ImageUtils.save_img(
                    save_dir + "stack/" + name + "_" + str(key) + "_stack.jpg", result
                )
            else:
                pass

    @staticmethod
    def resize_util(img, u_width, u_height=None):
        if u_height is None:
            h, w = img.shape[:2]
            u_height = int(h * u_width / w)
        return cv2.resize(img, (int(u_width), int(u_height)))

    @staticmethod
    def resize_util_h(img, u_height, u_width=None):
        if u_width is None:
            h, w = img.shape[:2]
            u_width = int(w * u_height / h)
        return cv2.resize(img, (int(u_width), int(u_height)))

    @staticmethod
    def grab_contours(cnts) -> list:
        """
        As an author I have little clue about contours, this is super confusing
        """
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        else:
            raise Exception(
                (
                    "Contours tuple must have length 2 or 3, "
                    "otherwise OpenCV changed their cv2.findContours return "
                    "signature yet again. Refer to OpenCV's documentation "
                    "in that case"
                )
            )
        return cnts


def normalize_util(img, alpha=0, beta=255):
    return cv2.normalize(img, alpha, beta, norm_type=cv2.NORM_MINMAX)


def draw_template_layout(img, template, shifted=True, draw_qvals=False, border=-1):
    img = ImageUtils.resize_util(img, template.dimensions[0], template.dimensions[1])
    final_align = img.copy()
    box_w, box_h = template.bubble_dimensions
    for q_block in template.q_blocks:
        s, d = q_block.orig, q_block.dimensions
        shift = q_block.shift
        if shifted:
            cv2.rectangle(
                final_align,
                (s[0] + shift, s[1]),
                (s[0] + shift + d[0], s[1] + d[1]),
                constants.CLR_BLACK,
                3,
            )
        else:
            cv2.rectangle(
                final_align,
                (s[0], s[1]),
                (s[0] + d[0], s[1] + d[1]),
                constants.CLR_BLACK,
                3,
            )
        for _, qbox_pts in q_block.traverse_pts:
            for Point in qbox_pts:
                x, y = (
                    (Point.x + q_block.shift, Point.y)
                    if shifted
                    else (Point.x, Point.y)
                )
                cv2.rectangle(
                    final_align,
                    (int(x + box_w / 10), int(y + box_h / 10)),
                    (int(x + box_w - box_w / 10), int(y + box_h - box_h / 10)),
                    constants.CLR_GRAY,
                    border,
                )
                if draw_qvals:
                    rect = [y, y + box_h, x, x + box_w]
                    cv2.putText(
                        final_align,
                        "%d" % (cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0]),
                        (rect[2] + 2, rect[0] + (box_h * 2) // 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        constants.CLR_BLACK,
                        2,
                    )
        if shifted:
            text_in_px = cv2.getTextSize(
                q_block.key, cv2.FONT_HERSHEY_SIMPLEX, constants.TEXT_SIZE, 4
            )
            cv2.putText(
                final_align,
                "%s" % (q_block.key),
                (int(s[0] + d[0] - text_in_px[0][0]), int(s[1] - text_in_px[0][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                constants.TEXT_SIZE,
                constants.CLR_BLACK,
                4,
            )
    return final_align


def order_points(pts) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    max_width = max(int(width_a), int(width_b))
    # max_width = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

    # compute the height of the new image, which will be the
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    return warped


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def get_global_threshold(
    q_vals_orig, plot_title=None, plot_show=True, sort_in_plot=True, looseness=1
):
    PAGE_TYPE_FOR_THRESHOLD, MIN_JUMP, JUMP_DELTA = map(
        config.threshold_params.get,
        [
            "PAGE_TYPE_FOR_THRESHOLD",
            "MIN_JUMP",
            "JUMP_DELTA",
        ],
    )

    global_default_threshold = (
        constants.GLOBAL_PAGE_THRESHOLD_WHITE
        if PAGE_TYPE_FOR_THRESHOLD == "white"
        else constants.GLOBAL_PAGE_THRESHOLD_BLACK
    )
    q_vals = sorted(q_vals_orig)
    ls = (looseness + 1) // 2
    l = len(q_vals) - ls
    max1, thr1 = MIN_JUMP, global_default_threshold
    for i in range(ls, l):
        jump = q_vals[i + ls] - q_vals[i - ls]
        if jump > max1:
            max1 = jump
            thr1 = q_vals[i - ls] + jump / 2
    max2, thr2 = MIN_JUMP, global_default_threshold
    for i in range(ls, l):
        jump = q_vals[i + ls] - q_vals[i - ls]
        new_thr = q_vals[i - ls] + jump / 2
        if jump > max2 and abs(thr1 - new_thr) > JUMP_DELTA:
            max2 = jump
            thr2 = new_thr
    global_thr, j_low, j_high = thr1, thr1 - max1 // 2, thr1 + max1 // 2

    return global_thr, j_low, j_high


def get_local_threshold(
    q_vals, global_thr, no_outliers, plot_title=None, plot_show=True
):
    q_vals = sorted(q_vals)
    if len(q_vals) < 3:
        thr1 = (
            global_thr
            if np.max(q_vals) - np.min(q_vals) < config.threshold_params.MIN_GAP
            else np.mean(q_vals)
        )
    else:
        l = len(q_vals) - 1
        max1, thr1 = config.threshold_params.MIN_JUMP, 255
        for i in range(1, l):
            jump = q_vals[i + 1] - q_vals[i - 1]
            if jump > max1:
                max1 = jump
                thr1 = q_vals[i - 1] + jump / 2

        confident_jump = (
            config.threshold_params.MIN_JUMP + config.threshold_params.CONFIDENT_SURPLUS
        )

        if max1 < confident_jump:
            if no_outliers:
                thr1 = global_thr
            else:
                pass
    return thr1


class MainOperations:
    """Perform primary functions such as displaying images and reading responses"""

    image_metrics = ImageMetrics()

    def __init__(self):
        self.image_utils = ImageUtils()

    @staticmethod
    def read_response(template, image, name, save_dir=None, auto_align=False):
        try:
            img = image.copy()
            # origDim = img.shape[:2]
            img = ImageUtils.resize_util(
                img, template.dimensions[0], template.dimensions[1]
            )
            if img.max() > img.min():
                img = normalize_util(img)
            transp_layer = img.copy()
            final_marked = img.copy()

            morph = img.copy()
            ImageUtils.append_save_img(3, morph)
            if auto_align:
                morph = MainOperations.image_metrics.clahe.apply(morph)
                ImageUtils.append_save_img(3, morph)
                morph = adjust_gamma(morph, config.threshold_params.GAMMA_LOW)
                _, morph = cv2.threshold(morph, 220, 220, cv2.THRESH_TRUNC)
                morph = normalize_util(morph)
                ImageUtils.append_save_img(3, morph)
                if config.outputs.show_image_level >= 4:
                    MainOperations.show("morph1", morph, 0, 1)

            # Move them to data class if needed
            # Overlay Transparencies
            alpha = 0.65
            box_w, box_h = template.bubble_dimensions
            omr_response = {}
            multi_marked, multi_roll = 0, 0
            if config.outputs.show_image_level >= 5:
                all_c_box_vals = {"int": [], "mcq": []}
                q_nums = {"int": [], "mcq": []}
            if auto_align:
                v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
                morph_v = cv2.morphologyEx(
                    morph, cv2.MORPH_OPEN, v_kernel, iterations=3
                )
                _, morph_v = cv2.threshold(morph_v, 200, 200, cv2.THRESH_TRUNC)
                morph_v = 255 - normalize_util(morph_v)
                ImageUtils.append_save_img(3, morph_v)
                morph_thr = 60  # for Mobile images, 40 for scanned Images
                _, morph_v = cv2.threshold(morph_v, morph_thr, 255, cv2.THRESH_BINARY)
                morph_v = cv2.erode(morph_v, np.ones((5, 5), np.uint8), iterations=2)

                ImageUtils.append_save_img(3, morph_v)
                ImageUtils.append_save_img(6, morph_v)
                for q_block in template.q_blocks:
                    s, d = q_block.orig, q_block.dimensions
                    match_col, max_steps, align_stride, thk = map(
                        config.alignment_params.get,
                        [
                            "match_col",
                            "max_steps",
                            "stride",
                            "thickness",
                        ],
                    )
                    shift, steps = 0, 0
                    while steps < max_steps:
                        left_mean = np.mean(
                            morph_v[
                                s[1] : s[1] + d[1],
                                s[0] + shift - thk : -thk + s[0] + shift + match_col,
                            ]
                        )
                        right_mean = np.mean(
                            morph_v[
                                s[1] : s[1] + d[1],
                                s[0]
                                + shift
                                - match_col
                                + d[0]
                                + thk : thk
                                + s[0]
                                + shift
                                + d[0],
                            ]
                        )

                        left_shift, right_shift = left_mean > 100, right_mean > 100
                        if left_shift:
                            if right_shift:
                                break
                            else:
                                shift -= align_stride
                        else:
                            if right_shift:
                                shift += align_stride
                            else:
                                break
                        steps += 1

                    q_block.shift = shift

            final_align = None
            if config.outputs.show_image_level >= 2:
                initial_align = draw_template_layout(img, template, shifted=False)
                final_align = draw_template_layout(
                    img, template, shifted=True, draw_qvals=True
                )
                ImageUtils.append_save_img(2, initial_align)
                ImageUtils.append_save_img(2, final_align)

                if auto_align:
                    final_align = np.hstack((initial_align, final_align))
            ImageUtils.append_save_img(5, img)

            all_q_vals, all_q_strip_arrs, all_q_std_vals = [], [], []
            total_q_strip_no = 0
            for q_block in template.q_blocks:
                q_std_vals = []
                for _, qbox_pts in q_block.traverse_pts:
                    q_strip_vals = []
                    for Point in qbox_pts:
                        # shifted
                        x, y = (Point.x + q_block.shift, Point.y)
                        rect = [y, y + box_h, x, x + box_w]
                        q_strip_vals.append(
                            cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0]
                        )
                    q_std_vals.append(round(np.std(q_strip_vals), 2))
                    all_q_strip_arrs.append(q_strip_vals)
                    all_q_vals.extend(q_strip_vals)
                    total_q_strip_no += 1
                all_q_std_vals.extend(q_std_vals)

            global_std_thresh, _, _ = get_global_threshold(
                all_q_std_vals
            )  # , "Q-wise Std-dev Plot", plot_show=True, sort_in_plot=True)

            global_thr, _, _ = get_global_threshold(all_q_vals, looseness=4)
            per_omr_threshold_avg, total_q_strip_no, total_q_box_no = 0, 0, 0
            non_empty_qnos = set()
            for q_block in template.q_blocks:
                block_q_strip_no = 1
                shift = q_block.shift
                s, d = q_block.orig, q_block.dimensions
                key = q_block.key[:3]
                for _, qbox_pts in q_block.traverse_pts:
                    no_outliers = all_q_std_vals[total_q_strip_no] < global_std_thresh
                    per_q_strip_threshold = get_local_threshold(
                        all_q_strip_arrs[total_q_strip_no],
                        global_thr,
                        no_outliers,
                        "Mean Intensity Histogram for "
                        + key
                        + "."
                        + qbox_pts[0].q_no
                        + "."
                        + str(block_q_strip_no),
                        config.outputs.show_image_level >= 6,
                    )
                    per_omr_threshold_avg += per_q_strip_threshold
                    for Point in qbox_pts:
                        x, y = (Point.x + q_block.shift, Point.y)
                        boxval0 = all_q_vals[total_q_box_no]
                        detected = per_q_strip_threshold > boxval0
                        if detected:
                            cv2.rectangle(
                                final_marked,
                                (int(x + box_w / 12), int(y + box_h / 12)),
                                (
                                    int(x + box_w - box_w / 12),
                                    int(y + box_h - box_h / 12),
                                ),
                                constants.CLR_DARK_GRAY,
                                3,
                            )
                        else:
                            cv2.rectangle(
                                final_marked,
                                (int(x + box_w / 10), int(y + box_h / 10)),
                                (
                                    int(x + box_w - box_w / 10),
                                    int(y + box_h - box_h / 10),
                                ),
                                constants.CLR_GRAY,
                                -1,
                            )

                        # TODO Make this part useful! (Abstract visualizer to check status)
                        if detected:
                            q, val = Point.q_no, str(Point.val)
                            cv2.putText(
                                final_marked,
                                val,
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                constants.TEXT_SIZE,
                                (20, 20, 10),
                                int(1 + 3.5 * constants.TEXT_SIZE),
                            )
                            # Only send rolls multi-marked in the directory
                            multi_marked_l = q in omr_response
                            multi_marked = multi_marked_l or multi_marked
                            omr_response[q] = (
                                (omr_response[q] + val) if multi_marked_l else val
                            )
                            non_empty_qnos.add(q)
                            multi_roll = multi_marked_l and "Roll" in str(q)
                            # blackVals.append(boxval0)
                        # else:
                        # whiteVals.append(boxval0)

                        total_q_box_no += 1
                        # /for qbox_pts
                    # /for qStrip

                    if config.outputs.show_image_level >= 5:
                        if key in all_c_box_vals:
                            q_nums[key].append(key[:2] + "_c" + str(block_q_strip_no))
                            all_c_box_vals[key].append(
                                all_q_strip_arrs[total_q_strip_no]
                            )

                    block_q_strip_no += 1
                    total_q_strip_no += 1
            for concatQ in template.concatenations:
                for q in concatQ:
                    if q not in non_empty_qnos:
                        omr_response[q] = q_block.empty_val

            for q in template.singles:
                if q not in non_empty_qnos:
                    omr_response[q] = q_block.empty_val

            if total_q_strip_no == 0:
                logger.error(
                    "\n\t UNEXPECTED Organizer Incorrect Error: \
                    total_q_strip_no is zero! q_blocks: ",
                    template.q_blocks,
                )
                exit(21)

            per_omr_threshold_avg /= total_q_strip_no
            per_omr_threshold_avg = round(per_omr_threshold_avg, 2)
            cv2.addWeighted(
                final_marked, alpha, transp_layer, 1 - alpha, 0, final_marked
            )
            if config.outputs.show_image_level >= 5:
                pass

            if config.outputs.show_image_level >= 3 and final_align is not None:
                final_align = ImageUtils.resize_util_h(
                    final_align, int(config.dimensions.display_height)
                )
                # [final_align.shape[1],0])

            if config.outputs.save_detections and save_dir is not None:
                if multi_roll:
                    save_dir = save_dir + "_MULTI_/"
                ImageUtils.save_img(save_dir + name, final_marked)

            ImageUtils.append_save_img(2, final_marked)

            for i in range(config.outputs.save_image_level):
                ImageUtils.save_or_show_stacks(i + 1, name, save_dir)

            return omr_response, final_marked, multi_marked, multi_roll

        except Exception as e:
            raise e

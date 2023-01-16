from deepmerge import Merger

OVERRIDE_MERGER = Merger(
    [(dict, ["merge"])],
    ["override"],
    ["override"],
)

"""Shared label definitions and dataset metadata."""

DATASET_SELECTIONS = {
    "vision1": {
        "description": "Initial matched Zeiss lesion dataset assembled from batches 202952, 202997, 203011, and 203093.",
        "annotation_roots": [
            "/path/to/private_source/eyedataset/_recheck_extract",
        ],
        "selected_annotations": {
            "202952": {
                "annotation_path": "/path/to/private_source/eyedataset/_recheck_extract/202952_20250729 数据（第三批）-厦眼-蔡司_20260403104633_1_1/annotator_1标注.xlsx",
                "lesion_sheet": "检查维度",
                "check_sheet": "检查维度",
            },
            "202997": {
                "annotation_path": "/path/to/private_source/eyedataset/_recheck_extract/202997_20250929 数据（第五批）-厦眼-蔡司_20260403104604_1_1/annotator_2标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
            "203011": {
                "annotation_path": "/path/to/private_source/eyedataset/_recheck_extract/203011_20251017 数据（第六批）-厦眼-蔡司_20260403105145_1_1/annotator_2标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
            "203093": {
                "annotation_path": "/path/to/private_source/eyedataset/_recheck_extract/203093_20251205 数据（第七批）-厦眼-蔡司_20260403104535_1_1/annotator_3标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
        },
        "notes": [
            "Raw data is preserved in place and is not copied into this project.",
            "Later releases should be registered as vision2, vision3, and so on.",
            "Batch 202952 uses a different worksheet schema and should be parsed with special handling.",
            "Per-batch files under labels/ store label truth only.",
            "Paths, split membership, and source references are centralized in manifest.csv.",
        ],
    },
    "vision2": {
        "description": "Union release built from all unique vision1 samples plus newly provided 2026 Q1 platform annotations that are not already covered by vision1.",
        "annotation_roots": [
            "/path/to/private_source/eyedataset/_recheck_extract",
            "/path/to/private_source/eyedataset/_incoming_2026q1/2026-1月-3月平台标注",
        ],
        "selected_annotations": {
            "202952": {
                "annotation_path": "/path/to/private_source/eyedataset/_recheck_extract/202952_20250729 数据（第三批）-厦眼-蔡司_20260403104633_1_1/annotator_1标注.xlsx",
                "lesion_sheet": "检查维度",
                "check_sheet": "检查维度",
            },
            "202997": {
                "annotation_path": "/path/to/private_source/eyedataset/_recheck_extract/202997_20250929 数据（第五批）-厦眼-蔡司_20260403104604_1_1/annotator_2标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
            "203011": {
                "annotation_path": "/path/to/private_source/eyedataset/_incoming_2026q1/2026-1月-3月平台标注/203011_20251017 数据（第六批）-厦眼-蔡司_20260225164320_1_1/annotator_2标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
            "203093": {
                "annotation_path": "/path/to/private_source/eyedataset/_incoming_2026q1/2026-1月-3月平台标注/203093_20251205 数据（第七批）-厦眼-蔡司_20260225164225_1_1/annotator_3标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
            "203257": {
                "annotation_path": "/path/to/private_source/eyedataset/_incoming_2026q1/2026-1月-3月平台标注/203257_蔡司-视网膜脱离补充数据_20260330105644_1_1/annotator_2标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
        },
        "notes": [
            "This release is deduplicated by unique image_key rather than raw annotation rows.",
            "Lesion labels from the incoming 2026 Q1 bundle are read from 序列维度 whenever that sheet exists.",
            "The incoming 2026 Q1 bundle contains multiple annotator files with substantial overlap and label disagreement.",
            "Overlapping samples already present in vision1 are kept from vision1, while only newly covered image keys are added from the incoming bundle.",
            "Per-batch files under labels/ store label truth only.",
            "Paths, split membership, and source references are centralized in manifest.csv.",
        ],
    },
    "vision3": {
        "description": "Vision2 plus 203257 recovered retinal detachment images that previously had labels but no matched image files.",
        "annotation_roots": [
            "/path/to/private_source/eyedataset/_recheck_extract",
            "/path/to/private_source/eyedataset/_incoming_2026q1/2026-1月-3月平台标注",
            "/path/to/project/网脱",
        ],
        "selected_annotations": {
            "202952": {
                "annotation_path": "/path/to/private_source/eyedataset/_recheck_extract/202952_20250729 数据（第三批）-厦眼-蔡司_20260403104633_1_1/annotator_1标注.xlsx",
                "lesion_sheet": "检查维度",
                "check_sheet": "检查维度",
            },
            "202997": {
                "annotation_path": "/path/to/private_source/eyedataset/_recheck_extract/202997_20250929 数据（第五批）-厦眼-蔡司_20260403104604_1_1/annotator_2标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
            "203011": {
                "annotation_path": "/path/to/private_source/eyedataset/_incoming_2026q1/2026-1月-3月平台标注/203011_20251017 数据（第六批）-厦眼-蔡司_20260225164320_1_1/annotator_2标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
            "203093": {
                "annotation_path": "/path/to/private_source/eyedataset/_incoming_2026q1/2026-1月-3月平台标注/203093_20251205 数据（第七批）-厦眼-蔡司_20260225164225_1_1/annotator_3标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
            "203257": {
                "annotation_path": "/path/to/private_source/eyedataset/_incoming_2026q1/2026-1月-3月平台标注/203257_蔡司-视网膜脱离补充数据_20260330105644_1_1/annotator_2标注.xlsx",
                "lesion_sheet": "序列维度",
                "check_sheet": "检查维度",
            },
        },
        "notes": [
            "This release starts from vision2 and appends recovered 203257 images from the 网脱 supplement directory.",
            "Only previously missing 203257 samples with both labels and recovered image files are added.",
            "Existing vision2 samples keep their original sample_id values so the old image cache can be reused.",
            "Per-batch files under labels/ store label truth only.",
            "Paths, split membership, and source references are centralized in manifest.csv.",
        ],
    },
}

VISION1_SELECTED_ANNOTATIONS = DATASET_SELECTIONS["vision1"]["selected_annotations"]

LESION_LABELS = {
    1: "屈光介质混浊",
    2: "玻璃体出血",
    3: "玻璃体星状小体",
    4: "玻璃体后脱离",
    5: "其他玻璃体异常",
    6: "玻璃膜疣",
    7: "黄斑视网膜下纤维膜",
    8: "黄斑近视性萎缩斑",
    9: "黄斑地图样萎缩",
    10: "黄斑前膜",
    11: "黄斑出血",
    12: "黄斑视网膜下出血",
    13: "黄斑区浆液性视网膜脱离",
    14: "黄斑裂孔",
    15: "其他黄斑病变",
    16: "视盘侧枝循环",
    17: "视盘边界不清",
    18: "先天性视盘发育异常",
    19: "视盘大视杯",
    20: "视神经萎缩",
    21: "视盘新生血管",
    22: "高度近视视盘萎缩弧",
    23: "高度近视视盘萎缩环",
    24: "视网膜脱离",
    25: "视网膜出血象限性",
    26: "视网膜下出血",
    27: "视网膜前出血",
    28: "视网膜裂孔",
    29: "视网膜纤维膜",
    30: "视网膜脉络膜占位",
    31: "视网膜激光斑",
    32: "全视网膜光凝",
    33: "单象限性视网膜光凝斑",
    34: "视网膜骨细胞样色素改变",
    35: "豹纹状眼底",
    36: "晚霞状眼底",
    37: "出血点出血斑",
    38: "硬性渗出",
    39: "棉绒斑",
    40: "视网膜新生血管",
    41: "视网膜陈旧色素病灶",
    42: "视网膜周边变性区",
    43: "其他视网膜病变",
    44: "视网膜动脉阻塞",
    45: "视网膜血管白线",
    46: "视网膜动脉硬化",
    47: "视网膜血管鞘-视网膜血管炎",
}

DIAGNOSIS_LABELS = {
    1: "正常眼底",
    2: "视网膜色素变性",
    3: "视网膜中央静脉阻塞",
    4: "视网膜分支静脉阻塞",
    5: "结晶样视网膜营养不良",
    6: "FEVR",
    7: "先天性脉络膜缺损",
    8: "糖尿病视网膜病变NPDR",
    9: "糖尿病视网膜病变PDR",
    10: "湿性年龄相关黄斑变性",
    11: "干性年龄相关黄斑变性",
    12: "病理性高度近视",
    13: "VKH",
}

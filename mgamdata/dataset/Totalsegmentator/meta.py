import os


DATA_ROOT = os.environ['Totalsegmentator_data_root']
DATA_ROOT_SLICE2D_TIFF = os.path.join(DATA_ROOT, 'slice2D_tiff')
META_CSV_PATH = os.path.join(DATA_ROOT, 'meta_v2.csv')

CLASS_INDEX_MAP = {
    'background': 0,
    'adrenal_gland_left': 1,
    'adrenal_gland_right': 2,
    'aorta': 3,
    'atrial_appendage_left': 4,
    'autochthon_left': 5,
    'autochthon_right': 6,
    'brachiocephalic_trunk': 7,
    'brachiocephalic_vein_left': 8,
    'brachiocephalic_vein_right': 9,
    'brain': 10,
    'clavicula_left': 11,
    'clavicula_right': 12,
    'colon': 13,
    'common_carotid_artery_left': 14,
    'common_carotid_artery_right': 15,
    'costal_cartilages': 16,
    'ct': 17,
    'duodenum': 18,
    'esophagus': 19,
    'femur_left': 20,
    'femur_right': 21,
    'gallbladder': 22,
    'gluteus_maximus_left': 23,
    'gluteus_maximus_right': 24,
    'gluteus_medius_left': 25,
    'gluteus_medius_right': 26,
    'gluteus_minimus_left': 27,
    'gluteus_minimus_right': 28,
    'heart': 29,
    'hip_left': 30,
    'hip_right': 31,
    'humerus_left': 32,
    'humerus_right': 33,
    'iliac_artery_left': 34,
    'iliac_artery_right': 35,
    'iliac_vena_left': 36,
    'iliac_vena_right': 37,
    'iliopsoas_left': 38,
    'iliopsoas_right': 39,
    'inferior_vena_cava': 40,
    'kidney_cyst_left': 41,
    'kidney_cyst_right': 42,
    'kidney_left': 43,
    'kidney_right': 44,
    'liver': 45,
    'lung_lower_lobe_left': 46,
    'lung_lower_lobe_right': 47,
    'lung_middle_lobe_right': 48,
    'lung_upper_lobe_left': 49,
    'lung_upper_lobe_right': 50,
    'pancreas': 51,
    'portal_vein_and_splenic_vein': 52,
    'prostate': 53,
    'pulmonary_vein': 54,
    'rib_left_1': 55,
    'rib_left_10': 56,
    'rib_left_11': 57,
    'rib_left_12': 58,
    'rib_left_2': 59,
    'rib_left_3': 60,
    'rib_left_4': 61,
    'rib_left_5': 62,
    'rib_left_6': 63,
    'rib_left_7': 64,
    'rib_left_8': 65,
    'rib_left_9': 66,
    'rib_right_1': 67,
    'rib_right_10': 68,
    'rib_right_11': 69,
    'rib_right_12': 70,
    'rib_right_2': 71,
    'rib_right_3': 72,
    'rib_right_4': 73,
    'rib_right_5': 74,
    'rib_right_6': 75,
    'rib_right_7': 76,
    'rib_right_8': 77,
    'rib_right_9': 78,
    'sacrum': 79,
    'scapula_left': 80,
    'scapula_right': 81,
    'skull': 82,
    'small_bowel': 83,
    'spinal_cord': 84,
    'spleen': 85,
    'sternum': 86,
    'stomach': 87,
    'subclavian_artery_left': 88,
    'subclavian_artery_right': 89,
    'superior_vena_cava': 90,
    'thyroid_gland': 91,
    'trachea': 92,
    'urinary_bladder': 93,
    'vertebrae_C1': 94,
    'vertebrae_C2': 95,
    'vertebrae_C3': 96,
    'vertebrae_C4': 97,
    'vertebrae_C5': 98,
    'vertebrae_C6': 99,
    'vertebrae_C7': 100,
    'vertebrae_L1': 101,
    'vertebrae_L2': 102,
    'vertebrae_L3': 103,
    'vertebrae_L4': 104,
    'vertebrae_L5': 105,
    'vertebrae_S1': 106,
    'vertebrae_T1': 107,
    'vertebrae_T10': 108,
    'vertebrae_T11': 109,
    'vertebrae_T12': 110,
    'vertebrae_T2': 111,
    'vertebrae_T3': 112,
    'vertebrae_T4': 113,
    'vertebrae_T5': 114,
    'vertebrae_T6': 115,
    'vertebrae_T7': 116,
    'vertebrae_T8': 117,
    'vertebrae_T9': 118,
}

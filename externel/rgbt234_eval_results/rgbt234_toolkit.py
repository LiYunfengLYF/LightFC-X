from rgbt import RGBT234

rgbt234 = RGBT234()
tracker_name = r'lightfcx'

# Register your tracker
rgbt234(
    tracker_name=tracker_name,
    result_path=fr'E:\code\LightFCX-review\output\test\tracking_results\RGBT_baseline_ep15_update\rgbt234',
    bbox_type="ltwh")

# result_path = fr'E:\code\rgbt-light\output\test\tracking_results\{tracker_name}\{yaml_name}\rgbt234'
# result_path = r'E:\winfred\dev\SMAT-main\output\test\tracking_results\mobilevitv2_track\mobilevitv2_256_128x1_ep300'


# rgbt234(
#     tracker_name=tracker_name,
#     result_path=result_path,
#     bbox_type="ltwh")

# Evaluate single tracker
apf_pr, _ = rgbt234.MPR(f"{tracker_name}")
print(apf_pr)

# Evaluate single tracker
apf_pr, _ = rgbt234.MSR(f"{tracker_name}")
print(apf_pr)

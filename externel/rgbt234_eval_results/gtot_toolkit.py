from rgbt import GTOT


gtot = GTOT()
tracker_name = r'lightfc'
yaml_name = r'lightfc_vit'

# Register your tracker
# rgbt234(
#     tracker_name="rgbt_light",
#     result_path=fr'E:\code\rgbt-light\output\test\tracking_results\rgbter_light\{yaml_name}\rgbt234',
#     bbox_type="ltwh")

# result_path = fr'E:\code\rgbt-light\output\test\tracking_results\{tracker_name}\{yaml_name}\rgbt234'
# result_path = r'E:\code\rgbt-light\output\test\tracking_results\rgbter_light\baseline_updatev1_cropath_linear2conv1_update_direct\gtot'
result_path =r'E:\winfred\dev\SMAT-main\output\test\tracking_results\mobilevitv2_track\mobilevitv2_256_128x1_ep300'

gtot(
    tracker_name=tracker_name,
    result_path=result_path,
   )

# Evaluate single tracker
apf_pr, _ = gtot.MPR(f"{tracker_name}")
print(apf_pr)

# Evaluate single tracker
apf_pr, _ = gtot.MSR(f"{tracker_name}")
print(apf_pr)

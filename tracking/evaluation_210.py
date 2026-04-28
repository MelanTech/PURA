from rgbt import RGBT210

rgbt210 = RGBT210()

# Register your tracker
rgbt210(
    tracker_name="Ours",
    result_path="/data/szk/Project/TBSI_TU_0509/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/rgbt210/",
    bbox_type="ltwh")

# Evaluate multiple trackers
pr_dict = rgbt210.PR()
print(pr_dict["Ours"][0])

sr_dict = rgbt210.SR()
print(sr_dict["Ours"][0])

# Draw a curve plot.
# rgbt210.draw_plot(metric_fun=rgbt210.PR)
# rgbt210.draw_plot(metric_fun=rgbt210.SR)

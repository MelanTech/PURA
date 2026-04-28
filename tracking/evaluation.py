from rgbt import LasHeR

lasher = LasHeR()

"""
LasHeR have 3 benchmarks: PR, NPR, SR
"""

# Register your tracker
lasher(
    tracker_name="Ours",
    result_path="/data/szk/Project/TBSI_TU_0509/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/lasher/",
    bbox_type="ltwh")

# Evaluate multiple trackers
pr_dict = lasher.PR()
print(pr_dict["Ours"][0])

npr_dict = lasher.NPR()
print(npr_dict["Ours"][0])

sr_dict = lasher.SR()
print(sr_dict["Ours"][0])

# lasher.draw_plot(metric_fun=lasher.PR)
# lasher.draw_plot(metric_fun=lasher.NPR)
# lasher.draw_plot(metric_fun=lasher.SR)

# lasher.draw_attributeRadar(metric_fun=lasher.PR)
# lasher.draw_attributeRadar(metric_fun=lasher.SR)

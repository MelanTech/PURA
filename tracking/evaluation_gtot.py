from rgbt import GTOT
from rgbt.utils import RGBT_start, RGBT_end

RGBT_start()
gtot = GTOT()

# Register your tracker
gtot(
    tracker_name="Ours",
    result_path="/data/szk/Project/TBSI_TU_0509/output.new.gtot.pura/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/",
    bbox_type="ltwh",
    prefix="")

pr_dict = gtot.MPR()
sr_dict = gtot.MSR()

print(pr_dict["Ours"][0])
print(sr_dict["Ours"][0])

# gtot.draw_plot(metric_fun=gtot.MPR)
# gtot.draw_plot(metric_fun=gtot.MSR)

RGBT_end()

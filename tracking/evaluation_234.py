from rgbt import RGBT234

rgbt234 = RGBT234()

trackers = [
    ('Ours', '/data1/szk/Project/PURA/output/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/rgbt234/', 'ltwh'),
    # ('Ours', './output/RGBT234/Ours', 'ltwh'),
    # ('ADRNet', './output/RGBT234/ADRNet', 'ltwh'),
    # ('APFNet', './output/RGBT234/APFNet', 'ltwh'),
    # ('HMFT', './output/RGBT234/HMFT', 'ltwh'),
    # ('MPLT', './output/RGBT234/MPLT', 'ltwh'),
    # ('TBSI', './output/RGBT234/TBSI', 'ltwh'),
    # ('ViPT', './output/RGBT234/ViPT', 'ltwh'),
]

for name, path, bbox_type in trackers:
    rgbt234(
        tracker_name=name,
        result_path=path,
        bbox_type=bbox_type,
        prefix='')

# Evaluate multiple trackers
pr_dict = rgbt234.MPR()
print(pr_dict["Ours"][0])

sr_dict = rgbt234.MSR()
print(sr_dict["Ours"][0])

# Draw a curve plot.
# rgbt234.draw_plot(metric_fun=rgbt234.MPR)
# rgbt234.draw_plot(metric_fun=rgbt234.MSR)

# rgbt234.draw_attributeRadar(metric_fun=rgbt234.MPR)
# rgbt234.draw_attributeRadar(metric_fun=rgbt234.MSR)

from got10k.experiments import ExperimentGOT10k

# report_files = ['/home/etvuz/project/siam_rcnn/experiments/got10k/ao.json']
report_files = ['/home/etvuz/project3/siamrcnn2/results/got10k_v21/100000/ao.json']
tracker_names = ['SiamFCv2', 'GOTURN', 'CCOT', 'MDNet']

# setup experiment and plot curves
experiment = ExperimentGOT10k('/data/zhbli/Dataset/got10k', subset='test')
experiment.plot_curves(report_files, tracker_names)
import numpy as np
import modisco
import modisco.visualization
from modisco.visualization import viz_sequence
import pandas as pd
import scipy.special as sp
import h5py
import modisco.util

# load processed importance scores
head_proc = pd.read_csv('../insilico/proc_insil_head_logit.csv')
testis_proc = pd.read_csv('../insilico/proc_insil_testis_logit.csv')

# convert nuc to index
nuc2idx = {'a': 0,
           'c': 1,
           'g': 2,
           't': 3}


def get_importances(example_proc, nuc2idx):
    example_proc = example_proc.reset_index()
    hyp = np.zeros([example_proc.shape[0], 4])
    obs = np.zeros([example_proc.shape[0], 4])
    onehot = np.zeros([example_proc.shape[0], 4])
    for index, row in example_proc.iterrows():
        mean_logit = np.mean(sp.logit(row[['a', 't', 'c', 'g']].to_list()))
        hyp[index, 0] = sp.logit(row['a']) - mean_logit
        hyp[index, 1] = sp.logit(row['c']) - mean_logit
        hyp[index, 2] = sp.logit(row['g']) - mean_logit
        hyp[index, 3] = sp.logit(row['t']) - mean_logit

        if row.ref in ['a', 't', 'c', 'g']:
            refidx = nuc2idx[row.ref]
            obs[index, refidx] = sp.logit(row[row.ref]) - mean_logit
            onehot[index, refidx] = 1.0

    return hyp, obs, onehot


seqlen = 1000
l = head_proc.sample_ind.max() + 1

# head importances
hyp_imp_head = np.zeros([l, seqlen, 4])
obs_imp_head = np.zeros([l, seqlen, 4])
# testis importances
hyp_imp_testis = np.zeros([l, seqlen, 4])
obs_imp_testis = np.zeros([l, seqlen, 4])

onehot_ref = np.zeros([l, seqlen, 4])
for i in range(l):
    hyp_imp_head[i, :, :], obs_imp_head[i, :, :], onehot_ref[i, :, :] = get_importances(
        head_proc[head_proc.sample_ind == i], nuc2idx)

    hyp_imp_testis[i, :, :], obs_imp_testis[i, :, :], _ = get_importances(
        testis_proc[testis_proc.sample_ind == i], nuc2idx)


hyp_imp_dict = {'head': hyp_imp_head, 'testis': hyp_imp_testis}
obs_imp_dict = {'head': obs_imp_head, 'testis': obs_imp_testis}


null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                    #Slight modifications from the default settings
                    sliding_window_size=10,
                    flank_size=5,
                    target_seqlet_fdr=0.1,
                    seqlets_to_patterns_factory=
                     modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                        #Note: as of version 0.5.6.0, it's possible to use the results of a motif discovery
                        # software like MEME to improve the TF-MoDISco clustering. To use the meme-based
                        # initialization, you would specify the initclusterer_factory as shown in the
                        # commented-out code below:
                        #initclusterer_factory=modisco.clusterinit.memeinit.MemeInitClustererFactory(
                        #    meme_command="meme", base_outdir="meme_out",
                        #    max_num_seqlets_to_use=10000, nmotifs=10, n_jobs=1),
                        #use_pynnd=True can be used for faster nn comp at coarse grained step
                        # (it will use pynndescent), but note that pynndescent may crash
                        #use_pynnd=True,
                        trim_to_window_size=10,
                        initial_flank_to_add=0,
                        final_flank_to_add=0,
                        final_min_cluster_size=30,
                        n_cores=24)
                )(
                 task_names=["head", "testis"],
                 contrib_scores=obs_imp_dict,
                 hypothetical_contribs=hyp_imp_dict,
                 one_hot=onehot_ref,
                 null_per_pos_scores=null_per_pos_scores)



grp = h5py.File("../tf_modisco/multitask_results.hdf5", "w")
tfmodisco_results.save_hdf5(grp)
grp.close()

np.save("../tf_modisco/onehot_ref", onehot_ref)

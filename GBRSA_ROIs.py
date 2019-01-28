
# coding: utf-8

# In[1]:


import numpy as np
import scipy
from brainiak.reprsimil.brsa import GBRSA
import brainiak.utils.utils as utils
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(10)


# In[2]:


# read FSL style event files
dir_design = '/home/tselab/studies/vr1_data/code/BIDS/brainiak/design/'
path_designs = [dir_design + 'e' + str(i+1) for i in range(8)]


# In[3]:


# generate the design matrix
TR_per_run = 189
nruns = 6
scan_duration = TR_per_run * nruns
tr = 2
design = utils.gen_design(path_designs, scan_duration*tr, tr)


# In[4]:


# visulization of design
fig = plt.figure(num=None, figsize=(30, 8),
                 facecolor='w', edgecolor='k')
plt.plot(design)
plt.ylim([-.5, 3])
plt.title('hypothetic fMRI response time courses '
          'of all conditions for one subject\n'
         '(design matrix)')
plt.xlabel('time')
plt.show()


# In[5]:


# load fmri data with ROIs
# (it's tricky with pyMVPA..since I use Python 3... so I did data masking in matlab with CoSMoMVPA)
matdata = scipy.io.loadmat('roi_data.mat', squeeze_me = True)


# In[6]:


# split to mask/session
ppa_s1 = list(matdata['data1'][0])
rsc_s1 = list(matdata['data1'][1])
opa_s1 = list(matdata['data1'][2])
hippo_s1 = list(matdata['data1'][3])
v1_s1 = list(matdata['data1'][4])

ppa_s2 = list(matdata['data2'][0])
rsc_s2 = list(matdata['data2'][1])
opa_s2 = list(matdata['data2'][2])
hippo_s2 = list(matdata['data2'][3])
v1_s2 = list(matdata['data2'][4])


# In[7]:


def gbrsa_fit(ldata, ldesign, lonset):
    # function to fit GBRSA and plot resulting RSA
    gbrsa = GBRSA()
    gbrsa.fit(X = ldata, design = ldesign, scan_onsets = lonset)
    
    return gbrsa

def rsa_plot(ldesign, model):
    nconds = ldesign[0].shape[1]
    fig = plt.figure(num=None, figsize=(4, 4), dpi=100)
    plt.pcolor(model.C_)
    plt.xlim([0, nconds])
    plt.ylim([0, nconds])
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title('Estimated correlation structure\n shared between voxels\n'
             'This constitutes the output of Bayesian RSA\n')
    plt.show()

    fig = plt.figure(num=None, figsize=(4, 4), dpi=100)
    plt.pcolor(model.U_)
    plt.xlim([0, nconds])
    plt.ylim([0, nconds])
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title('Estimated covariance structure\n shared between voxels\n')
    plt.show()


# In[8]:


## for v1
nsubjs = 21
designs = [design.copy() for _ in range(nsubjs)]
scan_onsets = [np.array(range(0, scan_duration, TR_per_run)) for _ in range(nsubjs)]


# In[12]:


mapmat = scipy.io.loadmat('remaps.mat', squeeze_me = True)
maps = mapmat['maps']
designs = [d[:, m-1] for d, m in zip(designs, maps)]


# In[9]:


# individual correlations
def cal_t(data, designs, onsets, contrast):
    con = []
    for d, design, onset in zip(data, designs, onsets):
        model = gbrsa_fit(d, design, onset)
        con.append(np.sum(np.multiply(contrast, model.C_)))
    return con


# In[10]:


r = np.array([[0,  6, -1, -1, -1, -1, -1, -1],
     [6,  0, -1, -1, -1, -1, -1, -1],
     [-1, -1,  0,  6, -1, -1, -1, -1],
     [-1, -1,  6,  0, -1, -1, -1, -1],
     [-1, -1, -1, -1,  0,  6, -1, -1],
     [-1, -1, -1, -1,  6,  0, -1, -1],
     [-1, -1, -1, -1, -1, -1,  0,  6],
     [-1, -1, -1, -1, -1, -1,  6,  0]]);


# In[13]:


ppa_s2_con = cal_t(ppa_s2, designs, scan_onsets, r)


# In[14]:


ppa_s1_con = cal_t(ppa_s1, designs, scan_onsets, r)


# In[16]:


rsc_s1_con = cal_t(rsc_s1, designs, scan_onsets, r)


# In[17]:


rsc_s2_con = cal_t(rsc_s2, designs, scan_onsets, r)


# In[18]:


opa_s1_con = cal_t(opa_s1, designs, scan_onsets, r)


# In[19]:


opa_s2_con = cal_t(opa_s2, designs, scan_onsets, r)


# In[20]:


hip_s1_con = cal_t(hippo_s1, designs, scan_onsets, r)


# In[21]:


hip_s2_con = cal_t(hippo_s2, designs, scan_onsets, r)


# In[24]:


[t, p] = scipy.stats.ttest_1samp(ppa_s1_con, 0)
print(t)
print(p)


# In[25]:


[t, p] = scipy.stats.ttest_1samp(ppa_s2_con, 0)
print(t)
print(p)


# In[26]:


[t, p] = scipy.stats.ttest_1samp(rsc_s1_con, 0)
print(t)
print(p)


# In[27]:


[t, p] = scipy.stats.ttest_1samp(rsc_s2_con, 0)
print(t)
print(p)


# In[28]:


[t, p] = scipy.stats.ttest_1samp(opa_s1_con, 0)
print(t)
print(p)


# In[30]:


[t, p] = scipy.stats.ttest_1samp(opa_s2_con, 0)
print(t)
print(p)


# In[32]:


[t, p] = scipy.stats.ttest_1samp(hip_s1_con, 0)
print(t)
print(p)


# In[33]:


[t, p] = scipy.stats.ttest_1samp(hip_s2_con, 0)
print(t)
print(p)


# In[ ]:


# def rsa(Y, designs):
#     # plot GLM/beta-based RSA
#     n_subj = 21
#     Y = ppa_s2
#     n_C = 8
#     sum_point_corr = np.zeros((n_C, n_C))
#     sum_point_cov = np.zeros((n_C, n_C))
#     betas_point = [None] * n_subj
#     for subj in range(n_subj):
#         regressor = np.insert(designs[subj],
#                               0, 1, axis=1)
#         betas_point[subj] = np.linalg.lstsq(regressor, Y[subj])[0]
#         point_corr = np.corrcoef(betas_point[subj][1:, :])
#         point_cov = np.cov(betas_point[subj][1:, :])
#         sum_point_corr += point_corr
#         sum_point_cov += point_cov
#         if subj == 0:
#             fig = plt.figure(num=None, figsize=(4, 4), dpi=100)
#             plt.pcolor(point_corr, vmin=-0.1, vmax=1)
#             plt.xlim([0, 8])
#             plt.ylim([0, 8])
#             plt.colorbar()
#             ax = plt.gca()
#             ax.set_aspect(1)
#             plt.title('Correlation structure estimated\n'
#                      'based on point estimates of betas\n'
#                      'for subject {}'.format(subj))
#             plt.show()
    
#             fig = plt.figure(num=None, figsize=(4, 4), dpi=100)
#             plt.pcolor(point_cov)
#             plt.xlim([0, 8])
#             plt.ylim([0, 8])
#             plt.colorbar()
#             ax = plt.gca()
#             ax.set_aspect(1)
#             plt.title('Covariance structure of\n'
#                      'point estimates of betas\n'
#                      'for subject {}'.format(subj))
#             plt.show()

#     fig = plt.figure(num=None, figsize=(4, 4), dpi=100)
#     plt.pcolor(sum_point_corr / n_subj, vmin=-0.1, vmax=1)
#     plt.xlim([0, 8])
#     plt.ylim([0, 8])
#     plt.colorbar()
#     ax = plt.gca()
#     ax.set_aspect(1)
#     plt.title('Correlation structure estimated\n'
#          'based on point estimates of betas\n'
#          'averaged over subjects')
#     plt.show()

#     fig = plt.figure(num=None, figsize=(4, 4), dpi=100)
#     plt.pcolor(sum_point_cov / n_subj)
#     plt.xlim([0, 8])
#     plt.ylim([0, 8])
#     plt.colorbar()
#     ax = plt.gca()
#     ax.set_aspect(1)
#     plt.title('Covariance structure of\n'
#              'point estimates of betas\n'
#              'averaged over subjects')
#     plt.show()


# In[ ]:


# r = np.array([[ 7, -1, -1, -1, -1, -1, -1, -1],
#               [-1,  7, -1, -1, -1, -1, -1, -1],
#               [-1, -1,  7, -1, -1, -1, -1, -1],
#               [-1, -1, -1,  7, -1, -1, -1, -1],
#               [-1, -1, -1, -1,  7, -1, -1, -1],
#               [-1, -1, -1, -1, -1,  7, -1, -1],
#               [-1, -1, -1, -1, -1, -1,  7, -1],
#               [-1, -1, -1, -1, -1, -1, -1,  7]]);


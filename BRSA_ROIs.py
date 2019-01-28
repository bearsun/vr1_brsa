
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import nibabel
from brainiak.reprsimil.brsa import BRSA
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


# In[98]:


matsubs = scipy.io.loadmat('subjs.mat', squeeze_me = True)
subs = matsubs['subjs']


# In[99]:


dirsubs = ['/home/tselab/studies/vr1_data/BIDS/fmriprep_mni/fsl/' + s + '/roi/' for s in subs]


# In[7]:


def read_coords(file):
    mask = nibabel.load(file)
    I, J, K = mask.shape
    all_coords = np.zeros((I, J, K, 3)) 
    all_coords[...,0] = np.arange(I)[:, np.newaxis, np.newaxis] 
    all_coords[...,1] = np.arange(J)[np.newaxis, :, np.newaxis] 
    all_coords[...,2] = np.arange(K)[np.newaxis, np.newaxis, :]
    ROI_coords = nibabel.affines.apply_affine(mask.affine, all_coords[mask.get_data().astype(bool)])
    return ROI_coords


# In[87]:


# load mask coordinates
masks = ['v1', 'ppa', 'rsc', 'opa', 'hippo']
pathmasks = []
mcoords = []
for m in masks:
    allmasks = []
    allcoords = []
    for d in dirsubs:
        allmasks.append(d + m + '.nii.gz')
        allcoords.append(read_coords(submasks[-1]))
    pathmasks.append(allmasks)
    mcoords.append(allcoords)


# In[84]:


# load fmri data with ROIs
# (it's tricky with pyMVPA..since I use Python 3... so I did data masking in matlab with CoSMoMVPA)
matdata = scipy.io.loadmat('roi_data.mat', squeeze_me = True)


# In[85]:


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


# In[18]:


# # calculate intensity
# def cal_inten(data):
#     inten = []
#     for d in data:
#         inten.append(d.mean(axis = 0))
#     return inten


# In[59]:


# individual correlations
def brsa_fit(ldata, ldesign, lonset, lcoords):
    # function to fit GBRSA and plot resulting RSA
    linten = ldata.mean(axis = 0)
    brsa = BRSA()
    brsa.fit(X = ldata, design = ldesign, scan_onsets = lonset, coords = lcoords, inten = linten)
    return brsa

def cal_t(data, designs, onsets, coords, contrast):
    con = []
    for d, design, onset, coord in zip(data, designs, onsets, coords):
        model = brsa_fit(d, design, onset, coord)
        con.append(np.sum(np.multiply(contrast, model.C_)))
    return con


# In[44]:


nsubjs = 21
designs = [design.copy() for _ in range(nsubjs)]
scan_onsets = [np.array(range(0, scan_duration, TR_per_run)) for _ in range(nsubjs)]


# In[45]:


mapmat = scipy.io.loadmat('remaps.mat', squeeze_me = True)
maps = mapmat['maps']


# In[48]:


designs = [d[:, m-1] for d, m in zip(designs, maps)]


# In[64]:


r = np.array([[0,  6, -1, -1, -1, -1, -1, -1],
     [6,  0, -1, -1, -1, -1, -1, -1],
     [-1, -1,  0,  6, -1, -1, -1, -1],
     [-1, -1,  6,  0, -1, -1, -1, -1],
     [-1, -1, -1, -1,  0,  6, -1, -1],
     [-1, -1, -1, -1,  6,  0, -1, -1],
     [-1, -1, -1, -1, -1, -1,  0,  6],
     [-1, -1, -1, -1, -1, -1,  6,  0]]);
ppa_t1 = cal_t(ppa_s1, designs, scan_onsets, mcoords[1], r)


# In[65]:


ppa_t2 = cal_t(ppa_s2, designs, scan_onsets, mcoords[1], r)


# In[66]:


rsc_t1 = cal_t(rsc_s1, designs, scan_onsets, mcoords[2], r)


# In[67]:


rsc_t2 = cal_t(rsc_s2, designs, scan_onsets, mcoords[2], r)


# In[68]:


opa_t1 = cal_t(opa_s1, designs, scan_onsets, mcoords[3], r)


# In[69]:


opa_t2 = cal_t(opa_s2, designs, scan_onsets, mcoords[3], r)


# In[88]:


hip_t1 = cal_t(hippo_s1, designs, scan_onsets, mcoords[4], r)


# In[89]:


hip_t2 = cal_t(hippo_s2, designs, scan_onsets, mcoords[4], r)


# In[74]:


[t, p] = scipy.stats.ttest_1samp(ppa_t1, 0)
print(t)
print(p)


# In[75]:


[t, p] = scipy.stats.ttest_1samp(ppa_t2, 0)
print(t)
print(p)


# In[76]:


[t, p] = scipy.stats.ttest_1samp(rsc_t1, 0)
print(t)
print(p)


# In[77]:


[t, p] = scipy.stats.ttest_1samp(rsc_t2, 0)
print(t)
print(p)


# In[79]:


[t, p] = scipy.stats.ttest_1samp(opa_t1, 0)
print(t)
print(p)


# In[80]:


[t, p] = scipy.stats.ttest_1samp(opa_t2, 0)
print(t)
print(p)


# In[90]:


[t, p] = scipy.stats.ttest_1samp(hip_t1, 0)
print(t)
print(p)


# In[91]:


[t, p] = scipy.stats.ttest_1samp(hip_t2, 0)
print(t)
print(p)


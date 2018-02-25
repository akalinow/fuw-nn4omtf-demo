# coding: utf-8
"""
    Copyright(C) 2018 Jacek Łysiak

    Test performance of OMTFInputPipe vs batch_size parameter
"""

import tensorflow as tf
import nn4omtf as no
import numpy as np
import time
import matplotlib.pyplot as plt
from nn4omtf.network import PIPE_MAPPING_TYPE



def set_pipe(batch, mapping_type=0):
    pipe = no.network.OMTFInputPipe(dataset=dataset,
                                name='train', 
                                hits_type=no.dataset.const.HITS_TYPE.REDUCED, 
                                batch_size=batch, 
                                out_class_bins=[5, 10, 50, 100, 500],
                                mapping_type=mapping_type)
    return pipe


# In[170]:


def measure(pipe, N=1):
    runs = []
    for _ in range(N):
        with tf.Session() as sess:
            pipe.initialize(sess)
            cnt = 0
            b = 0
            start = time.time()
            while True:
                x , _, _ = pipe.fetch()
                if x is None:
                    break
                cnt += x.shape[0]
                b += 1
            end = time.time()
            print(end-start, cnt, b)
            runs.append([end-start, cnt, b])
    return runs


# In[85]:


data = []

labels = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
times = [[t[0] for t in d[1]] for d in data]
events = [[t[1] for t in d[1]] for d in data]
batches = [[t[2] for t in d[1]] for d in data]
t_per_batch = [[ t/b for b, t in zip(batch, time)] for batch, time in zip(batches, times)]
ev_per_sec = [[ b/t for b, t in zip(event, time)] for event, time in zip(events, times)]


# In[177]:


ev_per_sec


# In[115]:


np.std(times,axis=1)


# In[191]:


fig, ax = plt.subplots(1, figsize=(9,5.5))
ax.errorbar(labels, np.mean(ev_per_sec,axis=1), yerr=np.std(ev_per_sec,axis=1), fmt='o')
# ax.set_title('Czas wczytywania całego zbioru', size=20)
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xticks(labels)
ax.set_xticklabels(labels, size=14)
# yt = [5, 10, 20, 50]
# ax.set_yticklabels(yt, size=14)
# ax.set_yticks(yt)
ax.tick_params(axis='y', labelsize=14)
ax.set_xlabel("Rozmiar paczki", size=15)
ax.set_ylabel("Odczytane przykłady (1/s)", size=15)
fig.tight_layout()
fig.savefig("4-img-ev-per-sec-read-time.png")


# In[166]:


fig, ax = plt.subplots(1, figsize=(9,5.5))
ax.errorbar(labels, np.mean(t_per_batch,axis=1), yerr=np.std(t_per_batch,axis=1), fmt='o', c='r')
# ax.set_title('Czas odczytu paczki', size=20)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(labels)
ax.set_xticklabels(labels, size=14)
# yt = [5, 10, 20, 50]
# ax.set_yticklabels(yt, size=14)
# ax.set_yticks(yt)
ax.tick_params(axis='y', labelsize=14)
ax.set_xlabel("Rozmiar paczki", size=15)
ax.set_ylabel("czas (s)", size=15)
fig.tight_layout()
fig.savefig("4-img-single-batch-read-time.png")


desc = """Test your env and select optimal batch_size 
for reading examples by OMTFInputPipe.
"""

def run(FLAGS):
    N = FLAGS.N
    batch_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    dataset = no.dataset.OMTFDataset.load(path=FLAGS.dataset)
    vp = lambda x: print(x) if FLAGS.v else None

    p = set_pipe(size)
    d = measure(p, N=N)
    data.append([size, d])
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-v", action="store_true", help="Be verbose!")
    parser.add_argument("-N", type=int, default=10, help="# of reps for each batch_size value")
    parser.add_argument("dataset", metavar="path", help="Path to test dataset")

    FLAGS, unparsed = parser.parse_known_args()

    run(FLAGS)


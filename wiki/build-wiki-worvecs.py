import sys, time, gzip
sys.path.append("..")
from worvecs import worvecs

t0 = time.time()
print('loading text...')
sentences = []
with gzip.open('./data/abstract.txt.gz', 'rt', encoding='utf8') as f:
    for l in f:
        sentences.append(l.strip().split(' '))

print('building worvecs...')
model = worvecs(sentences)

print('saving...')
model.save('./data/wiki-worvecs.txt.gz')

print('finished: %d min' % ((time.time()-t0)/60))

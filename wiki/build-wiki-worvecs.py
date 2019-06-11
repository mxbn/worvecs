import sys, time, gzip, logging
sys.path.append("..")
from worvecs import worvecs

if __name__ == '__main__':
    t0 = time.time()
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', \
                        level=logging.INFO)
    logging.info('loading text...')
    sentences = []
    with gzip.open('./data/abstract.txt.gz', 'rt', encoding='utf8') as f:
        for l in f:
            sentences.append(l.strip().split(' '))
    logging.info('building worvecs...')
    model = worvecs(sentences, verbose=1)
    logging.info('saving...')
    model.save('./data/wiki-worvecs.txt.gz')
    logging.info('finished: %d min' % ((time.time()-t0)/60))

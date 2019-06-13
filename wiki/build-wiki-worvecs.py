import sys, time, gzip, logging
sys.path.append("..")
from worvecs import worvecs

if __name__ == '__main__':
    t0 = time.time()
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', \
                        level=logging.INFO)

    logging.info('loading text...')
    sentences = []
    with gzip.open('./data/abstract.txt.gz', 'rt', encoding='utf-8', errors='replace') as f:
        for l in f:
            sentences.append(l.strip().split(' '))
    logging.info('loaded %d sentences' % len(sentences))

    logging.info('building worvecs with encoding 0...')
    with worvecs(sentences, encoding=0, verbose=1) as model0:
        logging.info('saving...')
        model0.save('./data/wiki-worvecs-0.txt.gz')

    logging.info('building worvecs with encoding 1...')
    with worvecs(sentences, encoding=1, verbose=1) as model1:
        logging.info('saving...')
        model1.save('./data/wiki-worvecs-1.txt.gz')
    
    logging.info('finished: %d min' % ((time.time()-t0)/60))
